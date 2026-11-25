use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use plait_core::OpId;
use plait_core::serving::{GroupPhase, WorkerGrant};
use plait_core::tensor::KPartial;
use plait_net::tensor::{MAX_PARTIAL_BYTES, PartialListener, partial_size};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinSet;

use super::{
    ControlPeers, DeviceSession, Digest, GroupControl, NativeW4A16, Sha256, W4Decision, W4Device,
    W4Partition, W4Scope, W4WeightInput, activation_format, check_expiry, validate_decision,
};
const MAX_PUBLISHED: usize = 8;
const MAX_PULLS: usize = 8;
const PULL_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Deserialize)]
pub(in crate::tensor_cli) struct WorkerW4 {
    #[serde(flatten)]
    device: W4Device,
    grant: WorkerGrant,
    k_range: [u32; 2],
    output_columns: u32,
    listen: SocketAddr,
    advertise: Option<SocketAddr>,
    device_lock: Option<PathBuf>,
    control: ControlPeers,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "request", rename_all = "snake_case")]
pub(in crate::tensor_cli) enum Request {
    Execute(Box<Invocation>),
    Release { partial: KPartial },
}

#[derive(Deserialize, Serialize)]
pub(in crate::tensor_cli) struct Invocation {
    decision: W4Decision,
    activations_f16: Vec<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    retention_ms: Option<u64>,
}

impl Request {
    pub(in crate::tensor_cli) fn execute(
        decision: W4Decision,
        activations_f16: Vec<u16>,
        retention_ms: u64,
    ) -> Self {
        Self::Execute(Box::new(Invocation {
            decision,
            activations_f16,
            retention_ms: Some(retention_ms),
        }))
    }

    pub(in crate::tensor_cli) fn partial(&self) -> KPartial {
        match self {
            Self::Execute(invocation) => invocation.decision.shard.partial,
            Self::Release { partial } => *partial,
        }
    }
}
struct Published {
    decision: W4Decision,
    activation_digest: [u8; 32],
    payload: Bytes,
    expires_at: Option<Instant>,
}

impl Published {
    fn expired(&self, now: Instant) -> bool {
        self.expires_at.is_some_and(|deadline| deadline <= now)
    }
}

#[derive(Default)]
struct Publications {
    entries: BTreeMap<OpId, Published>,
}

impl Publications {
    fn check(
        &mut self,
        decision: W4Decision,
        activation_digest: [u8; 32],
        payload_bytes: usize,
        now: Instant,
    ) -> io::Result<bool> {
        self.expire(now);
        if let Some(entry) = self.entries.get(&decision.shard.partial.op) {
            if entry.decision != decision || entry.activation_digest != activation_digest {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "operation already published a different resident W4 contribution",
                ));
            }
            return Ok(true);
        }
        let retained: usize = self.entries.values().map(|entry| entry.payload.len()).sum();
        if self.entries.len() >= MAX_PUBLISHED || payload_bytes > MAX_PARTIAL_BYTES - retained {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "resident W4 publication capacity exhausted",
            ));
        }
        Ok(false)
    }

    fn publish(&mut self, entry: Published) {
        // The worker serializes invocations. Other tasks only read or expire entries.
        let prior = self.entries.insert(entry.decision.shard.partial.op, entry);
        assert!(prior.is_none(), "publication was already present");
    }

    fn release(&mut self, partial: KPartial) -> bool {
        let std::collections::btree_map::Entry::Occupied(entry) = self.entries.entry(partial.op)
        else {
            return false;
        };
        if entry.get().decision.shard.partial != partial {
            return false;
        }
        entry.remove();
        true
    }

    fn fetch(&self, partial: KPartial, now: Instant) -> Option<Bytes> {
        self.entries
            .get(&partial.op)
            .filter(|entry| entry.decision.shard.partial == partial && !entry.expired(now))
            .map(|entry| entry.payload.clone())
    }

    fn expire(&mut self, now: Instant) {
        self.entries.retain(|_, entry| !entry.expired(now));
    }
}

struct ResidentWorker<'lib> {
    grant: WorkerGrant,
    scope: W4Scope,
    device: DeviceSession<'lib>,
    linear: W4Partition<'lib>,
    control: GroupControl,
    published: Arc<Mutex<Publications>>,
    addr: SocketAddr,
}

pub(in crate::tensor_cli) async fn run(
    command: WorkerW4,
) -> Result<(), Box<dyn std::error::Error>> {
    let [start, end] = command.k_range;
    let group_size = command.device.group_size;
    if start >= end || group_size == 0 || start % group_size != 0 {
        return Err("invalid resident W4 interval or group size".into());
    }
    if command
        .advertise
        .is_some_and(|addr| addr.ip().is_unspecified() || addr.port() == 0)
    {
        return Err("resident W4 advertised address must specify an IP and a nonzero port".into());
    }
    let mut lines = io::stdin().lock().lines();
    let input: W4WeightInput =
        serde_json::from_str(&lines.next().ok_or("resident W4 worker needs weights")??)?;
    let weights = input.layout(end - start, command.output_columns, group_size)?;
    let mut control = GroupControl::connect(command.control).await?;
    let view = control.version(command.grant.version).await?;
    if !view.holds_grant(command.grant)
        || matches!(view.phase, GroupPhase::Revoked | GroupPhase::Retired)
    {
        return Err("resident W4 worker has no grant to load weights".into());
    }
    let _lock = command
        .device_lock
        .as_deref()
        .map(crate::tensor_cli::runtime::enter_worker)
        .transpose()?;
    let scope = W4Scope {
        device: command.grant.device,
        incarnation: command.grant.incarnation,
        ordinal: command.device.ordinal,
        k_range: command.k_range,
        output_shape: [0, command.output_columns],
        group_size,
        library_sha256: Sha256::digest(std::fs::read(&command.device.library)?).into(),
        weights_sha256: input.digest(end - start, command.output_columns, group_size),
    };
    // The local deployment configuration chooses a trusted native device library.
    let backend = unsafe { NativeW4A16::load(&command.device.library) }?;
    let device = backend.open_device(command.device.ordinal)?;
    let linear = W4Partition::prepare(&device, weights)?;
    if let Some((kernel, _)) = &linear.optimized_a16 {
        eprintln!(
            "resident W4 using {kernel} FP32 output for A16, N={} K={}",
            command.output_columns,
            end - start
        );
    }
    drop(input);

    let listener = PartialListener::bind(command.listen).await?;
    let addr = command.advertise.unwrap_or(listener.local_addr()?);
    let published = Arc::new(Mutex::new(Publications::default()));
    let mut worker = ResidentWorker {
        grant: command.grant,
        scope,
        device,
        linear,
        control,
        published: published.clone(),
        addr,
    };
    let server = tokio::spawn(serve(listener, published));
    let outcome = async {
        respond(&json!({ "ready": addr }))?;
        for line in lines {
            if server.is_finished() {
                return Err(io::Error::other("resident W4 tensor listener stopped"));
            }
            let response = match serde_json::from_str::<Request>(&line?) {
                Ok(request) => worker
                    .handle(request)
                    .await
                    .unwrap_or_else(|error| json!({ "error": error.to_string() })),
                Err(error) => json!({ "error": error.to_string() }),
            };
            respond(&response)?;
        }
        Ok(())
    }
    .await;
    server.abort();
    let _ = server.await;
    Ok(outcome?)
}

impl ResidentWorker<'_> {
    async fn handle(&mut self, request: Request) -> Result<Value, Box<dyn std::error::Error>> {
        match request {
            Request::Execute(invocation) => self.invoke(*invocation).await,
            Request::Release { partial } => {
                let released = self.published.lock().unwrap().release(partial);
                Ok(json!({ "released": released }))
            }
        }
    }

    async fn invoke(
        &mut self,
        invocation: Invocation,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let Invocation {
            decision,
            activations_f16,
            retention_ms,
        } = invocation;
        if retention_ms.is_some_and(|ms| ms == 0 || ms > 120_000) {
            return Err("result retention must be between 1 and 120000 ms".into());
        }
        let partial = decision.shard.partial;
        let payload_bytes = partial_size(partial)?;
        if partial.grant != self.grant {
            return Err("W4 decision has a different worker grant".into());
        }
        let rows = partial.output_shape[0];
        let mut scope = self.scope;
        scope.output_shape[0] = rows;
        if !scope.matches(partial, self.linear.canonical.group_size()) {
            return Err("W4 decision is outside the resident worker scope".into());
        }
        validate_decision(decision, partial, scope)?;
        if u64::from(rows) * u64::from(self.linear.canonical.input_columns())
            != activations_f16.len() as u64
        {
            return Err("activation shape does not match the resident W4 invocation".into());
        }
        let mut digest = Sha256::new();
        for value in &activations_f16 {
            digest.update(value.to_le_bytes());
        }
        let activation_digest = digest.finalize().into();
        self.control
            .check_worker(self.grant, Some(partial.op))
            .await?;
        check_expiry(decision)?;
        let replay = self.published.lock().unwrap().check(
            decision,
            activation_digest,
            payload_bytes,
            Instant::now(),
        )?;
        if !replay {
            let activation = activation_format(
                decision.profile.activation_bits,
                decision.profile.quantization,
                self.device.supports_f16_quantization(),
            )?;
            let values = self
                .linear
                .project(&self.device, &activations_f16, activation)?;
            self.control
                .check_worker(self.grant, Some(partial.op))
                .await?;
            self.published.lock().unwrap().publish(Published {
                decision,
                activation_digest,
                payload: values
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>()
                    .into(),
                expires_at: retention_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
            });
        }
        Ok(json!({ "addr": self.addr, "partial": partial }))
    }
}

async fn serve(listener: PartialListener, published: Arc<Mutex<Publications>>) -> io::Result<()> {
    let mut pulls = JoinSet::new();
    let mut expiry = tokio::time::interval(Duration::from_secs(1));
    expiry.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            connection = listener.accept(), if pulls.len() < MAX_PULLS => {
                let connection = connection?;
                let published = published.clone();
                pulls.spawn(async move {
                    connection.serve(PULL_TIMEOUT, |request| {
                        published.lock().unwrap().fetch(request, Instant::now())
                    }).await
                });
            }
            Some(result) = pulls.join_next(), if !pulls.is_empty() => {
                if let Err(error) = result? {
                    eprintln!("tensor source request failed: {error}");
                }
            }
            _ = expiry.tick() => {
                published.lock().unwrap().expire(Instant::now());
            }
        }
    }
}

fn respond(value: &Value) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    writeln!(stdout)?;
    stdout.flush()
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{GroupId, GroupVersion};
    use plait_core::{DeviceId, NodeId, ShardId};
    use plait_net::tensor::fetch_partial;
    use plait_parallel::precision::{W4Profile, W4Shard};
    use tokio::net::TcpStream;

    use super::*;

    fn publication(counter: u64, expires_at: Option<Instant>) -> Published {
        let node = NodeId::derive("resident-publications");
        let device = DeviceId::new(node, 0);
        let partial = KPartial {
            grant: WorkerGrant {
                version: GroupVersion {
                    group: GroupId(node),
                    epoch: 1,
                },
                device,
                incarnation: 1,
                fence: 2,
            },
            op: OpId::new(node, counter),
            shard: ShardId(1),
            k_start: 0,
            k_end: 2,
            output_shape: [1, 1],
        };
        let scope = W4Scope {
            device,
            incarnation: 1,
            ordinal: 0,
            library_sha256: [0; 32],
            weights_sha256: [0; 32],
            k_range: [0, 2],
            output_shape: [1, 1],
            group_size: 1,
        };
        Published {
            decision: W4Decision {
                shard: W4Shard { partial, scope },
                profile: W4Profile {
                    scope,
                    calibration_id: [0; 32],
                    activation_bits: 16,
                    quantization: None,
                    latency_ns: 1,
                    error_ppm: 0,
                    valid_until_ms: u64::MAX,
                },
            },
            activation_digest: [0; 32],
            payload: Bytes::copy_from_slice(&(counter as f32).to_le_bytes()),
            expires_at,
        }
    }

    #[test]
    fn retries_and_capacity_preserve_unreleased_publications() {
        let mut retained = Publications::default();
        let now = Instant::now();
        let deadline = now + Duration::from_secs(1);
        let first = publication(0, Some(deadline));
        let decision = first.decision;
        let partial = decision.shard.partial;
        retained.publish(first);
        let next = publication(1, None).decision;
        assert_eq!(
            retained
                .check(next, [0; 32], MAX_PARTIAL_BYTES, now)
                .unwrap_err()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        for counter in 1..MAX_PUBLISHED as u64 {
            let entry = publication(counter, None);
            assert!(!retained.check(entry.decision, [0; 32], 4, now).unwrap());
            retained.publish(entry);
        }
        let another = publication(MAX_PUBLISHED as u64, None).decision;
        assert_eq!(
            retained.check(another, [0; 32], 4, now).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(retained.check(decision, [0; 32], 4, now).unwrap());
        assert_eq!(
            retained
                .check(decision, [1; 32], 4, now)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let mut stale = partial;
        stale.grant.fence -= 1;
        assert!(!retained.release(stale));
        assert!(retained.fetch(partial, now).is_some());
        assert!(retained.fetch(partial, deadline).is_none());
        assert!(!retained.check(another, [0; 32], 4, deadline).unwrap());
    }

    #[tokio::test]
    async fn multiple_publications_serve_pulls_while_another_connection_stalls() {
        let mut retained = Publications::default();
        let first = publication(1, None);
        let second = publication(2, None);
        let (first_partial, second_partial) =
            (first.decision.shard.partial, second.decision.shard.partial);
        retained.publish(first);
        retained.publish(second);
        let retained = Arc::new(Mutex::new(retained));
        let listener = PartialListener::bind("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(serve(listener, retained.clone()));
        let stalled = TcpStream::connect(addr).await.unwrap();
        let timeout = Duration::from_secs(1);
        for (partial, expected) in [(second_partial, 2.0f32), (first_partial, 1.0f32)] {
            assert_eq!(
                fetch_partial(addr, partial, timeout)
                    .await
                    .unwrap()
                    .as_ref(),
                expected.to_le_bytes()
            );
        }
        assert!(retained.lock().unwrap().release(second_partial));
        assert_eq!(
            fetch_partial(addr, second_partial, timeout)
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        assert!(fetch_partial(addr, first_partial, timeout).await.is_ok());
        drop(stalled);
        server.abort();
        let _ = server.await;
    }
}
