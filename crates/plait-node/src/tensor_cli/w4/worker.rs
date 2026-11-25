use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bytes::Bytes;
use plait_core::serving::{GroupPhase, WorkerGrant};
use plait_core::tensor::KPartial;
use plait_net::tensor::{PartialListener, partial_size};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::watch;

use super::{
    ControlPeers, DeviceSession, Digest, GroupControl, NativeW4A16, PreparedW4A16, Sha256,
    W4Decision, W4Device, W4Scope, W4WeightInput, activation_format, check_expiry,
    validate_decision,
};
use crate::tensor_cli::retryable_pull_error;

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
    fn expired(&self) -> bool {
        self.expires_at
            .is_some_and(|deadline| deadline <= Instant::now())
    }
}

struct ResidentWorker<'lib> {
    grant: WorkerGrant,
    scope: W4Scope,
    device: DeviceSession<'lib>,
    linear: PreparedW4A16<'lib>,
    control: GroupControl,
    published: watch::Sender<Option<Published>>,
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
    let linear = device.prepare(weights)?;
    drop(input);

    let listener = PartialListener::bind(command.listen).await?;
    let addr = command.advertise.unwrap_or(listener.local_addr()?);
    let (published, reader) = watch::channel(None);
    let mut worker = ResidentWorker {
        grant: command.grant,
        scope,
        device,
        linear,
        control,
        published,
        addr,
    };
    let server = tokio::spawn(serve(listener, reader));
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
                let released = self.published.send_if_modified(|slot| {
                    if slot
                        .as_ref()
                        .is_some_and(|item| item.decision.shard.partial == partial)
                    {
                        *slot = None;
                        true
                    } else {
                        false
                    }
                });
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
        partial_size(partial)?;
        if partial.grant != self.grant {
            return Err("W4 decision has a different worker grant".into());
        }
        let rows = partial.output_shape[0];
        let mut scope = self.scope;
        scope.output_shape[0] = rows;
        if !scope.matches(partial, self.linear.group_size()) {
            return Err("W4 decision is outside the resident worker scope".into());
        }
        validate_decision(decision, partial, scope)?;
        if u64::from(rows) * u64::from(self.linear.input_columns()) != activations_f16.len() as u64
        {
            return Err("activation shape does not match the resident W4 invocation".into());
        }
        let mut digest = Sha256::new();
        for value in &activations_f16 {
            digest.update(value.to_le_bytes());
        }
        let activation_digest = digest.finalize().into();
        self.published.send_if_modified(|slot| {
            if slot.as_ref().is_some_and(Published::expired) {
                *slot = None;
                true
            } else {
                false
            }
        });
        let replay = if let Some(item) = self.published.borrow().as_ref() {
            if item.decision != decision || item.activation_digest != activation_digest {
                return Err(
                    "release the current resident W4 contribution before submitting another".into(),
                );
            }
            true
        } else {
            false
        };
        self.control
            .check_worker(self.grant, Some(partial.op))
            .await?;
        check_expiry(decision)?;
        if !replay {
            let activation = activation_format(
                decision.profile.activation_bits,
                decision.profile.quantization,
                self.device.supports_f16_quantization(),
            )?;
            let values =
                self.device
                    .project_w4_f32(&self.linear, &activations_f16, None, activation)?;
            self.control
                .check_worker(self.grant, Some(partial.op))
                .await?;
            self.published.send_replace(Some(Published {
                decision,
                activation_digest,
                payload: values
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>()
                    .into(),
                expires_at: retention_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
            }));
        }
        Ok(json!({ "addr": self.addr, "partial": partial }))
    }
}

async fn serve(
    listener: PartialListener,
    reader: watch::Receiver<Option<Published>>,
) -> io::Result<()> {
    loop {
        let result = listener
            .serve_next(Duration::from_secs(3), |request| {
                reader
                    .borrow()
                    .as_ref()
                    .filter(|item| item.decision.shard.partial == request && !item.expired())
                    .map(|item| item.payload.clone())
            })
            .await;
        if let Err(error) = result {
            if !retryable_pull_error(&error) {
                return Err(error);
            }
            eprintln!("tensor source request failed: {error}");
        }
    }
}

fn respond(value: &Value) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    writeln!(stdout)?;
    stdout.flush()
}
