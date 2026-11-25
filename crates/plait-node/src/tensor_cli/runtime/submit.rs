use std::collections::BTreeMap;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use plait_core::serving::{
    GroupId, GroupPhase, GroupVersion, ServingAction, ServingLookup, ServingOperation,
    ServingOutcome, ServingView, W4_MAX_ADMISSION_MS,
};
use plait_core::{NodeId, OpId};
use plait_fabric::reduction::KSum;
use plait_net::tensor::partial_size;
use plait_parallel::precision::{
    W4Decision, W4PlacementRequest, W4Profile, W4Scope, W4Selection, plan_w4,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use super::{CALL_TIMEOUT, DispatchServer, read_frame, read_payload, write_frame, write_payload};
use crate::tensor_cli::control::GroupControl;
use crate::tensor_cli::w4::{self, worker::Request};
use crate::tensor_cli::{Source, pull_contribution, reduce_transfers};

const SHARD_TIMEOUT: Duration = Duration::from_secs(40);
const PULL_TIMEOUT: Duration = Duration::from_secs(5);
const RELEASE_TIMEOUT: Duration = Duration::from_secs(3);
const CLOSE_ADMISSION_TIMEOUT: Duration = Duration::from_secs(2);
const RESULT_RETENTION_MS: u64 = 90_000;

#[derive(Deserialize)]
#[serde(default)]
pub(super) struct Limits {
    host: usize,
    per_epoch: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            host: 16,
            per_epoch: 8,
        }
    }
}

pub(super) struct Submissions {
    pub(super) host: Semaphore,
    per_epoch: usize,
    epochs: Mutex<BTreeMap<GroupVersion, usize>>,
}

impl Submissions {
    pub(super) fn new(limits: Limits) -> io::Result<Self> {
        if limits.per_epoch == 0
            || limits.per_epoch >= limits.host
            || limits.host > Semaphore::MAX_PERMITS
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "W4 submission limits require 0 < per_epoch < host",
            ));
        }
        Ok(Self {
            host: Semaphore::new(limits.host),
            per_epoch: limits.per_epoch,
            epochs: Mutex::new(BTreeMap::new()),
        })
    }

    fn try_epoch(&self, version: GroupVersion) -> Option<EpochSlot<'_>> {
        let mut epochs = self.epochs.lock().unwrap();
        let in_flight = epochs.entry(version).or_default();
        if *in_flight == self.per_epoch {
            return None;
        }
        *in_flight += 1;
        Some(EpochSlot {
            submissions: self,
            version,
        })
    }
}

struct EpochSlot<'a> {
    submissions: &'a Submissions,
    version: GroupVersion,
}

impl Drop for EpochSlot<'_> {
    fn drop(&mut self) {
        let mut epochs = self.submissions.epochs.lock().unwrap();
        let count = epochs
            .get_mut(&self.version)
            .expect("epoch slot is counted");
        *count -= 1;
        if *count == 0 {
            epochs.remove(&self.version);
        }
    }
}

#[derive(Deserialize, Serialize)]
pub(super) struct Operator {
    group: GroupId,
    k: u32,
    group_size: u32,
    output_shape: [u32; 2],
    calibration_id: [u8; 32],
    max_error_ppm: u64,
    scopes: Vec<W4Scope>,
    profiles: Vec<W4Profile>,
    activations_f16: Vec<u16>,
}

#[derive(Clone, Copy)]
struct Shard {
    addr: SocketAddr,
    decision: W4Decision,
}

pub(super) async fn serve(
    stream: &mut TcpStream,
    server: &Arc<DispatchServer>,
    mut operator: Operator,
) -> io::Result<()> {
    let (mut control, work, version, tracked) = match admit(server, &operator).await {
        Ok(admission) => admission,
        Err(error) => return write_frame(stream, &json!({ "error": error.to_string() })).await,
    };
    let epoch_slot = server.submissions.try_epoch(version);
    let planned = match &epoch_slot {
        Some(_) => plan_admitted(&mut control, version, work, &mut operator).await,
        None => Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            format!("W4 submit group epoch {} is busy", version.epoch),
        )),
    };
    let (result, shards) = match planned {
        Ok((selection, plan, shards)) => {
            let result = async {
                let output = dispatch_and_reduce(
                    server,
                    &plan,
                    &shards,
                    operator.k as usize,
                    &operator.activations_f16,
                )
                .await?;
                complete(&mut control, version, work, &shards).await?;
                Ok((selection, output))
            }
            .await;
            (result, shards)
        }
        Err(error) => (Err(error), Vec::new()),
    };
    drop(operator);
    let close_admission = tracked && result.is_err();
    let reply = async {
        match result {
            Ok((plan, output)) => {
                write_frame(stream, &json!({ "plan": plan })).await?;
                write_payload(stream, &output).await
            }
            Err(error) => write_frame(stream, &json!({ "error": error.to_string() })).await,
        }
    };
    let cleanup = async {
        if close_admission {
            // Fence any late dispatch before sending exact releases. Admission
            // expiry and publication retention cover unreachable peers.
            let finish = ServingOperation {
                id: OpId::new(NodeId::random(), 1),
                action: ServingAction::FinishW4 { version, work },
            };
            let _ = tokio::time::timeout(CLOSE_ADMISSION_TIMEOUT, control.apply(&finish)).await;
        }
        release(server, &shards).await;
    };
    // Both the host and epoch slots account for cleanup until both futures finish.
    let (written, ()) = tokio::join!(reply, cleanup);
    drop(epoch_slot);
    written
}

async fn admit(
    server: &DispatchServer,
    operator: &Operator,
) -> io::Result<(GroupControl, OpId, GroupVersion, bool)> {
    let [rows, _] = operator.output_shape;
    if operator.k == 0
        || rows == 0
        || u64::from(rows) * u64::from(operator.k) != operator.activations_f16.len() as u64
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "activation shape does not match W4 operator",
        ));
    }
    let mut control = GroupControl::connect(server.control.clone()).await?;
    let work = OpId::new(NodeId::random(), 1);
    let now_ms = w4::now_ms()?;
    let admission = ServingOperation {
        id: work,
        action: ServingAction::AdmitW4 {
            group: operator.group,
            now_ms,
            expires_at_ms: now_ms.saturating_add(W4_MAX_ADMISSION_MS),
        },
    };
    let ServingOutcome::W4Admitted { version, tracked } = control
        .apply(&admission)
        .await?
        .map_err(|error| io::Error::other(format!("W4 admission rejected: {error:?}")))?
    else {
        return Err(io::Error::other("expected W4 admission receipt"));
    };
    Ok((control, work, version, tracked))
}

async fn complete(
    control: &mut GroupControl,
    version: GroupVersion,
    work: OpId,
    shards: &[Shard],
) -> io::Result<()> {
    let completion = ServingOperation {
        id: OpId::new(NodeId::random(), 1),
        action: ServingAction::CompleteW4 {
            version,
            work,
            grants: shards
                .iter()
                .map(|shard| shard.decision.shard.partial.grant)
                .collect(),
        },
    };
    control
        .apply(&completion)
        .await?
        .map_err(|error| io::Error::other(format!("W4 completion rejected: {error:?}")))?;
    Ok(())
}

async fn plan_admitted(
    control: &mut GroupControl,
    version: GroupVersion,
    work: OpId,
    operator: &mut Operator,
) -> io::Result<(W4Selection, KSum, Vec<Shard>)> {
    let view = control.version(version).await?;
    if !matches!(view.phase, GroupPhase::Active | GroupPhase::Draining) {
        return Err(io::Error::other("the admitted W4 version lost its grants"));
    }
    let ServingView::Dispatch(registered) =
        control.query(ServingLookup::Dispatch(view.version)).await?
    else {
        return Err(io::Error::other(
            "expected registered W4 execution endpoints",
        ));
    };
    let endpoints: BTreeMap<_, _> = registered
        .into_iter()
        .filter(|item| view.allows_grant(item.grant))
        .map(|item| (item.grant.device, item.addr))
        .collect();
    let request = W4PlacementRequest {
        k: operator.k,
        group_size: operator.group_size,
        output_shape: operator.output_shape,
        op: work,
        now_ms: w4::now_ms()?,
        calibration_id: operator.calibration_id,
        max_error_ppm: operator.max_error_ppm,
        workers: view
            .grants
            .values()
            .filter(|grant| endpoints.contains_key(&grant.device))
            .copied()
            .collect(),
        scopes: std::mem::take(&mut operator.scopes),
        profiles: std::mem::take(&mut operator.profiles),
    };
    let selection = plan_w4(&request).map_err(io::Error::other)?;
    let shards: Vec<_> = selection
        .decisions
        .iter()
        .map(|&decision| Shard {
            addr: endpoints[&decision.shard.partial.grant.device],
            decision,
        })
        .collect();
    let plan = KSum::new(
        operator.k,
        operator.group_size,
        shards
            .iter()
            .map(|shard| shard.decision.shard.partial)
            .collect(),
    )
    .map_err(io::Error::other)?;
    for shard in &shards {
        partial_size(shard.decision.shard.partial)?;
    }
    plan.check_grants(&view).map_err(io::Error::other)?;
    Ok((selection, plan, shards))
}

async fn dispatch_and_reduce(
    server: &Arc<DispatchServer>,
    plan: &KSum,
    shards: &[Shard],
    k: usize,
    activations: &[u16],
) -> io::Result<Vec<u8>> {
    let mut calls = JoinSet::new();
    for &shard in shards {
        let server = server.clone();
        let partial = shard.decision.shard.partial;
        let slice = partial.k_start as usize..partial.k_end as usize;
        let input = activations
            .chunks_exact(k)
            .flat_map(|row| row[slice.clone()].iter().copied())
            .collect();
        calls.spawn(async move {
            let request = Request::execute(shard.decision, input, RESULT_RETENTION_MS);
            let reply =
                tokio::time::timeout(SHARD_TIMEOUT, server.route(shard.addr, request)).await??;
            if let Some(error) = reply.get("error").and_then(Value::as_str) {
                return Err(io::Error::other(format!(
                    "shard {}: {error}",
                    partial.shard.0
                )));
            }
            let source: Source = serde_json::from_value(reply)?;
            if source.partial != partial {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "worker returned a different W4 contribution",
                ));
            }
            pull_contribution(source, PULL_TIMEOUT).await
        });
    }
    // Dropping the JoinSet on the first failure cancels sibling client tasks.
    reduce_transfers(plan, &mut calls).await
}

async fn release(server: &Arc<DispatchServer>, shards: &[Shard]) {
    let mut calls = JoinSet::new();
    for &shard in shards {
        let server = server.clone();
        let partial = shard.decision.shard.partial;
        calls.spawn(async move {
            let _ = tokio::time::timeout(
                RELEASE_TIMEOUT,
                server.route(shard.addr, Request::Release { partial }),
            )
            .await;
        });
    }
    while calls.join_next().await.is_some() {}
}

#[derive(Serialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum SubmitMessage {
    Submit(Operator),
}

pub(in crate::tensor_cli) async fn run(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let operator = serde_json::from_reader(io::stdin().lock())?;
    let (header, output) = tokio::time::timeout(CALL_TIMEOUT, async {
        let mut stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        write_frame(&mut stream, &SubmitMessage::Submit(operator)).await?;
        let header: Value = read_frame(&mut stream).await?;
        if let Some(error) = header.get("error").and_then(Value::as_str) {
            return Err(io::Error::other(error.to_owned()));
        }
        let output = read_payload(&mut stream).await?;
        Ok::<_, io::Error>((header, output))
    })
    .await??;
    eprintln!("{header}");
    io::stdout().write_all(&output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use plait_core::serving::WorkerGrant;
    use plait_core::tensor::KPartial;
    use plait_core::{DeviceId, ShardId};
    use plait_net::tensor::{PartialListener, PartialSource};
    use plait_parallel::precision::W4Shard;
    use serde_json::json;
    use tokio::sync::{mpsc, oneshot};

    use super::super::Invocation;
    use super::*;

    #[test]
    fn saturation_stays_in_the_admitted_version_and_slots_return_on_drop() {
        let budget = Submissions::new(Limits {
            host: 4,
            per_epoch: 2,
        })
        .unwrap();
        let group = GroupId(NodeId::derive("switching-submissions"));
        let old = GroupVersion { group, epoch: 1 };
        let new = GroupVersion { group, epoch: 2 };
        let other = GroupVersion {
            group: GroupId(NodeId::derive("other-submissions")),
            epoch: 1,
        };
        let first = (
            budget.host.try_acquire().unwrap(),
            budget.try_epoch(old).unwrap(),
        );
        let second = (
            budget.host.try_acquire().unwrap(),
            budget.try_epoch(old).unwrap(),
        );
        assert!(budget.try_epoch(old).is_none());
        let next = (
            budget.host.try_acquire().unwrap(),
            budget.try_epoch(new).unwrap(),
        );
        let unrelated = (
            budget.host.try_acquire().unwrap(),
            budget.try_epoch(other).unwrap(),
        );
        assert!(budget.host.try_acquire().is_err());
        drop((first, second));
        assert!(!budget.epochs.lock().unwrap().contains_key(&old));
        assert_eq!(budget.host.available_permits(), 2);
        drop((next, unrelated));
        assert!(budget.epochs.lock().unwrap().is_empty());
        assert_eq!(budget.host.available_permits(), 4);
        assert!(
            Submissions::new(Limits {
                host: 2,
                per_epoch: 2
            })
            .is_err()
        );
    }

    fn test_shards() -> [Shard; 2] {
        let host = NodeId::derive("async-reduction-test");
        let addr = "127.0.0.1:0".parse().unwrap();
        let shard = |index| {
            let device = DeviceId::new(host, index);
            let partial = KPartial {
                grant: WorkerGrant {
                    version: GroupVersion {
                        group: GroupId(host),
                        epoch: 1,
                    },
                    device,
                    incarnation: 1,
                    fence: 1,
                },
                op: OpId::new(host, 1),
                shard: ShardId(index),
                k_start: index,
                k_end: index + 1,
                output_shape: [1, 1],
            };
            let scope = W4Scope {
                device,
                incarnation: 1,
                ordinal: index as i32,
                library_sha256: [0; 32],
                weights_sha256: [0; 32],
                k_range: [index, index + 1],
                output_shape: [1, 1],
                group_size: 1,
            };
            Shard {
                addr,
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
            }
        };
        [shard(0), shard(1)]
    }

    #[tokio::test]
    async fn pulls_a_ready_shard_before_the_last_worker_finishes() {
        let shards = test_shards();
        let addr = shards[0].addr;
        let [slow, fast] = shards.map(|shard| shard.decision.shard.partial);
        let slow_data = PartialSource::bind(addr, slow, 2.0f32.to_le_bytes().to_vec())
            .await
            .unwrap();
        let fast_data = PartialSource::bind(addr, fast, 3.0f32.to_le_bytes().to_vec())
            .await
            .unwrap();
        let (send_slow, mut slow_worker) = mpsc::channel::<Invocation>(1);
        let (send_fast, mut fast_worker) = mpsc::channel::<Invocation>(1);
        let (pulled, fast_finished) = oneshot::channel();
        let fast_worker = tokio::spawn(async move {
            let invocation = fast_worker.recv().await.unwrap();
            assert_eq!(invocation.request.partial(), fast);
            invocation
                .response
                .send(json!({"addr": fast_data.local_addr().unwrap(), "partial": fast}))
                .unwrap();
            fast_data.serve_next(PULL_TIMEOUT).await.unwrap();
            pulled.send(()).unwrap();
        });
        let slow_worker = tokio::spawn(async move {
            let invocation = slow_worker.recv().await.unwrap();
            assert_eq!(invocation.request.partial(), slow);
            fast_finished.await.unwrap();
            invocation
                .response
                .send(json!({"addr": slow_data.local_addr().unwrap(), "partial": slow}))
                .unwrap();
            slow_data.serve_next(PULL_TIMEOUT).await.unwrap();
        });
        let server = Arc::new(DispatchServer {
            addr,
            control: crate::tensor_cli::ControlPeers::new(Vec::new(), Vec::new(), addr.to_string()),
            routes: BTreeMap::from([
                (slow.grant.device, send_slow),
                (fast.grant.device, send_fast),
            ]),
            submissions: Submissions::new(Limits::default()).unwrap(),
        });
        let plan = KSum::new(2, 1, vec![fast, slow]).unwrap();
        let output = tokio::time::timeout(
            Duration::from_secs(3),
            dispatch_and_reduce(&server, &plan, &shards, 2, &[0, 0]),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(output, 5.0f32.to_le_bytes());
        fast_worker.await.unwrap();
        slow_worker.await.unwrap();
    }

    #[tokio::test]
    async fn a_held_pull_does_not_block_the_next_reduction_on_the_same_workers() {
        let first = test_shards();
        let second = first.map(|mut shard| {
            let partial = &mut shard.decision.shard.partial;
            partial.op = OpId::new(partial.grant.device.node, 2);
            shard
        });
        let addr = first[0].addr;
        let first_parts = first.map(|shard| shard.decision.shard.partial);
        let second_parts = second.map(|shard| shard.decision.shard.partial);
        let (hold, mut held) = mpsc::channel(1);
        let mut routes = BTreeMap::new();
        let mut workers = JoinSet::new();
        for (index, (first, second)) in first_parts.into_iter().zip(second_parts).enumerate() {
            let (route, mut requests) = mpsc::channel::<Invocation>(1);
            routes.insert(first.grant.device, route);
            let hold = hold.clone();
            workers.spawn(async move {
                let listener = PartialListener::bind(addr).await.unwrap();
                let endpoint = listener.local_addr().unwrap();
                let contributions = [(first, [2.0f32, 3.0][index]), (second, [7.0, 11.0][index])];
                for (partial, value) in contributions {
                    let invocation = requests.recv().await.unwrap();
                    assert_eq!(invocation.request.partial(), partial);
                    invocation
                        .response
                        .send(json!({"addr": endpoint, "partial": partial}))
                        .unwrap();
                    let connection = listener.accept().await.unwrap();
                    let bytes = Bytes::copy_from_slice(&value.to_le_bytes());
                    if index == 1 && partial == first {
                        assert!(hold.send((connection, partial, bytes)).await.is_ok());
                    } else {
                        connection
                            .serve(PULL_TIMEOUT, |requested| {
                                (requested == partial).then_some(bytes)
                            })
                            .await
                            .unwrap();
                    }
                }
            });
        }
        drop(hold);
        let server = Arc::new(DispatchServer {
            addr,
            control: crate::tensor_cli::ControlPeers::new(Vec::new(), Vec::new(), addr.to_string()),
            routes,
            submissions: Submissions::new(Limits::default()).unwrap(),
        });
        let slow_server = server.clone();
        let in_flight = tokio::spawn(async move {
            let plan = KSum::new(2, 1, first_parts.to_vec()).unwrap();
            dispatch_and_reduce(&slow_server, &plan, &first, 2, &[0, 0]).await
        });

        tokio::time::timeout(Duration::from_secs(3), async {
            let (connection, partial, bytes) = held.recv().await.unwrap();
            let plan = KSum::new(2, 1, second_parts.to_vec()).unwrap();
            let output = dispatch_and_reduce(&server, &plan, &second, 2, &[0, 0])
                .await
                .unwrap();
            assert_eq!(output, 18.0f32.to_le_bytes());
            assert!(!in_flight.is_finished());

            connection
                .serve(PULL_TIMEOUT, |requested| {
                    (requested == partial).then_some(bytes)
                })
                .await
                .unwrap();
            assert_eq!(in_flight.await.unwrap().unwrap(), 5.0f32.to_le_bytes());
            while let Some(result) = workers.join_next().await {
                result.unwrap();
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_failed_shard_cancels_a_sibling_waiting_for_its_worker() {
        let shards = test_shards();
        let addr = shards[0].addr;
        let [failed, stalled] = shards.map(|shard| shard.decision.shard.partial);
        let (send_failed, mut failed_worker) = mpsc::channel::<Invocation>(1);
        let (send_stalled, mut stalled_worker) = mpsc::channel::<Invocation>(1);
        let workers = tokio::spawn(async move {
            let mut waiting = stalled_worker.recv().await.unwrap().response;
            failed_worker
                .recv()
                .await
                .unwrap()
                .response
                .send(json!({ "error": "fixture rejected the shard" }))
                .unwrap();
            waiting.closed().await;
        });
        let server = Arc::new(DispatchServer {
            addr,
            control: crate::tensor_cli::ControlPeers::new(Vec::new(), Vec::new(), addr.to_string()),
            routes: BTreeMap::from([
                (failed.grant.device, send_failed),
                (stalled.grant.device, send_stalled),
            ]),
            submissions: Submissions::new(Limits::default()).unwrap(),
        });
        let plan = KSum::new(2, 1, vec![failed, stalled]).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            let error = dispatch_and_reduce(&server, &plan, &shards, 2, &[0, 0])
                .await
                .unwrap_err();
            assert_eq!(error.to_string(), "shard 0: fixture rejected the shard");
            workers.await.unwrap();
        })
        .await
        .unwrap();
    }
}
