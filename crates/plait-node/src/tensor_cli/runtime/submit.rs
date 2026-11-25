use std::collections::BTreeMap;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::sync::Arc;
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
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::task::JoinSet;

use super::{CALL_TIMEOUT, DispatchServer, read_frame, read_payload, write_frame};
use crate::tensor_cli::control::GroupControl;
use crate::tensor_cli::w4::{self, worker::Request};
use crate::tensor_cli::{Source, pull_contribution, reduce_transfers};

const SHARD_TIMEOUT: Duration = Duration::from_secs(40);
const PULL_TIMEOUT: Duration = Duration::from_secs(5);
const RELEASE_TIMEOUT: Duration = Duration::from_secs(3);
const RESULT_RETENTION_MS: u64 = 90_000;

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

pub(super) async fn execute(
    server: &Arc<DispatchServer>,
    operator: Operator,
) -> io::Result<(W4Selection, Vec<u8>)> {
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
    let (selection, output) =
        match execute_admitted(server, &mut control, version, work, operator).await {
            Ok(result) => result,
            Err(error) => {
                if tracked {
                    let finish = ServingOperation {
                        id: OpId::new(NodeId::random(), 1),
                        action: ServingAction::FinishW4 { version, work },
                    };
                    let _ = control.apply(&finish).await;
                }
                return Err(error);
            }
        };
    let completion = ServingOperation {
        id: OpId::new(NodeId::random(), 1),
        action: ServingAction::CompleteW4 {
            version,
            work,
            grants: selection
                .decisions
                .iter()
                .map(|decision| decision.shard.partial.grant)
                .collect(),
        },
    };
    control
        .apply(&completion)
        .await?
        .map_err(|error| io::Error::other(format!("W4 completion rejected: {error:?}")))?;
    Ok((selection, output))
}

async fn execute_admitted(
    server: &Arc<DispatchServer>,
    control: &mut GroupControl,
    version: GroupVersion,
    work: OpId,
    operator: Operator,
) -> io::Result<(W4Selection, Vec<u8>)> {
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
        scopes: operator.scopes,
        profiles: operator.profiles,
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
    let result = dispatch_and_reduce(
        server,
        &plan,
        &shards,
        operator.k as usize,
        &operator.activations_f16,
    )
    .await;
    release(server, &shards).await;
    Ok((selection, result?))
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
    // Each shard pulls as soon as its worker publishes. Drain the tasks before
    // releasing their immutable results, including when a sibling failed.
    let result = reduce_transfers(plan, &mut calls).await;
    while calls.join_next().await.is_some() {}
    result
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
    use plait_core::serving::WorkerGrant;
    use plait_core::tensor::KPartial;
    use plait_core::{DeviceId, ShardId};
    use plait_net::tensor::PartialSource;
    use plait_parallel::precision::W4Shard;
    use serde_json::json;
    use tokio::sync::{Semaphore, mpsc, oneshot};

    use super::super::Invocation;
    use super::*;

    #[tokio::test]
    async fn pulls_a_ready_shard_before_the_last_worker_finishes() {
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
        let shards = [shard(0), shard(1)];
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
            submissions: Arc::new(Semaphore::new(1)),
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
}
