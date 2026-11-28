//! Join a communicator using device grants and endpoints from the control plane.

use std::collections::BTreeSet;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use plait_core::group::{
    DeviceFailureScope, GrantStatus, GroupAction, GroupFault, GroupLookup, GroupOperation,
    GroupReply, GroupVersion, GroupView, WorkerFaultEvidence, WorkerGrant,
};
use plait_core::{DeviceId, NodeId, OpId};
use plait_net::control::{Client, discover_voters};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::time::{Instant, MissedTickBehavior, interval, sleep, timeout, timeout_at};

use crate::{BackendKind, Comm, PeerGroup, RanklessComm, RanklessListener, create_runtime};

/// A caller's local device grant identity and the desired collective slot order.
pub struct JoinGroup {
    pub version: GroupVersion,
    pub member: DeviceId,
    pub layout: Vec<DeviceId>,
    pub listen: SocketAddr,
    pub seeds: Vec<SocketAddr>,
}

/// Publish the local listener, connect to the committed members, and report ready.
/// The caller decides when to activate the group and admit work to the communicator.
pub async fn join_committed_group(
    request: JoinGroup,
    backend: BackendKind,
    device: usize,
    timeout: Duration,
) -> io::Result<RanklessComm> {
    join_committed_group_with_egress(request, backend, device, timeout, None).await
}

pub(crate) async fn join_committed_group_with_egress(
    request: JoinGroup,
    backend: BackendKind,
    device: usize,
    timeout: Duration,
    replicated_egress_bytes: Option<u64>,
) -> io::Result<RanklessComm> {
    join(
        request,
        backend,
        device,
        timeout,
        replicated_egress_bytes,
        None,
    )
    .await
    .map(|(group, _)| group)
}

/// Join both communication axes of a rectangular vLLM-style model layout.
/// The device order lists the tensor slots of each pipeline stage in order.
pub async fn join_committed_model_group(
    request: JoinGroup,
    backend: BackendKind,
    device: usize,
    timeout: Duration,
    tensor_parallel: usize,
    replicated_egress_bytes: Option<u64>,
) -> io::Result<(RanklessComm, RanklessComm)> {
    let width = request.layout.len();
    if tensor_parallel <= 1 || tensor_parallel >= width || width % tensor_parallel != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mixed layout requires nontrivial TP and PP sizes",
        ));
    }
    let (tensor, pipeline) = join(
        request,
        backend,
        device,
        timeout,
        replicated_egress_bytes,
        Some(tensor_parallel),
    )
    .await?;
    Ok((
        tensor,
        pipeline.expect("mixed group has a pipeline communicator"),
    ))
}

async fn join(
    request: JoinGroup,
    backend: BackendKind,
    device: usize,
    timeout: Duration,
    replicated_egress_bytes: Option<u64>,
    tensor_parallel: Option<usize>,
) -> io::Result<(RanklessComm, Option<RanklessComm>)> {
    let layout: BTreeSet<_> = request.layout.iter().copied().collect();
    if layout.len() != request.layout.len()
        || !layout.contains(&request.member)
        || request.version.epoch == 0
        || request.listen.ip().is_unspecified()
        || request.seeds.is_empty()
        || request
            .seeds
            .iter()
            .any(|seed| seed.ip().is_unspecified() || seed.port() == 0)
        || timeout.is_zero()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Plait group join",
        ));
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid group join timeout"))?;
    timeout_at(
        deadline,
        connect(
            request,
            layout,
            backend,
            device,
            deadline,
            timeout,
            (replicated_egress_bytes, tensor_parallel),
        ),
    )
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "Plait group join timed out"))?
}

async fn connect(
    request: JoinGroup,
    layout: BTreeSet<DeviceId>,
    backend: BackendKind,
    device: usize,
    deadline: Instant,
    timeout: Duration,
    transport_options: (Option<u64>, Option<usize>),
) -> io::Result<(RanklessComm, Option<RanklessComm>)> {
    let (replicated_egress_bytes, tensor_parallel) = transport_options;
    let failure_scope = match std::env::var("PLAIT_DEVICE_FAILURE_SCOPE") {
        Err(std::env::VarError::NotPresent) => DeviceFailureScope::default(),
        Ok(value) if value == "host" => DeviceFailureScope::Host,
        Ok(value) if value == "device" => DeviceFailureScope::Device,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "PLAIT_DEVICE_FAILURE_SCOPE must be host or device",
            ));
        }
    };
    let JoinGroup {
        version,
        member,
        layout: order,
        listen,
        seeds,
    } = request;
    let executor = control_runtime()?;
    let mut control = executor
        .spawn(Rendezvous::new(listen, seeds.clone()))
        .await
        .map_err(io::Error::other)??;
    let session = NodeId::random();
    let view = control
        .call(|client, voter| {
            Box::pin(client.observe_group(
                voter,
                OpId::new(session, 1),
                version.group,
                Some(version.epoch),
            ))
        })
        .await
        .map_err(|error| io::Error::other(format!("observe group: {error:?}")))?;
    check_open(&view)?;
    let granted = view
        .grants
        .iter()
        .map(|grant| grant.device)
        .collect::<BTreeSet<_>>();
    if layout != granted {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "slot layout differs from the committed device grants",
        ));
    }
    let grant = view
        .grants
        .into_iter()
        .find(|grant| grant.device == member)
        .ok_or_else(|| io::Error::other("this device has no grant for this group"))?;
    let runtime = create_runtime(backend, device).map_err(io::Error::other)?;
    let pipeline_runtime = tensor_parallel
        .map(|_| create_runtime(backend, device).map_err(io::Error::other))
        .transpose()?;
    let mut listener = RanklessListener::bind(listen, *session.as_bytes(), grant)?;
    if let Some(bytes) = replicated_egress_bytes {
        listener.set_replicated_egress_bytes(bytes);
    }
    control
        .apply(&GroupOperation {
            id: OpId::new(session, 2),
            action: GroupAction::PublishEndpoint(listener.member()),
        })
        .await?;

    let manifest = loop {
        if let Some(view) = control.view(version).await {
            check_open(&view)?;
            if let Some(manifest) = view.manifest {
                break manifest;
            }
        }
        sleep(POLL).await;
    };

    let listener = listener.with_staging_ceiling(control.staging_bytes(grant).await);
    // Other local participants may be using this runtime to finish their joins.
    let (group, pipeline) = tokio::task::spawn_blocking(move || {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if let Some(tp) = tensor_parallel {
            let (group, pipeline) =
                listener.connect_axes(manifest.members, &order, tp, timeout, remaining)?;
            Ok((group, Some(pipeline)))
        } else {
            let group = listener.connect(manifest.members, Some(&order), timeout, remaining)?;
            Ok::<_, io::Error>((group, None))
        }
    })
    .await
    .map_err(io::Error::other)??;
    let digest = group
        .manifest_digest()
        .ok_or_else(|| io::Error::other("peer group failed during startup"))?;
    control
        .apply(&GroupOperation {
            id: OpId::new(session, 3),
            action: GroupAction::Ready {
                grant,
                manifest: digest,
            },
        })
        .await?;
    let (notify, closes) = mpsc::channel(1);
    let reporter = Arc::new(GrantReporter {
        notify,
        listen,
        seeds,
        operation: GroupOperation {
            id: OpId::new(session, 5),
            action: WorkerFaultEvidence::Granted(grant).report(failure_scope),
        },
        attempted: OnceLock::new(),
    });
    group.watch_grant(reporter.clone());
    let mut watched = vec![Arc::downgrade(&group)];
    let pipeline = pipeline.zip(pipeline_runtime).map(|(pipeline, runtime)| {
        pipeline.watch_grant(reporter);
        watched.push(Arc::downgrade(&pipeline));
        Comm::rankless_from_peers(pipeline, device, runtime)
    });
    executor.spawn(control.watch(watched, grant, OpId::new(session, 4), closes));
    Ok((Comm::rankless_from_peers(group, device, runtime), pipeline))
}

#[cfg(feature = "python")]
pub(super) async fn compute_rate_revision(
    listen: SocketAddr,
    seeds: Vec<SocketAddr>,
    grant: WorkerGrant,
    calibration: NodeId,
    update: Option<(u64, std::num::NonZeroU64)>,
) -> io::Result<u64> {
    use plait_core::group::GroupOutcome;

    let mut control = Rendezvous::new(listen, seeds).await?;
    if let Some((expected_revision, operations_per_second)) = update {
        let operation = GroupOperation {
            id: OpId::new(NodeId::random(), 1),
            action: GroupAction::PublishGrantedComputeRate {
                grant,
                calibration,
                expected_revision,
                operations_per_second,
            },
        };
        let result = control
            .call(|client, voter| {
                let operation = operation.clone();
                Box::pin(async move { client.apply_group(voter, &operation).await })
            })
            .await;
        return match result {
            Ok(GroupOutcome::ComputeRate(rate)) => Ok(rate.revision),
            Err(error) => Err(io::Error::other(format!(
                "compute rate rejected: {error:?}"
            ))),
            Ok(_) => Err(io::Error::other("unexpected compute rate receipt")),
        };
    }
    let reply = control
        .call(|client, voter| {
            Box::pin(client.query_group(
                voter,
                GroupLookup::ComputeRate {
                    device: grant.device,
                    calibration,
                },
            ))
        })
        .await;
    match reply {
        GroupReply::ComputeRate(rate) => Ok(rate
            .filter(|rate| rate.incarnation == grant.incarnation)
            .map_or(0, |rate| rate.revision)),
        _ => Err(io::Error::other("unexpected compute rate reply")),
    }
}

fn control_runtime() -> io::Result<&'static Runtime> {
    static EXECUTOR: OnceLock<io::Result<Runtime>> = OnceLock::new();
    EXECUTOR
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(1)
                .thread_name("plait-peer-control")
                .build()
        })
        .as_ref()
        .map_err(|error| io::Error::new(error.kind(), error.to_string()))
}

fn check_open(view: &GroupView) -> io::Result<()> {
    if view.phase.accepts_peer_join() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "Plait group cannot be joined in phase {:?}",
            view.phase
        )))
    }
}

const POLL: Duration = Duration::from_millis(100);
const WATCH_INTERVAL: Duration = Duration::from_millis(250);
const METADATA_QUERY_TIMEOUT: Duration = Duration::from_millis(750);
const WATCH_DISCOVERY_INTERVAL: Duration = Duration::from_secs(10);
const REPORT_TIMEOUT: Duration = Duration::from_secs(15);
type PeerCall<'a, T> = Pin<Box<dyn Future<Output = io::Result<T>> + Send + 'a>>;

pub(crate) struct GrantReporter {
    notify: mpsc::Sender<Option<GroupFault>>,
    listen: SocketAddr,
    seeds: Vec<SocketAddr>,
    operation: GroupOperation,
    attempted: OnceLock<()>,
}

impl GrantReporter {
    pub(crate) fn closed(&self, fault: Option<GroupFault>) {
        let _ = self.notify.try_send(fault);
    }

    pub(crate) fn withdraw(&self) {
        // A confirmed local error can arrive after the normal group watcher
        // has already observed revocation or reported a network failure.
        self.attempted.get_or_init(|| {
            if let Err(error) = self.report() {
                eprintln!("Plait unavailable worker report: {error}");
            }
        });
    }

    fn report(&self) -> io::Result<()> {
        let executor = control_runtime()?;
        let listen = self.listen;
        let seeds = self.seeds.clone();
        let operation = self.operation.clone();
        let (deliver, receipt) = std::sync::mpsc::sync_channel(1);
        executor.spawn(async move {
            let result = timeout(REPORT_TIMEOUT, async {
                let mut control = Rendezvous::new(listen, seeds).await?;
                control.apply(&operation).await
            })
            .await;
            let _ = deliver.send(result);
        });
        // Wait for the receipt before returning an error that may exit the worker.
        receipt
            .recv_timeout(REPORT_TIMEOUT)
            .map_err(io::Error::other)?
            .map_err(io::Error::other)?
    }
}

struct Rendezvous {
    client: Client,
    seeds: Vec<SocketAddr>,
    voters: Vec<NodeId>,
}

impl Rendezvous {
    async fn new(listen: SocketAddr, seeds: Vec<SocketAddr>) -> io::Result<Self> {
        let reply = SocketAddr::new(listen.ip(), 0);
        Ok(Self {
            client: Client::new_on(&reply.to_string()).await?,
            seeds,
            voters: Vec::new(),
        })
    }

    async fn apply(&mut self, operation: &GroupOperation) -> io::Result<()> {
        self.call(|client, voter| {
            let operation = operation.clone();
            Box::pin(async move { client.apply_group(voter, &operation).await })
        })
        .await
        .map(|_| ())
        .map_err(|error| io::Error::other(format!("group join rejected: {error:?}")))
    }

    async fn view(&mut self, version: GroupVersion) -> Option<GroupView> {
        let reply = self
            .call(|client, voter| {
                Box::pin(client.query_group(voter, GroupLookup::Version(version)))
            })
            .await;
        // A reachable replica can be behind on publications from other workers.
        self.voters.rotate_left(1);
        match reply {
            GroupReply::Version(view) => view,
            _ => None,
        }
    }

    async fn staging_bytes(&mut self, grant: WorkerGrant) -> u64 {
        let read = async {
            for &voter in &self.voters {
                if let Ok(GroupReply::PeerStagingBytes(Some(bytes))) = self
                    .client
                    .query_group(voter, GroupLookup::PeerStagingBytes(grant))
                    .await
                {
                    return bytes;
                }
            }
            0
        };
        // An older voter or a slow lookup limits this join to the blocking path.
        timeout(METADATA_QUERY_TIMEOUT, read).await.unwrap_or(0)
    }

    async fn watch(
        mut self,
        groups: Vec<Weak<PeerGroup>>,
        grant: WorkerGrant,
        id: OpId,
        mut closes: mpsc::Receiver<Option<GroupFault>>,
    ) {
        let fault = tokio::select! {
            biased;
            closed = closes.recv() => closed.flatten(),
            () = self.wait_for_revocation(grant) => None,
        };
        for group in groups.iter().filter_map(Weak::upgrade) {
            group.cancel();
        }
        if let Some(fault) = fault {
            let operation = GroupOperation {
                id,
                action: GroupAction::ReportFailure { grant, fault },
            };
            // Keep the operation alive briefly even if the caller drops the failed handle.
            let _ = timeout(REPORT_TIMEOUT, self.apply(&operation)).await;
        }
    }

    async fn wait_for_revocation(&mut self, grant: WorkerGrant) {
        let mut ticks = interval(WATCH_INTERVAL);
        ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut refresh_at = Instant::now() + WATCH_DISCOVERY_INTERVAL;
        let mut legacy = BTreeSet::new();
        loop {
            ticks.tick().await;
            let status = self.poll_grant(grant, &mut legacy).await;
            if status == GrantStatus::Ended {
                return;
            }
            if status == GrantStatus::Unknown || Instant::now() >= refresh_at {
                self.refresh().await;
                refresh_at = Instant::now() + WATCH_DISCOVERY_INTERVAL;
            }
        }
    }

    async fn poll_grant(
        &mut self,
        grant: WorkerGrant,
        legacy: &mut BTreeSet<NodeId>,
    ) -> GrantStatus {
        for _ in 0..self.voters.len() {
            let voter = self.voters[0];
            self.voters.rotate_left(1);
            if !legacy.contains(&voter) {
                let response = timeout(
                    METADATA_QUERY_TIMEOUT,
                    self.client
                        .query_group(voter, GroupLookup::GrantStatus(grant)),
                )
                .await;
                match response {
                    Ok(Ok(GroupReply::GrantStatus(GrantStatus::Unknown))) => continue,
                    Ok(Ok(GroupReply::GrantStatus(status))) => return status,
                    // Older voters close the connection on an unknown query variant.
                    Ok(Err(error)) if error.kind() == io::ErrorKind::ConnectionAborted => {}
                    _ => continue,
                }
            }
            let response = timeout(
                METADATA_QUERY_TIMEOUT,
                self.client
                    .query_group(voter, GroupLookup::Version(grant.version)),
            )
            .await;
            if let Ok(Ok(GroupReply::Version(view))) = response {
                legacy.insert(voter);
                let status = view.map_or(GrantStatus::Unknown, |view| view.grant_status(grant));
                if status != GrantStatus::Unknown {
                    return status;
                }
            }
        }
        GrantStatus::Unknown
    }

    async fn refresh(&mut self) {
        if let Ok(voters) = discover_voters(&self.seeds).await {
            self.voters.clear();
            for (id, address) in voters {
                self.client.add_peer(id, address);
                self.voters.push(id);
            }
        }
    }

    async fn call<T>(
        &mut self,
        invoke: impl for<'a> Fn(&'a mut Client, NodeId) -> PeerCall<'a, T>,
    ) -> T {
        loop {
            for (index, &voter) in self.voters.iter().enumerate() {
                if let Ok(result) = invoke(&mut self.client, voter).await {
                    self.voters.rotate_left(index);
                    return result;
                }
            }
            self.refresh().await;
            sleep(POLL).await;
        }
    }
}
