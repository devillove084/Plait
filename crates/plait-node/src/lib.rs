//! A leaderless Plait process. Every node can accept submissions, propose
//! commands, serve snapshots, and recover stalled instances.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use tokio::task::JoinHandle;

use plait_consensus::{Options, Replica};
use plait_core::group::{GroupLookup, GroupReply};
use plait_core::serving::{ServingAction, ServingLookup, ServingOperation, ServingView};
use plait_core::{
    AcceleratorKind, ClusterConfig, Command, ComputeRequirement, ComputeTask, DeviceId, FuseKind,
    NodeId, OpId, OpSpec, ShardId, TensorId, TensorKey,
};
use plait_fabric::Runtime;
use plait_net::{TcpTransport, Wire};
use serde::{Deserialize, Serialize};

mod execution_host;
mod observation;
mod storage;
use observation::PendingObservation;
pub use plait_net::control::{Client, QueryResult, discover_voters};
pub use storage::StorageMode;
use storage::{Image, Journal};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JoinTarget {
    pub node: NodeId,
    pub addr: String,
}

#[derive(Clone, Debug)]
pub struct NodeConfig {
    pub id: NodeId,
    pub listen: String,
    pub peers: BTreeMap<NodeId, String>,
    pub join: Option<JoinTarget>,
    pub options: Options,
    pub advertise_devices: bool,
}

impl NodeConfig {
    pub fn bootstrap(
        id: NodeId,
        listen: impl Into<String>,
        peers: BTreeMap<NodeId, String>,
    ) -> Self {
        Self {
            id,
            listen: listen.into(),
            peers,
            join: None,
            options: Options::default(),
            advertise_devices: true,
        }
    }
}

pub struct RunningNode {
    id: NodeId,
    addr: String,
    stop: Arc<Notify>,
    handle: Option<JoinHandle<io::Result<()>>>,
}

impl RunningNode {
    pub async fn start(config: NodeConfig) -> io::Result<Self> {
        Self::start_configured(config, None).await
    }

    pub async fn start_durable(config: NodeConfig, storage: StorageMode) -> io::Result<Self> {
        Self::start_configured(config, Some(storage)).await
    }

    async fn start_configured(
        mut config: NodeConfig,
        storage: Option<StorageMode>,
    ) -> io::Result<Self> {
        let journal = storage
            .map(|mode| Journal::open(mode, config.id))
            .transpose()?;
        let (replica, journal) = if let Some((mut journal, saved)) = journal {
            let mut replica = if let Some(image) = saved {
                config.join = image.pending_join;
                for (id, address) in image.peers {
                    config.peers.entry(id).or_insert(address);
                }
                let replica = Replica::<Runtime>::from_snapshot(
                    config.id,
                    image.snapshot,
                    config.options.clone(),
                );
                if !replica.violations().is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "replica journal cannot be replayed",
                    ));
                }
                replica
            } else {
                let replica = Self::bootstrap_replica(&config)?;
                journal.install(Image {
                    id: config.id,
                    snapshot: replica.snapshot(),
                    peers: config.peers.clone(),
                    pending_join: config.join.clone(),
                })?;
                replica
            };
            replica.track_deltas();
            (replica, Some(journal))
        } else {
            (Self::bootstrap_replica(&config)?, None)
        };
        let mut peers = config.peers.clone();
        if let Some(join) = &config.join {
            peers.insert(join.node, join.addr.clone());
        }
        let transport = TcpTransport::bind(config.id, &config.listen, peers).await?;
        Ok(Self::spawn(config, transport, replica, journal))
    }

    fn bootstrap_replica(config: &NodeConfig) -> io::Result<Replica<Runtime>> {
        let peers = config
            .peers
            .keys()
            .copied()
            .filter(|_| config.join.is_none());
        let cluster = ClusterConfig::new(std::iter::once(config.id).chain(peers))
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
        Ok(Replica::new(
            config.id,
            cluster,
            Runtime::new(),
            config.options.clone(),
        ))
    }

    pub fn start_with_transport(
        config: NodeConfig,
        transport: TcpTransport,
        cluster: ClusterConfig,
    ) -> io::Result<Self> {
        let replica = Replica::new(config.id, cluster, Runtime::new(), config.options.clone());
        Ok(Self::spawn(config, transport, replica, None))
    }

    fn spawn(
        config: NodeConfig,
        transport: TcpTransport,
        replica: Replica<Runtime>,
        journal: Option<Journal>,
    ) -> Self {
        let id = config.id;
        let addr = transport.local_addr().to_owned();
        transport.advertise_voter(config.join.is_none() && replica.config().is_voter(id));
        let stop = Arc::new(Notify::new());
        let task_stop = stop.clone();
        let state = NodeState {
            id,
            transport,
            replica,
            journal,
            options: config.options,
            joined: config.join.is_none(),
            join: config.join,
            advertise_devices: config.advertise_devices,
            staged_proposed: BTreeSet::new(),
            commit_proposed: BTreeSet::new(),
            devices_registered: false,
            last_join_poll: 0,
            last_ready: 0,
            started: Instant::now(),
            observations: BTreeMap::new(),
            execution_hosts: execution_host::HostWatch::default(),
            next_hosted_pd_scan: 0,
        };
        let handle = tokio::spawn(state.run(task_stop));
        Self {
            id,
            addr,
            stop,
            handle: Some(handle),
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    pub async fn wait(&mut self) -> io::Result<()> {
        let Some(handle) = &mut self.handle else {
            return Ok(());
        };
        let result = handle.await;
        self.handle = None;
        result.map_err(io::Error::other)?
    }
}

impl Drop for RunningNode {
    fn drop(&mut self) {
        self.stop.notify_waiters();
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

struct NodeState {
    id: NodeId,
    transport: TcpTransport,
    replica: Replica<Runtime>,
    journal: Option<Journal>,
    options: Options,
    join: Option<JoinTarget>,
    advertise_devices: bool,
    joined: bool,
    staged_proposed: BTreeSet<NodeId>,
    commit_proposed: BTreeSet<NodeId>,
    devices_registered: bool,
    last_join_poll: u64,
    last_ready: u64,
    started: Instant,
    observations: BTreeMap<(NodeId, u64), PendingObservation>,
    execution_hosts: execution_host::HostWatch,
    next_hosted_pd_scan: u64,
}

impl NodeState {
    fn now(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    async fn run(mut self, stop: Arc<Notify>) -> io::Result<()> {
        let mut ticker = tokio::time::interval(Duration::from_millis(10));
        loop {
            #[allow(clippy::large_enum_variant)]
            enum Event {
                Stop,
                Incoming(Option<(NodeId, Wire)>),
                Tick,
            }
            let event = tokio::select! {
                _ = stop.notified() => Event::Stop,
                message = self.transport.recv() => Event::Incoming(message),
                _ = ticker.tick() => Event::Tick,
            };
            match event {
                Event::Stop => break,
                Event::Incoming(None) => break,
                Event::Incoming(Some((from, wire))) => {
                    let now = self.now();
                    self.handle(from, wire, now).await?;
                    self.persist()?;
                    self.advertise_voter();
                    self.reply_to_observations().await;
                    self.flush().await;
                }
                Event::Tick => {
                    let now = self.now();
                    self.replica.tick(now);
                    if self.joined && self.replica.config().is_voter(self.id) {
                        for operation in self
                            .execution_hosts
                            .expirations(self.replica.machine(), now)
                        {
                            self.propose(Command::Serving(Box::new(operation)), now);
                        }
                        if now >= self.next_hosted_pd_scan {
                            self.next_hosted_pd_scan = now + 1000;
                            for intent in self.replica.machine().serving_hosted_work() {
                                self.propose(
                                    Command::Serving(Box::new(ServingOperation {
                                        id: OpId::new(NodeId::random(), 1),
                                        action: ServingAction::ResolveHostedServing {
                                            intent,
                                            cancel: false,
                                        },
                                    })),
                                    now,
                                );
                            }
                            for intent in self.replica.machine().serving_hosted_pd_work() {
                                self.propose(
                                    Command::Serving(Box::new(ServingOperation {
                                        id: OpId::new(NodeId::random(), 1),
                                        action: ServingAction::ResolveHostedPd {
                                            intent,
                                            cancel: false,
                                        },
                                    })),
                                    now,
                                );
                            }
                            for action in self.replica.machine().serving_hosted_pd_transition_work()
                            {
                                self.propose(
                                    Command::Serving(Box::new(ServingOperation {
                                        id: OpId::new(NodeId::random(), 1),
                                        action,
                                    })),
                                    now,
                                );
                            }
                        }
                    }
                    self.maybe_register_devices(now);
                    self.persist()?;
                    self.advertise_voter();
                    self.reply_to_observations().await;
                    self.drive_join(now).await;
                    self.refresh_peers();
                    self.flush().await;
                }
            }
        }
        Ok(())
    }

    async fn handle(&mut self, from: NodeId, wire: Wire, now: u64) -> io::Result<()> {
        match wire {
            Wire::Consensus(message) => {
                if self.joined {
                    self.replica.handle(from, message, now);
                }
            }
            Wire::JoinRequest { node, addr } => {
                self.transport.set_peer(node, addr.clone());
                if !self.joined || node == self.id {
                    return Ok(());
                }
                if self.replica.config().is_member(node) {
                    self.send_snapshot(node).await;
                } else if self.replica.config().is_voter(self.id)
                    && self.staged_proposed.insert(node)
                {
                    self.propose(
                        Command::Membership(plait_core::MembershipChange::StageAdd { node, addr }),
                        now,
                    );
                }
            }
            Wire::SnapshotRequest { node } => {
                if self.joined && self.replica.config().is_member(node) {
                    self.send_snapshot(node).await;
                }
            }
            Wire::Snapshot {
                snapshot,
                mut peers,
            } => {
                if self.joined {
                    return Ok(());
                }
                if !snapshot.config.is_member(self.id) {
                    return Ok(());
                }
                // The sender's peer table may omit its own address. Keep the
                // joining node's known addresses for the installed membership.
                for (node, address) in self.transport.peers().await {
                    peers.entry(node).or_insert(address);
                }
                peers.retain(|node, _| *node != self.id && snapshot.config.is_member(*node));
                let mut replica = Replica::from_snapshot(self.id, snapshot, self.options.clone());
                if !replica.violations().is_empty() {
                    return Ok(());
                }
                if let Some(journal) = &mut self.journal {
                    journal.install(Image {
                        id: self.id,
                        snapshot: replica.snapshot(),
                        peers: peers.clone(),
                        pending_join: None,
                    })?;
                    replica.track_deltas();
                }
                self.replica = replica;
                self.execution_hosts = execution_host::HostWatch::default();
                self.transport.set_peers(peers);
                self.joined = true;
                self.refresh_peers();
                self.send_ready(now).await;
            }
            Wire::Ready { node } => {
                if self.joined
                    && self.replica.config().is_voter(self.id)
                    && self.replica.config().staged_add().contains(&node)
                    && self.commit_proposed.insert(node)
                {
                    self.propose(
                        Command::Membership(plait_core::MembershipChange::CommitAdd { node }),
                        now,
                    );
                }
            }
            Wire::Submit {
                request,
                command,
                reply_to,
            } => {
                self.transport.set_peer(from, reply_to);
                let can_propose = self.joined && self.replica.config().is_voter(self.id);
                let result = if !can_propose {
                    Err("this node is not a voter".to_owned())
                } else {
                    match self.replica.machine().validate(&command) {
                        Err(error) => Err(error),
                        Ok(()) => self
                            .replica
                            .propose(command, now)
                            .map(|_| String::new())
                            .map_err(|error| error.to_string()),
                    }
                };
                let (ok, message) = match result {
                    Ok(message) => (true, message),
                    Err(message) => (false, message),
                };
                self.persist()?;
                let _ = self
                    .transport
                    .send(
                        from,
                        Wire::Ack {
                            request,
                            ok,
                            message,
                        },
                    )
                    .await;
            }
            Wire::Query {
                request,
                op,
                reply_to,
            } => {
                self.transport.set_peer(from, reply_to);
                let runtime = self.replica.machine();
                let reply = Wire::QueryReply {
                    request,
                    op,
                    declared: runtime.op(op).is_some(),
                    values: runtime.result(op).map(<[f64]>::to_vec),
                    executed: self.replica.executed_count(),
                    voters: self.replica.config().voters().iter().copied().collect(),
                };
                let _ = self.transport.send(from, reply).await;
            }
            Wire::Ping => {
                let _ = self.transport.send(from, Wire::Pong).await;
            }
            Wire::GroupQuery {
                request,
                lookup,
                reply_to,
            } => {
                self.transport.set_peer(from, reply_to);
                let runtime = self.replica.machine();
                let view = match lookup {
                    GroupLookup::Version(version) => {
                        GroupReply::Version(runtime.peer_group(version))
                    }
                    GroupLookup::Receipt(id) => {
                        GroupReply::Receipt(runtime.peer_group_receipt(id).cloned())
                    }
                    GroupLookup::Placement(group) => {
                        GroupReply::Placement(runtime.placement_inventory(group))
                    }
                    GroupLookup::LinkProbes => GroupReply::LinkProbes(runtime.link_probes()),
                };
                let _ = self
                    .transport
                    .send(
                        from,
                        Wire::GroupQueryReply {
                            request,
                            lookup,
                            view: Box::new(view),
                        },
                    )
                    .await;
            }
            Wire::ServingQuery {
                request,
                lookup,
                reply_to,
            } => {
                self.transport.set_peer(from, reply_to);
                let runtime = self.replica.machine();
                let view = match lookup {
                    ServingLookup::Group(version) => {
                        ServingView::Group(runtime.serving_group(version).cloned())
                    }
                    ServingLookup::Operation(operation) => {
                        ServingView::Operation(runtime.serving_receipt(operation).cloned())
                    }
                    ServingLookup::Resources => ServingView::Resources(runtime.serving_resources()),
                    ServingLookup::Routes(id) => ServingView::Routes(runtime.serving_routes(id)),
                    ServingLookup::Dispatch(version) => {
                        ServingView::Dispatch(runtime.serving_dispatch(version))
                    }
                    ServingLookup::W4Drain(version) => {
                        ServingView::W4Drain(runtime.serving_w4_drain(version))
                    }
                    ServingLookup::DecoderRoutes(id) => {
                        ServingView::DecoderRoutes(runtime.serving_decoder_routes(id))
                    }
                    ServingLookup::DecoderDrain(version) => {
                        ServingView::DecoderDrain(runtime.serving_decoder_drain(version))
                    }
                    ServingLookup::ResidentOwner(grant) => {
                        ServingView::ResidentOwner(runtime.serving_resident_owner(grant))
                    }
                    ServingLookup::DecoderFailover(id) => {
                        ServingView::DecoderFailover(runtime.serving_decoder_failover(id))
                    }
                    ServingLookup::DecoderSwitch(id) => {
                        ServingView::DecoderSwitch(runtime.serving_decoder_switch(id))
                    }
                    ServingLookup::DecoderRouteWork(id) => {
                        ServingView::DecoderRouteWork(runtime.serving_decoder_route_work(id))
                    }
                    ServingLookup::DecoderAdmissionRoutes(id) => {
                        ServingView::DecoderAdmissionRoutes(
                            runtime.serving_decoder_admission_routes(id),
                        )
                    }
                    ServingLookup::ExecutionHostLiveness(host) => {
                        ServingView::ExecutionHostLiveness(runtime.serving_execution_host(host))
                    }
                    ServingLookup::HostedAssignments(host) => {
                        ServingView::HostedAssignments(runtime.serving_hosted_assignments(host))
                    }
                    ServingLookup::HostedManagedAssignments(host) => {
                        ServingView::HostedManagedAssignments(
                            runtime.serving_hosted_managed_assignments(host),
                        )
                    }
                    ServingLookup::HostedPdPending(key) => {
                        ServingView::HostedPdPending(runtime.serving_hosted_pd_pending(key))
                    }
                    ServingLookup::HostedRoutes(key) => {
                        ServingView::HostedRoutes(runtime.serving_hosted_routes(key))
                    }
                    ServingLookup::HostedServingPending(key) => {
                        ServingView::HostedServingPending(runtime.serving_hosted_pending(key))
                    }
                    ServingLookup::HostedServingPlanning(key) => {
                        ServingView::HostedServingPlanning(runtime.serving_mode_planning(key))
                    }
                    ServingLookup::HostedEvacuationPlanning(key) => {
                        ServingView::HostedEvacuationPlanning(
                            runtime.serving_hosted_evacuation(key),
                        )
                    }
                    ServingLookup::ExecutionDrain(target) => {
                        ServingView::ExecutionDrain(runtime.serving_execution_drain(target))
                    }
                    ServingLookup::HostedPdPlanning(key) => {
                        ServingView::HostedPdPlanning(runtime.serving_hosted_pd_planning(key))
                    }
                    ServingLookup::HostedPdReconfiguration(key) => {
                        ServingView::HostedPdReconfiguration(
                            runtime.serving_hosted_pd_reconfiguration(key),
                        )
                    }
                    ServingLookup::HostedPdRecovery(key) => {
                        ServingView::HostedPdRecovery(runtime.serving_hosted_pd_recovery(key))
                    }
                    ServingLookup::HostedPdTransition(key) => ServingView::HostedPdTransition(
                        runtime.serving_hosted_pd_transition_status(key),
                    ),
                    ServingLookup::HostedProbeEndpoints => {
                        ServingView::HostedProbeEndpoints(runtime.serving_hosted_probe_endpoints())
                    }
                    ServingLookup::HostedPlanning => {
                        ServingView::HostedPlanning(runtime.serving_hosted_planning())
                    }
                    ServingLookup::DecoderClaims(version) => {
                        ServingView::DecoderClaims(runtime.serving_decoder_claims(version))
                    }
                    ServingLookup::DecoderCheckpoints {
                        model,
                        work,
                        transfer,
                    } => ServingView::DecoderCheckpoints(
                        runtime.serving_decoder_checkpoints(model, work, transfer),
                    ),
                    ServingLookup::DecoderReplay {
                        model,
                        work,
                        offset,
                    } => {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .ok()
                            .and_then(|elapsed| u64::try_from(elapsed.as_millis()).ok());
                        ServingView::DecoderReplay(now.and_then(|now| {
                            runtime.serving_decoder_replay(model, work, offset, now)
                        }))
                    }
                };
                let _ = self
                    .transport
                    .send(
                        from,
                        Wire::ServingQueryReply {
                            request,
                            lookup,
                            view: Box::new(view),
                        },
                    )
                    .await;
            }
            Wire::ObserveServingGroup {
                request,
                operation,
                group,
                epoch,
                reply_to,
            } => {
                self.transport.set_peer(from, reply_to);
                let operation = ServingOperation {
                    id: operation,
                    action: ServingAction::ObserveGroup { group, epoch },
                };
                self.start_observation(from, request, operation, now).await;
            }
            Wire::Hello { .. }
            | Wire::Pong
            | Wire::DiscoverVoter
            | Wire::VoterIdentity(_)
            | Wire::Ack { .. }
            | Wire::QueryReply { .. }
            | Wire::ServingQueryReply { .. }
            | Wire::ServingGroupObserved { .. }
            | Wire::GroupQueryReply { .. } => {}
        }
        Ok(())
    }

    fn advertise_voter(&self) {
        self.transport
            .advertise_voter(self.joined && self.replica.config().is_voter(self.id));
    }

    fn persist(&mut self) -> io::Result<()> {
        if let Some(journal) = &mut self.journal
            && let Some(delta) = self.replica.take_delta()
        {
            journal.record(delta)?;
        }
        Ok(())
    }

    fn maybe_register_devices(&mut self, now: u64) {
        if !self.advertise_devices
            || self.devices_registered
            || !self.replica.config().is_voter(self.id)
            || (self.join.is_some() && !self.joined)
        {
            return;
        }
        let profiles = plait_device::discover(self.id);
        for profile in &profiles {
            self.propose(Command::RegisterDevice(profile.clone()), now);
        }
        self.devices_registered = true;
    }

    async fn drive_join(&mut self, now: u64) {
        if let Some(join) = &self.join {
            if !self.joined && now.saturating_sub(self.last_join_poll) >= 300 {
                self.last_join_poll = now;
                let _ = self
                    .transport
                    .send(
                        join.node,
                        Wire::JoinRequest {
                            node: self.id,
                            addr: self.transport.local_addr().to_owned(),
                        },
                    )
                    .await;
                let _ = self
                    .transport
                    .send(join.node, Wire::SnapshotRequest { node: self.id })
                    .await;
            }
        }
        if self.joined
            && !self.replica.config().is_voter(self.id)
            && self.replica.config().is_member(self.id)
            && now.saturating_sub(self.last_ready) >= 300
        {
            self.last_ready = now;
            self.send_ready(now).await;
        }
    }

    async fn send_ready(&mut self, _now: u64) {
        let voters: Vec<NodeId> = self.replica.config().voters().iter().copied().collect();
        for voter in voters {
            let _ = self
                .transport
                .send(voter, Wire::Ready { node: self.id })
                .await;
        }
    }

    async fn send_snapshot(&mut self, to: NodeId) {
        let peers = self.transport.peers().await;
        if !peers.contains_key(&to) {
            return;
        }
        let wire = Wire::Snapshot {
            snapshot: self.replica.snapshot(),
            peers,
        };
        let _ = self.transport.send(to, wire).await;
    }

    fn refresh_peers(&mut self) {
        for (node, addr) in self.replica.machine().members() {
            if !addr.is_empty() {
                self.transport.set_peer(*node, addr.clone());
            }
        }
    }

    fn propose(&mut self, command: Command, now: u64) {
        if let Err(error) = self.replica.propose(command, now) {
            eprintln!("{}: proposal failed: {error}", self.id);
        }
    }

    async fn flush(&mut self) {
        for outbound in self.replica.take_outbox() {
            if let Err(error) = self
                .transport
                .send(outbound.to, Wire::Consensus(outbound.message))
                .await
            {
                eprintln!("{}: send to {} failed: {error}", self.id, outbound.to);
            }
        }
    }
}

/// Start three TCP nodes, propose a fused sum through different nodes, and
/// verify that all replicas materialize the same result. This is the shortest
/// end-to-end proof that submission, consensus, membership, and fusion work
/// over real sockets.
pub async fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    let ids = [
        NodeId::derive("demo-a"),
        NodeId::derive("demo-b"),
        NodeId::derive("demo-c"),
    ];
    let mut transports = Vec::new();
    for id in ids {
        transports.push(TcpTransport::bind(id, "127.0.0.1:0", BTreeMap::new()).await?);
    }
    let addrs: Vec<String> = transports
        .iter()
        .map(|transport| transport.local_addr().to_owned())
        .collect();
    let mut peers = BTreeMap::new();
    for (id, addr) in ids.iter().zip(&addrs) {
        peers.insert(*id, addr.clone());
    }
    let mut nodes = Vec::new();
    for (id, transport) in ids.iter().zip(transports) {
        transport.set_peers(peers.clone());
        let config = NodeConfig {
            id: *id,
            listen: String::new(),
            peers: peers.clone(),
            join: None,
            options: Options::default(),
            advertise_devices: true,
        };
        let cluster = ClusterConfig::new(peers.keys().copied())?;
        nodes.push(RunningNode::start_with_transport(
            config, transport, cluster,
        )?);
    }

    let mut client = Client::new().await?;
    for (id, addr) in ids.iter().zip(&addrs) {
        client.add_peer(*id, addr.clone());
    }

    let op = OpId::new(NodeId::derive("demo-plan"), 1);
    let spec = OpSpec {
        id: op,
        kind: FuseKind::Sum,
        contributors: ids.iter().copied().collect(),
        tensor: TensorKey::new(TensorId(7), ShardId(0)),
        epoch: 1,
    };
    client.submit(ids[0], Command::DeclareOp(spec)).await?;
    wait_declared(&mut client, ids[0], op).await?;

    let partials = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]];
    for (index, node) in ids.iter().enumerate() {
        client
            .submit(
                *node,
                Command::Contribute {
                    op,
                    contributor: *node,
                    values: partials[index].to_vec(),
                },
            )
            .await?;
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    let result = loop {
        let query = client.query(ids[0], op).await?;
        if let Some(values) = query.values {
            break values;
        }
        if Instant::now() >= deadline {
            return Err("fused result did not materialize".into());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let expected = vec![111.0, 222.0, 333.0];
    if result != expected {
        return Err(format!("unexpected fused value: {result:?}").into());
    }
    let voters = client.query(ids[1], op).await?.voters;
    println!(
        "plait demo: 3 TCP peers, {} voters, fused sum {:?} on every query",
        voters.len(),
        result
    );
    drop(nodes);
    Ok(())
}

async fn wait_declared(client: &mut Client, node: NodeId, op: OpId) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if client.query(node, op).await?.declared {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "operator declaration timed out",
            ));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Run the distributed DSA reference operator over three in-process Plait ordering
/// replicas and compare it with the single-node reference.
pub fn run_dsa_demo() -> Result<(), Box<dyn std::error::Error>> {
    use plait_parallel::{AxisSizes, DsaConfig, DsaRunner, GlobalDsa, Topology};

    let config = DsaConfig::small();
    let global = GlobalDsa::new(&config);
    let expected = global.reference(&config);

    for (label, sizes) in [
        ("context-parallel", AxisSizes::new(1, 1, 1, 1, 3, 1)),
        ("tensor-parallel", AxisSizes::new(1, 1, 2, 1, 1, 1)),
    ] {
        let topology = Topology::with_prefix(sizes, label)?;
        let runner = DsaRunner::new(topology, config.clone(), &global)?;
        let output = runner.run()?;
        let mut max_error = 0.0f64;
        for (left, right) in output.values.iter().zip(&expected) {
            for (a, b) in left.iter().zip(right) {
                max_error = max_error.max((a - b).abs());
            }
        }
        if max_error > 1e-9 {
            return Err(format!("{label} DSA error {max_error}").into());
        }
        println!(
            "plait dsa: {label} output matches reference (max error {max_error:.3e}, {} instances)",
            output.instances
        );
    }
    Ok(())
}

/// Execute the three remaining parallelism axes for real: expert-parallel MoE,
/// data-parallel gradient accumulation, and pipeline-parallel stage handoffs.
/// Each runner uses the same Plait ordering cluster and fused runtime as the node.
pub fn run_parallel_demo() -> Result<(), Box<dyn std::error::Error>> {
    use plait_parallel::{
        AxisSizes, DpConfig, DpRunner, GlobalDp, GlobalMoe, GlobalPp, MoeConfig, MoeRunner,
        PpConfig, PpRunner, Topology,
    };

    let moe_config = MoeConfig::small();
    let moe_global = GlobalMoe::new(&moe_config);
    let moe_topology = Topology::with_prefix(
        AxisSizes::new(1, 1, 1, moe_config.experts, 1, 1),
        "demo-moe",
    )?;
    let moe = MoeRunner::new(moe_topology, moe_config.clone())?.run(&moe_global)?;
    let moe_expected = moe_global.reference(&moe_config);
    let moe_error = max_error(&moe.values, &moe_expected);
    println!(
        "plait moe: EP output matches reference (max error {moe_error:.3e}, {} instances)",
        moe.instances
    );

    let dp_config = DpConfig::small();
    let dp_global = GlobalDp::new(&dp_config);
    let dp_topology =
        Topology::with_prefix(AxisSizes::new(dp_config.replicas, 1, 1, 1, 1, 1), "demo-dp")?;
    let dp = DpRunner::new(dp_topology, dp_config.clone())?.run(&dp_global)?;
    let (dp_weights, dp_losses) = dp_global.reference(&dp_config);
    let dp_error = dp
        .weights
        .iter()
        .zip(&dp_weights)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        .max(
            dp.losses
                .iter()
                .zip(&dp_losses)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
        );
    println!(
        "plait dp: {}-replica optimizer matches reference (max error {dp_error:.3e}, {} instances)",
        dp_config.replicas, dp.instances
    );

    let pp_config = PpConfig::small();
    let pp_global = GlobalPp::new(&pp_config);
    let pp_topology =
        Topology::with_prefix(AxisSizes::new(1, pp_config.stages, 1, 1, 1, 1), "demo-pp")?;
    let pp = PpRunner::new(pp_topology, pp_config.clone())?.run(&pp_global)?;
    let pp_expected = pp_global.reference(&pp_config);
    let pp_error = max_error(&pp.values, &pp_expected);
    println!(
        "plait pp: {} stages / {} microbatches match reference (max error {pp_error:.3e}, max in flight {}, {} instances)",
        pp_config.stages, pp_config.microbatches, pp.max_in_flight, pp.instances
    );
    Ok(())
}

fn gemm_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a = (0..m * k)
        .map(|index| ((index * 17 + 3) % 23) as f32 * 0.125 - 1.0)
        .collect();
    let b = (0..k * n)
        .map(|index| ((index * 11 + 5) % 19) as f32 * 0.25 - 2.0)
        .collect();
    (a, b, vec![0.0; m * n])
}

fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}

fn verify_gemm(actual: &[f32], expected: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err("GEMM length mismatch".into());
    }
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        if (a - b).abs() > 1e-2 {
            return Err(format!("GEMM mismatch at {index}: {a} vs {b}").into());
        }
    }
    Ok(())
}

fn max_error(actual: &[Vec<f64>], expected: &[Vec<f64>]) -> f64 {
    let mut error = 0.0f64;
    for (left, right) in actual.iter().zip(expected) {
        for (a, b) in left.iter().zip(right) {
            error = error.max((a - b).abs());
        }
    }
    error
}

/// Run the real ibverbs self-test. Returns a skip message when no device is
/// configured; this is expected on hosts without an RNIC or Soft-RoCE.
#[cfg(all(target_os = "linux", feature = "rdma"))]
pub async fn run_rdma_selftest() -> Result<(), Box<dyn std::error::Error>> {
    let result = tokio::task::spawn_blocking(plait_rdma::selftest)
        .await
        .map_err(|error| io::Error::other(error.to_string()))??;
    match result {
        plait_rdma::SelfTest::Skipped(reason) => {
            println!("plait rdma: skipped ({reason})");
            println!(
                "Set up Soft-RoCE: sudo modprobe rdma_rxe && sudo rdma link add rxe0 type rxe netdev <iface>"
            );
        }
        plait_rdma::SelfTest::Passed { device, bytes } => {
            println!("plait rdma: {device} loopback write/read passed ({bytes} bytes)");
        }
    }
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "rdma")))]
pub async fn run_rdma_selftest() -> Result<(), Box<dyn std::error::Error>> {
    println!("plait rdma: skipped (requires Linux and the rdma feature)");
    Ok(())
}

/// Print the accelerators visible to this process.
pub fn run_devices() -> Result<(), Box<dyn std::error::Error>> {
    let node = NodeId::derive("local-device-discovery");
    let profiles = plait_device::discover(node);
    if profiles.is_empty() {
        println!("plait devices: none");
        return Ok(());
    }
    for profile in profiles {
        println!(
            "{}  arch={}  name={}  memory_mib={}  flops={}  rdma={}",
            profile.id,
            profile.arch,
            profile.name,
            profile.memory_bytes / (1024 * 1024),
            profile.flops,
            profile.rdma_capable
        );
    }
    Ok(())
}

/// Discover local devices, replicate their profiles, then schedule planned
/// operations onto eligible devices through the same Plait ordering cluster. NVIDIA
/// tasks execute real CUTLASS GEMMs; CPU tasks execute a CPU GEMM reference.
/// Placements and compute results are replicated commands.
pub fn run_hetero_demo() -> Result<(), Box<dyn std::error::Error>> {
    use plait_device::{DeviceRegistry, SchedulingPolicy};
    use plait_parallel::{AxisSizes, DsaConfig, Topology, dsa_plan};

    let local = NodeId::derive("hetero-discovery");
    let discovered = plait_device::discover(local);
    if discovered.is_empty() {
        return Err("no devices discovered".into());
    }
    let topology = Topology::with_prefix(AxisSizes::new(3, 1, 1, 1, 1, 1), "hetero")?;
    let nodes = topology.nodes().to_vec();
    let mut cluster = plait_parallel::LocalCluster::new(nodes.iter().copied());
    let mut registry = DeviceRegistry::new();

    // Spread discovered devices over the three simulated replicas. A real
    // cluster would discover profiles on each node and replicate them directly.
    let mut profiles = Vec::new();
    for (index, profile) in discovered.into_iter().enumerate() {
        let node = nodes[index % nodes.len()];
        let mut profile = profile;
        profile.id = DeviceId::new(node, profile.id.ordinal);
        profiles.push(profile);
    }
    for profile in &profiles {
        registry.register(profile.clone());
        cluster.run_command(profile.id.node, Command::RegisterDevice(profile.clone()))?;
    }

    let plan = dsa_plan(&topology, &DsaConfig::small());
    let operations: Vec<_> = plan.plan.ops.iter().take(32).cloned().collect();
    let arch_order: Vec<AcceleratorKind> = profiles.iter().map(|p| p.arch).collect();
    let mut executors: BTreeMap<u32, plait_gpu::GpuExecutor> = BTreeMap::new();
    let mut assignments = Vec::new();
    let mut gpu_tasks = 0usize;
    let mut cpu_tasks = 0usize;

    // A small GEMM is enough to exercise the real CUTLASS path while keeping the
    // demo result small enough to travel inline.
    let (rows, cols, inner) = (16usize, 16usize, 16usize);
    let (a, b, zeros) = gemm_inputs(rows, cols, inner);
    let cpu_expected = cpu_gemm(&a, &b, rows, cols, inner);

    for (index, op) in operations.iter().enumerate() {
        let requirement = ComputeRequirement {
            arch: Some(arch_order[index % arch_order.len()]),
            ..op.requirement()
        };
        let Some(device) = registry.select(&requirement, SchedulingPolicy::LeastLoaded) else {
            continue;
        };
        let Some(arch) = registry.profiles().get(&device).map(|profile| profile.arch) else {
            continue;
        };
        registry.reserve(device);

        let values: Vec<f64> = match arch {
            AcceleratorKind::Nvidia => {
                let ordinal = device.ordinal;
                if let std::collections::btree_map::Entry::Vacant(entry) = executors.entry(ordinal)
                {
                    entry.insert(plait_gpu::GpuExecutor::new(ordinal as usize)?);
                }
                let executor = executors.get(&ordinal).expect("executor inserted");
                let output = executor.sgemm(&a, &b, &zeros, rows, cols, inner, 1.0, 0.0)?;
                verify_gemm(&output, &cpu_expected)?;
                gpu_tasks += 1;
                output.into_iter().map(f64::from).collect()
            }
            AcceleratorKind::Cpu => {
                let output = cpu_gemm(&a, &b, rows, cols, inner);
                verify_gemm(&output, &cpu_expected)?;
                cpu_tasks += 1;
                output.into_iter().map(f64::from).collect()
            }
            other => {
                return Err(format!(
                    "no kernel adapter for {other}; refusing to use a placeholder"
                )
                .into());
            }
        };

        let task = OpId::new(NodeId::derive("hetero-plan"), index as u64 + 1);
        let output = TensorKey::new(TensorId(30_000 + index as u128), ShardId(0));
        cluster.run_command(
            device.node,
            Command::Compute(ComputeTask {
                task,
                kernel: op.name.clone(),
                device,
                requirement,
                inputs: Vec::new(),
                output,
                version: 1,
                values,
            }),
        )?;
        assignments.push((task, device, op.name.clone()));
    }

    for (task, device, kernel) in &assignments {
        for node in &nodes {
            if cluster.runtime(*node).placement(*task) != Some(*device) {
                return Err(format!(
                    "placement mismatch for {kernel} on {node}: expected {device}"
                )
                .into());
            }
        }
    }
    let mut counts: BTreeMap<AcceleratorKind, usize> = BTreeMap::new();
    for (_, device, _) in &assignments {
        if let Some(profile) = registry.profiles().get(device) {
            *counts.entry(profile.arch).or_default() += 1;
        }
    }
    println!(
        "plait hetero: {} devices, {} compute tasks, placements {:?}, CUTLASS GEMM tasks {}, CPU GEMM tasks {}",
        profiles.len(),
        assignments.len(),
        counts,
        gpu_tasks,
        cpu_tasks
    );
    for (task, device, kernel) in assignments.iter().take(8) {
        println!("  {task:?} -> {device} ({kernel})");
    }
    Ok(())
}

/// Execute real CUDA work on every visible NVIDIA device. The AXPY path uses a
/// compact embedded cubin; the GEMM path goes through the CUTLASS C++ FFI.
/// Benchmark one CUTLASS SGEMM per visible NVIDIA device.
pub fn run_gpu_bench() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    let node = NodeId::derive("gpu-bench");
    let profiles = plait_device::discover(node);
    let gpus: Vec<_> = profiles
        .iter()
        .filter(|profile| profile.arch == AcceleratorKind::Nvidia)
        .collect();
    if gpus.is_empty() {
        println!("plait gpu-bench: no NVIDIA device discovered");
        return Ok(());
    }
    let (m, n, k) = (1024usize, 1024usize, 1024usize);
    let a: Vec<f32> = (0..m * k)
        .map(|index| ((index * 17 + 3) % 23) as f32 * 0.125 - 1.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|index| ((index * 11 + 5) % 19) as f32 * 0.25 - 2.0)
        .collect();
    let c = vec![0.0f32; m * n];
    for profile in gpus {
        let ordinal = profile.id.ordinal as usize;
        let executor = match plait_gpu::GpuExecutor::new(ordinal) {
            Ok(executor) => executor,
            Err(error) => {
                println!("plait gpu-bench: ordinal={ordinal} skipped: {error}");
                continue;
            }
        };
        // Warmup, then median-ish mean of five launches.
        let _ = executor.sgemm(&a, &b, &c, m, n, k, 1.0, 0.0)?;
        let iterations = 5usize;
        let start = Instant::now();
        for _ in 0..iterations {
            let output = executor.sgemm(&a, &b, &c, m, n, k, 1.0, 0.0)?;
            if output.len() != m * n {
                return Err("SGEMM output length mismatch".into());
            }
        }
        let seconds = start.elapsed().as_secs_f64() / iterations as f64;
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / seconds / 1e9;
        println!(
            "plait gpu-bench: ordinal={ordinal} name=\"{}\" m={m} n={n} k={k} ms={:.3} gflops={:.1}",
            profile.name,
            seconds * 1e3,
            gflops
        );
    }
    Ok(())
}

pub fn run_gpu_demo() -> Result<(), Box<dyn std::error::Error>> {
    let node = NodeId::derive("gpu-demo");
    let profiles = plait_device::discover(node);
    let gpus: Vec<_> = profiles
        .iter()
        .filter(|profile| profile.arch == AcceleratorKind::Nvidia)
        .collect();
    if gpus.is_empty() {
        println!("plait gpu: no NVIDIA device discovered");
        return Ok(());
    }
    let mut executed = 0usize;
    for profile in &gpus {
        let executor = match plait_gpu::GpuExecutor::new(profile.id.ordinal as usize) {
            Ok(executor) => executor,
            Err(error) => {
                println!("plait gpu: {} skipped: {error}", profile.name);
                continue;
            }
        };
        let x: Vec<f32> = (0..4096).map(|index| index as f32 * 0.001).collect();
        let y: Vec<f32> = (0..4096).map(|index| 1.0 - index as f32 * 0.0005).collect();
        let alpha = 1.5f32;
        let output = executor.axpy(&x, &y, alpha)?;
        for (index, value) in output.iter().enumerate() {
            let expected = alpha * x[index] + y[index];
            if (value - expected).abs() > 1e-4 {
                return Err(format!("{} AXPY mismatch at {index}", profile.name).into());
            }
        }
        executed += 1;
        println!(
            "plait gpu: {} ({}) AXPY ok ({} elements, ordinal {})",
            profile.name,
            profile.arch,
            output.len(),
            profile.id.ordinal
        );
    }

    let m = 128usize;
    let n = 128usize;
    let k = 128usize;
    let a: Vec<f32> = (0..m * k)
        .map(|index| (index % 13) as f32 * 0.125)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|index| (index % 11) as f32 * 0.25 - 0.5)
        .collect();
    let c = vec![0.0f32; m * n];
    let cutlass_result = plait_gpu::GpuExecutor::new(0)
        .and_then(|executor| executor.sgemm(&a, &b, &c, m, n, k, 1.0, 0.0));
    match cutlass_result {
        Ok(output) => {
            for row in 0..m {
                for col in 0..n {
                    let mut expected = 0.0f32;
                    for inner in 0..k {
                        expected += a[row * k + inner] * b[inner * n + col];
                    }
                    if (output[row * n + col] - expected).abs() > 1e-2 {
                        return Err(format!("CUTLASS SGEMM mismatch at [{row}][{col}]").into());
                    }
                }
            }
            println!(
                "plait gpu: CUTLASS device-pointer SGEMM {m}x{n}x{k} ok on {} visible GPU(s)",
                executed
            );
        }
        Err(error) => {
            println!("plait gpu: CUTLASS SGEMM skipped: {error}");
        }
    }
    Ok(())
}

/// Benchmark the peer-memory all-reduce against the same byte count used by
/// the NCCL `all_reduce_perf` harness. This is intentionally separate from the
/// consensus matrix runner: it measures the GPU data plane, not agreement.
#[allow(clippy::too_many_arguments)]
pub fn run_collectives_demo(
    ranks: usize,
    bytes: usize,
    iters: usize,
    algorithm: &str,
    op: &str,
    channels: usize,
    resident: bool,
    device_ring: bool,
    protocol: &str,
    trace: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use plait_collective::{AllReduceAlgorithm, PeerGroup};
    use std::time::Instant;

    let requested = algorithm.to_owned();
    let mut algorithm = match algorithm {
        "ring" => AllReduceAlgorithm::Ring,
        "recursive" => AllReduceAlgorithm::RecursiveDoubling,
        "tree" => AllReduceAlgorithm::Tree,
        "reduce-broadcast" => AllReduceAlgorithm::ReduceBroadcast,
        "auto" => AllReduceAlgorithm::Ring,
        other => return Err(format!("unknown allreduce algorithm {other}").into()),
    };

    let available = cudarc_device_count();
    let ranks = if ranks == 0 { available } else { ranks };
    if ranks < 2 {
        return Err(format!("collectives need at least 2 ranks, found {available}").into());
    }
    if op == "alltoall" && bytes % (ranks * std::mem::size_of::<f32>()) != 0 {
        return Err("alltoall bytes must divide by ranks".into());
    }
    if matches!(op, "allgather" | "reducescatter")
        && bytes % (ranks * std::mem::size_of::<f32>()) != 0
    {
        return Err(format!("{op} bytes must divide by ranks").into());
    }
    let mut group = PeerGroup::new_with_channels(ranks, channels)?;
    if requested == "auto" {
        algorithm = group.recommended_allreduce(bytes);
    }
    let count = (bytes / std::mem::size_of::<f32>()).max(1);
    let inputs: Vec<Vec<f32>> = (0..ranks)
        .map(|rank| {
            (0..count)
                .map(|index| (rank + 1) as f32 + (index % 17) as f32 * 0.001)
                .collect()
        })
        .collect();
    let expected: Vec<f32> = (0..count)
        .map(|index| {
            let sum = (1..=ranks).sum::<usize>() as f32;
            sum + (index % 17) as f32 * 0.001 * ranks as f32
        })
        .collect();

    for _ in 0..5 {
        run_collective_op(&group, op, &inputs, &expected, algorithm)?;
    }
    if trace.is_some() {
        group.enable_trace(iters.saturating_mul(group.world_size()).max(1));
    }
    let start = Instant::now();
    for _ in 0..iters {
        run_collective_op(&group, op, &inputs, &expected, algorithm)?;
    }
    let seconds = start.elapsed().as_secs_f64() / iters as f64;
    if op == "allreduce" && protocol != "simple" {
        let checked = match protocol {
            "ll" => group.all_reduce_device_ring(&inputs)?,
            "ll128" => group.all_reduce_ll128(&inputs)?,
            other => return Err(format!("unknown protocol {other}").into()),
        };
        validate_device_output("allreduce", &inputs, &expected, &checked)?;
        let start = Instant::now();
        if protocol == "ll" {
            for _ in 0..iters {
                group.all_reduce_device_ring(&inputs)?;
            }
        } else {
            let _ = group.all_reduce_ll128_resident(&inputs, iters)?;
        }
        let elapsed = start.elapsed().as_secs_f64() / iters as f64;
        let bw = allreduce_bytes_f64(bytes) / elapsed / 1e9;
        let busbw = bw * 2.0 * (ranks - 1) as f64 / ranks as f64;
        println!(
            "plait collectives: protocol={protocol} allreduce ranks={ranks} bytes={} time_ms={:.6} algbw_GB/s={bw:.3} busbw_GB/s={busbw:.3}",
            (bytes / std::mem::size_of::<f32>()).max(1) * std::mem::size_of::<f32>(),
            elapsed * 1e3
        );
        return Ok(());
    }
    if device_ring && op == "alltoall" {
        let checked = group.all_to_all_device(&inputs)?;
        validate_device_output("alltoall", &inputs, &expected, &checked)?;
        let (duration, output) = group.all_to_all_device_resident(&inputs, iters)?;
        let _ = output;
        let elapsed = duration.as_secs_f64() / iters as f64;
        let bw = allreduce_bytes_f64(bytes) / elapsed / 1e9;
        let busbw = bw * 2.0 * (ranks - 1) as f64 / ranks as f64;
        println!(
            "plait collectives: device-ring alltoall ranks={ranks} bytes={} time_ms={:.6} algbw_GB/s={bw:.3} busbw_GB/s={busbw:.3}",
            (bytes / std::mem::size_of::<f32>()).max(1) * std::mem::size_of::<f32>(),
            elapsed * 1e3
        );
    } else if device_ring {
        if op == "broadcast" {
            let checked = group.broadcast_device(inputs[0].clone())?;
            validate_device_output("broadcast", &inputs, &expected, &checked)?;
            let (duration, _) = group.broadcast_device_resident(inputs[0].clone(), iters)?;
            let elapsed = duration.as_secs_f64() / iters as f64;
            let bw = allreduce_bytes_f64(bytes) / elapsed / 1e9;
            let busbw = bw * 2.0 * (ranks - 1) as f64 / ranks as f64;
            println!(
                "plait collectives: device-ring broadcast ranks={ranks} bytes={} time_ms={:.6} algbw_GB/s={bw:.3} busbw_GB/s={busbw:.3}",
                (bytes / std::mem::size_of::<f32>()).max(1) * std::mem::size_of::<f32>(),
                elapsed * 1e3
            );
            return Ok(());
        }
        if op == "allgather" {
            let width = inputs[0].len() / ranks;
            let chunks: Vec<Vec<f32>> = inputs
                .iter()
                .map(|values| values[..width].to_vec())
                .collect();
            let checked = group.all_gather_device(&chunks)?;
            validate_device_output(op, &inputs, &expected, &checked)?;
            let (duration, _) = group.all_gather_device_resident(&chunks, iters)?;
            let elapsed = duration.as_secs_f64() / iters as f64;
            let bw = allreduce_bytes_f64(bytes) / elapsed / 1e9;
            let busbw = bw * 2.0 * (ranks - 1) as f64 / ranks as f64;
            println!(
                "plait collectives: device-ring allgather ranks={ranks} bytes={} time_ms={:.6} algbw_GB/s={bw:.3} busbw_GB/s={busbw:.3}",
                (bytes / std::mem::size_of::<f32>()).max(1) * std::mem::size_of::<f32>(),
                elapsed * 1e3
            );
            return Ok(());
        }
        if op == "reducescatter" {
            let checked = group.reduce_scatter_device(&inputs)?;
            validate_device_output(op, &inputs, &expected, &checked)?;
            let (duration, _) = group.reduce_scatter_device_resident(&inputs, iters)?;
            let elapsed = duration.as_secs_f64() / iters as f64;
            let bw = allreduce_bytes_f64(bytes) / elapsed / 1e9;
            let busbw = bw * 2.0 * (ranks - 1) as f64 / ranks as f64;
            println!(
                "plait collectives: device-ring reducescatter ranks={ranks} bytes={} time_ms={:.6} algbw_GB/s={bw:.3} busbw_GB/s={busbw:.3}",
                (bytes / std::mem::size_of::<f32>()).max(1) * std::mem::size_of::<f32>(),
                elapsed * 1e3
            );
            return Ok(());
        }
        let contributions = device_collective_inputs(op, &inputs)?;
        let checked = group.all_reduce_device_ring(&contributions)?;
        let checked = device_collective_output(op, &inputs, checked);
        validate_device_output(op, &inputs, &expected, &checked)?;
        let (duration, output) = group.all_reduce_device_ring_resident(&contributions, iters)?;
        let _ = device_collective_output(op, &inputs, output);
        let elapsed = duration.as_secs_f64() / iters as f64;
        let bw = allreduce_bytes_f64(bytes) / elapsed / 1e9;
        let busbw = bw * 2.0 * (ranks - 1) as f64 / ranks as f64;
        println!(
            "plait collectives: device-ring {op} ranks={ranks} bytes={} time_ms={:.6} algbw_GB/s={bw:.3} busbw_GB/s={busbw:.3}",
            (bytes / std::mem::size_of::<f32>()).max(1) * std::mem::size_of::<f32>(),
            elapsed * 1e3
        );
    }
    if resident && op == "allreduce" {
        let (elapsed, output) = group.all_reduce_resident(&inputs, iters)?;
        if output.iter().any(|values| values.len() != inputs[0].len()) {
            return Err("resident allreduce output length mismatch".into());
        }
        let resident_seconds = elapsed.as_secs_f64() / iters as f64;
        let resident_algbw = allreduce_bytes_f64(bytes) / resident_seconds / 1e9;
        let resident_busbw = resident_algbw * 2.0 * (ranks - 1) as f64 / ranks as f64;
        println!(
            "plait collectives: resident allreduce ranks={ranks} bytes={} time_ms={:.6} algbw_GB/s={resident_algbw:.3} busbw_GB/s={resident_busbw:.3}",
            (bytes / std::mem::size_of::<f32>()).max(1) * std::mem::size_of::<f32>(),
            resident_seconds * 1e3
        );
    }
    let allreduce_bytes = count * std::mem::size_of::<f32>();
    let algbw = allreduce_bytes as f64 / seconds / 1e9;
    let busbw = algbw * 2.0 * (ranks - 1) as f64 / ranks as f64;
    println!(
        "plait collectives: op={op} algorithm={algorithm:?} ranks={} channels={} bytes={allreduce_bytes} time_ms={:.6} algbw_GB/s={algbw:.3} busbw_GB/s={busbw:.3}",
        group.world_size(),
        group.channels(),
        seconds * 1e3
    );
    if let Some(path) = trace {
        group.write_trace(path)?;
        println!(
            "plait collectives: chrome trace {} events at {path}",
            group.trace_events().len()
        );
    }
    Ok(())
}

fn allreduce_bytes_f64(bytes: usize) -> f64 {
    (bytes / std::mem::size_of::<f32>()).max(1) as f64 * std::mem::size_of::<f32>() as f64
}

fn device_collective_inputs(
    op: &str,
    inputs: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let n = inputs.len();
    let count = inputs[0].len();
    let width = count / n;
    match op {
        "allreduce" | "reduce" | "reducescatter" => Ok(inputs.to_vec()),
        "broadcast" => {
            let mut result = vec![vec![0.0f32; count]; n];
            result[0].copy_from_slice(&inputs[0]);
            Ok(result)
        }
        "allgather" => {
            let mut result = vec![vec![0.0f32; count]; n];
            for (rank, values) in inputs.iter().enumerate() {
                result[rank][rank * width..(rank + 1) * width].copy_from_slice(&values[..width]);
            }
            Ok(result)
        }
        "alltoall" => {
            let mut result = vec![vec![0.0f32; count]; n];
            for (source, values) in inputs.iter().enumerate() {
                for destination in 0..n {
                    let dst = destination * width + source * width;
                    result[source][dst..dst + width]
                        .copy_from_slice(&values[destination * width..(destination + 1) * width]);
                }
            }
            Ok(result)
        }
        other => Err(format!("unknown device collective op {other}").into()),
    }
}

fn device_collective_output(
    op: &str,
    inputs: &[Vec<f32>],
    reduced: Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let n = inputs.len();
    let count = inputs[0].len();
    let width = count / n;
    match op {
        "alltoall" => reduced
            .into_iter()
            .enumerate()
            .map(|(rank, values)| values[rank * width..(rank + 1) * width].to_vec())
            .collect(),
        _ => reduced,
    }
}

fn validate_device_output(
    op: &str,
    inputs: &[Vec<f32>],
    expected: &[f32],
    output: &[Vec<f32>],
) -> Result<(), Box<dyn std::error::Error>> {
    let n = inputs.len();
    let count = inputs[0].len();
    let width = count / n;
    match op {
        "allreduce" => {
            for values in output {
                if values
                    .iter()
                    .zip(expected)
                    .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                {
                    return Err("allreduce device correctness failed".into());
                }
            }
        }
        "reducescatter" => {
            for (rank, values) in output.iter().enumerate() {
                let chunk = (rank + 1) % n;
                let wanted = &expected[chunk * width..(chunk + 1) * width];
                if values[chunk * width..(chunk + 1) * width]
                    .iter()
                    .zip(wanted)
                    .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                {
                    return Err("reducescatter device correctness failed".into());
                }
            }
        }
        "broadcast" => {
            for values in output {
                if values
                    .iter()
                    .zip(&inputs[0])
                    .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                {
                    return Err("broadcast device correctness failed".into());
                }
            }
        }
        "reduce" => {
            if output[0]
                .iter()
                .zip(expected)
                .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
            {
                return Err("reduce device correctness failed".into());
            }
        }
        "allgather" => {
            let wanted: Vec<f32> = inputs
                .iter()
                .flat_map(|v| v[..width].iter().copied())
                .collect();
            for values in output {
                if values
                    .iter()
                    .zip(&wanted)
                    .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                {
                    return Err("allgather device correctness failed".into());
                }
            }
        }
        "alltoall" => {
            for destination in 0..n {
                for source in 0..n {
                    let want = &inputs[source][destination * width..(destination + 1) * width];
                    let actual = &output[destination][source * width..(source + 1) * width];
                    if actual
                        .iter()
                        .zip(want)
                        .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                    {
                        return Err("alltoall device correctness failed".into());
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn run_collective_op(
    group: &plait_collective::PeerGroup,
    op: &str,
    inputs: &[Vec<f32>],
    expected: &[f32],
    algorithm: plait_collective::AllReduceAlgorithm,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = group.world_size();
    let count = inputs[0].len();
    let output = match op {
        "allreduce" => group.all_reduce(inputs, algorithm)?,
        "broadcast" => group.broadcast(inputs[0].clone(), 0)?,
        "reduce" => {
            let reduced = group.reduce(inputs, 0)?;
            if reduced
                .iter()
                .zip(expected)
                .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
            {
                return Err("reduce correctness check failed".into());
            }
            return Ok(());
        }
        "allgather" => {
            let width = count / n;
            let chunks: Vec<Vec<f32>> = inputs
                .iter()
                .map(|values| values[..width].to_vec())
                .collect();
            group.all_gather(&chunks)?
        }
        "reducescatter" => group.reduce_scatter(inputs)?,
        "alltoall" => group.all_to_all(inputs)?,
        other => return Err(format!("unknown collective op {other}").into()),
    };
    match op {
        "allreduce" | "broadcast" => {
            let wanted = if op == "broadcast" {
                &inputs[0][..]
            } else {
                expected
            };
            for values in output {
                if values
                    .iter()
                    .zip(wanted)
                    .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                {
                    return Err(format!("{op} correctness check failed").into());
                }
            }
        }
        "allgather" => {
            let width = count / n;
            let wanted: Vec<f32> = inputs
                .iter()
                .flat_map(|v| v[..width].iter().copied())
                .collect();
            for values in output {
                if values
                    .iter()
                    .zip(&wanted)
                    .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                {
                    return Err("allgather correctness check failed".into());
                }
            }
        }
        "reducescatter" => {
            let width = count / n;
            for (rank, values) in output.iter().enumerate() {
                let chunk = (rank + 1) % n;
                let wanted = &expected[chunk * width..(chunk + 1) * width];
                if values[chunk * width..(chunk + 1) * width]
                    .iter()
                    .zip(wanted)
                    .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                {
                    return Err("reducescatter correctness check failed".into());
                }
            }
        }
        "alltoall" => {
            let width = count / n;
            for destination in 0..n {
                for source in 0..n {
                    let send = &inputs[source][destination * width..(destination + 1) * width];
                    let actual = &output[destination][source * width..(source + 1) * width];
                    if actual
                        .iter()
                        .zip(send)
                        .any(|(actual, expected)| (*actual - *expected).abs() > 1e-2)
                    {
                        return Err("alltoall correctness check failed".into());
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn cudarc_device_count() -> usize {
    // Keep the dependency-contained probe local to the demo; a host without
    // CUDA reports zero rather than panicking.
    std::process::Command::new("nvidia-smi")
        .args(["-L"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).lines().count())
        .unwrap_or(0)
}

/// Probe GPUDirect registration: a VMM device allocation registered with the
/// first verbs device. A skip means the host lacks an RDMA device, not that the
/// path is fast.
#[cfg(all(target_os = "linux", feature = "rdma"))]
pub fn run_gpu_rdma_probe() -> Result<(), Box<dyn std::error::Error>> {
    use plait_collective::ipc::VmmRegion;
    use plait_rdma::RdmaDevice;

    let Some(device) = RdmaDevice::open_first()? else {
        println!("plait gpu-rdma: skipped (no active ibverbs device)");
        return Ok(());
    };
    let mut region = VmmRegion::alloc(0, 1 << 20)?;
    let len = region.bytes();
    let probe = unsafe { device.probe_gpu_region(region.as_mut_ptr(), len) };
    match probe {
        Ok(()) => println!(
            "plait gpu-rdma: registered {} bytes of VMM memory on {}",
            len,
            device.name()
        ),
        Err(error) => println!(
            "plait gpu-rdma: provider {} rejected GPU registration: {error}",
            device.name()
        ),
    }
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "rdma")))]
pub fn run_gpu_rdma_probe() -> Result<(), Box<dyn std::error::Error>> {
    println!("plait gpu-rdma: skipped (requires Linux and the rdma feature)");
    Ok(())
}

/// Run one host's share of a two-host 12-replica matrix collective.
///
/// Manifest lines are `hex-node-id@listen-address`. The first six entries are
/// host A, the last six are host B. Only the driver host submits the command;
/// both hosts keep their nodes alive for `hold_seconds`.
/// Demonstrate the heterogeneous max-min planner on the real 4+8 mix.
pub fn run_hetero_plan(
    local_gflops: f64,
    remote_gflops: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use plait_parallel::{
        AsymmetricLayout, HeterogeneousLayout, LinkKind, RankProfile, RankWeight,
    };

    let mut profiles = Vec::new();
    for index in 0..4 {
        profiles.push(RankProfile {
            node: NodeId::derive(&format!("ampere-{index}")),
            gpus: 1,
            compute_units: local_gflops / 1000.0,
            link: LinkKind::Pcie,
            bandwidth_gbps: Some(25.0),
            latency_us: Some(5.0),
        });
    }
    for index in 0..8 {
        profiles.push(RankProfile {
            node: NodeId::derive(&format!("blackwell-{index}")),
            gpus: 1,
            compute_units: remote_gflops / 1000.0,
            link: LinkKind::Ub,
            bandwidth_gbps: Some(200.0),
            latency_us: Some(2.0),
        });
    }

    let message_bytes = 64 * 1024 * 1024;
    let plan = HeterogeneousLayout::plan_detailed(&profiles, message_bytes)?;
    println!(
        "plait hetero-plan: local_gflops={local_gflops:.1} remote_gflops={remote_gflops:.1} ranks=12 message_mib=64 local_share={:.3} remote_share={:.3}",
        plan.shares[..4].iter().map(|(_, share)| share).sum::<f64>(),
        plan.shares[4..].iter().map(|(_, share)| share).sum::<f64>()
    );
    for rows in [8usize, 64, 256] {
        let ranges = plan.layout.row_ranges(rows);
        let local: usize = ranges[..4].iter().map(|range| range.range.len()).sum();
        let remote: usize = ranges[4..].iter().map(|range| range.range.len()).sum();
        println!(
            "plait hetero-plan: rows={rows} local_rows={local} remote_rows={remote} predicted_makespan_ms={:.3}",
            plan.makespan * 1e3
        );
    }

    let equal = AsymmetricLayout::new(
        profiles
            .iter()
            .map(|profile| RankWeight {
                node: profile.node,
                gpus: profile.gpus,
                weight: 1,
            })
            .collect(),
    )?;
    let equal_ranges = equal.row_ranges(64);
    let local_equal: usize = equal_ranges[..4]
        .iter()
        .map(|range| range.range.len())
        .sum();
    let remote_equal: usize = equal_ranges[4..]
        .iter()
        .map(|range| range.range.len())
        .sum();
    println!(
        "plait hetero-plan: rows=64 equal_split_local={local_equal} equal_split_remote={remote_equal}"
    );
    Ok(())
}

pub async fn run_matrix_node(
    manifest: &str,
    local_start: usize,
    local_count: usize,
    driver: bool,
    hold_seconds: u64,
    shape: (usize, usize, usize),
    asymmetric: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use plait_parallel::{AxisSizes, HeterogeneousLayout, LinkKind, RankProfile, Topology};

    let text = std::fs::read_to_string(manifest)?;
    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (id, addr) = line
            .split_once('@')
            .ok_or_else(|| format!("manifest line must be id@addr: {line}"))?;
        entries.push((NodeId::from_hex(id)?, addr.to_owned()));
    }
    if entries.len() != 12 {
        return Err(format!("matrix manifest needs 12 entries, got {}", entries.len()).into());
    }
    let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
    let peers: BTreeMap<NodeId, String> = entries.iter().cloned().collect();
    let mut nodes = Vec::new();
    for (id, addr) in entries.iter().skip(local_start).take(local_count) {
        let transport = TcpTransport::bind(*id, addr, peers.clone()).await?;
        let config = NodeConfig {
            id: *id,
            listen: String::new(),
            peers: peers.clone(),
            join: None,
            options: Options::default(),
            advertise_devices: false,
        };
        let cluster = ClusterConfig::new(ids.iter().copied())?;
        nodes.push(RunningNode::start_with_transport(
            config, transport, cluster,
        )?);
    }
    println!(
        "plait matrix: host process started {} local replicas ({local_start}..{}), 12 voters",
        nodes.len(),
        local_start + nodes.len()
    );

    if driver {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let advertise = entries
            .get(local_start)
            .map(|(_, addr)| {
                addr.rsplit_once(':')
                    .map(|(host, _)| host)
                    .unwrap_or("127.0.0.1")
            })
            .unwrap_or("127.0.0.1");
        let mut client = Client::new_on(&format!("{advertise}:0")).await?;
        for (id, addr) in &entries {
            client.add_peer(*id, addr.clone());
        }
        let topology = Topology::new(AxisSizes::new(2, 1, 3, 2, 1, 1), ids.clone())?;
        let (m, n, k) = shape;
        let (a, b) = matrix_inputs(m, n, k);
        let expected = matrix_reference(&a, &b, m, n, k);
        let op = OpId::new(NodeId::derive("matrix-plan"), 1);
        let spec = OpSpec {
            id: op,
            kind: FuseKind::Sum,
            contributors: ids.iter().copied().collect(),
            tensor: TensorKey::new(TensorId(40_000), ShardId(0)),
            epoch: 1,
        };
        client.submit(ids[0], Command::DeclareOp(spec)).await?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let mut declared_everywhere = true;
            for (id, _) in &entries {
                let declared = client
                    .query(*id, op)
                    .await
                    .ok()
                    .is_some_and(|reply| reply.declared);
                if !declared {
                    declared_everywhere = false;
                    break;
                }
            }
            if declared_everywhere {
                break;
            }
            if Instant::now() >= deadline {
                return Err("matrix declaration timed out".into());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let collective_start = Instant::now();
        let row_ranges = if asymmetric {
            let profiles: Vec<RankProfile> = entries
                .iter()
                .enumerate()
                .map(|(index, (node, _))| {
                    if index < 4 {
                        RankProfile {
                            node: *node,
                            gpus: 1,
                            compute_units: 1.0,
                            link: LinkKind::Pcie,
                            bandwidth_gbps: Some(25.0),
                            latency_us: Some(5.0),
                        }
                    } else {
                        RankProfile {
                            node: *node,
                            gpus: 1,
                            compute_units: 4.0,
                            link: LinkKind::Ub,
                            bandwidth_gbps: Some(200.0),
                            latency_us: Some(2.0),
                        }
                    }
                })
                .collect();
            let plan = HeterogeneousLayout::plan_detailed(&profiles, m * k * 4)?;
            let ranges = plan.layout.row_ranges(m);
            let local_share: f64 = plan.shares.iter().take(4).map(|(_, share)| share).sum();
            let remote_share: f64 = plan.shares.iter().skip(4).map(|(_, share)| share).sum();
            let local: usize = ranges.iter().take(4).map(|range| range.range.len()).sum();
            let remote: usize = ranges.iter().skip(4).map(|range| range.range.len()).sum();
            println!(
                "plait matrix: heterogeneous LP local_share={local_share:.3} remote_share={remote_share:.3} makespan={:.6} local_rows={local} remote_rows={remote} message_bytes={}",
                plan.makespan,
                m * k * 4
            );
            Some(ranges)
        } else {
            None
        };
        let mut contributions = 0usize;
        for (index, (id, _)) in entries.iter().enumerate() {
            let coord = topology.coord(*id).ok_or("node missing from topology")?;
            let values = if let Some(ranges) = &row_ranges {
                matrix_partial_rows(&a, &b, m, n, k, ranges[index].range.clone())
            } else {
                matrix_partial(
                    &a,
                    &b,
                    m,
                    n,
                    k,
                    coord.data as usize,
                    coord.tensor as usize,
                    coord.expert as usize,
                    2,
                    3,
                    2,
                )
            };
            client
                .submit(
                    *id,
                    Command::Contribute {
                        op,
                        contributor: *id,
                        values,
                    },
                )
                .await?;
            contributions += 1;
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        let values = loop {
            if let Some(values) = client
                .query(ids[0], op)
                .await
                .ok()
                .and_then(|reply| reply.values)
            {
                break values;
            }
            if Instant::now() >= deadline {
                return Err("matrix collective timed out".into());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        };
        if values.len() != expected.len() {
            return Err("matrix result length mismatch".into());
        }
        for (index, (actual, expected)) in values.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1e-9 {
                return Err(format!("matrix mismatch at {index}: {actual} vs {expected}").into());
            }
        }
        let checksum: f64 = values.iter().sum();
        println!(
            "plait matrix: 12 replicas {}, {m}x{n}x{k}, {contributions} contributions, checksum {checksum}, collective_ms {:.3}",
            if asymmetric {
                "4+8 asymmetric row split"
            } else {
                "DP2×TP3×EP2"
            },
            collective_start.elapsed().as_secs_f64() * 1e3
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
    } else {
        tokio::time::sleep(Duration::from_secs(hold_seconds)).await;
    }
    drop(nodes);
    Ok(())
}

fn matrix_partial_rows(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for row in rows {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}

fn matrix_inputs(m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let a = (0..m * k)
        .map(|index| ((index * 7 + 3) % 17) as f64 - 8.0)
        .collect();
    let b = (0..k * n)
        .map(|index| ((index * 11 + 5) % 13) as f64 - 6.0)
        .collect();
    (a, b)
}

fn matrix_reference(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn matrix_partial(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    dp: usize,
    tp: usize,
    ep: usize,
    dp_size: usize,
    tp_size: usize,
    ep_size: usize,
) -> Vec<f64> {
    let row_start = dp * m / dp_size;
    let row_end = (dp + 1) * m / dp_size;
    let col_start = ep * n / ep_size;
    let col_end = (ep + 1) * n / ep_size;
    let k_start = tp * k / tp_size;
    let k_end = (tp + 1) * k / tp_size;
    let mut out = vec![0.0; m * n];
    for row in row_start..row_end {
        for col in col_start..col_end {
            let mut sum = 0.0;
            for inner in k_start..k_end {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}
