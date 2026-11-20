//! A leaderless Plait process. Every node can accept submissions, propose
//! commands, serve snapshots, and recover stalled instances.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use tokio::task::JoinHandle;

use plait_consensus::{Options, Replica};
use plait_core::{
    AcceleratorKind, ClusterConfig, Command, ComputeRequirement, ComputeTask, DeviceId, FuseKind,
    NodeId, OpId, OpSpec, ShardId, TensorId, TensorKey,
};
use plait_fabric::Runtime;
use plait_net::{TcpTransport, Wire};

#[derive(Clone, Debug)]
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
        }
    }
}

pub struct RunningNode {
    id: NodeId,
    addr: String,
    stop: Arc<Notify>,
    handle: Option<JoinHandle<()>>,
}

impl RunningNode {
    pub async fn start(config: NodeConfig) -> io::Result<Self> {
        let mut transport_peers = config.peers.clone();
        if let Some(join) = &config.join {
            transport_peers.insert(join.node, join.addr.clone());
        }
        let transport = TcpTransport::bind(config.id, &config.listen, transport_peers).await?;
        let voters = if config.join.is_some() {
            vec![config.id]
        } else {
            let mut voters: Vec<NodeId> = config.peers.keys().copied().collect();
            voters.push(config.id);
            voters
        };
        let cluster = ClusterConfig::new(voters)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
        Self::start_with_transport(config, transport, cluster)
    }

    pub fn start_with_transport(
        config: NodeConfig,
        transport: TcpTransport,
        cluster: ClusterConfig,
    ) -> io::Result<Self> {
        let id = config.id;
        let addr = transport.local_addr().to_owned();
        let stop = Arc::new(Notify::new());
        let task_stop = stop.clone();
        let state = NodeState {
            id,
            transport,
            replica: Replica::new(id, cluster, Runtime::new(), config.options.clone()),
            options: config.options,
            join: config.join,
            joined: false,
            staged_proposed: BTreeSet::new(),
            commit_proposed: BTreeSet::new(),
            devices_registered: false,
            last_join_poll: 0,
            last_ready: 0,
            started: Instant::now(),
        };
        let handle = tokio::spawn(state.run(task_stop));
        Ok(Self {
            id,
            addr,
            stop,
            handle: Some(handle),
        })
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn addr(&self) -> &str {
        &self.addr
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
    options: Options,
    join: Option<JoinTarget>,
    joined: bool,
    staged_proposed: BTreeSet<NodeId>,
    commit_proposed: BTreeSet<NodeId>,
    devices_registered: bool,
    last_join_poll: u64,
    last_ready: u64,
    started: Instant,
}

impl NodeState {
    fn now(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    async fn run(mut self, stop: Arc<Notify>) {
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
                    self.handle(from, wire, now).await;
                }
                Event::Tick => {
                    let now = self.now();
                    self.replica.tick(now);
                    self.maybe_register_devices(now);
                    self.drive_join(now).await;
                    self.refresh_peers();
                    self.flush().await;
                }
            }
        }
    }

    async fn handle(&mut self, from: NodeId, wire: Wire, now: u64) {
        match wire {
            Wire::Consensus(message) => {
                if self.joined || self.join.is_none() {
                    self.replica.handle(from, message, now);
                }
            }
            Wire::JoinRequest { node, addr } => {
                self.transport.set_peer(node, addr.clone());
                if node == self.id {
                    return;
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
                if self.replica.config().is_member(node) {
                    self.send_snapshot(node).await;
                }
            }
            Wire::Snapshot { snapshot, peers } => {
                if self.joined {
                    return;
                }
                if !snapshot.config.is_member(self.id) && !snapshot.config.is_voter(self.id) {
                    return;
                }
                self.replica = Replica::from_snapshot(self.id, snapshot, self.options.clone());
                self.transport.set_peers(peers);
                self.joined = true;
                self.refresh_peers();
                self.send_ready(now).await;
            }
            Wire::Ready { node } => {
                if self.replica.config().is_voter(self.id)
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
                let can_propose = self.replica.config().is_voter(self.id);
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
            Wire::Hello { .. } | Wire::Pong | Wire::Ack { .. } | Wire::QueryReply { .. } => {}
        }
    }

    fn maybe_register_devices(&mut self, now: u64) {
        if self.devices_registered
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

pub struct Client {
    transport: TcpTransport,
    next: u64,
}

impl Client {
    pub async fn new() -> io::Result<Self> {
        let id = NodeId::random();
        let transport = TcpTransport::bind(id, "127.0.0.1:0", BTreeMap::new()).await?;
        Ok(Self { transport, next: 1 })
    }

    pub fn add_peer(&self, node: NodeId, addr: impl Into<String>) {
        self.transport.set_peer(node, addr);
    }

    pub async fn submit(&mut self, node: NodeId, command: Command) -> io::Result<String> {
        let request = self.next;
        self.next += 1;
        let reply_to = self.transport.local_addr().to_owned();
        self.transport
            .send(
                node,
                Wire::Submit {
                    request,
                    command,
                    reply_to,
                },
            )
            .await?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let next = tokio::time::timeout(
                remaining.min(Duration::from_millis(100)),
                self.transport.recv(),
            )
            .await;
            if let Ok(Some((
                _,
                Wire::Ack {
                    request: answer,
                    ok,
                    message,
                },
            ))) = next
            {
                if answer == request {
                    if ok {
                        return Ok(message);
                    }
                    return Err(io::Error::other(message));
                }
            }
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "submit acknowledgement timed out",
        ))
    }

    pub async fn query(&mut self, node: NodeId, op: OpId) -> io::Result<QueryResult> {
        let request = self.next;
        self.next += 1;
        let reply_to = self.transport.local_addr().to_owned();
        self.transport
            .send(
                node,
                Wire::Query {
                    request,
                    op,
                    reply_to,
                },
            )
            .await?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let next = tokio::time::timeout(
                remaining.min(Duration::from_millis(100)),
                self.transport.recv(),
            )
            .await;
            if let Ok(Some((
                _,
                Wire::QueryReply {
                    request: answer,
                    op: answer_op,
                    declared,
                    values,
                    executed,
                    voters,
                },
            ))) = next
            {
                if answer == request && answer_op == op {
                    return Ok(QueryResult {
                        declared,
                        values,
                        executed,
                        voters,
                    });
                }
            }
        }
        Err(io::Error::new(io::ErrorKind::TimedOut, "query timed out"))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    pub declared: bool,
    pub values: Option<Vec<f64>>,
    pub executed: usize,
    pub voters: Vec<NodeId>,
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

/// Run the distributed DSA reference operator over three in-process EPaxos
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
/// Each runner uses the same EPaxos cluster and fused runtime as the node.
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
/// operations onto eligible devices through the same EPaxos cluster. NVIDIA
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
    match plait_gpu::cutlass_sgemm(&a, &b, &c, m, n, k, 1.0, 0.0) {
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
                "plait gpu: CUTLASS SGEMM {m}x{n}x{k} ok on {} visible GPU(s)",
                executed
            );
        }
        Err(error) => {
            println!("plait gpu: CUTLASS SGEMM skipped: {error}");
        }
    }
    Ok(())
}
