//! A leaderless Plait process. Every node can accept submissions, propose
//! commands, serve snapshots, and recover stalled instances.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

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
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RunningNode {
    pub fn start(config: NodeConfig) -> io::Result<Self> {
        let mut transport_peers = config.peers.clone();
        if let Some(join) = &config.join {
            transport_peers.insert(join.node, join.addr.clone());
        }
        let transport = TcpTransport::bind(config.id, &config.listen, transport_peers)?;
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
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
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
        let handle = thread::spawn(move || state.run(thread_stop));
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
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
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

    fn run(mut self, stop: Arc<AtomicBool>) {
        while !stop.load(Ordering::Relaxed) {
            let now = self.now();
            if let Some((from, wire)) = self.transport.recv_timeout(Duration::from_millis(15)) {
                self.handle(from, wire, now);
            }
            self.replica.tick(now);
            self.maybe_register_devices(now);
            self.drive_join(now);
            self.refresh_peers();
            self.flush();
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn handle(&mut self, from: NodeId, wire: Wire, now: u64) {
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
                    self.send_snapshot(node);
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
                    self.send_snapshot(node);
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
                self.send_ready(now);
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
                let _ = self.transport.send(
                    from,
                    Wire::Ack {
                        request,
                        ok,
                        message,
                    },
                );
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
                let _ = self.transport.send(from, reply);
            }
            Wire::Ping => {
                let _ = self.transport.send(from, Wire::Pong);
            }
            Wire::Hello { .. } | Wire::Pong | Wire::Ack { .. } | Wire::QueryReply { .. } => {}
        }
    }

    fn maybe_register_devices(&mut self, now: u64) {
        if self.devices_registered || !self.replica.config().is_voter(self.id) {
            return;
        }
        let profiles = plait_device::discover(self.id);
        for profile in &profiles {
            self.propose(Command::RegisterDevice(profile.clone()), now);
        }
        self.devices_registered = true;
    }

    fn drive_join(&mut self, now: u64) {
        if let Some(join) = &self.join {
            if !self.joined && now.saturating_sub(self.last_join_poll) >= 300 {
                self.last_join_poll = now;
                let _ = self.transport.send(
                    join.node,
                    Wire::JoinRequest {
                        node: self.id,
                        addr: self.transport.local_addr().to_owned(),
                    },
                );
                let _ = self
                    .transport
                    .send(join.node, Wire::SnapshotRequest { node: self.id });
            }
        }
        if self.joined
            && !self.replica.config().is_voter(self.id)
            && self.replica.config().is_member(self.id)
            && now.saturating_sub(self.last_ready) >= 300
        {
            self.last_ready = now;
            self.send_ready(now);
        }
    }

    fn send_ready(&mut self, _now: u64) {
        let voters: Vec<NodeId> = self.replica.config().voters().iter().copied().collect();
        for voter in voters {
            let _ = self.transport.send(voter, Wire::Ready { node: self.id });
        }
    }

    fn send_snapshot(&mut self, to: NodeId) {
        if self.transport.peer_addr(to).is_none() {
            return;
        }
        let wire = Wire::Snapshot {
            snapshot: self.replica.snapshot(),
            peers: self.transport.peers(),
        };
        let _ = self.transport.send(to, wire);
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

    fn flush(&mut self) {
        for outbound in self.replica.take_outbox() {
            if let Err(error) = self
                .transport
                .send(outbound.to, Wire::Consensus(outbound.message))
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
    pub fn new() -> io::Result<Self> {
        let id = NodeId::random();
        let transport = TcpTransport::bind(id, "127.0.0.1:0", BTreeMap::new())?;
        Ok(Self { transport, next: 1 })
    }

    pub fn add_peer(&self, node: NodeId, addr: impl Into<String>) {
        self.transport.set_peer(node, addr);
    }

    pub fn submit(&mut self, node: NodeId, command: Command) -> io::Result<String> {
        let request = self.next;
        self.next += 1;
        let reply_to = self.transport.local_addr().to_owned();
        self.transport.send(
            node,
            Wire::Submit {
                request,
                command,
                reply_to,
            },
        )?;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some((
                _,
                Wire::Ack {
                    request: answer,
                    ok,
                    message,
                },
            )) = self.transport.recv_timeout(Duration::from_millis(100))
            {
                if answer == request {
                    if ok {
                        return Ok(message);
                    }
                    return Err(io::Error::other(message));
                }
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "submit acknowledgement timed out",
                ));
            }
        }
    }

    pub fn query(&mut self, node: NodeId, op: OpId) -> io::Result<QueryResult> {
        let request = self.next;
        self.next += 1;
        let reply_to = self.transport.local_addr().to_owned();
        self.transport.send(
            node,
            Wire::Query {
                request,
                op,
                reply_to,
            },
        )?;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some((
                _,
                Wire::QueryReply {
                    request: answer,
                    op: answer_op,
                    declared,
                    values,
                    executed,
                    voters,
                },
            )) = self.transport.recv_timeout(Duration::from_millis(100))
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
            if Instant::now() >= deadline {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "query timed out"));
            }
        }
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
pub fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    let ids = [
        NodeId::derive("demo-a"),
        NodeId::derive("demo-b"),
        NodeId::derive("demo-c"),
    ];
    let mut transports = Vec::new();
    for id in ids {
        transports.push(TcpTransport::bind(id, "127.0.0.1:0", BTreeMap::new())?);
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

    let mut client = Client::new()?;
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
    client.submit(ids[0], Command::DeclareOp(spec))?;
    wait_declared(&mut client, ids[0], op)?;

    let partials = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]];
    for (index, node) in ids.iter().enumerate() {
        client.submit(
            *node,
            Command::Contribute {
                op,
                contributor: *node,
                values: partials[index].to_vec(),
            },
        )?;
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    let result = loop {
        let query = client.query(ids[0], op)?;
        if let Some(values) = query.values {
            break values;
        }
        if Instant::now() >= deadline {
            return Err("fused result did not materialize".into());
        }
        thread::sleep(Duration::from_millis(50));
    };
    let expected = vec![111.0, 222.0, 333.0];
    if result != expected {
        return Err(format!("unexpected fused value: {result:?}").into());
    }
    let voters = client.query(ids[1], op)?.voters;
    println!(
        "plait demo: 3 TCP peers, {} voters, fused sum {:?} on every query",
        voters.len(),
        result
    );
    drop(nodes);
    Ok(())
}

fn wait_declared(client: &mut Client, node: NodeId, op: OpId) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if client.query(node, op)?.declared {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "operator declaration timed out",
            ));
        }
        thread::sleep(Duration::from_millis(50));
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
pub fn run_rdma_selftest() -> Result<(), Box<dyn std::error::Error>> {
    match plait_rdma::selftest()? {
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
/// operations onto eligible devices through the same EPaxos cluster. The
/// compute result is a deterministic placeholder; the placement and device
/// reservation are real replicated commands.
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
    let mut assignments = Vec::new();
    for (index, op) in operations.iter().enumerate() {
        let requirement = ComputeRequirement {
            arch: Some(arch_order[index % arch_order.len()]),
            ..op.requirement()
        };
        let Some(device) = registry.select(&requirement, SchedulingPolicy::LeastLoaded) else {
            continue;
        };
        registry.reserve(device);
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
                values: vec![index as f64],
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
        "plait hetero: {} devices, {} compute tasks, placements {:?}",
        profiles.len(),
        assignments.len(),
        counts
    );
    for (task, device, kernel) in assignments.iter().take(8) {
        println!("  {task:?} -> {device} ({kernel})");
    }
    Ok(())
}
