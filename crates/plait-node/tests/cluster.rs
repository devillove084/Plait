use std::collections::BTreeMap;
use std::thread;
use std::time::{Duration, Instant};

use plait_core::{
    ClusterConfig, Command, FuseKind, NodeId, OpId, OpSpec, ShardId, TensorId, TensorKey,
};
use plait_net::TcpTransport;
use plait_node::{Client, JoinTarget, NodeConfig, RunningNode};

fn ids(prefix: &str, count: usize) -> Vec<NodeId> {
    (0..count)
        .map(|index| NodeId::derive(&format!("{prefix}-{index}")))
        .collect()
}

fn start_bootstrap(ids: &[NodeId]) -> (Vec<RunningNode>, Vec<String>) {
    let mut transports = Vec::new();
    for id in ids {
        transports.push(TcpTransport::bind(*id, "127.0.0.1:0", BTreeMap::new()).unwrap());
    }
    let addrs: Vec<String> = transports
        .iter()
        .map(|transport| transport.local_addr().to_owned())
        .collect();
    let peers: BTreeMap<NodeId, String> = ids.iter().copied().zip(addrs.iter().cloned()).collect();
    let mut nodes = Vec::new();
    for (id, transport) in ids.iter().zip(transports) {
        transport.set_peers(peers.clone());
        let config = NodeConfig {
            id: *id,
            listen: String::new(),
            peers: peers.clone(),
            join: None,
            options: Default::default(),
        };
        let cluster = ClusterConfig::new(ids.iter().copied()).unwrap();
        nodes.push(RunningNode::start_with_transport(config, transport, cluster).unwrap());
    }
    (nodes, addrs)
}

fn wait_for<T>(what: &str, mut check: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn tcp_cluster_proposes_and_fuses_from_every_node() {
    let ids = ids("tcp-fuse", 3);
    let (_nodes, addrs) = start_bootstrap(&ids);
    let mut client = Client::new().unwrap();
    for (id, addr) in ids.iter().zip(&addrs) {
        client.add_peer(*id, addr.clone());
    }
    let op = OpId::new(NodeId::derive("tcp-op"), 7);
    let spec = OpSpec {
        id: op,
        kind: FuseKind::Sum,
        contributors: ids.iter().copied().collect(),
        tensor: TensorKey::new(TensorId(9), ShardId(0)),
        epoch: 2,
    };
    client.submit(ids[0], Command::DeclareOp(spec)).unwrap();
    wait_for("declaration", || {
        client
            .query(ids[0], op)
            .ok()
            .filter(|reply| reply.declared)
            .map(|_| ())
    });
    let partials = [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]];
    for (index, id) in ids.iter().enumerate() {
        client
            .submit(
                *id,
                Command::Contribute {
                    op,
                    contributor: *id,
                    values: partials[index].to_vec(),
                },
            )
            .unwrap();
    }
    let values = wait_for("fused result", || {
        client.query(ids[2], op).ok().and_then(|reply| reply.values)
    });
    assert_eq!(values, vec![7.0, 7.0]);
}

#[test]
fn a_joining_node_receives_a_snapshot_and_becomes_a_voter() {
    let ids = ids("join", 3);
    let bootstrap = [ids[0], ids[1]];
    let (mut nodes, addrs) = start_bootstrap(&bootstrap);
    let mut client = Client::new().unwrap();
    for (id, addr) in bootstrap.iter().zip(&addrs) {
        client.add_peer(*id, addr.clone());
    }

    let joining_config = NodeConfig {
        id: ids[2],
        listen: "127.0.0.1:0".to_owned(),
        peers: BTreeMap::new(),
        join: Some(JoinTarget {
            node: ids[0],
            addr: addrs[0].clone(),
        }),
        options: Default::default(),
    };
    let joining = RunningNode::start(joining_config).unwrap();
    client.add_peer(ids[2], joining.addr().to_owned());

    let op = OpId::new(NodeId::derive("join-op"), 1);
    wait_for("three voters", || {
        client
            .query(ids[2], op)
            .ok()
            .filter(|reply| reply.voters.len() == 3)
            .map(|_| ())
    });
    // The new voter can propose and the old nodes can still form a quorum.
    let spec = OpSpec {
        id: op,
        kind: FuseKind::Max,
        contributors: ids.iter().copied().collect(),
        tensor: TensorKey::new(TensorId(10), ShardId(0)),
        epoch: 3,
    };
    client.submit(ids[2], Command::DeclareOp(spec)).unwrap();
    wait_for("declaration", || {
        client
            .query(ids[0], op)
            .ok()
            .filter(|reply| reply.declared)
            .map(|_| ())
    });
    for (index, id) in ids.iter().enumerate() {
        client
            .submit(
                *id,
                Command::Contribute {
                    op,
                    contributor: *id,
                    values: vec![(index + 1) as f64],
                },
            )
            .unwrap();
    }
    let values = wait_for("joined result", || {
        client.query(ids[1], op).ok().and_then(|reply| reply.values)
    });
    assert_eq!(values, vec![3.0]);

    // Keep handles alive until all assertions finish.
    drop(joining);
    nodes.clear();
}
