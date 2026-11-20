use std::collections::{BTreeMap, VecDeque};

use plait_consensus::{InstanceId, Message, Options, ProposeError, Replica};
use plait_core::{ClusterConfig, Command, NodeId, OpId};
use plait_fabric::Runtime;

/// An in-process EPaxos cluster with the real consensus engine and the same
/// command/runtime path as the TCP node. It is used by the DSA reference
/// operator and by tests that need deterministic message delivery.
pub struct LocalCluster {
    nodes: BTreeMap<NodeId, Replica<Runtime>>,
    queue: VecDeque<(NodeId, NodeId, Message)>,
    now: u64,
}

impl LocalCluster {
    pub fn new(ids: impl IntoIterator<Item = NodeId>) -> Self {
        let ids: Vec<NodeId> = ids.into_iter().collect();
        let config = ClusterConfig::bootstrap(ids.iter().copied());
        let options = Options::default();
        let nodes = ids
            .iter()
            .map(|id| {
                (
                    *id,
                    Replica::new(*id, config.clone(), Runtime::new(), options.clone()),
                )
            })
            .collect();
        Self {
            nodes,
            queue: VecDeque::new(),
            now: 1,
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    pub fn propose(&mut self, node: NodeId, command: Command) -> Result<InstanceId, ProposeError> {
        self.now = self.now.saturating_add(1);
        let instance = self
            .nodes
            .get_mut(&node)
            .expect("node belongs to the cluster")
            .propose(command, self.now)?;
        self.drain_outbox();
        Ok(instance)
    }

    pub fn run(&mut self) {
        let mut steps = 0usize;
        while let Some((from, to, message)) = self.queue.pop_front() {
            if let Some(replica) = self.nodes.get_mut(&to) {
                self.now = self.now.saturating_add(1);
                replica.handle(from, message, self.now);
            }
            self.drain_outbox();
            steps += 1;
            assert!(steps < 2_000_000, "local EPaxos cluster did not quiesce");
        }
    }

    pub fn result(&self, node: NodeId, op: OpId) -> Option<Vec<f64>> {
        self.nodes
            .get(&node)
            .and_then(|replica| replica.machine().result(op))
            .map(<[f64]>::to_vec)
    }

    pub fn runtime(&self, node: NodeId) -> &Runtime {
        self.nodes
            .get(&node)
            .expect("node belongs to the cluster")
            .machine()
    }

    pub fn executed(&self, node: NodeId) -> usize {
        self.nodes
            .get(&node)
            .expect("node belongs to the cluster")
            .executed_count()
    }

    pub fn violations(&self) -> BTreeMap<NodeId, Vec<String>> {
        self.nodes
            .iter()
            .filter_map(|(id, replica)| {
                if replica.violations().is_empty() {
                    None
                } else {
                    Some((*id, replica.violations().to_vec()))
                }
            })
            .collect()
    }

    fn drain_outbox(&mut self) {
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for from in ids {
            if let Some(replica) = self.nodes.get_mut(&from) {
                for outbound in replica.take_outbox() {
                    self.queue.push_back((from, outbound.to, outbound.message));
                }
            }
        }
    }
}
