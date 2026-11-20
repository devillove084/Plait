use std::collections::{BTreeMap, VecDeque};

use plait_consensus::{InstanceId, Message, Options, ProposeError, Replica};
use plait_core::{ClusterConfig, Command, NodeId, OpId, OpSpec, ShardId, TensorId, TensorKey};
use plait_fabric::Runtime;

use crate::plan::{ParallelOp, fuse_kind};

/// An in-process EPaxos cluster with the real consensus engine and the same
/// command/runtime path as the TCP node. It is used by the DSA reference
/// operator and by tests that need deterministic message delivery.
pub struct LocalCluster {
    nodes: BTreeMap<NodeId, Replica<Runtime>>,
    queue: VecDeque<(NodeId, NodeId, Message)>,
    now: u64,
    declared: std::collections::BTreeSet<OpId>,
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
            declared: std::collections::BTreeSet::new(),
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

    /// Propose a non-fused command and run consensus to quiescence.
    pub fn run_command(&mut self, node: NodeId, command: Command) -> Result<(), OpError> {
        self.propose(node, command)
            .map_err(|error| OpError::new(error.to_string()))?;
        self.run();
        Ok(())
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

    /// Declare one planned operation, submit every contributor's payload, run
    /// consensus to quiescence, and return the fused result. The operation is
    /// declared at most once per cluster.
    pub fn run_op(
        &mut self,
        op: &ParallelOp,
        contributions: BTreeMap<NodeId, Vec<f64>>,
    ) -> Result<Vec<f64>, OpError> {
        let coordinator = *op
            .participants
            .iter()
            .min()
            .ok_or_else(|| OpError::new("operation has no participants"))?;
        if self.declared.insert(op.op_id) {
            let kind = fuse_kind(op.sync)
                .ok_or_else(|| OpError::new("operation is not a fused reduction"))?;
            let spec = OpSpec {
                id: op.op_id,
                kind,
                contributors: op.participants.iter().copied().collect(),
                tensor: TensorKey::new(tensor_id(op.op_id), ShardId(0)),
                epoch: 0,
            };
            self.propose(coordinator, Command::DeclareOp(spec))
                .map_err(|error| OpError::new(error.to_string()))?;
            self.run();
        }
        for (node, values) in contributions {
            self.propose(
                node,
                Command::Contribute {
                    op: op.op_id,
                    contributor: node,
                    values,
                },
            )
            .map_err(|error| OpError::new(error.to_string()))?;
        }
        self.run();
        self.result(coordinator, op.op_id)
            .ok_or_else(|| OpError::new(format!("no fused result for {}", op.name)))
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

fn tensor_id(op: OpId) -> TensorId {
    TensorId(u128::from_be_bytes(*op.origin.as_bytes()) ^ u128::from(op.counter))
}

#[derive(Clone, Debug)]
pub struct OpError(String);

impl OpError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OpError {}
