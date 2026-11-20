use std::collections::{BTreeMap, BTreeSet};

use plait_consensus::{InstanceId, StateMachine};
use plait_core::{
    AcceleratorKind, Command, ComputeTask, DeviceId, DeviceProfile, FuseKind, MembershipChange,
    NodeId, OpId, OpSpec, TensorKey, TensorVersion, TensorView,
};

/// A deterministic event emitted while applying replicated commands. Nodes use
/// these to wake local consumers without polling the whole runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    OpDeclared(OpId),
    Contribution {
        op: OpId,
        contributor: NodeId,
    },
    Fused {
        op: OpId,
        tensor: TensorKey,
        epoch: u64,
        values: Vec<f64>,
    },
    Published {
        key: TensorKey,
        version: u64,
    },
    Member {
        node: NodeId,
        addr: String,
    },
    MemberRemoved(NodeId),
    DeviceRegistered {
        device: DeviceId,
        arch: AcceleratorKind,
        memory_bytes: u64,
    },
    Computed {
        task: OpId,
        kernel: String,
        device: DeviceId,
        output: TensorKey,
    },
    Error(String),
}

/// The fused-operator runtime. Every replica applies the same consensus
/// commands to its own local state; a node's partial result is delivered by
/// replication rather than by a separate collective phase.
///
/// Once every declared contributor to an operator has been applied, the
/// runtime combines the partials locally and installs the resulting tensor
/// shard. There is no rank-zero driver, no global barrier, and no separate
/// all-reduce message.
#[derive(Clone, Default)]
pub struct Runtime {
    ops: BTreeMap<OpId, OpSpec>,
    pending: BTreeMap<OpId, BTreeMap<NodeId, Vec<f64>>>,
    results: BTreeMap<OpId, Vec<f64>>,
    shards: BTreeMap<TensorKey, TensorView>,
    members: BTreeMap<NodeId, String>,
    devices: BTreeMap<DeviceId, DeviceProfile>,
    placements: BTreeMap<OpId, DeviceId>,
    applied: BTreeSet<InstanceId>,
    events: Vec<Event>,
}

impl Runtime {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn op(&self, op: OpId) -> Option<&OpSpec> {
        self.ops.get(&op)
    }

    pub fn result(&self, op: OpId) -> Option<&[f64]> {
        self.results.get(&op).map(Vec::as_slice)
    }

    pub fn shard(&self, key: TensorKey) -> Option<&TensorView> {
        self.shards.get(&key)
    }

    pub fn members(&self) -> &BTreeMap<NodeId, String> {
        &self.members
    }

    pub fn devices(&self) -> &BTreeMap<DeviceId, DeviceProfile> {
        &self.devices
    }

    pub fn placement(&self, task: OpId) -> Option<DeviceId> {
        self.placements.get(&task).copied()
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// Validate a command against local replicated state before proposing it.
    /// This is an optimization and an early error path; the apply path remains
    /// total and deterministic because peers may have proposed concurrently.
    pub fn validate(&self, command: &Command) -> Result<(), String> {
        match command {
            Command::DeclareOp(spec) => {
                if spec.contributors.is_empty() {
                    return Err("an operator needs at least one contributor".to_owned());
                }
                if let Some(existing) = self.ops.get(&spec.id) {
                    if existing != spec {
                        return Err(format!(
                            "operator {:?} already declared differently",
                            spec.id
                        ));
                    }
                }
                Ok(())
            }
            Command::Contribute {
                op,
                contributor,
                values,
            } => {
                let Some(spec) = self.ops.get(op) else {
                    return Err(format!("operator {op:?} is not declared yet"));
                };
                if !spec.contributors.contains(contributor) {
                    return Err(format!("{contributor} is not a contributor to {op:?}"));
                }
                match spec.kind {
                    FuseKind::TopK { .. } => {
                        if values.len() % 2 != 0 {
                            return Err(
                                "top-k contributions must be [score, index] pairs".to_owned()
                            );
                        }
                    }
                    FuseKind::OnlineSoftmax => {
                        if values.len() < 2 {
                            return Err("online-softmax needs [max, sum, acc...]".to_owned());
                        }
                    }
                    FuseKind::Sum | FuseKind::Max => {
                        if let Some(width) = self
                            .pending
                            .get(op)
                            .and_then(|pending| pending.values().next().map(Vec::len))
                        {
                            if values.len() != width {
                                return Err(format!(
                                    "contribution width {} does not match {width}",
                                    values.len()
                                ));
                            }
                        }
                    }
                    FuseKind::Gather => {}
                }
                Ok(())
            }
            Command::Publish { values, .. } | Command::Stage { values, .. } => {
                if values.is_empty() {
                    return Err("a published shard cannot be empty".to_owned());
                }
                Ok(())
            }
            Command::RegisterDevice(profile) => {
                if profile.name.is_empty() {
                    return Err("a device profile needs a name".to_owned());
                }
                Ok(())
            }
            Command::Compute(task) => {
                if task.values.is_empty() {
                    return Err("a compute result cannot be empty".to_owned());
                }
                let Some(profile) = self.devices.get(&task.device) else {
                    return Err(format!("device {} is not registered", task.device));
                };
                if !task.requirement.matches(profile) {
                    return Err(format!(
                        "device {} does not satisfy the compute requirement",
                        task.device
                    ));
                }
                Ok(())
            }
            Command::Nop | Command::Membership(_) => Ok(()),
        }
    }

    fn apply_command(&mut self, command: &Command) {
        match command {
            Command::Nop => {}
            Command::DeclareOp(spec) => self.declare(spec.clone()),
            Command::Contribute {
                op,
                contributor,
                values,
            } => self.contribute(*op, *contributor, values.clone()),
            Command::Publish {
                key,
                version,
                values,
            } => {
                self.publish(*key, *version, values);
            }
            Command::Stage {
                output,
                version,
                values,
                ..
            } => {
                self.publish(*output, *version, values);
            }
            Command::RegisterDevice(profile) => self.register_device(profile.clone()),
            Command::Compute(task) => self.compute(task.clone()),
            Command::Membership(change) => self.membership(change),
        }
    }

    fn register_device(&mut self, profile: DeviceProfile) {
        let device = profile.id;
        let arch = profile.arch;
        let memory_bytes = profile.memory_bytes;
        self.devices.insert(device, profile);
        self.events.push(Event::DeviceRegistered {
            device,
            arch,
            memory_bytes,
        });
    }

    fn compute(&mut self, task: ComputeTask) {
        let Some(profile) = self.devices.get(&task.device).cloned() else {
            self.events.push(Event::Error(format!(
                "compute task {:?} targets unknown device {}",
                task.task, task.device
            )));
            return;
        };
        if !task.requirement.matches(&profile) {
            self.events.push(Event::Error(format!(
                "device {} does not satisfy task {:?}",
                task.device, task.task
            )));
            return;
        }
        self.publish(task.output, task.version, &task.values);
        self.placements.insert(task.task, task.device);
        self.events.push(Event::Computed {
            task: task.task,
            kernel: task.kernel,
            device: task.device,
            output: task.output,
        });
    }

    fn publish(&mut self, key: TensorKey, version: u64, values: &[f64]) {
        self.shards.insert(
            key,
            TensorView {
                key,
                version: TensorVersion(version),
                values: values.to_vec(),
            },
        );
        self.events.push(Event::Published { key, version });
    }

    fn declare(&mut self, spec: OpSpec) {
        if let Some(existing) = self.ops.get(&spec.id) {
            if existing != &spec {
                self.events.push(Event::Error(format!(
                    "operator {:?} was already declared differently",
                    spec.id
                )));
                return;
            }
            return;
        }
        self.ops.insert(spec.id, spec.clone());
        self.events.push(Event::OpDeclared(spec.id));
        // A contribution can be ordered before its declaration if two nodes
        // propose concurrently. It waits here until the declaration arrives.
        if let Some(pending) = self.pending.get(&spec.id) {
            let invalid: Vec<NodeId> = pending
                .keys()
                .filter(|node| !spec.contributors.contains(node))
                .copied()
                .collect();
            if !invalid.is_empty() {
                self.events.push(Event::Error(format!(
                    "operator {:?} has contributions from non-contributors: {invalid:?}",
                    spec.id
                )));
                self.pending.remove(&spec.id);
                return;
            }
            self.try_fuse(spec.id);
        }
    }

    fn contribute(&mut self, op: OpId, contributor: NodeId, values: Vec<f64>) {
        if let Some(spec) = self.ops.get(&op) {
            if !spec.contributors.contains(&contributor) {
                self.events.push(Event::Error(format!(
                    "{contributor} is not a contributor to {op:?}"
                )));
                return;
            }
        }
        let pending = self.pending.entry(op).or_default();
        if pending.contains_key(&contributor) {
            return;
        }
        let fixed_width = self
            .ops
            .get(&op)
            .map(|spec| {
                matches!(
                    spec.kind,
                    FuseKind::Sum | FuseKind::Max | FuseKind::OnlineSoftmax
                )
            })
            .unwrap_or(true);
        if fixed_width {
            if let Some(width) = pending.values().next().map(Vec::len) {
                if values.len() != width {
                    self.events.push(Event::Error(format!(
                        "operator {op:?}: contribution width {} does not match {width}",
                        values.len()
                    )));
                    return;
                }
            }
        }
        pending.insert(contributor, values);
        self.events.push(Event::Contribution { op, contributor });
        if self.ops.contains_key(&op) {
            self.try_fuse(op);
        }
    }

    fn try_fuse(&mut self, op: OpId) {
        let Some(spec) = self.ops.get(&op).cloned() else {
            return;
        };
        let Some(pending) = self.pending.get(&op) else {
            return;
        };
        if pending.len() != spec.contributors.len() {
            return;
        }
        if !spec
            .contributors
            .iter()
            .all(|node| pending.contains_key(node))
        {
            return;
        }
        let fields = match spec.kind {
            FuseKind::TopK { k } => {
                let mut candidates: Vec<(f64, u64)> = Vec::new();
                for values in pending.values() {
                    if values.len() % 2 != 0 {
                        self.events.push(Event::Error(format!(
                            "operator {op:?}: odd top-k contribution"
                        )));
                        return;
                    }
                    for pair in values.chunks_exact(2) {
                        if !pair[1].is_finite() || pair[1] < 0.0 || pair[1].fract() != 0.0 {
                            self.events.push(Event::Error(format!(
                                "operator {op:?}: invalid candidate index"
                            )));
                            return;
                        }
                        candidates.push((pair[0], pair[1] as u64));
                    }
                }
                candidates.sort_by(|left, right| {
                    right
                        .0
                        .total_cmp(&left.0)
                        .then_with(|| left.1.cmp(&right.1))
                });
                let mut seen = BTreeSet::new();
                let mut selected = Vec::new();
                for (_, index) in candidates {
                    if seen.insert(index) {
                        selected.push(index as f64);
                        if selected.len() == k as usize {
                            break;
                        }
                    }
                }
                selected
            }
            FuseKind::OnlineSoftmax => {
                let widths: BTreeSet<usize> = pending.values().map(Vec::len).collect();
                if widths.len() != 1 {
                    self.events.push(Event::Error(format!(
                        "operator {op:?}: ragged online-softmax partials"
                    )));
                    return;
                }
                let width = *widths.iter().next().expect("non-empty");
                if width < 2 {
                    self.events.push(Event::Error(format!(
                        "operator {op:?}: online-softmax partial too short"
                    )));
                    return;
                }
                let mut max = f64::NEG_INFINITY;
                let mut sum = 0.0;
                let mut acc = vec![0.0; width - 2];
                for values in pending.values() {
                    let local_max = values[0];
                    let local_sum = values[1];
                    let new_max = max.max(local_max);
                    let old_weight = if max == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (max - new_max).exp()
                    };
                    let new_weight = if local_max == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (local_max - new_max).exp()
                    };
                    sum = sum * old_weight + local_sum * new_weight;
                    for (target, value) in acc.iter_mut().zip(&values[2..]) {
                        *target = *target * old_weight + value * new_weight;
                    }
                    max = new_max;
                }
                let mut combined = vec![max, sum];
                combined.extend(acc);
                combined
            }
            FuseKind::Sum | FuseKind::Max | FuseKind::Gather => {
                let width = pending.values().next().map(Vec::len).unwrap_or(0);
                if matches!(spec.kind, FuseKind::Sum | FuseKind::Max)
                    && pending.values().any(|values| values.len() != width)
                {
                    self.events
                        .push(Event::Error(format!("operator {op:?}: ragged partials")));
                    return;
                }
                let mut fused = match spec.kind {
                    FuseKind::Sum => vec![0.0; width],
                    FuseKind::Max => vec![f64::NEG_INFINITY; width],
                    FuseKind::Gather => Vec::with_capacity(width * spec.contributors.len()),
                    _ => unreachable!(),
                };
                for contributor in &spec.contributors {
                    let Some(values) = pending.get(contributor) else {
                        return;
                    };
                    match spec.kind {
                        FuseKind::Sum => {
                            for (target, value) in fused.iter_mut().zip(values) {
                                *target += value;
                            }
                        }
                        FuseKind::Max => {
                            for (target, value) in fused.iter_mut().zip(values) {
                                *target = target.max(*value);
                            }
                        }
                        FuseKind::Gather => fused.extend_from_slice(values),
                        _ => unreachable!(),
                    }
                }
                fused
            }
        };
        let fused = fields;

        self.shards.insert(
            spec.tensor,
            TensorView {
                key: spec.tensor,
                version: TensorVersion(spec.epoch),
                values: fused.clone(),
            },
        );
        self.results.insert(op, fused.clone());
        self.pending.remove(&op);
        self.events.push(Event::Fused {
            op,
            tensor: spec.tensor,
            epoch: spec.epoch,
            values: fused,
        });
    }

    fn membership(&mut self, change: &MembershipChange) {
        match change {
            MembershipChange::StageAdd { node, addr } => {
                self.members.insert(*node, addr.clone());
                self.events.push(Event::Member {
                    node: *node,
                    addr: addr.clone(),
                });
            }
            MembershipChange::CommitAdd { node } => {
                if !self.members.contains_key(node) {
                    self.events.push(Event::Error(format!(
                        "{node} was promoted without a recorded address"
                    )));
                }
                if let Some(addr) = self.members.get(node).cloned() {
                    self.events.push(Event::Member { node: *node, addr });
                }
            }
            MembershipChange::StageRemove { .. } => {}
            MembershipChange::CommitRemove { node } => {
                self.members.remove(node);
                self.events.push(Event::MemberRemoved(*node));
            }
        }
    }
}

impl StateMachine for Runtime {
    fn apply(&mut self, instance: InstanceId, command: &Command) {
        if !self.applied.insert(instance) {
            return;
        }
        self.apply_command(command);
    }

    fn supports_parallel(&self) -> bool {
        true
    }

    /// Independent SCCs have disjoint write keys. Each worker starts from the
    /// same replicated state and applies one complete SCC; the parent then
    /// merges the resulting maps in deterministic batch order.
    fn fork_batch(&self) -> Option<Self> {
        let mut fork = self.clone();
        fork.events.clear();
        Some(fork)
    }

    fn merge_batch(&mut self, mut other: Self) {
        self.applied.extend(other.applied);
        self.ops.extend(other.ops);
        self.pending.extend(other.pending);
        self.results.extend(other.results);
        self.shards.extend(other.shards);
        self.members.extend(other.members);
        self.devices.extend(other.devices);
        self.placements.extend(other.placements);
        self.events.append(&mut other.events);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::{ComputeRequirement, ShardId, TensorId};

    fn node(label: &str) -> NodeId {
        NodeId::derive(label)
    }

    fn spec(op: OpId, kind: FuseKind, contributors: [NodeId; 3]) -> OpSpec {
        OpSpec {
            id: op,
            kind,
            contributors: contributors.into_iter().collect(),
            tensor: TensorKey::new(TensorId(1), ShardId(0)),
            epoch: 4,
        }
    }

    fn apply(runtime: &mut Runtime, sequence: u64, command: Command) {
        let id = InstanceId {
            proposer: node("test"),
            counter: sequence,
        };
        runtime.apply(id, &command);
    }

    fn contributions() -> [Command; 3] {
        let nodes = [node("a"), node("b"), node("c")];
        [
            Command::Contribute {
                op: OpId::new(nodes[0], 1),
                contributor: nodes[0],
                values: vec![1.0, 2.0],
            },
            Command::Contribute {
                op: OpId::new(nodes[0], 1),
                contributor: nodes[1],
                values: vec![3.0, 4.0],
            },
            Command::Contribute {
                op: OpId::new(nodes[0], 1),
                contributor: nodes[2],
                values: vec![5.0, 6.0],
            },
        ]
    }

    #[test]
    fn parallel_forks_merge_independent_writes() {
        let base = Runtime::new();
        let id = |counter| InstanceId {
            proposer: node("fork"),
            counter,
        };
        let publish = |tensor, value| Command::Publish {
            key: TensorKey::new(TensorId(tensor), ShardId(0)),
            version: 1,
            values: vec![value],
        };
        let mut left = base
            .fork_batch()
            .expect("runtime supports parallel forking");
        left.apply(id(1), &publish(1, 1.0));
        let mut right = base
            .fork_batch()
            .expect("runtime supports parallel forking");
        right.apply(id(2), &publish(2, 2.0));

        let mut merged = base;
        merged.merge_batch(left);
        merged.merge_batch(right);
        assert_eq!(
            merged
                .shard(TensorKey::new(TensorId(1), ShardId(0)))
                .unwrap()
                .values,
            vec![1.0]
        );
        assert_eq!(
            merged
                .shard(TensorKey::new(TensorId(2), ShardId(0)))
                .unwrap()
                .values,
            vec![2.0]
        );
        assert_eq!(merged.events().len(), 2);
    }

    #[test]
    fn sum_waits_for_every_contributor_then_fuses() {
        let nodes = [node("a"), node("b"), node("c")];
        let op = OpId::new(nodes[0], 1);
        let mut runtime = Runtime::new();
        apply(
            &mut runtime,
            1,
            Command::DeclareOp(spec(op, FuseKind::Sum, nodes)),
        );
        let partials = contributions();
        apply(&mut runtime, 2, partials[0].clone());
        apply(&mut runtime, 3, partials[1].clone());
        assert!(runtime.result(op).is_none());
        apply(&mut runtime, 4, partials[2].clone());
        assert_eq!(runtime.result(op), Some([9.0, 12.0].as_slice()));
        assert_eq!(
            runtime
                .shard(TensorKey::new(TensorId(1), ShardId(0)))
                .unwrap()
                .values,
            vec![9.0, 12.0]
        );
    }

    #[test]
    fn max_and_gather_use_contributor_identity_order() {
        let nodes = [node("a"), node("b"), node("c")];
        let max_op = OpId::new(nodes[0], 1);
        let gather_op = OpId::new(nodes[0], 2);
        let mut runtime = Runtime::new();
        apply(
            &mut runtime,
            1,
            Command::DeclareOp(spec(max_op, FuseKind::Max, nodes)),
        );
        apply(
            &mut runtime,
            2,
            Command::DeclareOp(OpSpec {
                tensor: TensorKey::new(TensorId(2), ShardId(0)),
                ..spec(gather_op, FuseKind::Gather, nodes)
            }),
        );
        let mut sequence = 3;
        for command in contributions() {
            apply(&mut runtime, sequence, command.clone());
            sequence += 1;
            apply(
                &mut runtime,
                sequence,
                match command {
                    Command::Contribute {
                        op,
                        contributor,
                        values,
                    } => Command::Contribute {
                        op: OpId::new(op.origin, gather_op.counter),
                        contributor,
                        values,
                    },
                    _ => unreachable!(),
                },
            );
            sequence += 1;
        }
        assert_eq!(runtime.result(max_op), Some([5.0, 6.0].as_slice()));
        let partials = BTreeMap::from([
            (nodes[0], vec![1.0, 2.0]),
            (nodes[1], vec![3.0, 4.0]),
            (nodes[2], vec![5.0, 6.0]),
        ]);
        let expected: Vec<f64> = partials.values().flatten().copied().collect();
        assert_eq!(runtime.result(gather_op), Some(expected.as_slice()));
    }

    #[test]
    fn duplicate_contribution_is_idempotent() {
        let nodes = [node("a"), node("b"), node("c")];
        let op = OpId::new(nodes[0], 1);
        let mut runtime = Runtime::new();
        apply(
            &mut runtime,
            1,
            Command::DeclareOp(spec(op, FuseKind::Sum, nodes)),
        );
        for (index, command) in contributions().into_iter().enumerate() {
            apply(&mut runtime, index as u64 + 2, command.clone());
            apply(&mut runtime, index as u64 + 10, command);
        }
        assert_eq!(runtime.result(op), Some([9.0, 12.0].as_slice()));
    }

    #[test]
    fn contribution_before_declaration_waits() {
        let nodes = [node("a"), node("b"), node("c")];
        let op = OpId::new(nodes[0], 1);
        let mut runtime = Runtime::new();
        let partials = contributions();
        apply(&mut runtime, 1, partials[0].clone());
        apply(&mut runtime, 2, partials[1].clone());
        apply(&mut runtime, 3, partials[2].clone());
        assert!(runtime.result(op).is_none());
        apply(
            &mut runtime,
            4,
            Command::DeclareOp(spec(op, FuseKind::Sum, nodes)),
        );
        assert_eq!(runtime.result(op), Some([9.0, 12.0].as_slice()));
    }

    #[test]
    fn top_k_merges_local_candidate_lists() {
        let nodes = [node("a"), node("b"), node("c")];
        let op = OpId::new(nodes[0], 1);
        let mut runtime = Runtime::new();
        apply(
            &mut runtime,
            1,
            Command::DeclareOp(OpSpec {
                id: op,
                kind: FuseKind::TopK { k: 3 },
                contributors: BTreeSet::from([nodes[0], nodes[1], nodes[2]]),
                tensor: TensorKey::new(TensorId(3), ShardId(0)),
                epoch: 1,
            }),
        );
        apply(
            &mut runtime,
            2,
            Command::Contribute {
                op,
                contributor: nodes[0],
                values: vec![0.9, 1.0, 0.8, 2.0],
            },
        );
        apply(
            &mut runtime,
            3,
            Command::Contribute {
                op,
                contributor: nodes[1],
                values: vec![0.95, 7.0, 0.1, 3.0],
            },
        );
        apply(
            &mut runtime,
            4,
            Command::Contribute {
                op,
                contributor: nodes[2],
                values: vec![],
            },
        );
        assert_eq!(runtime.result(op), Some([7.0, 1.0, 2.0].as_slice()));
    }

    #[test]
    fn online_softmax_matches_direct_softmax() {
        let nodes = [node("a"), node("b"), node("c")];
        let op = OpId::new(nodes[0], 1);
        let mut runtime = Runtime::new();
        apply(
            &mut runtime,
            1,
            Command::DeclareOp(OpSpec {
                id: op,
                kind: FuseKind::OnlineSoftmax,
                contributors: BTreeSet::from([nodes[0], nodes[1]]),
                tensor: TensorKey::new(TensorId(4), ShardId(0)),
                epoch: 1,
            }),
        );
        let e = (-1.0f64).exp();
        apply(
            &mut runtime,
            2,
            Command::Contribute {
                op,
                contributor: nodes[0],
                values: vec![2.0, 1.0 + e, 20.0 + 10.0 * e],
            },
        );
        apply(
            &mut runtime,
            3,
            Command::Contribute {
                op,
                contributor: nodes[1],
                values: vec![3.0, 1.0, 30.0],
            },
        );
        let combined = runtime.result(op).unwrap();
        let sum = combined[1];
        let weights = [(-1.0f64).exp(), (-2.0f64).exp(), 1.0];
        let denominator: f64 = weights.iter().sum();
        let expected = (20.0 * weights[0] + 10.0 * weights[1] + 30.0 * weights[2]) / denominator;
        assert!((combined[2] / sum - expected).abs() < 1e-12);
    }

    #[test]
    fn registered_device_receives_compute_placement() {
        let nodes = [node("a"), node("b"), node("c")];
        let mut runtime = Runtime::new();
        let device = plait_core::DeviceProfile {
            id: plait_core::DeviceId::new(nodes[0], 0),
            name: "gpu-0".into(),
            arch: AcceleratorKind::Nvidia,
            memory_bytes: 1024,
            flops: 10,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: true,
            numa_node: None,
        };
        apply(&mut runtime, 1, Command::RegisterDevice(device.clone()));
        let task = OpId::new(nodes[0], 1);
        let output = TensorKey::new(TensorId(22), ShardId(0));
        apply(
            &mut runtime,
            2,
            Command::Compute(ComputeTask {
                task,
                kernel: "dsa.attention".into(),
                device: device.id,
                requirement: ComputeRequirement::default(),
                inputs: Vec::new(),
                output,
                version: 1,
                values: vec![4.0, 5.0],
            }),
        );
        assert_eq!(runtime.placement(task), Some(device.id));
        assert_eq!(runtime.shard(output).unwrap().values, vec![4.0, 5.0]);
        assert!(runtime.devices().contains_key(&device.id));
    }

    #[test]
    fn membership_records_addresses() {
        let mut runtime = Runtime::new();
        let new = node("new");
        apply(
            &mut runtime,
            1,
            Command::Membership(MembershipChange::StageAdd {
                node: new,
                addr: "127.0.0.1:9004".to_owned(),
            }),
        );
        assert_eq!(
            runtime.members().get(&new).map(String::as_str),
            Some("127.0.0.1:9004")
        );
        apply(
            &mut runtime,
            2,
            Command::Membership(MembershipChange::CommitRemove { node: new }),
        );
        assert!(!runtime.members().contains_key(&new));
    }
}
