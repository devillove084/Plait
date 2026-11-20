//! Online strongly-connected-component scheduler for committed instances.
//!
//! The graph is not rebuilt on every scheduling decision. A committed instance
//! creates a singleton component; every dependency edge either joins the
//! condensation DAG or closes a cycle, in which case exactly the components on
//! paths through the new edge are merged. Execution removes a whole component.
//!
//! The ready frontier is the set of components with no outgoing dependency
//! edges and no unresolved dependency on an instance that has not committed.
//! This is the online equivalent of a topological-sort frontier; components in
//! the frontier can execute in parallel.
//!
//! The algorithm is a dynamic-SCC/condensation algorithm, not an amortized
//! Haeupler-style implementation. A cycle merge performs a reachability search
//! on the current condensation, which is bounded by the pending frontier rather
//! than the entire replicated log.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::{Command, KeyClass, Mode};

use crate::types::InstanceId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeKinds(u8);

impl EdgeKinds {
    pub const DATA: Self = Self(1);
    pub const DEVICE: Self = Self(2);
    pub const CONFIG: Self = Self(4);
    pub const CONTROL: Self = Self(8);

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

#[derive(Clone, Debug)]
struct NodeMeta {
    seq: u64,
    deps: BTreeSet<InstanceId>,
    unresolved: BTreeSet<InstanceId>,
    accesses: Vec<plait_core::Access>,
    command: Command,
}

#[derive(Clone, Debug, Default)]
struct Component {
    members: BTreeSet<InstanceId>,
    /// Components this component depends on. Empty means all dependencies are
    /// either executed or part of this component.
    succ: BTreeSet<usize>,
    /// Reverse edges, used to find cycle participants.
    pred: BTreeSet<usize>,
    /// Number of dependencies that have not committed yet.
    blocked: usize,
}

/// Incrementally maintained SCC condensation and ready frontier.
pub struct ExecutionGraph {
    nodes: BTreeMap<InstanceId, NodeMeta>,
    components: BTreeMap<usize, Component>,
    node_component: BTreeMap<InstanceId, usize>,
    next_component: usize,
    /// Raw dependent edges, classified by the conflicting access keys.
    raw_edges: BTreeMap<(InstanceId, InstanceId), EdgeKinds>,
    /// All nodes that have ever depended on a key, including instances that
    /// have already executed. Useful when a late dependency commits.
    dependents: BTreeMap<InstanceId, BTreeSet<InstanceId>>,
    executed: BTreeSet<InstanceId>,
    ready: BTreeSet<usize>,
    merges: u64,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            components: BTreeMap::new(),
            node_component: BTreeMap::new(),
            next_component: 0,
            raw_edges: BTreeMap::new(),
            dependents: BTreeMap::new(),
            executed: BTreeSet::new(),
            ready: BTreeSet::new(),
            merges: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Number of online cycle merges performed. A value of zero does not mean
    /// the graph has no cycles; it is telemetry for dynamic updates only.
    pub fn merges(&self) -> u64 {
        self.merges
    }

    pub fn node_ids(&self) -> impl Iterator<Item = InstanceId> + '_ {
        self.nodes.keys().copied()
    }

    pub fn command(&self, id: InstanceId) -> Option<&Command> {
        self.nodes.get(&id).map(|meta| &meta.command)
    }

    pub fn deps(&self, id: InstanceId) -> Option<&BTreeSet<InstanceId>> {
        self.nodes.get(&id).map(|meta| &meta.deps)
    }

    pub fn is_executed(&self, id: InstanceId) -> bool {
        self.executed.contains(&id)
    }

    /// Add a committed instance and wire all edges whose dependencies are
    /// already present. If a dependency commits later, the edge is added then.
    pub fn commit(
        &mut self,
        id: InstanceId,
        command: Command,
        seq: u64,
        deps: &BTreeSet<InstanceId>,
    ) {
        debug_assert!(!self.nodes.contains_key(&id), "instance committed twice");
        let accesses = command.accesses();
        let mut unresolved = BTreeSet::new();
        for dep in deps {
            if *dep == id {
                continue;
            }
            self.dependents.entry(*dep).or_default().insert(id);
            if !self.nodes.contains_key(dep) && !self.executed.contains(dep) {
                unresolved.insert(*dep);
            }
        }
        self.nodes.insert(
            id,
            NodeMeta {
                seq,
                deps: deps.clone(),
                unresolved: unresolved.clone(),
                accesses,
                command,
            },
        );
        let component = self.next_component;
        self.next_component += 1;
        self.components.insert(
            component,
            Component {
                members: BTreeSet::from([id]),
                succ: BTreeSet::new(),
                pred: BTreeSet::new(),
                blocked: unresolved.len(),
            },
        );
        self.node_component.insert(id, component);
        self.sync_ready(component);

        for dep in deps {
            if *dep != id && self.nodes.contains_key(dep) {
                let kinds = edge_kinds(&self.nodes[&id], &self.nodes[dep]);
                self.raw_edges.insert((id, *dep), kinds);
                self.add_edge(id, *dep);
            }
        }

        // This instance may have been the missing dependency of instances that
        // committed earlier. Resolve them now.
        let dependents = self.dependents.get(&id).cloned().unwrap_or_default();
        for dependent in dependents {
            if dependent == id {
                continue;
            }
            let unresolved_here = self
                .nodes
                .get(&dependent)
                .is_some_and(|meta| meta.unresolved.contains(&id));
            if unresolved_here {
                self.resolve_unresolved(dependent, id);
            }
        }
        if let Some(&component) = self.node_component.get(&id) {
            self.sync_ready(component);
        }
    }

    /// Mark an instance executed. Replica execution batches are SCCs, so the
    /// entire component is removed on the first member; later members are
    /// idempotent. Removing a component can make its predecessors ready.
    pub fn mark_executed(&mut self, id: InstanceId) {
        self.executed.insert(id);
        let Some(component) = self.node_component.get(&id).copied() else {
            return;
        };
        let Some(entry) = self.components.remove(&component) else {
            return;
        };
        self.ready.remove(&component);

        for predecessor in &entry.pred {
            if let Some(pred) = self.components.get_mut(predecessor) {
                pred.succ.remove(&component);
            }
            self.sync_ready(*predecessor);
        }
        for successor in &entry.succ {
            if let Some(succ) = self.components.get_mut(successor) {
                succ.pred.remove(&component);
            }
        }

        for member in entry.members {
            self.executed.insert(member);
            self.node_component.remove(&member);
            if let Some(meta) = self.nodes.remove(&member) {
                for dep in meta.deps {
                    if let Some(dependents) = self.dependents.get_mut(&dep) {
                        dependents.remove(&member);
                    }
                }
            }
            self.raw_edges
                .retain(|(from, to), _| *from != member && *to != member);
        }
    }

    /// Raw dependency edges, used by diagnostics and the Tarjan oracle tests.
    pub fn edges(&self) -> Vec<(InstanceId, InstanceId, EdgeKinds)> {
        self.raw_edges
            .iter()
            .map(|((from, to), kinds)| (*from, *to, *kinds))
            .collect()
    }

    /// Strongly connected components in the current pending graph.
    pub fn sccs(&self) -> Vec<Vec<InstanceId>> {
        let mut sccs: Vec<Vec<InstanceId>> = self
            .components
            .values()
            .map(|component| component.members.iter().copied().collect())
            .collect();
        sccs.sort();
        sccs
    }

    /// Components that can execute now. The ready set is maintained online; a
    /// component is ready exactly when it has no outgoing edges and no
    /// unresolved dependency.
    pub fn ready_batches(&self) -> Vec<Vec<InstanceId>> {
        let mut batches: Vec<Vec<InstanceId>> = self
            .ready
            .iter()
            .filter_map(|component| self.components.get(component))
            .map(|component| {
                let mut batch: Vec<InstanceId> = component.members.iter().copied().collect();
                batch.sort_by_key(|id| (self.nodes.get(id).map(|meta| meta.seq).unwrap_or(0), *id));
                batch
            })
            .collect();
        batches.sort_by_key(|batch| {
            batch
                .iter()
                .map(|id| (self.nodes.get(id).map(|meta| meta.seq).unwrap_or(0), *id))
                .min()
        });
        batches
    }

    fn sync_ready(&mut self, component: usize) {
        let ready = self
            .components
            .get(&component)
            .is_some_and(|entry| entry.blocked == 0 && entry.succ.is_empty());
        if ready {
            self.ready.insert(component);
        } else {
            self.ready.remove(&component);
        }
    }

    fn resolve_unresolved(&mut self, dependent: InstanceId, dep: InstanceId) {
        let removed = self
            .nodes
            .get_mut(&dependent)
            .is_some_and(|meta| meta.unresolved.remove(&dep));
        if !removed {
            return;
        }
        let component = self.node_component.get(&dependent).copied();
        if let Some(component) = component {
            if let Some(entry) = self.components.get_mut(&component) {
                entry.blocked = entry.blocked.saturating_sub(1);
            }
        }
        if self.nodes.contains_key(&dep) {
            let kinds = edge_kinds(&self.nodes[&dependent], &self.nodes[&dep]);
            self.raw_edges.insert((dependent, dep), kinds);
            self.add_edge(dependent, dep);
        }
        if let Some(component) = component {
            self.sync_ready(component);
        }
    }

    fn add_edge(&mut self, from: InstanceId, to: InstanceId) {
        let Some(&from_component) = self.node_component.get(&from) else {
            return;
        };
        let Some(&to_component) = self.node_component.get(&to) else {
            return;
        };
        if from_component == to_component {
            return;
        }
        if self
            .components
            .get(&from_component)
            .is_some_and(|entry| entry.succ.contains(&to_component))
        {
            return;
        }
        // from -> to closes a cycle exactly when to can already reach from.
        if self.path_exists(to_component, from_component) {
            self.merge_cycle(from_component, to_component);
        } else {
            if let Some(entry) = self.components.get_mut(&from_component) {
                entry.succ.insert(to_component);
            }
            if let Some(entry) = self.components.get_mut(&to_component) {
                entry.pred.insert(from_component);
            }
            self.sync_ready(from_component);
        }
    }

    fn path_exists(&self, from: usize, to: usize) -> bool {
        if from == to {
            return true;
        }
        let mut stack = vec![from];
        let mut seen = BTreeSet::new();
        while let Some(component) = stack.pop() {
            if !seen.insert(component) {
                continue;
            }
            let Some(entry) = self.components.get(&component) else {
                continue;
            };
            for next in &entry.succ {
                if *next == to {
                    return true;
                }
                stack.push(*next);
            }
        }
        false
    }

    fn reachable(&self, start: usize) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![start];
        while let Some(component) = stack.pop() {
            if !seen.insert(component) {
                continue;
            }
            if let Some(entry) = self.components.get(&component) {
                stack.extend(entry.succ.iter().copied());
            }
        }
        seen
    }

    fn reverse_reachable(&self, start: usize) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![start];
        while let Some(component) = stack.pop() {
            if !seen.insert(component) {
                continue;
            }
            if let Some(entry) = self.components.get(&component) {
                stack.extend(entry.pred.iter().copied());
            }
        }
        seen
    }

    /// Merge every component on a path from `dependency` to `dependent` after
    /// adding `dependent -> dependency`. The intersection of forward
    /// reachability from `dependency` and reverse reachability from
    /// `dependent` is exactly the set of components on such paths.
    fn merge_cycle(&mut self, dependent: usize, dependency: usize) {
        let mut cycle = self.reachable(dependency);
        let back = self.reverse_reachable(dependent);
        cycle.retain(|component| back.contains(component));
        cycle.insert(dependent);
        cycle.insert(dependency);
        if cycle.len() < 2 {
            return;
        }

        let keep = *cycle.iter().min().expect("cycle is non-empty");
        let snapshots: Vec<(usize, Component)> = cycle
            .iter()
            .filter_map(|component| {
                self.components
                    .get(component)
                    .cloned()
                    .map(|entry| (*component, entry))
            })
            .collect();
        let mut members = BTreeSet::new();
        let mut blocked = 0usize;
        let mut succ = BTreeSet::new();
        let mut pred = BTreeSet::new();
        for (component, entry) in &snapshots {
            members.extend(entry.members.iter().copied());
            blocked += entry.blocked;
            for next in &entry.succ {
                if !cycle.contains(next) {
                    succ.insert(*next);
                    if let Some(target) = self.components.get_mut(next) {
                        target.pred.remove(component);
                        target.pred.insert(keep);
                    }
                }
            }
            for previous in &entry.pred {
                if !cycle.contains(previous) {
                    pred.insert(*previous);
                    if let Some(source) = self.components.get_mut(previous) {
                        source.succ.remove(component);
                        source.succ.insert(keep);
                    }
                }
            }
        }

        for component in &cycle {
            self.ready.remove(component);
            self.components.remove(component);
        }
        let member_ids: Vec<InstanceId> = members.iter().copied().collect();
        self.components.insert(
            keep,
            Component {
                members,
                succ,
                pred,
                blocked,
            },
        );
        for member in &member_ids {
            self.node_component.insert(*member, keep);
        }
        self.sync_ready(keep);
        self.merges += 1;
    }
}

impl Default for ExecutionGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn edge_kinds(left: &NodeMeta, right: &NodeMeta) -> EdgeKinds {
    let mut kinds = EdgeKinds::default();
    for a in &left.accesses {
        for b in &right.accesses {
            // A dependency exists because this pair conflicts; read/read
            // configuration accesses are shared by every command and must not
            // turn every edge into a configuration edge.
            if a.key == b.key && (a.mode != Mode::Read || b.mode != Mode::Read) {
                kinds.insert(match a.key.class() {
                    KeyClass::Data => EdgeKinds::DATA,
                    KeyClass::Device => EdgeKinds::DEVICE,
                    KeyClass::Config => EdgeKinds::CONFIG,
                });
            }
        }
    }
    if kinds == EdgeKinds::default() {
        kinds.insert(EdgeKinds::CONTROL);
    }
    kinds
}
#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::algo::tarjan_scc;
    use petgraph::graphmap::DiGraphMap;
    use plait_core::{
        AcceleratorKind, DeviceId, DeviceProfile, MembershipChange, NodeId, ShardId, TensorId,
        TensorKey,
    };

    fn id(name: &str, counter: u64) -> InstanceId {
        InstanceId {
            proposer: NodeId::derive(name),
            counter,
        }
    }

    fn commit(graph: &mut ExecutionGraph, id: InstanceId, seq: u64, deps: &[InstanceId]) {
        let deps: BTreeSet<InstanceId> = deps.iter().copied().collect();
        graph.commit(id, Command::Nop, seq, &deps);
    }

    fn oracle(graph: &ExecutionGraph) -> Vec<Vec<InstanceId>> {
        let mut expected: DiGraphMap<InstanceId, ()> = DiGraphMap::new();
        for id in graph.node_ids() {
            expected.add_node(id);
        }
        for (from, to, _) in graph.edges() {
            expected.add_edge(from, to, ());
        }
        let mut sccs: Vec<Vec<InstanceId>> = tarjan_scc(&expected)
            .into_iter()
            .map(|mut component| {
                component.sort();
                component
            })
            .collect();
        sccs.sort();
        sccs
    }

    fn expected_ready(graph: &ExecutionGraph) -> Vec<Vec<InstanceId>> {
        let components = graph.sccs();
        let mut ready = Vec::new();
        for component in &components {
            let members: BTreeSet<InstanceId> = component.iter().copied().collect();
            let is_ready = component.iter().all(|id| {
                graph.deps(*id).is_some_and(|deps| {
                    deps.iter()
                        .all(|dep| members.contains(dep) || graph.is_executed(*dep))
                })
            });
            if is_ready {
                ready.push(component.clone());
            }
        }
        ready.sort();
        ready
    }

    fn assert_ready_matches(graph: &ExecutionGraph) {
        let mut actual = graph.ready_batches();
        actual.sort();
        assert_eq!(actual, expected_ready(graph));
    }

    #[test]
    fn online_scc_matches_tarjan_on_random_commits() {
        let ids: Vec<InstanceId> = (0..48).map(|index| id("random", index)).collect();
        let mut graph = ExecutionGraph::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 7;
            seed ^= seed >> 9;
            seed ^= seed << 8;
            seed
        };
        for (index, node) in ids.iter().enumerate() {
            let mut deps = BTreeSet::new();
            for candidate in &ids {
                if candidate != node && next() % 7 == 0 {
                    deps.insert(*candidate);
                }
            }
            commit(
                &mut graph,
                *node,
                index as u64,
                &deps.iter().copied().collect::<Vec<_>>(),
            );
            assert_eq!(graph.sccs(), oracle(&graph), "mismatch after {index}");
            assert_ready_matches(&graph);
            let ready = graph.ready_batches();
            for batch in &ready {
                for node in batch {
                    assert!(graph.nodes.contains_key(node));
                }
            }
        }
        assert!(graph.merges() > 0, "test graph should contain a cycle");
    }

    #[test]
    fn online_scc_matches_tarjan_while_components_execute() {
        let ids: Vec<InstanceId> = (0..36).map(|index| id("execute", index)).collect();
        let mut graph = ExecutionGraph::new();
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed << 7;
            seed ^= seed >> 9;
            seed ^= seed << 8;
            seed
        };
        for (index, node) in ids.iter().enumerate() {
            let mut deps = BTreeSet::new();
            for candidate in &ids {
                if candidate != node && next() % 6 == 0 {
                    deps.insert(*candidate);
                }
            }
            let deps: Vec<InstanceId> = deps.into_iter().collect();
            commit(&mut graph, *node, index as u64, &deps);
            assert_eq!(
                graph.sccs(),
                oracle(&graph),
                "mismatch after commit {index}"
            );
            if index % 3 == 0 {
                if let Some(batch) = graph.ready_batches().into_iter().next() {
                    for executed in batch {
                        graph.mark_executed(executed);
                    }
                    assert_eq!(
                        graph.sccs(),
                        oracle(&graph),
                        "mismatch after execute {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn late_dependency_creates_an_edge_and_blocks() {
        let a = id("late-a", 1);
        let b = id("late-b", 1);
        let mut graph = ExecutionGraph::new();
        commit(&mut graph, a, 1, &[b]);
        assert!(graph.ready_batches().is_empty());

        commit(&mut graph, b, 2, &[]);
        assert_eq!(graph.ready_batches(), vec![vec![b]]);
        assert!(
            graph.edges().iter().any(|(from, to, kinds)| *from == a
                && *to == b
                && kinds.contains(EdgeKinds::CONTROL))
        );

        graph.mark_executed(b);
        assert_eq!(graph.ready_batches(), vec![vec![a]]);
    }

    #[test]
    fn a_cycle_becomes_one_ready_component() {
        let a = id("cycle-a", 1);
        let b = id("cycle-b", 1);
        let c = id("cycle-c", 1);
        let mut graph = ExecutionGraph::new();
        commit(&mut graph, a, 1, &[b]);
        commit(&mut graph, b, 2, &[a]);
        commit(&mut graph, c, 3, &[a]);

        assert_eq!(graph.ready_batches(), vec![vec![a, b]]);
        graph.mark_executed(a);
        assert_eq!(graph.ready_batches(), vec![vec![c]]);
    }

    #[test]
    fn edges_are_classified_by_key_kind() {
        let node = NodeId::derive("edge-node");
        let tensor = TensorKey::new(TensorId(1), ShardId(0));
        let device = DeviceId::new(node, 0);
        let profile = DeviceProfile {
            id: device,
            name: "test".into(),
            arch: AcceleratorKind::Cpu,
            memory_bytes: 1,
            flops: 1,
            compute_units: 1,
            supports_fp16: false,
            supports_bf16: false,
            supports_fp64: false,
            rdma_capable: false,
            numa_node: None,
        };

        let a = id("edge-a", 1);
        let b = id("edge-b", 1);
        let c = id("edge-c", 1);
        let d = id("edge-d", 1);
        let mut graph = ExecutionGraph::new();
        graph.commit(
            a,
            Command::Publish {
                key: tensor,
                version: 1,
                values: vec![1.0],
            },
            1,
            &BTreeSet::new(),
        );
        graph.commit(
            b,
            Command::Publish {
                key: tensor,
                version: 2,
                values: vec![2.0],
            },
            2,
            &BTreeSet::from([a]),
        );
        graph.commit(
            c,
            Command::RegisterDevice(profile.clone()),
            3,
            &BTreeSet::from([a]),
        );
        graph.commit(d, Command::RegisterDevice(profile), 4, &BTreeSet::from([c]));
        let edge = |from, to| {
            graph
                .edges()
                .into_iter()
                .find(|(f, t, _)| *f == from && *t == to)
                .map(|(_, _, kinds)| kinds)
                .expect("edge exists")
        };
        assert!(edge(b, a).contains(EdgeKinds::DATA));
        assert!(!edge(b, a).contains(EdgeKinds::CONFIG));
        assert!(edge(d, c).contains(EdgeKinds::DEVICE));
        assert!(!edge(d, c).contains(EdgeKinds::CONFIG));

        let membership = id("edge-m", 1);
        let member = id("edge-n", 1);
        graph.commit(
            membership,
            Command::Membership(MembershipChange::StageRemove { node }),
            5,
            &BTreeSet::new(),
        );
        graph.commit(
            member,
            Command::Membership(MembershipChange::CommitRemove { node }),
            6,
            &BTreeSet::from([membership]),
        );
        assert!(graph.edges().iter().any(|(f, t, kinds)| {
            *f == member && *t == membership && kinds.contains(EdgeKinds::CONFIG)
        }));
    }

    #[test]
    fn key_classes_match_command_accesses() {
        assert_eq!(
            plait_core::Key::Tensor(TensorKey::new(TensorId(1), ShardId(0))).class(),
            plait_core::KeyClass::Data
        );
        assert_eq!(
            plait_core::Key::Device(DeviceId::new(NodeId::derive("x"), 0)).class(),
            plait_core::KeyClass::Device
        );
        assert_eq!(
            plait_core::Key::Config.class(),
            plait_core::KeyClass::Config
        );
    }
}
