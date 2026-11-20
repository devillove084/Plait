//! Explicit, cached execution graph over committed consensus instances.
//!
//! Nodes are committed instances; edges are EPaxos dependencies classified as
//! data, device, or configuration edges. The graph is dynamic: it grows as
//! commits arrive and shrinks as SCCs execute. SCCs are cached until a commit
//! changes the graph; execution removes whole components without invalidating
//! unrelated components.

use std::collections::{BTreeMap, BTreeSet};

use petgraph::algo::tarjan_scc;
use petgraph::graphmap::DiGraphMap;
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
    accesses: Vec<plait_core::Access>,
    command: Command,
}

/// Dynamic dependency graph and cached SCC condensation.
pub struct ExecutionGraph {
    nodes: BTreeMap<InstanceId, NodeMeta>,
    graph: DiGraphMap<InstanceId, EdgeKinds>,
    dependents: BTreeMap<InstanceId, BTreeSet<InstanceId>>,
    components: Vec<Vec<InstanceId>>,
    dirty: bool,
    rebuilds: u64,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            graph: DiGraphMap::new(),
            dependents: BTreeMap::new(),
            components: Vec::new(),
            dirty: false,
            rebuilds: 0,
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

    pub fn component_count(&mut self) -> usize {
        self.refresh();
        self.components.len()
    }

    /// Add or update a committed instance. Edges to already-present deps and
    /// from already-present dependents are created immediately; SCCs are
    /// recomputed lazily.
    pub fn commit(
        &mut self,
        id: InstanceId,
        command: Command,
        seq: u64,
        deps: &BTreeSet<InstanceId>,
    ) {
        let has_pending_dep = deps.iter().any(|dep| self.nodes.contains_key(dep));
        let has_pending_dependent = self.dependents.get(&id).is_some_and(|dependents| {
            dependents
                .iter()
                .any(|dependent| self.nodes.contains_key(dependent))
        });
        let accesses = command.accesses();
        self.nodes.insert(
            id,
            NodeMeta {
                seq,
                deps: deps.clone(),
                accesses,
                command,
            },
        );
        self.graph.add_node(id);
        for dep in deps {
            self.dependents.entry(*dep).or_default().insert(id);
            if let Some(dep_meta) = self.nodes.get(dep) {
                let kinds = edge_kinds(&self.nodes[&id], dep_meta);
                self.graph.add_edge(id, *dep, kinds);
            }
        }
        if let Some(dependents) = self.dependents.get(&id).cloned() {
            for dependent in dependents {
                if let Some(meta) = self.nodes.get(&dependent) {
                    let kinds = edge_kinds(meta, &self.nodes[&id]);
                    self.graph.add_edge(dependent, id, kinds);
                }
            }
        }
        if has_pending_dep || has_pending_dependent {
            // A new edge can merge existing components; defer SCC work until
            // the next scheduling decision.
            self.dirty = true;
        } else if !self.dirty {
            // No edges to the pending frontier: this instance is its own SCC,
            // so the cached condensation stays valid.
            self.components.push(vec![id]);
        }
    }

    /// Remove an executed instance. Removing a singleton leaves unrelated SCCs
    /// valid; removing part of a multi-node component invalidates the cache so
    /// the remainder can be recomputed before the next scheduling decision.
    pub fn mark_executed(&mut self, id: InstanceId) {
        self.nodes.remove(&id);
        self.graph.remove_node(id);
        let mut split = false;
        for component in &mut self.components {
            if component.contains(&id) {
                split |= component.len() > 1;
                component.retain(|candidate| *candidate != id);
            }
        }
        self.components.retain(|component| !component.is_empty());
        // Removing one member of a multi-node SCC can split the remainder;
        // recompute before the next scheduling decision.
        if split {
            self.dirty = true;
        }
    }

    /// All dependency edges currently in the graph, for diagnostics/tests.
    pub fn edges(&mut self) -> Vec<(InstanceId, InstanceId, EdgeKinds)> {
        self.refresh();
        self.graph
            .all_edges()
            .map(|(from, to, kinds)| (from, to, *kinds))
            .collect()
    }

    /// Components that can execute now: every external dependency is executed
    /// or part of the same component.
    pub fn ready_batches(&mut self, executed: &BTreeSet<InstanceId>) -> Vec<Vec<InstanceId>> {
        self.refresh();
        let mut ready = Vec::new();
        for component in &self.components {
            let set: BTreeSet<InstanceId> = component.iter().copied().collect();
            let is_ready = component.iter().all(|id| {
                self.nodes
                    .get(id)
                    .map(|meta| {
                        meta.deps
                            .iter()
                            .all(|dep| executed.contains(dep) || set.contains(dep))
                    })
                    .unwrap_or(false)
            });
            if is_ready {
                let mut batch = component.clone();
                batch.sort_by_key(|id| (self.nodes.get(id).map(|meta| meta.seq).unwrap_or(0), *id));
                ready.push(batch);
            }
        }
        ready.sort_by_key(|batch| {
            batch
                .iter()
                .map(|id| (self.nodes.get(id).map(|meta| meta.seq).unwrap_or(0), *id))
                .min()
        });
        ready
    }

    /// Number of full SCC recomputations. Used to verify the cache.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    pub fn command(&self, id: InstanceId) -> Option<&Command> {
        self.nodes.get(&id).map(|meta| &meta.command)
    }

    pub fn deps(&self, id: InstanceId) -> Option<&BTreeSet<InstanceId>> {
        self.nodes.get(&id).map(|meta| &meta.deps)
    }

    fn refresh(&mut self) {
        if !self.dirty {
            return;
        }
        let components = tarjan_scc(&self.graph);
        let mut components: Vec<Vec<InstanceId>> = components
            .into_iter()
            .map(|mut component| {
                component.sort();
                component
            })
            .collect();
        components.sort_by_key(|component| {
            component
                .iter()
                .min()
                .copied()
                .unwrap_or_else(|| InstanceId {
                    proposer: plait_core::NodeId::derive("empty"),
                    counter: 0,
                })
        });
        self.components = components;
        self.dirty = false;
        self.rebuilds += 1;
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
    use plait_core::{
        AcceleratorKind, Command, DeviceId, DeviceProfile, Key, MembershipChange, NodeId, ShardId,
        TensorId, TensorKey,
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

    #[test]
    fn cycles_are_one_component_and_execute_before_dependents() {
        let a = id("graph-a", 1);
        let b = id("graph-b", 1);
        let c = id("graph-c", 1);
        let mut graph = ExecutionGraph::new();
        commit(&mut graph, a, 1, &[b]);
        commit(&mut graph, b, 2, &[a]);
        commit(&mut graph, c, 3, &[a]);

        let executed = BTreeSet::new();
        let ready = graph.ready_batches(&executed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], vec![a, b]);

        graph.mark_executed(a);
        graph.mark_executed(b);
        let executed = BTreeSet::from([a, b]);
        let ready = graph.ready_batches(&executed);
        assert_eq!(ready, vec![vec![c]]);
    }

    #[test]
    fn a_dependency_committed_later_creates_an_edge_and_blocks() {
        let a = id("late-a", 1);
        let b = id("late-b", 1);
        let mut graph = ExecutionGraph::new();
        commit(&mut graph, a, 1, &[b]);
        let executed = BTreeSet::new();
        assert!(graph.ready_batches(&executed).is_empty());

        commit(&mut graph, b, 2, &[]);
        let ready = graph.ready_batches(&executed);
        assert_eq!(ready, vec![vec![b]]);
        assert!(
            graph.edges().iter().any(|(from, to, kinds)| *from == a
                && *to == b
                && kinds.contains(EdgeKinds::CONTROL))
        );

        graph.mark_executed(b);
        let executed = BTreeSet::from([b]);
        let ready = graph.ready_batches(&executed);
        assert_eq!(ready, vec![vec![a]]);
    }

    #[test]
    fn scc_cache_is_reused_until_a_commit_changes_the_graph() {
        let a = id("cache-a", 1);
        let b = id("cache-b", 1);
        let mut graph = ExecutionGraph::new();
        commit(&mut graph, a, 1, &[]);
        let executed = BTreeSet::new();
        // A singleton with no pending edges is inserted without Tarjan.
        let _ = graph.ready_batches(&executed);
        assert_eq!(graph.rebuilds(), 0);
        let _ = graph.ready_batches(&executed);
        assert_eq!(graph.rebuilds(), 0);

        graph.mark_executed(a);
        let _ = graph.ready_batches(&executed);
        assert_eq!(graph.rebuilds(), 0);

        commit(&mut graph, b, 2, &[]);
        let _ = graph.ready_batches(&executed);
        assert_eq!(graph.rebuilds(), 0);
        assert_eq!(graph.ready_batches(&executed).len(), 1);

        let c = id("cache-c", 1);
        commit(&mut graph, c, 3, &[b]);
        let _ = graph.ready_batches(&executed);
        assert_eq!(graph.rebuilds(), 1);
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
        let edges = graph.edges();
        let kind = |from, to| {
            edges
                .iter()
                .find(|(f, t, _)| *f == from && *t == to)
                .map(|(_, _, kinds)| *kinds)
                .expect("edge exists")
        };
        assert!(kind(b, a).contains(EdgeKinds::DATA));
        assert!(!kind(b, a).contains(EdgeKinds::CONFIG));
        assert!(kind(d, c).contains(EdgeKinds::DEVICE));
        assert!(!kind(d, c).contains(EdgeKinds::CONFIG));

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
        assert!(graph.edges().iter().any(|(f, t, kinds)| *f == member
            && *t == membership
            && kinds.contains(EdgeKinds::CONFIG)));
    }

    #[test]
    fn key_classes_match_command_accesses() {
        assert_eq!(
            Key::Tensor(TensorKey::new(TensorId(1), ShardId(0))).class(),
            plait_core::KeyClass::Data
        );
        assert_eq!(
            Key::Device(DeviceId::new(NodeId::derive("x"), 0)).class(),
            plait_core::KeyClass::Device
        );
        assert_eq!(Key::Config.class(), plait_core::KeyClass::Config);
    }
}
