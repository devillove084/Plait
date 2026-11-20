use std::collections::{BTreeMap, BTreeSet};

use plait_core::{ComputeRequirement, DType, FuseKind, NodeId, OpId};
use serde::{Deserialize, Serialize};

use crate::dsa::DsaConfig;
use crate::topology::{Axis, Coord, Topology};

/// A collective is not a global step. Each variant is expressed as a fused
/// state-machine operation whose conflict key and participant group come from
/// the topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncKind {
    FusedSum,
    FusedMax,
    Gather,
    TopK {
        k: u32,
    },
    OnlineSoftmax,
    /// One pipeline stage hands a microbatch activation to the next stage.
    PipelineHandoff,
    /// Tokens move to their selected experts.
    Dispatch,
    /// Expert outputs for the same token are summed.
    Combine,
    /// Independent replicas advance without cross-replica synchronization.
    Replicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferPolicy {
    /// Small control data travels inside the consensus command.
    Inline,
    /// Bulk tensor bytes stay in registered memory; the log carries a region
    /// descriptor and peers use one-sided reads/writes.
    Registered,
}

#[derive(Clone, Debug)]
pub struct ParallelOp {
    pub name: String,
    pub op_id: OpId,
    pub axis: Axis,
    pub sync: SyncKind,
    pub participants: Vec<NodeId>,
    pub conflict_key: String,
    pub transfer: TransferPolicy,
}

impl ParallelOp {
    /// Compute requirements inferred from the axis. A planner can override this
    /// when a concrete kernel needs more memory or a different dtype.
    pub fn requirement(&self) -> ComputeRequirement {
        let dtype = match self.axis {
            Axis::Context | Axis::Expert | Axis::Pipeline | Axis::Sequence => DType::F16,
            Axis::Data | Axis::Tensor => DType::F32,
        };
        ComputeRequirement {
            dtype,
            ..ComputeRequirement::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub topology: Topology,
    pub ops: Vec<ParallelOp>,
}

impl Plan {
    pub fn op(&self, name: &str) -> Option<&ParallelOp> {
        self.ops.iter().find(|op| op.name == name)
    }

    pub fn by_axis(&self, axis: Axis) -> impl Iterator<Item = &ParallelOp> {
        self.ops.iter().filter(move |op| op.axis == axis)
    }
}

/// The concrete DSA plan plus the lookup tables the executor needs. The plan
/// names every axis; the executor uses TP/CP/SP for the reference DSA run and
/// exposes EP/DP/PP mappings for larger topologies.
#[derive(Clone, Debug)]
pub struct DsaPlan {
    pub plan: Plan,
    pub q_index: BTreeMap<u32, OpId>,
    pub candidates: BTreeMap<(u32, u32), OpId>,
    pub attention: BTreeMap<(u32, u32), OpId>,
    pub output: BTreeMap<u32, OpId>,
    pub q_index_participants: BTreeMap<u32, Vec<NodeId>>,
    pub candidate_participants: BTreeMap<u32, Vec<NodeId>>,
    pub attention_participants: BTreeMap<u32, Vec<NodeId>>,
    pub output_participants: BTreeMap<u32, Vec<NodeId>>,
}

pub fn tp_for_head(head: u32, config: &DsaConfig, topology: &Topology) -> u32 {
    let tp = topology.sizes().tensor;
    (head * tp / config.heads).min(tp.saturating_sub(1))
}

pub fn head_range(tp: u32, config: &DsaConfig, topology: &Topology) -> std::ops::Range<u32> {
    let tp_size = topology.sizes().tensor;
    let start = tp * config.heads / tp_size;
    let end = (tp + 1) * config.heads / tp_size;
    start..end
}

pub fn seq_range(cp: u32, config: &DsaConfig, topology: &Topology) -> std::ops::Range<u32> {
    let cp_size = topology.sizes().context;
    let start = cp * config.seq / cp_size;
    let end = (cp + 1) * config.seq / cp_size;
    start..end
}

fn mint(name: &str) -> OpId {
    OpId::new(NodeId::derive(name), 1)
}

pub fn dsa_plan(topology: &Topology, config: &DsaConfig) -> DsaPlan {
    let mut plan = Plan {
        topology: topology.clone(),
        ops: Vec::new(),
    };
    let mut seen = BTreeSet::new();
    let mut push = |op: ParallelOp| {
        if seen.insert(op.name.clone()) {
            plan.ops.push(op);
        }
    };

    let sizes = topology.sizes();
    let tp_size = sizes.tensor;
    let cp_size = sizes.context;
    let seq = config.seq;

    for tp in 0..tp_size {
        let base = Coord {
            tensor: tp,
            ..Coord::default()
        };
        for head in head_range(tp, config, topology) {
            let name = format!("sp.q_index:{tp}:{head}");
            // CP and SP are the same physical sequence-shard group in this
            // execution model. SP is a labeling of that group, not a second
            // multiplicative dimension.
            let participants = topology.group(Axis::Context, base);
            push(ParallelOp {
                op_id: mint(&name),
                name: name.clone(),
                axis: Axis::Sequence,
                sync: SyncKind::Gather,
                participants,
                conflict_key: format!("seq:{tp}:{head}:q_index"),
                transfer: TransferPolicy::Inline,
            });
            let bulk = format!("sp.q_bulk:{tp}:{head}");
            push(ParallelOp {
                op_id: mint(&bulk),
                name: bulk,
                axis: Axis::Sequence,
                sync: SyncKind::Gather,
                participants: topology.group(Axis::Context, base),
                conflict_key: format!("seq:{tp}:{head}:q"),
                transfer: TransferPolicy::Registered,
            });
            for query in 0..seq {
                let candidates = format!("cp.candidates:{tp}:{head}:{query}");
                push(ParallelOp {
                    op_id: mint(&candidates),
                    name: candidates,
                    axis: Axis::Context,
                    sync: SyncKind::TopK { k: config.top_k },
                    participants: topology.group(Axis::Context, base),
                    conflict_key: format!("ctx:{tp}:{head}:{query}:candidates"),
                    transfer: TransferPolicy::Inline,
                });
                let attention = format!("cp.attention:{tp}:{head}:{query}");
                push(ParallelOp {
                    op_id: mint(&attention),
                    name: attention,
                    axis: Axis::Context,
                    sync: SyncKind::OnlineSoftmax,
                    participants: topology.group(Axis::Context, base),
                    conflict_key: format!("ctx:{tp}:{head}:{query}:attention"),
                    transfer: TransferPolicy::Inline,
                });
            }
        }
    }

    for cp in 0..cp_size {
        let coord = Coord {
            context: cp,
            ..Coord::default()
        };
        let name = format!("tp.output:{cp}");
        push(ParallelOp {
            op_id: mint(&name),
            name,
            axis: Axis::Tensor,
            sync: SyncKind::FusedSum,
            participants: topology.group(Axis::Tensor, coord),
            conflict_key: format!("tp:{cp}:output"),
            transfer: TransferPolicy::Registered,
        });
    }

    // The remaining axes are part of the same plan. Their operations become
    // active when the corresponding model stages exist; the DSA reference
    // operator below executes TP/CP/SP and records these mappings for
    // reconfiguration and scheduling.
    for dp in 0..sizes.data {
        let coord = Coord {
            data: dp,
            ..Coord::default()
        };
        for shard in 0..config.hidden {
            let name = format!("dp.gradient:{dp}:{shard}");
            push(ParallelOp {
                op_id: mint(&name),
                name,
                axis: Axis::Data,
                sync: SyncKind::FusedSum,
                participants: topology.group(Axis::Data, coord),
                conflict_key: format!("dp:{dp}:gradient:{shard}"),
                transfer: TransferPolicy::Registered,
            });
        }
    }

    for stage in 0..sizes.pipeline {
        for microbatch in 0..seq {
            let name = format!("pp.handoff:{stage}:{microbatch}");
            push(ParallelOp {
                op_id: mint(&name),
                name,
                axis: Axis::Pipeline,
                sync: SyncKind::PipelineHandoff,
                participants: topology.group(
                    Axis::Pipeline,
                    Coord {
                        pipeline: stage,
                        ..Coord::default()
                    },
                ),
                conflict_key: format!("pp:{stage}:{microbatch}"),
                transfer: TransferPolicy::Registered,
            });
        }
    }

    for expert in 0..sizes.expert {
        for (label, sync) in [
            ("dispatch", SyncKind::Dispatch),
            ("combine", SyncKind::Combine),
        ] {
            let name = format!("ep.{label}:{expert}");
            push(ParallelOp {
                op_id: mint(&name),
                name,
                axis: Axis::Expert,
                sync,
                participants: topology.group(
                    Axis::Expert,
                    Coord {
                        expert,
                        ..Coord::default()
                    },
                ),
                conflict_key: format!("ep:{expert}:{label}"),
                transfer: if label == "combine" {
                    TransferPolicy::Inline
                } else {
                    TransferPolicy::Registered
                },
            });
        }
    }

    let mut dsa = DsaPlan {
        plan,
        q_index: BTreeMap::new(),
        candidates: BTreeMap::new(),
        attention: BTreeMap::new(),
        output: BTreeMap::new(),
        q_index_participants: BTreeMap::new(),
        candidate_participants: BTreeMap::new(),
        attention_participants: BTreeMap::new(),
        output_participants: BTreeMap::new(),
    };

    for op in &dsa.plan.ops {
        if let Some(rest) = op.name.strip_prefix("sp.q_index:") {
            let mut parts = rest.split(':');
            let tp: u32 = parts.next().unwrap().parse().unwrap();
            let head: u32 = parts.next().unwrap().parse().unwrap();
            let _ = tp;
            dsa.q_index.insert(head, op.op_id);
            dsa.q_index_participants
                .insert(head, op.participants.clone());
        } else if let Some(rest) = op.name.strip_prefix("cp.candidates:") {
            let parts: Vec<&str> = rest.split(':').collect();
            let tp: u32 = parts[0].parse().unwrap();
            let head: u32 = parts[1].parse().unwrap();
            let query: u32 = parts[2].parse().unwrap();
            let _ = tp;
            dsa.candidates.insert((head, query), op.op_id);
            dsa.candidate_participants
                .insert(head, op.participants.clone());
        } else if let Some(rest) = op.name.strip_prefix("cp.attention:") {
            let parts: Vec<&str> = rest.split(':').collect();
            let head: u32 = parts[1].parse().unwrap();
            let query: u32 = parts[2].parse().unwrap();
            dsa.attention.insert((head, query), op.op_id);
            dsa.attention_participants
                .insert(head, op.participants.clone());
        } else if let Some(rest) = op.name.strip_prefix("tp.output:") {
            let cp: u32 = rest.parse().unwrap();
            dsa.output.insert(cp, op.op_id);
            dsa.output_participants.insert(cp, op.participants.clone());
        }
    }

    dsa
}

/// Convert a plan-level synchronization to the runtime fusion kind. Pipeline
/// handoffs and dispatch are not reductions; they are represented by the
/// scheduler, not by this function.
pub fn fuse_kind(sync: SyncKind) -> Option<FuseKind> {
    match sync {
        SyncKind::FusedSum => Some(FuseKind::Sum),
        SyncKind::FusedMax => Some(FuseKind::Max),
        SyncKind::Gather => Some(FuseKind::Gather),
        SyncKind::TopK { k } => Some(FuseKind::TopK { k }),
        SyncKind::OnlineSoftmax => Some(FuseKind::OnlineSoftmax),
        // Expert combine is a token-wise sum. Dispatch is a transfer;
        // pipeline handoff is scheduled as a dependency.
        SyncKind::Combine => Some(FuseKind::Sum),
        SyncKind::PipelineHandoff | SyncKind::Dispatch | SyncKind::Replicate => None,
    }
}

pub fn spec_for(
    op: &ParallelOp,
    tensor: plait_core::TensorKey,
    epoch: u64,
) -> Option<plait_core::OpSpec> {
    let kind = fuse_kind(op.sync)?;
    Some(plait_core::OpSpec {
        id: op.op_id,
        kind,
        contributors: op.participants.iter().copied().collect(),
        tensor,
        epoch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::AxisSizes;

    fn config() -> DsaConfig {
        DsaConfig {
            seq: 6,
            heads: 4,
            head_dim: 4,
            index_dim: 8,
            top_k: 2,
            hidden: 16,
            experts: 2,
            ffn: 16,
        }
    }

    #[test]
    fn every_axis_has_an_ordering_mapping() {
        let topology = Topology::with_prefix(AxisSizes::new(2, 2, 2, 2, 3, 3), "plan").unwrap();
        let plan = dsa_plan(&topology, &config());
        for axis in [
            Axis::Data,
            Axis::Pipeline,
            Axis::Tensor,
            Axis::Expert,
            Axis::Context,
            Axis::Sequence,
        ] {
            assert!(
                plan.plan.by_axis(axis).next().is_some(),
                "axis {axis:?} has no operation"
            );
        }
        let dp = plan.plan.by_axis(Axis::Data).next().unwrap();
        assert_eq!(dp.sync, SyncKind::FusedSum);
        assert_eq!(dp.transfer, TransferPolicy::Registered);
        let pp = plan.plan.by_axis(Axis::Pipeline).next().unwrap();
        assert_eq!(pp.sync, SyncKind::PipelineHandoff);
        let ep = plan.plan.by_axis(Axis::Expert).next().unwrap();
        assert!(matches!(ep.sync, SyncKind::Dispatch | SyncKind::Combine));
        let cp = plan.plan.by_axis(Axis::Context).next().unwrap();
        assert!(matches!(
            cp.sync,
            SyncKind::TopK { .. } | SyncKind::OnlineSoftmax
        ));
        let sp = plan.plan.by_axis(Axis::Sequence).next().unwrap();
        assert_eq!(sp.sync, SyncKind::Gather);
    }

    #[test]
    fn participants_match_the_group_size() {
        let topology = Topology::with_prefix(AxisSizes::new(1, 1, 2, 1, 3, 3), "group").unwrap();
        let plan = dsa_plan(&topology, &config());
        let op = plan.plan.op("cp.candidates:0:0:0").unwrap();
        assert_eq!(op.participants.len(), 3);
        let op = plan.plan.op("tp.output:0").unwrap();
        assert_eq!(op.participants.len(), 2);
    }
}
