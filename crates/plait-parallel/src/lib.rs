//! Parallelism planning and a runnable distributed DSA operator.
//!
//! TP/PP/EP/DP/CP/SP are placement coordinates. Each axis maps to a fused
//! EPaxos operation with an explicit conflict key and participant group; the
//! DSA reference operator executes the TP/CP/SP path over the real consensus
//! and registered-memory data plane.

pub mod cluster;
pub mod dsa;
pub mod plan;
pub mod topology;

pub use cluster::LocalCluster;
pub use dsa::{DsaConfig, DsaError, DsaOutput, DsaRunner, GlobalDsa};
pub use plan::{
    DsaPlan, ParallelOp, Plan, SyncKind, TransferPolicy, dsa_plan, fuse_kind, head_range,
    seq_range, spec_for, tp_for_head,
};
pub use topology::{Axis, AxisSizes, Coord, Topology, TopologyError};
