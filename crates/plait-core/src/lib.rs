//! Backend-neutral vocabulary shared by every Plait crate.
//!
//! Nothing in this crate knows about sockets, threads, or a particular
//! accelerator. Nodes are named by stable identities, commands declare the
//! tensor/operator keys they touch, and membership is a replicated
//! configuration rather than an array of ranks.

pub mod command;
pub mod config;
pub mod device;
pub mod group;
pub mod identity;
pub mod serving;
pub mod tensor;

pub use command::{
    Access, Command, ComputeTask, FuseKind, Key, KeyClass, MembershipChange, Mode, OpId, OpSpec,
};
pub use config::{ClusterConfig, ConfigError, QuorumPolicy};
pub use device::{AcceleratorKind, ComputeRequirement, DeviceId, DeviceProfile};
pub use identity::NodeId;
pub use tensor::{
    DType, Placement, Shape, ShardId, TensorId, TensorKey, TensorSpec, TensorVersion, TensorView,
};
