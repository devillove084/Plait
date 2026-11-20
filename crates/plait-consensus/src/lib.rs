//! Leaderless EPaxos replication.
//!
//! Every voter can propose, every voter can recover, and no operation depends
//! on a distinguished rank. The only privileged thing in the protocol is a
//! ballot, which any node may raise.

pub mod execution;
pub mod replica;
pub mod types;

pub use execution::{EdgeKinds, ExecutionGraph};
pub use replica::{Replica, StateMachine};
pub use types::{
    Attrs, Ballot, InstanceId, Message, Options, Outbound, Phase, ProposeError, Record, Snapshot,
    Value, ValueState,
};
