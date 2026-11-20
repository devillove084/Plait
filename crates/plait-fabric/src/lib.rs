//! Fused operator execution on top of leaderless consensus.
//!
//! A contribution is a consensus command. Replication delivers every node's
//! partial to every replica, and the runtime combines the partials as soon as
//! the last declared contributor is executed. Communication and computation
//! are therefore the same ordered state-machine transition.

pub mod runtime;

pub use runtime::{Event, Runtime};
