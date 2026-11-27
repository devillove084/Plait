//! Requests to a replicated Plait control plane.

mod client;
mod discovery;

pub use client::{Client, QueryResult};
pub use discovery::discover_voters;
