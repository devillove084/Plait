//! Network measurement endpoints and session-scoped samples.

use std::net::{IpAddr, SocketAddr};

use serde::{Deserialize, Serialize};

use crate::NodeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeEndpoint {
    pub host: NodeId,
    pub session: NodeId,
    pub address: SocketAddr,
}

impl ProbeEndpoint {
    pub fn valid(&self) -> bool {
        self.address.port() > 0
            && !self.address.ip().is_unspecified()
            && !self.address.ip().is_multicast()
            && !matches!(self.address.ip(), IpAddr::V4(ip) if ip.is_broadcast())
            && !matches!(self.address, SocketAddr::V6(address) if address.flowinfo() != 0 || address.scope_id() != 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeRegistration {
    pub revision: u64,
    pub endpoint: ProbeEndpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkSample {
    pub target: NodeId,
    pub target_session: NodeId,
    pub sampled_at_ms: u64,
    pub bytes_per_second: u64,
    pub rtt_ns: u64,
}
