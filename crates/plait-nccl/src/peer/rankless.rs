//! Bind the peer sockets and temporary slots to a committed device assignment.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub use plait_core::group::GroupMember as RanklessMember;
use plait_core::group::{GroupManifest, WorkerGrant};

use super::{GrantBinding, Listener, PeerGroup};

pub struct RanklessListener {
    listener: Listener,
    grant: WorkerGrant,
}

impl RanklessListener {
    pub fn bind(address: SocketAddr, ticket: [u8; 16], grant: WorkerGrant) -> io::Result<Self> {
        let listener = Listener::bind_in_epoch(address, ticket, grant.version)?;
        Ok(Self { listener, grant })
    }

    pub fn member(&self) -> RanklessMember {
        RanklessMember {
            grant: self.grant,
            endpoint: self.listener.endpoint(),
        }
    }

    pub(crate) fn connect(
        self,
        members: Vec<RanklessMember>,
        timeout: Duration,
    ) -> io::Result<Arc<PeerGroup>> {
        let manifest = GroupManifest::new(self.grant.version, members)
            .ok_or_else(|| invalid("invalid peer group manifest"))?;
        let slot = manifest
            .members
            .binary_search_by_key(&self.grant.device, |member| member.grant.device)
            .ok()
            .filter(|&slot| manifest.members[slot] == self.member())
            .ok_or_else(|| {
                invalid("the local device grant and endpoint must match the peer manifest")
            })?;
        let binding = GrantBinding {
            devices: manifest
                .members
                .iter()
                .map(|member| member.grant.device)
                .collect(),
            digest: manifest.digest(),
        };
        let endpoints = manifest
            .members
            .into_iter()
            .map(|member| member.endpoint)
            .collect();
        self.listener
            .connect_bound(slot, endpoints, Some(binding), timeout)
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
