//! Bind the peer sockets and temporary slots to a committed device assignment.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use plait_core::group::WorkerGrant;
use sha2::{Digest, Sha256};

use super::{Endpoint, GrantBinding, Listener, PeerGroup};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RanklessMember {
    pub grant: WorkerGrant,
    pub endpoint: Endpoint,
}

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
        mut members: Vec<RanklessMember>,
        timeout: Duration,
    ) -> io::Result<Arc<PeerGroup>> {
        members.sort_unstable_by_key(|member| member.grant.device);
        if members
            .windows(2)
            .any(|pair| pair[0].grant.device == pair[1].grant.device)
            || members
                .iter()
                .any(|member| member.grant.version != self.grant.version)
        {
            return Err(invalid(
                "duplicate device or wrong group version in the peer manifest",
            ));
        }
        let slot = members
            .binary_search_by_key(&self.grant.device, |member| member.grant.device)
            .ok()
            .filter(|&slot| members[slot] == self.member())
            .ok_or_else(|| {
                invalid("the local device grant and endpoint must match the peer manifest")
            })?;
        let mut digest = Sha256::new();
        let mut devices = Vec::with_capacity(members.len());
        let mut endpoints = Vec::with_capacity(members.len());
        for member in members {
            let grant = member.grant;
            digest.update(grant.device.node.as_bytes());
            digest.update(grant.device.ordinal.to_le_bytes());
            digest.update(grant.incarnation.to_le_bytes());
            digest.update(grant.fence.to_le_bytes());
            devices.push(grant.device);
            endpoints.push(member.endpoint);
        }
        self.listener.connect_bound(
            slot,
            endpoints,
            Some(GrantBinding {
                devices,
                digest: digest.finalize().into(),
            }),
            timeout,
        )
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
