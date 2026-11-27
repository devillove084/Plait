//! Bind the peer sockets and temporary slots to a committed device assignment.

use std::collections::BTreeMap;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use plait_core::DeviceId;
pub use plait_core::group::GroupMember as RanklessMember;
use plait_core::group::{GroupManifest, WorkerGrant};
use sha2::{Digest, Sha256};

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
        layout: Option<&[DeviceId]>,
        timeout: Duration,
        bootstrap_timeout: Duration,
    ) -> io::Result<Arc<PeerGroup>> {
        let manifest = GroupManifest::new(self.grant.version, members)
            .ok_or_else(|| invalid("invalid peer group manifest"))?;
        let digest = manifest.digest();
        let (members, has_custom_order) = match layout {
            Some(devices)
                if !devices
                    .iter()
                    .copied()
                    .eq(manifest.members.iter().map(|member| member.grant.device)) =>
            {
                (order_members(manifest.members, devices)?, true)
            }
            _ => (manifest.members, false),
        };
        let local = self.member();
        let slot = members
            .iter()
            .position(|member| *member == local)
            .ok_or_else(|| {
                invalid("the local device grant and endpoint must match the peer manifest")
            })?;
        let devices: Vec<_> = members.iter().map(|member| member.grant.device).collect();
        let handshake_digest = if has_custom_order {
            let mut binding = Sha256::new();
            binding.update(b"plait-peer-slot-layout-v1");
            binding.update(digest);
            for device in &devices {
                binding.update(device.node.as_bytes());
                binding.update(device.ordinal.to_le_bytes());
            }
            binding.finalize().into()
        } else {
            digest
        };
        let binding = GrantBinding {
            devices,
            digest,
            handshake_digest,
        };
        let endpoints = members.into_iter().map(|member| member.endpoint).collect();
        self.listener
            .connect_bound(slot, endpoints, Some(binding), timeout, bootstrap_timeout)
    }
}

fn order_members(
    members: Vec<RanklessMember>,
    layout: &[DeviceId],
) -> io::Result<Vec<RanklessMember>> {
    if layout.len() != members.len() {
        return Err(invalid(
            "slot layout must include each manifest device once",
        ));
    }
    let mut members: BTreeMap<_, _> = members
        .into_iter()
        .map(|member| (member.grant.device, member))
        .collect();
    layout
        .iter()
        .map(|device| {
            members
                .remove(device)
                .ok_or_else(|| invalid("slot layout must include each manifest device once"))
        })
        .collect()
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
