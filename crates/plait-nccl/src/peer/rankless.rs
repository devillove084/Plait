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

#[cfg(feature = "control")]
use super::{Bootstrap, requests};
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

    /// Set the per-worker replicated AllReduce payload cap before connecting.
    pub(crate) fn set_replicated_egress_bytes(&mut self, bytes: u64) {
        self.listener.set_replicated_egress_bytes(bytes);
    }

    #[cfg(feature = "control")]
    pub(crate) fn with_staging_ceiling(mut self, bytes: u64) -> Self {
        self.listener.staging_ceiling = usize::try_from(bytes).unwrap_or(usize::MAX);
        self
    }

    pub(crate) fn connect(
        self,
        members: Vec<RanklessMember>,
        layout: Option<&[DeviceId]>,
        timeout: Duration,
        bootstrap_timeout: Duration,
    ) -> io::Result<Arc<PeerGroup>> {
        let (slot, endpoints, binding) = self.resolve(members, layout)?;
        self.listener
            .connect_bound(slot, endpoints, Some(binding), timeout, bootstrap_timeout)
    }

    #[cfg(feature = "control")]
    pub(crate) fn connect_axes(
        self,
        members: Vec<RanklessMember>,
        layout: &[DeviceId],
        tensor_parallel: usize,
        timeout: Duration,
        bootstrap_timeout: Duration,
    ) -> io::Result<(Arc<PeerGroup>, Arc<PeerGroup>)> {
        let (slot, endpoints, binding) = self.resolve(members, Some(layout))?;
        let world = endpoints.len();
        if tensor_parallel <= 1 || tensor_parallel >= world || world % tensor_parallel != 0 {
            return Err(invalid("mixed layout requires nontrivial TP and PP sizes"));
        }
        let tensor_slot = slot % tensor_parallel;
        let pipeline_slot = slot / tensor_parallel;
        let row_start = pipeline_slot * tensor_parallel;
        let tensor_members = (row_start..row_start + tensor_parallel).collect::<Vec<_>>();
        let pipeline_members = (tensor_slot..world)
            .step_by(tensor_parallel)
            .collect::<Vec<_>>();
        let mut expected = vec![false; world];
        for &member in tensor_members.iter().chain(&pipeline_members) {
            expected[member] = true;
        }

        // Each pair shares at most one axis. Bootstrap those links once against
        // the full version and reject peers that derived a different topology.
        let mut digest = Sha256::new();
        digest.update(b"plait-peer-model-parallel-v1");
        digest.update(binding.handshake_digest);
        digest.update((tensor_parallel as u64).to_le_bytes());
        let manifest = self.listener.wire_digest(digest.finalize().into());
        let requests = requests::Dispatcher::from_env(self.listener.staging_ceiling)?;
        let mut links = self.listener.connect_links(
            &endpoints,
            &expected,
            Bootstrap::new(slot, manifest, timeout, bootstrap_timeout)?,
        )?;
        let mut axis = |members: Vec<usize>, slot, name: &[u8], requests, egress| {
            let mut digest = Sha256::new();
            digest.update(manifest);
            digest.update(name);
            digest.update((members[0] as u64).to_le_bytes());
            let profile = digest.finalize().into();
            let devices = members
                .iter()
                .map(|&index| binding.devices[index])
                .collect();
            let links = members
                .into_iter()
                .map(|index| links[index].take())
                .collect();
            PeerGroup::new(
                slot,
                links,
                profile,
                timeout,
                egress,
                requests,
                Some(GrantBinding {
                    devices,
                    digest: binding.digest,
                    handshake_digest: manifest,
                    #[cfg(feature = "python")]
                    local_grant: self.grant,
                }),
            )
        };
        let tensor = axis(
            tensor_members,
            tensor_slot,
            b"tp",
            requests.share_staging(),
            self.listener.replicated_egress_bytes,
        );
        let pipeline = axis(pipeline_members, pipeline_slot, b"pp", requests, None);
        Ok((tensor, pipeline))
    }

    fn resolve(
        &self,
        members: Vec<RanklessMember>,
        layout: Option<&[DeviceId]>,
    ) -> io::Result<(usize, Vec<super::Endpoint>, GrantBinding)> {
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
            #[cfg(feature = "python")]
            local_grant: self.grant,
        };
        let endpoints = members.into_iter().map(|member| member.endpoint).collect();
        Ok((slot, endpoints, binding))
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

#[cfg(all(test, feature = "control"))]
mod tests {
    use super::*;
    use crate::peer::Collective;
    use crate::{DataType, ReduceOp};
    use plait_core::NodeId;
    use plait_core::group::{GroupId, GroupVersion};

    #[test]
    fn mixed_axes_have_distinct_peers_and_share_the_grant_staging_budget() {
        let layout = [2, 1, 4, 3].map(|host| DeviceId::new(NodeId::from_bytes([host; 16]), 0));
        let version = GroupVersion {
            group: GroupId(NodeId::from_bytes([9; 16])),
            epoch: 1,
        };
        let listeners = layout
            .iter()
            .enumerate()
            .map(|(slot, &device)| {
                RanklessListener::bind(
                    "127.0.0.1:0".parse().unwrap(),
                    [slot as u8; 16],
                    WorkerGrant {
                        version,
                        device,
                        incarnation: 1,
                        fence: 1,
                    },
                )
                .unwrap()
                .with_staging_ceiling(4)
            })
            .collect::<Vec<_>>();
        let members = listeners
            .iter()
            .map(RanklessListener::member)
            .collect::<Vec<_>>();
        let digest = GroupManifest::new(version, members.clone())
            .unwrap()
            .digest();
        std::thread::scope(|scope| {
            for (slot, listener) in listeners.into_iter().enumerate() {
                let members = members.clone();
                scope.spawn(move || {
                    let timeout = Duration::from_secs(5);
                    let (tp, pp) = listener
                        .connect_axes(members, &layout, 2, timeout, timeout)
                        .unwrap();
                    assert_eq!((tp.slot(), pp.slot()), (slot % 2, slot / 2));
                    assert_eq!((tp.world(), pp.world()), (2, 2));
                    assert_eq!(tp.manifest_digest(), Some(digest));
                    assert_eq!(pp.manifest_digest(), Some(digest));
                    assert_eq!(tp.device_slot(layout[slot ^ 1]), Some((slot ^ 1) % 2));
                    assert_eq!(pp.device_slot(layout[slot ^ 2]), Some((slot ^ 2) / 2));
                    assert_eq!(tp.device_slot(layout[slot ^ 2]), None);
                    assert_eq!(pp.device_slot(layout[slot ^ 1]), None);
                    let reservation = tp.requests.admit(3).unwrap();
                    assert!(pp.requests.admit(2).is_none());
                    drop(reservation);
                    assert!(pp.requests.admit(4).is_some());

                    let reduction = tp
                        .execute(
                            Collective::AllReduce(DataType::Int32, ReduceOp::Sum),
                            &(slot as i32).to_ne_bytes(),
                        )
                        .unwrap();
                    assert_eq!(
                        reduction,
                        (if slot < 2 { 1_i32 } else { 5_i32 }).to_ne_bytes()
                    );
                    if pp.slot() == 0 {
                        pp.send(1, DataType::Uint8, vec![slot as u8]).unwrap();
                        assert_eq!(
                            pp.recv(1, DataType::Uint8, 1).unwrap(),
                            vec![slot as u8 + 2]
                        );
                    } else {
                        assert_eq!(
                            pp.recv(0, DataType::Uint8, 1).unwrap(),
                            vec![slot as u8 - 2]
                        );
                        pp.send(0, DataType::Uint8, vec![slot as u8]).unwrap();
                    }
                    let reduction = pp
                        .execute(
                            Collective::AllReduce(DataType::Int32, ReduceOp::Sum),
                            &(slot as i32).to_ne_bytes(),
                        )
                        .unwrap();
                    assert_eq!(reduction, (2 + 2 * (slot % 2) as i32).to_ne_bytes());
                });
            }
        });
    }
}
