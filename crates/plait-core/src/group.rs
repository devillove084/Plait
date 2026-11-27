//! Versioned identities and accelerator grants shared by transports and runtimes.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, SocketAddr};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Access, DeviceId, Key, Mode, NodeId, OpId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GroupId(pub NodeId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GroupVersion {
    pub group: GroupId,
    pub epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerGrant {
    pub version: GroupVersion,
    pub device: DeviceId,
    pub incarnation: u64,
    pub fence: u64,
}

/// Pinned bytes consume both the pinning quota and the host memory quota.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostMemory {
    pub bytes: u64,
    pub pinned_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerDemand {
    pub incarnation: u64,
    pub device_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupPhase {
    Preparing,
    Ready,
    Active,
    Draining,
    Revoked,
    Retired,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePlan {
    pub workers: BTreeMap<DeviceId, WorkerDemand>,
    pub hosts: BTreeMap<NodeId, HostMemory>,
}

impl ResourcePlan {
    pub fn valid(&self) -> bool {
        let hosts: BTreeSet<_> = self.workers.keys().map(|device| device.node).collect();
        !self.workers.is_empty()
            && hosts.iter().eq(self.hosts.keys())
            && self.workers.values().all(|demand| demand.device_bytes > 0)
            && self
                .hosts
                .values()
                .all(|demand| demand.pinned_bytes <= demand.bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerEndpoint {
    pub address: SocketAddr,
    pub ticket: [u8; 16],
}

impl PeerEndpoint {
    pub fn valid(&self) -> bool {
        self.address.port() != 0
            && !self.address.ip().is_unspecified()
            && !self.address.ip().is_multicast()
            && !matches!(self.address.ip(), IpAddr::V4(ip) if ip.is_broadcast())
            && !matches!(self.address, SocketAddr::V6(address) if address.flowinfo() != 0 || address.scope_id() != 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupMember {
    pub grant: WorkerGrant,
    pub endpoint: PeerEndpoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupManifest {
    pub version: GroupVersion,
    pub members: Vec<GroupMember>,
}

impl GroupManifest {
    pub fn new(version: GroupVersion, mut members: Vec<GroupMember>) -> Option<Self> {
        members.sort_unstable_by_key(|member| member.grant.device);
        let mut addresses = BTreeSet::new();
        let mut tickets = BTreeSet::new();
        if version.epoch == 0
            || members.is_empty()
            || members
                .windows(2)
                .any(|pair| pair[0].grant.device == pair[1].grant.device)
            || members.iter().any(|member| {
                member.grant.version != version
                    || !member.endpoint.valid()
                    || !addresses.insert(member.endpoint.address)
                    || !tickets.insert(member.endpoint.ticket)
            })
        {
            return None;
        }
        Some(Self { version, members })
    }

    /// The native peer handshake digest, independent of the input member order.
    /// Use the canonical manifest returned by `new`.
    pub fn digest(&self) -> [u8; 32] {
        let mut grants = Sha256::new();
        for member in &self.members {
            let grant = member.grant;
            grants.update(grant.device.node.as_bytes());
            grants.update(grant.device.ordinal.to_le_bytes());
            grants.update(grant.incarnation.to_le_bytes());
            grants.update(grant.fence.to_le_bytes());
        }
        let endpoints = self.members.iter().map(|member| member.endpoint);
        let mut digest = peer_digest(Some(self.version), endpoints);
        digest.update(b"plait-peer-device-grants-v1");
        digest.update(grants.finalize());
        digest.finalize().into()
    }
}

#[doc(hidden)]
pub fn peer_digest(
    version: Option<GroupVersion>,
    endpoints: impl IntoIterator<Item = PeerEndpoint>,
) -> Sha256 {
    let mut digest = Sha256::new();
    match version {
        Some(version) => {
            digest.update([1]);
            digest.update(version.group.0.as_bytes());
            digest.update(version.epoch.to_le_bytes());
        }
        None => digest.update([0]),
    }
    for member in endpoints {
        match member.address {
            SocketAddr::V4(address) => {
                digest.update([4]);
                digest.update(address.ip().octets());
            }
            SocketAddr::V6(address) => {
                digest.update([6]);
                digest.update(address.ip().octets());
                digest.update(address.scope_id().to_le_bytes());
            }
        }
        digest.update(member.address.port().to_le_bytes());
        digest.update(member.ticket);
    }
    digest
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupView {
    pub version: GroupVersion,
    pub phase: GroupPhase,
    pub grants: Vec<WorkerGrant>,
    pub manifest: Option<GroupManifest>,
    pub ready: BTreeSet<DeviceId>,
    pub released: BTreeSet<DeviceId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupOperation {
    pub id: OpId,
    pub action: GroupAction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupAction {
    SetHostMemory {
        host: NodeId,
        capacity: HostMemory,
    },
    RegisterWorker {
        device: DeviceId,
        incarnation: u64,
        device_bytes: u64,
    },
    Prepare {
        group: GroupId,
        expected_epoch: u64,
        resources: ResourcePlan,
    },
    PublishEndpoint(GroupMember),
    Ready {
        grant: WorkerGrant,
        manifest: [u8; 32],
    },
    Activate(GroupVersion),
    Drain(GroupVersion),
    Revoke(GroupVersion),
    Release(WorkerGrant),
    Observe {
        group: GroupId,
        epoch: Option<u64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupOutcome {
    HostConfigured,
    WorkerRegistered,
    Version {
        version: GroupVersion,
        phase: GroupPhase,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupError {
    OperationIdReused,
    InvalidPlan,
    InvalidCapacity,
    UnknownHost(NodeId),
    HostCapacityExceeded(NodeId),
    HostOffline(NodeId),
    HostDraining(NodeId),
    UnknownWorker(DeviceId),
    WorkerBusy(DeviceId),
    WorkerDraining(DeviceId),
    StaleIncarnation(DeviceId),
    DeviceCapacityExceeded(DeviceId),
    UnknownVersion(GroupVersion),
    NoActiveGroup(GroupId),
    StaleEpoch { current: u64 },
    CandidateExists(GroupVersion),
    InvalidPhase(GroupPhase),
    GrantMismatch,
    InvalidEndpoint,
    ManifestMismatch,
    SequenceExhausted,
}

pub type GroupResult = Result<GroupOutcome, GroupError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupReceipt {
    pub operation: GroupOperation,
    pub result: GroupResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupLookup {
    Version(GroupVersion),
    Receipt(OpId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupReply {
    Version(Option<GroupView>),
    Receipt(Option<GroupReceipt>),
}

impl GroupOperation {
    pub(crate) fn accesses(&self) -> Vec<Access> {
        let mut keys = BTreeSet::from([Key::ServingOperation(self.id)]);
        let mut mode = Mode::Write;
        match &self.action {
            GroupAction::SetHostMemory { host, .. } => {
                keys.insert(Key::HostMemory(*host));
            }
            GroupAction::RegisterWorker { device, .. } => {
                keys.insert(Key::Device(*device));
            }
            GroupAction::Prepare {
                group, resources, ..
            } => {
                keys.insert(Key::ServingGroup(*group));
                keys.extend(resources.hosts.keys().copied().map(Key::HostMemory));
                keys.extend(resources.workers.keys().copied().map(Key::Device));
            }
            GroupAction::PublishEndpoint(GroupMember { grant, .. })
            | GroupAction::Ready { grant, .. } => {
                keys.insert(Key::ServingGroup(grant.version.group));
            }
            GroupAction::Activate(version)
            | GroupAction::Drain(version)
            | GroupAction::Revoke(version) => {
                keys.insert(Key::ServingGroup(version.group));
            }
            GroupAction::Release(grant) => {
                keys.extend([
                    Key::ServingGroup(grant.version.group),
                    Key::Device(grant.device),
                    Key::HostMemory(grant.device.node),
                ]);
            }
            GroupAction::Observe { group, .. } => {
                keys.insert(Key::ServingGroup(*group));
                mode = Mode::Read;
            }
        }
        keys.into_iter()
            .map(|key| Access {
                key,
                mode: if key == Key::ServingOperation(self.id) {
                    Mode::Write
                } else {
                    mode
                },
            })
            .collect()
    }
}
