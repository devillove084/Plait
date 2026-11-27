//! Versioned identities and accelerator grants shared by transports and runtimes.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, SocketAddr};
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::link::{ProbeEndpoint, ProbeRegistration};
use crate::{Access, DeviceId, DeviceProfile, Key, Mode, NodeId, OpId};

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
pub struct PlacementInventory {
    pub group: GroupId,
    pub latest: Option<(GroupVersion, GroupPhase)>,
    pub active: Option<ActivePlacement>,
    pub hosts: Vec<PlacementHost>,
    pub workers: Vec<PlacementWorker>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivePlacement {
    pub version: GroupVersion,
    pub workers: Vec<(DeviceId, WorkerDemand)>,
    pub hosts: Vec<(NodeId, HostMemory)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementHost {
    pub host: NodeId,
    pub capacity: HostMemory,
    pub reserved: HostMemory,
    pub eligible: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementWorker {
    pub device: DeviceId,
    pub incarnation: u64,
    pub device_bytes: u64,
    pub grant: Option<WorkerGrant>,
    pub eligible: bool,
    pub profile: Option<DeviceProfile>,
}

/// A revision for one device incarnation within a caller's calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeRate {
    pub device: DeviceId,
    pub incarnation: u64,
    pub revision: u64,
    /// `None` invalidates an earlier measurement and preserves its revision history.
    pub operations_per_second: Option<NonZeroU64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupOperation {
    pub id: OpId,
    pub action: GroupAction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateEvidence {
    pub calibration: NodeId,
    pub compared_active: bool,
    pub rates: Vec<ComputeRate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupRecipe {
    pub version: GroupVersion,
    pub recipe: Option<NodeId>,
}

/// A compare-and-set for a group's active epoch. A missing target drains the
/// current epoch. Equal epochs check the group without changing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupTransition {
    pub group: GroupId,
    pub expected: Option<u64>,
    pub target: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupPreparation {
    pub group: GroupId,
    pub expected_epoch: u64,
    pub expected_active: Option<u64>,
    pub resources: ResourcePlan,
    pub recipe: Option<NodeId>,
    pub probes: Vec<ProbeRegistration>,
    pub rates: Option<RateEvidence>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedSetView {
    pub prepare: OpId,
    pub members: Vec<(GroupVersion, GroupPhase)>,
    pub activated: bool,
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
    AdvertiseWorker {
        profile: DeviceProfile,
        incarnation: u64,
        device_bytes: u64,
    },
    PreparePlanned {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        resources: ResourcePlan,
    },
    ObserveProbe(NodeId),
    PublishProbe {
        endpoint: ProbeEndpoint,
        expected_revision: u64,
    },
    PrepareMeasured {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        resources: ResourcePlan,
        probes: Vec<ProbeRegistration>,
    },
    SetStandby {
        active: GroupVersion,
        standby: Option<u64>,
    },
    ReportFailure {
        grant: WorkerGrant,
        fault: GroupFault,
    },
    PublishComputeRate {
        calibration: NodeId,
        device: DeviceId,
        incarnation: u64,
        expected_revision: u64,
        operations_per_second: Option<NonZeroU64>,
    },
    PrepareWithRates {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        resources: ResourcePlan,
        probes: Vec<ProbeRegistration>,
        calibration: NodeId,
        compared_active: bool,
        rates: Vec<ComputeRate>,
    },
    PublishGrantedComputeRate {
        grant: WorkerGrant,
        calibration: NodeId,
        expected_revision: u64,
        operations_per_second: NonZeroU64,
    },
    PrepareRecipe {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        resources: ResourcePlan,
        recipe: NodeId,
        probes: Vec<ProbeRegistration>,
        rates: Option<RateEvidence>,
    },
    SwitchActiveSet {
        transitions: Vec<GroupTransition>,
    },
    ObserveActiveSet {
        groups: Vec<GroupId>,
    },
    PrepareSet {
        preparations: Vec<GroupPreparation>,
        guards: Vec<(GroupId, Option<u64>)>,
    },
    AbortFailedSet {
        prepare: OpId,
        versions: Vec<GroupVersion>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupFault {
    Transport,
    Device,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupOutcome {
    HostConfigured,
    WorkerRegistered,
    Version {
        version: GroupVersion,
        phase: GroupPhase,
    },
    Probe(Option<ProbeRegistration>),
    ComputeRate(ComputeRate),
    ActiveSet(Vec<(GroupId, Option<u64>)>),
    PreparedSet(Vec<GroupVersion>),
    AbortedSet(Vec<GroupVersion>),
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
    StaleEpoch {
        current: u64,
    },
    CandidateExists(GroupVersion),
    InvalidPhase(GroupPhase),
    GrantMismatch,
    InvalidEndpoint,
    ManifestMismatch,
    SequenceExhausted,
    ProfileMismatch(DeviceId),
    ActiveEpochChanged {
        current: Option<u64>,
    },
    StaleProbe {
        host: NodeId,
        revision: u64,
    },
    StaleComputeRate {
        device: DeviceId,
        revision: u64,
    },
    GroupSetChanged {
        group: GroupId,
        current: Option<u64>,
    },
    UnknownPreparation(OpId),
    PreparationActivated,
    PreparationHealthy,
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
    Placement(GroupId),
    LinkProbes,
    GrantStatus(WorkerGrant),
    Standby(GroupId),
    PlacementWithRates {
        group: GroupId,
        calibration: NodeId,
    },
    ComputeRate {
        device: DeviceId,
        calibration: NodeId,
    },
    Recipe(GroupVersion),
    LatestPreparation(GroupId),
    PeerStagingBytes(WorkerGrant),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupReply {
    Version(Option<GroupView>),
    Receipt(Option<GroupReceipt>),
    Placement(PlacementInventory),
    LinkProbes(Vec<ProbeRegistration>),
    GrantStatus(GrantStatus),
    Standby(Option<GroupVersion>),
    PlacementWithRates {
        inventory: PlacementInventory,
        rates: Vec<ComputeRate>,
    },
    ComputeRate(Option<ComputeRate>),
    Recipe(Option<GroupRecipe>),
    Preparation(Option<PreparedSetView>),
    PeerStagingBytes(Option<u64>),
}

/// A replica's applied view of an exact communicator grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrantStatus {
    Unknown,
    Held,
    Ended,
}

impl GroupView {
    pub fn grant_status(&self, grant: WorkerGrant) -> GrantStatus {
        if self.version != grant.version {
            return GrantStatus::Unknown;
        }
        self.phase.grant_status(
            self.grants.contains(&grant),
            self.released.contains(&grant.device),
        )
    }
}

impl GroupPhase {
    pub fn accepts_peer_join(self) -> bool {
        matches!(self, Self::Preparing | Self::Ready | Self::Active)
    }

    pub fn grant_status(self, has_grant: bool, released: bool) -> GrantStatus {
        if has_grant && !released && !matches!(self, Self::Revoked | Self::Retired) {
            GrantStatus::Held
        } else {
            GrantStatus::Ended
        }
    }
}

impl GroupOperation {
    pub(crate) fn accesses(&self) -> Vec<Access> {
        let mut keys = BTreeSet::from([Key::ServingOperation(self.id)]);
        let mut reads = BTreeSet::new();
        match &self.action {
            GroupAction::SetHostMemory { host, .. } => {
                keys.insert(Key::HostMemory(*host));
            }
            GroupAction::RegisterWorker { device, .. }
            | GroupAction::PublishComputeRate { device, .. } => {
                keys.insert(Key::Device(*device));
            }
            GroupAction::AdvertiseWorker { profile, .. } => {
                keys.insert(Key::Device(profile.id));
            }
            GroupAction::PrepareSet {
                preparations,
                guards,
            } => {
                reads.extend(guards.iter().map(|(group, _)| Key::ServingGroup(*group)));
                for plan in preparations {
                    keys.insert(Key::ServingGroup(plan.group));
                    keys.extend(plan.resources.hosts.keys().copied().map(Key::HostMemory));
                    keys.extend(plan.resources.workers.keys().copied().map(Key::Device));
                    reads.extend(
                        plan.probes
                            .iter()
                            .map(|probe| Key::LinkProbe(probe.endpoint.host)),
                    );
                    if let Some(evidence) = &plan.rates {
                        reads.extend(evidence.rates.iter().map(|rate| Key::Device(rate.device)));
                    }
                }
            }
            GroupAction::AbortFailedSet { prepare, versions } => {
                reads.insert(Key::ServingOperation(*prepare));
                keys.extend(
                    versions
                        .iter()
                        .map(|version| Key::ServingGroup(version.group)),
                );
            }
            GroupAction::PublishGrantedComputeRate { grant, .. } => {
                reads.insert(Key::ServingGroup(grant.version.group));
                keys.insert(Key::Device(grant.device));
            }
            GroupAction::Prepare {
                group, resources, ..
            }
            | GroupAction::PreparePlanned {
                group, resources, ..
            }
            | GroupAction::PrepareMeasured {
                group, resources, ..
            }
            | GroupAction::PrepareWithRates {
                group, resources, ..
            }
            | GroupAction::PrepareRecipe {
                group, resources, ..
            } => {
                keys.insert(Key::ServingGroup(*group));
                keys.extend(resources.hosts.keys().copied().map(Key::HostMemory));
                keys.extend(resources.workers.keys().copied().map(Key::Device));
                if let GroupAction::PrepareMeasured { probes, .. }
                | GroupAction::PrepareWithRates { probes, .. }
                | GroupAction::PrepareRecipe { probes, .. } = &self.action
                {
                    reads.extend(
                        probes
                            .iter()
                            .map(|probe| Key::LinkProbe(probe.endpoint.host)),
                    );
                }
                match &self.action {
                    GroupAction::PrepareWithRates { rates, .. }
                    | GroupAction::PrepareRecipe {
                        rates: Some(RateEvidence { rates, .. }),
                        ..
                    } => reads.extend(rates.iter().map(|rate| Key::Device(rate.device))),
                    _ => {}
                }
            }
            GroupAction::PublishEndpoint(GroupMember { grant, .. })
            | GroupAction::Ready { grant, .. }
            | GroupAction::ReportFailure { grant, .. } => {
                keys.insert(Key::ServingGroup(grant.version.group));
            }
            GroupAction::Activate(version)
            | GroupAction::Drain(version)
            | GroupAction::Revoke(version)
            | GroupAction::SetStandby {
                active: version, ..
            } => {
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
                reads.insert(Key::ServingGroup(*group));
            }
            GroupAction::SwitchActiveSet { transitions } => {
                for change in transitions {
                    let key = Key::ServingGroup(change.group);
                    if change.expected == change.target {
                        reads.insert(key);
                    } else {
                        keys.insert(key);
                    }
                }
            }
            GroupAction::ObserveActiveSet { groups } => {
                reads.extend(groups.iter().copied().map(Key::ServingGroup));
            }
            GroupAction::ObserveProbe(host) => {
                reads.insert(Key::LinkProbe(*host));
            }
            GroupAction::PublishProbe { endpoint, .. } => {
                keys.insert(Key::LinkProbe(endpoint.host));
            }
        }
        reads.retain(|key| !keys.contains(key));
        keys.into_iter()
            .map(|key| Access {
                key,
                mode: Mode::Write,
            })
            .chain(reads.into_iter().map(|key| Access {
                key,
                mode: Mode::Read,
            }))
            .collect()
    }
}
