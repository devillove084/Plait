//! Serving plans and acknowledgements shared by planners and Plait nodes.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, SocketAddr, SocketAddrV4};

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

/// A pin consumes both the pinning quota and the host's total memory quota.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostMemory {
    pub bytes: u64,
    pub pinned_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingRole {
    Aggregated,
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostedActivation {
    #[default]
    OnReady,
    OnRevocation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedAddress {
    pub host: NodeId,
    pub address: IpAddr,
}

/// The portable inputs needed to launch one committed vLLM process group.
/// Each host supplies its own installed implementation of `profile`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedRecipe {
    pub profile: String,
    pub api_host: NodeId,
    pub hosts: Vec<HostedAddress>,
    pub master_port: u16,
    pub http_port: u16,
    pub backend_port: u16,
    pub api_model: String,
    pub max_model_len: u32,
    pub gpu_memory_bps: u16,
    pub prefix_caching: bool,
}

impl HostedRecipe {
    pub fn launch_digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"plait-hosted-recipe-v1");
        for name in [&self.profile, &self.api_model] {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
        }
        hash.update(self.api_host.as_bytes());
        hash.update((self.hosts.len() as u64).to_le_bytes());
        for host in &self.hosts {
            hash.update(host.host.as_bytes());
            match host.address {
                IpAddr::V4(address) => {
                    hash.update([4]);
                    hash.update(address.octets());
                }
                IpAddr::V6(address) => {
                    hash.update([6]);
                    hash.update(address.octets());
                }
            }
        }
        for port in [self.master_port, self.http_port, self.backend_port] {
            hash.update(port.to_le_bytes());
        }
        hash.update(self.max_model_len.to_le_bytes());
        hash.update(self.gpu_memory_bps.to_le_bytes());
        hash.update([self.prefix_caching.into()]);
        hash.finalize().into()
    }

    pub fn validate(&self, plan: &GroupPlan) -> Result<(), ServingError> {
        self.validate_topology(plan, 2)?;
        if self.launch_digest() != plan.launch_digest {
            return Err(ServingError::InvalidPlan);
        }
        Ok(())
    }

    pub fn validate_aggregated_route(&self, plan: &GroupPlan) -> Result<(), ServingError> {
        self.validate_topology(plan, 1)?;
        if plan.role != ServingRole::Aggregated || self.launch_digest() != plan.launch_digest {
            return Err(ServingError::InvalidPlan);
        }
        Ok(())
    }

    fn validate_topology(&self, plan: &GroupPlan, min_hosts: usize) -> Result<(), ServingError> {
        let hosts: BTreeSet<_> = self.hosts.iter().map(|host| host.host).collect();
        let ports = BTreeSet::from([self.master_port, self.http_port, self.backend_port]);
        if self.profile.is_empty()
            || self.profile.len() > 128
            || self
                .profile
                .bytes()
                .any(|c| !(c.is_ascii_alphanumeric() || b"._-".contains(&c)))
            || self.api_model.is_empty()
            || self.api_model.len() > 256
            || self.api_model.contains('\0')
            || hosts.len() < min_hosts
            || hosts.len() != self.hosts.len()
            || !hosts.contains(&self.api_host)
            || !hosts.iter().eq(plan.hosts.keys())
            || self
                .hosts
                .iter()
                .any(|host| host.address.is_unspecified() || host.address.is_multicast())
            || ports.len() != 3
            || ports.contains(&0)
            || self.max_model_len == 0
            || !(1..=10_000).contains(&self.gpu_memory_bps)
        {
            return Err(ServingError::InvalidPlan);
        }
        Ok(())
    }
}

/// Inventory identifier for an engine profile with a specific layer partition.
pub fn hosted_pipeline_profile(profile: &str, layers: &[u32]) -> Option<String> {
    if layers.len() < 2
        || layers.contains(&0)
        || profile.is_empty()
        || !profile
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return None;
    }
    let mut digest = Sha256::new();
    digest.update(b"plait-hosted-layer-partition-v1");
    for &count in layers {
        digest.update(count.to_le_bytes());
    }
    let name = format!("{profile}.pp.{:x}", digest.finalize());
    (name.len() <= 128).then_some(name)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedAssignment {
    pub version: GroupVersion,
    pub model: String,
    pub tensor_parallel: u32,
    pub phase: GroupPhase,
    pub holder: Option<NodeId>,
    pub recipe: HostedRecipe,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdKv {
    pub store: SocketAddrV4,
    pub namespace: String,
    pub block_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdLaunch {
    pub role: ServingRole,
    pub kv: HostedPdKv,
}

impl HostedPdLaunch {
    pub fn launch_digest(&self, recipe: &HostedRecipe) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"plait-hosted-pd-v1");
        hash.update(recipe.launch_digest());
        hash.update([match self.role {
            ServingRole::Prefill => 1,
            ServingRole::Decode => 2,
            ServingRole::Aggregated => 0,
        }]);
        hash.update(self.kv.store.ip().octets());
        hash.update(self.kv.store.port().to_le_bytes());
        hash.update(self.kv.block_size.to_le_bytes());
        hash.update(self.kv.namespace.as_bytes());
        hash.finalize().into()
    }

    pub fn validate(&self, recipe: &HostedRecipe, plan: &GroupPlan) -> Result<(), ServingError> {
        recipe.validate_topology(plan, 1)?;
        if plan.role != self.role
            || plan.role == ServingRole::Aggregated
            || self.kv.store.port() == 0
            || self.kv.store.ip().is_unspecified()
            || self.kv.store.ip().is_multicast()
            || self.kv.namespace.is_empty()
            || self.kv.namespace.len() > 256
            || self.kv.namespace.contains('\0')
            || self.kv.block_size == 0
            || self.launch_digest(recipe) != plan.launch_digest
        {
            return Err(ServingError::InvalidPlan);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedManagedAssignment {
    pub assignment: HostedAssignment,
    pub pd: Option<HostedPdLaunch>,
}

impl HostedManagedAssignment {
    pub fn launch_digest(&self) -> [u8; 32] {
        let recipe = &self.assignment.recipe;
        self.pd
            .as_ref()
            .map_or_else(|| recipe.launch_digest(), |pd| pd.launch_digest(recipe))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdPlacement {
    pub group: GroupId,
    pub expected_epoch: u64,
    pub expected_active: Option<u64>,
    pub plan: GroupPlan,
    pub recipe: HostedRecipe,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdIntent {
    pub staged: OpId,
    pub model: String,
    pub expected_revision: u64,
    pub previous: Vec<PdRoute>,
    pub paths: Vec<PdRoute>,
    pub prepared: BTreeSet<GroupVersion>,
    pub hosts: BTreeSet<NodeId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdStage {
    pub placements: Vec<HostedPdPlacement>,
    pub paths: Vec<PdRoute>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdTransition {
    pub staged: OpId,
    pub model: String,
    pub expected_revision: u64,
    pub kv: HostedPdKv,
    pub bridge: Vec<PdRoute>,
    pub release: BTreeSet<GroupVersion>,
    pub target: HostedPdStage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostedPdTransitionPhase {
    PreparingBridge,
    WaitingForRelease,
    PreparingTarget,
    Published,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdTransitionStatus {
    pub staged: OpId,
    pub revision: u64,
    pub phase: HostedPdTransitionPhase,
    pub error: Option<ServingError>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdInstance {
    pub version: GroupVersion,
    pub model: String,
    pub role: ServingRole,
    pub kv: HostedPdKv,
    pub ranks: BTreeMap<NodeId, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdPlanning {
    pub revision: u64,
    pub managed: bool,
    pub pending: Option<OpId>,
    pub active: Vec<HostedPdInstance>,
    pub paths: Vec<PdRoute>,
    pub resources: ServingResources,
    pub inventory: HostedPlanning,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdRelease {
    pub version: GroupVersion,
    pub hosts: BTreeMap<NodeId, HostMemory>,
    pub ports: HostedPortUse,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdReconfiguration {
    pub planning: HostedPdPlanning,
    pub releases: Vec<HostedPdRelease>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPdRecovery {
    pub planning: HostedPdPlanning,
    pub checkpoint: String,
    pub kv: HostedPdKv,
    pub paths: Vec<PdRoute>,
    pub failed: BTreeMap<GroupVersion, ServingRole>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedProbeEndpoint {
    pub host: NodeId,
    pub session: NodeId,
    pub address: SocketAddr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedLinkSample {
    pub target: NodeId,
    pub target_session: NodeId,
    pub sampled_at_ms: u64,
    pub bytes_per_second: u64,
    pub rtt_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedInventory {
    pub host: NodeId,
    pub address: IpAddr,
    pub profiles: BTreeSet<String>,
    /// Local accelerator ordinal to registered incarnation.
    pub devices: BTreeMap<u32, u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPortUse {
    pub host: NodeId,
    pub ports: [u16; 3],
}

impl From<&HostedRecipe> for HostedPortUse {
    fn from(recipe: &HostedRecipe) -> Self {
        Self {
            host: recipe.api_host,
            ports: [recipe.master_port, recipe.http_port, recipe.backend_port],
        }
    }
}

impl HostedPortUse {
    pub fn conflicts(self, other: Self) -> bool {
        self.host == other.host && self.ports.iter().any(|port| other.ports.contains(port))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedPlanning {
    pub hosts: Vec<HostedInventory>,
    pub occupied: Vec<HostedPortUse>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerDemand {
    pub incarnation: u64,
    pub device_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupPlan {
    pub role: ServingRole,
    pub model: String,
    pub tensor_parallel: u32,
    pub pipeline_parallel: u32,
    /// Digest of the engine configuration the workers are asked to install.
    pub launch_digest: [u8; 32],
    pub workers: BTreeMap<DeviceId, WorkerDemand>,
    pub hosts: BTreeMap<NodeId, HostMemory>,
}

impl GroupPlan {
    pub fn validate(&self) -> Result<(), ServingError> {
        let worker_count = u64::from(self.tensor_parallel) * u64::from(self.pipeline_parallel);
        let worker_hosts: BTreeSet<_> = self.workers.keys().map(|device| device.node).collect();
        if self.model.is_empty()
            || worker_count == 0
            || worker_count != self.workers.len() as u64
            || !worker_hosts.iter().eq(self.hosts.keys())
            || self.workers.values().any(|demand| demand.device_bytes == 0)
            || self
                .hosts
                .values()
                .any(|demand| demand.pinned_bytes > demand.bytes)
        {
            return Err(ServingError::InvalidPlan);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerGrant {
    pub version: GroupVersion,
    pub device: DeviceId,
    pub incarnation: u64,
    pub fence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentOwner {
    pub session: NodeId,
    pub supervisor_lock: [u64; 2],
    pub device_lock: [u64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionHostLiveness {
    pub pulse: OpId,
    pub online: bool,
    pub supervisor_lock: [u64; 2],
}

pub const EXECUTION_PULSE_MS: u64 = 2_000;
pub const EXECUTION_HOST_TIMEOUT_MS: u64 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerDispatch {
    pub grant: WorkerGrant,
    pub addr: SocketAddr,
}

pub const W4_MAX_ADMISSION_MS: u64 = 90_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct W4DrainStatus {
    pub pending: u64,
    pub next_expiry_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct W4ShardLayout {
    pub k_range: [u32; 2],
    pub group_size: u32,
    pub output_columns: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct W4GroupLayout {
    pub k: u32,
    pub group_size: u32,
    pub output_columns: u32,
    pub shards: BTreeMap<DeviceId, [u32; 2]>,
}

impl W4GroupLayout {
    pub fn covers(&self, plan: &GroupPlan) -> bool {
        if self.group_size == 0
            || self.k == 0
            || self.output_columns == 0
            || !self.shards.keys().eq(plan.workers.keys())
        {
            return false;
        }
        let mut ranges: Vec<_> = self.shards.values().copied().collect();
        ranges.sort_unstable();
        let mut next = 0;
        for [start, end] in ranges {
            if start != next || start % self.group_size != 0 || end <= start {
                return false;
            }
            next = end;
        }
        next == self.k
    }

    pub fn shard(&self, device: DeviceId) -> Option<W4ShardLayout> {
        Some(W4ShardLayout {
            k_range: *self.shards.get(&device)?,
            group_size: self.group_size,
            output_columns: self.output_columns,
        })
    }
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
pub struct GroupView {
    pub version: GroupVersion,
    pub phase: GroupPhase,
    pub plan: GroupPlan,
    pub grants: BTreeMap<DeviceId, WorkerGrant>,
    pub ready: BTreeSet<DeviceId>,
    pub stopped: BTreeSet<DeviceId>,
    pub launch_owner: Option<NodeId>,
    pub endpoint: Option<ServingEndpoint>,
}

impl GroupView {
    pub fn holds_grant(&self, grant: WorkerGrant) -> bool {
        grant.version == self.version
            && self.grants.get(&grant.device) == Some(&grant)
            && !self.stopped.contains(&grant.device)
    }

    pub fn allows_grant(&self, grant: WorkerGrant) -> bool {
        matches!(self.phase, GroupPhase::Active | GroupPhase::Draining) && self.holds_grant(grant)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingEndpoint {
    pub api_model: String,
    pub http: SocketAddr,
    pub prefill_kv: Option<SocketAddrV4>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RouteSetId(pub NodeId);

impl RouteSetId {
    pub fn for_model(model: &str) -> Self {
        Self(NodeId::derive(&format!("plait/serving/routes/{model}")))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PdRoute {
    pub prefill: GroupVersion,
    pub decode: GroupVersion,
}

mod decoder;
pub use decoder::{
    DECODER_REPLAY_MAX_TOKENS, DecoderAdmissionRoutes, DecoderCheckpoint, DecoderClaim,
    DecoderDirectory, DecoderDrainStatus, DecoderEndpoint, DecoderFailover, DecoderKvSource,
    DecoderOffer, DecoderPreparation, DecoderReplayPage, DecoderReplayState, DecoderRoute,
    DecoderRouteWeight, DecoderRouteWork, DecoderStandby, DecoderStep, DecoderSwitch,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPdRoute {
    pub route: PdRoute,
    pub prefill_http: SocketAddr,
    pub prefill_kv: SocketAddrV4,
    pub decode_http: SocketAddr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDirectory {
    pub model: String,
    pub revision: u64,
    pub paths: Vec<ResolvedPdRoute>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HostedRoute {
    Aggregated(GroupVersion),
    Split(PdRoute),
}

impl HostedRoute {
    pub fn groups(self) -> impl Iterator<Item = GroupVersion> {
        let (first, second) = match self {
            Self::Aggregated(group) => (group, None),
            Self::Split(path) => (path.prefill, Some(path.decode)),
        };
        std::iter::once(first).chain(second)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedServingPlan {
    pub kv: Option<HostedPdKv>,
    pub placements: Vec<HostedPdPlacement>,
    pub routes: Vec<HostedRoute>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedServingIntent {
    pub staged: OpId,
    pub model: String,
    pub expected_revision: u64,
    pub previous: Vec<HostedRoute>,
    pub routes: Vec<HostedRoute>,
    pub prepared: BTreeSet<GroupVersion>,
    pub hosts: BTreeSet<NodeId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedRouteDirectory {
    pub split: RouteDirectory,
    pub aggregated: Vec<(GroupVersion, SocketAddr)>,
}

/// Committed resources visible on one replica at the time of the query.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingResources {
    pub hosts: Vec<HostResources>,
    pub workers: Vec<WorkerResources>,
    pub groups: Vec<GroupResources>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostResources {
    pub host: NodeId,
    pub capacity: HostMemory,
    pub reserved: HostMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerResources {
    pub device: DeviceId,
    pub incarnation: u64,
    pub device_bytes: u64,
    pub grant: Option<WorkerGrant>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupResources {
    pub latest: GroupVersion,
    pub phase: GroupPhase,
    pub active: Option<ActiveGroup>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveGroup {
    pub version: GroupVersion,
    pub role: ServingRole,
    pub model: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingOperation {
    pub id: OpId,
    pub action: ServingAction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingAction {
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
        plan: GroupPlan,
    },
    WorkerReady {
        grant: WorkerGrant,
        launch_digest: [u8; 32],
    },
    Activate(GroupVersion),
    Drain(GroupVersion),
    Revoke(GroupVersion),
    /// The node has closed this grant's process and accelerator/transport access.
    WorkerStopped(WorkerGrant),
    Retire {
        version: GroupVersion,
        /// Carries the write set in the command. The runtime validates it against the plan.
        workers: BTreeSet<DeviceId>,
    },
    ClaimLaunch {
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
        launch_digest: [u8; 32],
    },
    LaunchReady {
        version: GroupVersion,
        holder: NodeId,
    },
    ObserveLaunch {
        version: GroupVersion,
        holder: NodeId,
    },
    LaunchStopped {
        version: GroupVersion,
        holder: NodeId,
    },
    LaunchReadyAt {
        version: GroupVersion,
        holder: NodeId,
        endpoint: ServingEndpoint,
    },
    PublishRoutes {
        model: String,
        expected_revision: u64,
        paths: Vec<PdRoute>,
        /// Optionally make a ready candidate active in the same transaction.
        activate: Option<GroupVersion>,
    },
    ObserveRoutes {
        model: String,
    },
    ObserveGroup {
        group: GroupId,
        /// None observes the version admitting new work. An explicit epoch
        /// observes a particular version, including draining and revoked ones.
        epoch: Option<u64>,
    },
    WorkerReadyAt {
        grant: WorkerGrant,
        launch_digest: [u8; 32],
        addr: SocketAddr,
    },
    /// Prepare a native W4 partition and activate it in the last ready transaction.
    PrepareW4Switch {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: GroupPlan,
        layout: W4GroupLayout,
    },
    WorkerReadyW4 {
        grant: WorkerGrant,
        launch_digest: [u8; 32],
        addr: SocketAddr,
        shard: W4ShardLayout,
    },
    PrepareManagedW4 {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: GroupPlan,
        layout: W4GroupLayout,
    },
    /// The serving operation ID is also the ID carried by every admitted shard.
    AdmitW4 {
        group: GroupId,
        now_ms: u64,
        expires_at_ms: u64,
    },
    CheckW4 {
        grant: WorkerGrant,
        work: Option<OpId>,
    },
    FinishW4 {
        version: GroupVersion,
        work: OpId,
    },
    ExpireW4 {
        version: GroupVersion,
        through_ms: u64,
    },
    DecoderReady {
        endpoint: DecoderEndpoint,
        launch_digest: [u8; 32],
    },
    PublishDecoderRoutes {
        model: String,
        expected_revision: u64,
        paths: Vec<PdRoute>,
        activate: Option<GroupVersion>,
    },
    AdmitDecoder {
        model: String,
        revision: u64,
        path: PdRoute,
        now_ms: u64,
    },
    AdvanceDecoder {
        model: String,
        work: OpId,
        path: PdRoute,
        grant: WorkerGrant,
        now_ms: u64,
        step: DecoderStep,
    },
    CancelDecoder {
        model: String,
        work: OpId,
        path: PdRoute,
    },
    DecoderReadyManaged {
        endpoint: DecoderEndpoint,
        launch_digest: [u8; 32],
    },
    ExpireDecoder {
        model: String,
        path: PdRoute,
        through_ms: u64,
    },
    ClaimResident {
        grant: WorkerGrant,
        owner: ResidentOwner,
        launch_digest: [u8; 32],
    },
    LoseResident {
        grant: WorkerGrant,
        owner: ResidentOwner,
    },
    PublishDecoderStandbys {
        model: String,
        expected_revision: u64,
        paths: Vec<PdRoute>,
        activate: Option<GroupVersion>,
        standbys: Vec<DecoderStandby>,
    },
    FailoverDecoder {
        model: String,
        intent: DecoderFailover,
    },
    PulseExecutionHost {
        host: NodeId,
        supervisor_lock: [u64; 2],
    },
    ExpireExecutionHost {
        host: NodeId,
        pulse: OpId,
        groups: BTreeSet<GroupVersion>,
    },
    AdmitDecoderWithRecovery {
        model: String,
        revision: u64,
        path: PdRoute,
        now_ms: u64,
    },
    RecoverDecoderWork {
        model: String,
        work: OpId,
        revision: u64,
        failed: PdRoute,
        replacement: PdRoute,
        now_ms: u64,
    },
    AdmitDecoderWithRetry {
        model: String,
        revision: u64,
        path: PdRoute,
        now_ms: u64,
    },
    RestartDecoderWork {
        model: String,
        work: OpId,
        revision: u64,
        failed: PdRoute,
        replacement: PdRoute,
        now_ms: u64,
    },
    AdmitDecoderReplay {
        model: String,
        revision: u64,
        path: PdRoute,
        now_ms: u64,
        max_tokens: u32,
    },
    AppendDecoderOutput {
        model: String,
        work: OpId,
        path: PdRoute,
        grant: WorkerGrant,
        now_ms: u64,
        owner: OpId,
        offset: u32,
        tokens: Vec<u32>,
        checkpoint: Option<DecoderCheckpoint>,
    },
    AdmitResumableDecoder {
        model: String,
        revision: u64,
        path: PdRoute,
        now_ms: u64,
        max_tokens: u32,
        input_sha256: [u8; 32],
    },
    TakeoverDecoder {
        model: String,
        work: OpId,
        expected_entry: OpId,
        revision: u64,
        previous: PdRoute,
        replacement: PdRoute,
        now_ms: u64,
        offset: u32,
        input_sha256: [u8; 32],
    },
    RebindDecoderEntry {
        model: String,
        work: OpId,
        entry: OpId,
        revision: u64,
        failed: PdRoute,
        replacement: PdRoute,
        now_ms: u64,
        restart_prefill: bool,
    },
    CancelDecoderEntry {
        model: String,
        work: OpId,
        entry: OpId,
        path: PdRoute,
    },
    TakeoverDecoderWithRetry {
        model: String,
        work: OpId,
        expected_entry: OpId,
        revision: u64,
        previous: PdRoute,
        replacement: PdRoute,
        now_ms: u64,
        offset: u32,
        input_sha256: [u8; 32],
        retry: DecoderEntryRetry,
    },
    SubmitDecoder {
        model: String,
        revision: u64,
        path: Option<PdRoute>,
        previous: Option<PdRoute>,
        now_ms: u64,
        request: DecoderSubmission,
    },
    StageDecoderSwitch {
        model: String,
        expected_revision: u64,
        paths: Vec<PdRoute>,
        standbys: Vec<DecoderStandby>,
    },
    CommitDecoderSwitch {
        model: String,
        intent: DecoderSwitch,
    },
    CancelDecoderSwitch {
        model: String,
        staged: OpId,
    },
    PrepareDecoderSwitch {
        model: String,
        expected_revision: u64,
        prepare: Vec<DecoderPreparation>,
        paths: Vec<PdRoute>,
        standbys: Vec<DecoderStandby>,
    },
    PrepareDecoderSwitchFromPool {
        model: String,
        expected_revision: u64,
        request_sha256: [u8; 32],
        prepare: Vec<DecoderPreparation>,
        paths: Vec<PdRoute>,
        standbys: Vec<DecoderStandby>,
    },
    ConfigureDecoderWeights {
        model: String,
        expected_revision: u64,
        weights: Vec<DecoderRouteWeight>,
    },
    AdmitDecoderRouted {
        model: String,
        routes: DecoderAdmissionRoutes,
        now_ms: u64,
        replay: Option<DecoderReplayInput>,
    },
    SubmitDecoderRouted {
        model: String,
        routes: DecoderAdmissionRoutes,
        previous: Option<PdRoute>,
        now_ms: u64,
        request: DecoderSubmission,
    },
    PrepareWeightedDecoderSwitch {
        model: String,
        expected_revision: u64,
        request_sha256: [u8; 32],
        from_pool: bool,
        prepare: Vec<DecoderPreparation>,
        paths: Vec<PdRoute>,
        standbys: Vec<DecoderStandby>,
        weights: Vec<DecoderRouteWeight>,
    },
    /// Authorize one fully reduced output and close its tracked admission.
    CompleteW4 {
        version: GroupVersion,
        work: OpId,
        grants: Vec<WorkerGrant>,
    },
    /// Release one stopped device of a revoked native group. The final stop
    /// also retires the version and releases its shared host reservations.
    ReclaimResident {
        grant: WorkerGrant,
        owner: Option<ResidentOwner>,
        workers: BTreeSet<DeviceId>,
    },
    /// Reserve the launch on one host of a multi-host engine.
    ClaimHostLaunch {
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
        launch_digest: [u8; 32],
    },
    /// Confirm this host's workers. The HTTP host also publishes the endpoint.
    HostLaunchReady {
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
        endpoint: Option<ServingEndpoint>,
    },
    ObserveHostLaunch {
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
    },
    /// Confirm local device and transport release. A draining engine releases
    /// its HTTP host first, after it finishes the requests it already admitted.
    HostLaunchStopped {
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
    },
    /// Arm an aggregated HTTP group for an ordered switch after host readiness.
    PrepareHostedSwitch {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: GroupPlan,
        activation: HostedActivation,
        request_sha256: Option<[u8; 32]>,
    },
    PrepareHostedDeployment {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: GroupPlan,
        activation: HostedActivation,
        request_sha256: Option<[u8; 32]>,
        recipe: HostedRecipe,
    },
    RegisterHostedProbe(HostedProbeEndpoint),
    AdvertiseHostedInventory {
        inventory: HostedInventory,
        supervisor_lock: [u64; 2],
    },
    PrepareHostedPoolDeployment {
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: GroupPlan,
        activation: HostedActivation,
        request_sha256: [u8; 32],
        recipe: HostedRecipe,
    },
    PrepareHostedPd {
        model: String,
        expected_revision: u64,
        kv: HostedPdKv,
        placements: Vec<HostedPdPlacement>,
        paths: Vec<PdRoute>,
    },
    ResolveHostedPd {
        intent: HostedPdIntent,
        cancel: bool,
    },
    PreparePlannedHostedPd {
        model: String,
        expected_revision: u64,
        kv: HostedPdKv,
        placements: Vec<HostedPdPlacement>,
        paths: Vec<PdRoute>,
        request_sha256: [u8; 32],
    },
    PrepareHostedPdTransition {
        model: String,
        expected_revision: u64,
        kv: HostedPdKv,
        bridge: HostedPdStage,
        release: BTreeSet<GroupVersion>,
        target: HostedPdStage,
    },
    ResolveHostedPdTransition {
        transition: HostedPdTransition,
        intent: HostedPdIntent,
    },
    AdvanceHostedPdTransition(HostedPdTransition),
    AdvertiseHostedInventoryWithRelease {
        inventory: HostedInventory,
        supervisor_lock: [u64; 2],
        release_profiles: BTreeSet<String>,
    },
    PreparePlannedHostedPdTransition {
        model: String,
        expected_revision: u64,
        kv: HostedPdKv,
        bridge: HostedPdStage,
        release: Vec<HostedPdRelease>,
        target: HostedPdStage,
        request_sha256: Option<[u8; 32]>,
    },
    PrepareHostedServing {
        model: String,
        expected_revision: u64,
        plan: HostedServingPlan,
    },
    ResolveHostedServing {
        intent: HostedServingIntent,
        cancel: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderReplayInput {
    pub max_tokens: u32,
    pub input_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderSubmission {
    pub id: NodeId,
    pub sequence: u64,
    pub offset: u32,
    pub max_tokens: u32,
    pub input_sha256: [u8; 32],
}

impl DecoderSubmission {
    pub const fn work(id: NodeId) -> OpId {
        OpId::new(id, 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderEntryRetry {
    /// Chosen by the caller before sending any takeover in this series.
    pub series: NodeId,
    /// Strictly increasing within a series anchored at the same entry.
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingOutcome {
    HostConfigured,
    WorkerRegistered,
    Group {
        version: GroupVersion,
        phase: GroupPhase,
    },
    RoutesPublished {
        revision: u64,
    },
    RoutesObserved {
        revision: u64,
    },
    W4Admitted {
        version: GroupVersion,
        tracked: bool,
    },
    DecoderAdmitted {
        revision: u64,
        route: DecoderRoute,
    },
    DecoderAdvanced,
    DecoderOffered(DecoderOffer),
    ExecutionHostOnline,
    ExecutionHostOffline,
    DecoderRecoveryAdmitted {
        revision: u64,
        route: DecoderRoute,
    },
    DecoderTakenOver {
        revision: u64,
        route: DecoderRoute,
        committed: u32,
        recoverable: bool,
    },
    DecoderSubmissionComplete {
        revision: u64,
        path: PdRoute,
        count: u32,
    },
    DecoderSwitchStaged {
        revision: u64,
    },
    DecoderSwitchCancelled,
    HostedProbeRegistered,
    HostedPdStaged {
        revision: u64,
    },
    HostedPdCancelled,
    HostedServingStaged {
        revision: u64,
    },
    HostedServingCancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingError {
    OperationIdReused,
    InvalidPlan,
    InvalidCapacity,
    UnknownHost(NodeId),
    HostCapacityExceeded(NodeId),
    UnknownWorker(DeviceId),
    WorkerBusy(DeviceId),
    StaleIncarnation(DeviceId),
    DeviceCapacityExceeded(DeviceId),
    UnknownVersion(GroupVersion),
    StaleEpoch { current: u64 },
    CandidateExists(GroupVersion),
    InvalidPhase(GroupPhase),
    GrantMismatch,
    LaunchDigestMismatch,
    WorkersStillRunning,
    SequenceExhausted,
    LaunchOwnerMismatch,
    InvalidEndpoint,
    InvalidRoute(PdRoute),
    InvalidDirectory,
    StaleDirectory { current: u64 },
    NoActiveGroup(GroupId),
    ActiveEpochChanged { current: Option<u64> },
    W4NotAdmitted,
    DecoderNotAdmitted,
    InvalidDecoderStage,
    ExecutionHostOffline(NodeId),
    StaleExecutionHost,
    InvalidDecoderOutput,
    DecoderReplayCapacity,
    StaleDecoderEntry,
    DecoderSubmissionRouteChanged(PdRoute),
    DecoderSwitchInProgress { staged: OpId },
    HostedInventoryChanged(NodeId),
    HostedPortsInUse(NodeId),
    HostedPdInProgress { staged: OpId },
}

pub type ServingResult = Result<ServingOutcome, ServingError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingReceipt {
    pub operation: ServingOperation,
    pub result: ServingResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingLookup {
    Group(GroupVersion),
    Operation(OpId),
    Resources,
    Routes(RouteSetId),
    Dispatch(GroupVersion),
    W4Drain(GroupVersion),
    DecoderRoutes(RouteSetId),
    DecoderDrain(GroupVersion),
    ResidentOwner(WorkerGrant),
    DecoderFailover(RouteSetId),
    ExecutionHostLiveness(NodeId),
    DecoderClaims(GroupVersion),
    DecoderCheckpoints {
        model: RouteSetId,
        work: OpId,
        transfer: NodeId,
    },
    DecoderReplay {
        model: RouteSetId,
        work: OpId,
        offset: u32,
    },
    DecoderSwitch(RouteSetId),
    DecoderRouteWork(RouteSetId),
    DecoderAdmissionRoutes(RouteSetId),
    HostedAssignments(NodeId),
    HostedProbeEndpoints,
    HostedPlanning,
    HostedManagedAssignments(NodeId),
    HostedPdPending(RouteSetId),
    HostedPdPlanning(RouteSetId),
    HostedPdTransition(RouteSetId),
    HostedPdReconfiguration(RouteSetId),
    HostedPdRecovery(RouteSetId),
    HostedRoutes(RouteSetId),
    HostedServingPending(RouteSetId),
}

/// A replica's locally applied state. A missing result may indicate replication lag.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingView {
    Group(Option<GroupView>),
    Operation(Option<ServingReceipt>),
    Resources(ServingResources),
    Routes(Option<RouteDirectory>),
    Dispatch(Vec<WorkerDispatch>),
    W4Drain(Option<W4DrainStatus>),
    DecoderRoutes(Option<DecoderDirectory>),
    DecoderDrain(Option<DecoderDrainStatus>),
    ResidentOwner(Option<ResidentOwner>),
    DecoderFailover(Option<DecoderFailover>),
    ExecutionHostLiveness(Option<ExecutionHostLiveness>),
    DecoderClaims(Vec<DecoderClaim>),
    DecoderCheckpoints(Vec<DecoderCheckpoint>),
    DecoderReplay(Option<DecoderReplayPage>),
    DecoderSwitch(Option<DecoderSwitch>),
    DecoderRouteWork(Option<DecoderRouteWork>),
    DecoderAdmissionRoutes(DecoderAdmissionRoutes),
    HostedAssignments(Vec<HostedAssignment>),
    HostedProbeEndpoints(Vec<HostedProbeEndpoint>),
    HostedPlanning(HostedPlanning),
    HostedManagedAssignments(Vec<HostedManagedAssignment>),
    HostedPdPending(Option<HostedPdIntent>),
    HostedPdPlanning(HostedPdPlanning),
    HostedPdTransition(Option<HostedPdTransitionStatus>),
    HostedPdReconfiguration(HostedPdReconfiguration),
    HostedPdRecovery(Option<HostedPdRecovery>),
    HostedRoutes(Option<HostedRouteDirectory>),
    HostedServingPending(Option<HostedServingIntent>),
}

impl ServingOperation {
    pub(crate) fn accesses(&self) -> Vec<Access> {
        let mut keys = BTreeSet::from([Key::ServingOperation(self.id)]);
        match &self.action {
            ServingAction::SetHostMemory { host, .. }
            | ServingAction::PulseExecutionHost { host, .. } => {
                keys.insert(Key::HostMemory(*host));
            }
            ServingAction::RegisterHostedProbe(endpoint) => {
                keys.insert(Key::HostMemory(endpoint.host));
            }
            ServingAction::AdvertiseHostedInventory { inventory, .. }
            | ServingAction::AdvertiseHostedInventoryWithRelease { inventory, .. } => {
                keys.insert(Key::HostMemory(inventory.host));
                keys.extend(
                    inventory
                        .devices
                        .keys()
                        .map(|&ordinal| Key::Device(DeviceId::new(inventory.host, ordinal))),
                );
            }
            ServingAction::ExpireExecutionHost { host, groups, .. } => {
                keys.insert(Key::HostMemory(*host));
                keys.extend(
                    groups
                        .iter()
                        .map(|version| Key::ServingGroup(version.group)),
                );
            }
            ServingAction::RegisterWorker { device, .. } => {
                keys.insert(Key::Device(*device));
            }
            ServingAction::PrepareHostedPd {
                model,
                placements,
                paths,
                ..
            }
            | ServingAction::PreparePlannedHostedPd {
                model,
                placements,
                paths,
                ..
            } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                hosted_pd_stage_keys(&mut keys, placements, paths);
            }
            ServingAction::ResolveHostedPd { intent, .. } => {
                hosted_pd_intent_keys(&mut keys, intent);
            }
            ServingAction::PrepareHostedServing { model, plan, .. } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                hosted_pd_stage_keys(&mut keys, &plan.placements, &[]);
                keys.extend(
                    plan.routes
                        .iter()
                        .flat_map(|route| route.groups())
                        .map(|version| Key::ServingGroup(version.group)),
                );
            }
            ServingAction::ResolveHostedServing { intent, .. } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(&intent.model)));
                keys.extend(
                    intent
                        .previous
                        .iter()
                        .chain(&intent.routes)
                        .flat_map(|route| route.groups())
                        .chain(intent.prepared.iter().copied())
                        .map(|version| Key::ServingGroup(version.group)),
                );
                keys.extend(intent.hosts.iter().copied().map(Key::HostMemory));
            }
            ServingAction::PrepareHostedPdTransition {
                model,
                bridge,
                release,
                target,
                ..
            } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                hosted_pd_stage_keys(&mut keys, &bridge.placements, &bridge.paths);
                hosted_pd_stage_keys(&mut keys, &target.placements, &target.paths);
                keys.extend(
                    release
                        .iter()
                        .map(|version| Key::ServingGroup(version.group)),
                );
            }
            ServingAction::PreparePlannedHostedPdTransition {
                model,
                bridge,
                release,
                target,
                ..
            } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                hosted_pd_stage_keys(&mut keys, &bridge.placements, &bridge.paths);
                hosted_pd_stage_keys(&mut keys, &target.placements, &target.paths);
                for retired in release {
                    keys.insert(Key::ServingGroup(retired.version.group));
                    keys.extend(retired.hosts.keys().copied().map(Key::HostMemory));
                }
            }
            ServingAction::AdvanceHostedPdTransition(transition)
            | ServingAction::ResolveHostedPdTransition { transition, .. } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(&transition.model)));
                hosted_pd_stage_keys(
                    &mut keys,
                    &transition.target.placements,
                    &transition.target.paths,
                );
                hosted_pd_stage_keys(&mut keys, &[], &transition.bridge);
                keys.extend(
                    transition
                        .release
                        .iter()
                        .map(|version| Key::ServingGroup(version.group)),
                );
                if let ServingAction::ResolveHostedPdTransition { intent, .. } = &self.action {
                    hosted_pd_intent_keys(&mut keys, intent);
                }
            }
            ServingAction::Prepare { group, plan, .. }
            | ServingAction::PrepareW4Switch { group, plan, .. }
            | ServingAction::PrepareManagedW4 { group, plan, .. }
            | ServingAction::PrepareHostedSwitch { group, plan, .. } => {
                keys.insert(Key::ServingGroup(*group));
                keys.extend(plan.workers.keys().copied().map(Key::Device));
                keys.extend(plan.hosts.keys().copied().map(Key::HostMemory));
            }
            ServingAction::PrepareHostedDeployment {
                group,
                plan,
                recipe,
                ..
            }
            | ServingAction::PrepareHostedPoolDeployment {
                group,
                plan,
                recipe,
                ..
            } => {
                keys.extend([Key::ServingGroup(*group), Key::HostMemory(recipe.api_host)]);
                keys.extend(plan.workers.keys().copied().map(Key::Device));
                keys.extend(plan.hosts.keys().copied().map(Key::HostMemory));
            }
            ServingAction::Retire { version, workers } => {
                keys.insert(Key::ServingGroup(version.group));
                keys.extend(workers.iter().copied().map(Key::Device));
                keys.extend(workers.iter().map(|device| Key::HostMemory(device.node)));
            }
            ServingAction::ReclaimResident { grant, workers, .. } => {
                keys.insert(Key::ServingGroup(grant.version.group));
                keys.extend(workers.iter().copied().map(Key::Device));
                keys.extend(workers.iter().map(|device| Key::HostMemory(device.node)));
            }
            ServingAction::WorkerReady { grant, .. }
            | ServingAction::WorkerReadyAt { grant, .. }
            | ServingAction::WorkerReadyW4 { grant, .. }
            | ServingAction::CheckW4 { grant, .. }
            | ServingAction::LoseResident { grant, .. }
            | ServingAction::WorkerStopped(grant) => {
                keys.insert(Key::ServingGroup(grant.version.group));
            }
            ServingAction::ClaimResident { grant, .. } => {
                keys.extend([
                    Key::ServingGroup(grant.version.group),
                    Key::HostMemory(grant.device.node),
                ]);
            }
            ServingAction::ClaimHostLaunch { version, host, .. } => {
                keys.extend([Key::ServingGroup(version.group), Key::HostMemory(*host)]);
            }
            ServingAction::Activate(version)
            | ServingAction::FinishW4 { version, .. }
            | ServingAction::CompleteW4 { version, .. }
            | ServingAction::ExpireW4 { version, .. }
            | ServingAction::Drain(version)
            | ServingAction::Revoke(version)
            | ServingAction::ClaimLaunch { version, .. }
            | ServingAction::LaunchReady { version, .. }
            | ServingAction::LaunchReadyAt { version, .. }
            | ServingAction::ObserveLaunch { version, .. }
            | ServingAction::LaunchStopped { version, .. }
            | ServingAction::HostLaunchReady { version, .. }
            | ServingAction::ObserveHostLaunch { version, .. }
            | ServingAction::HostLaunchStopped { version, .. } => {
                keys.insert(Key::ServingGroup(version.group));
            }
            ServingAction::DecoderReady { endpoint, .. }
            | ServingAction::DecoderReadyManaged { endpoint, .. } => {
                keys.insert(Key::ServingGroup(endpoint.grant.version.group));
            }
            ServingAction::PublishRoutes {
                model,
                paths,
                activate,
                ..
            }
            | ServingAction::PublishDecoderRoutes {
                model,
                paths,
                activate,
                ..
            } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                for path in paths {
                    keys.extend([
                        Key::ServingGroup(path.prefill.group),
                        Key::ServingGroup(path.decode.group),
                    ]);
                }
                keys.extend(activate.map(|version| Key::ServingGroup(version.group)));
            }
            ServingAction::PublishDecoderStandbys {
                model,
                paths,
                activate,
                standbys,
                ..
            } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                let paths = paths.iter().chain(
                    standbys
                        .iter()
                        .flat_map(|rule| std::iter::once(&rule.primary).chain(&rule.alternatives)),
                );
                for path in paths {
                    keys.extend([
                        Key::ServingGroup(path.prefill.group),
                        Key::ServingGroup(path.decode.group),
                    ]);
                }
                keys.extend(activate.map(|version| Key::ServingGroup(version.group)));
            }
            ServingAction::FailoverDecoder { model, intent } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                for path in
                    std::iter::once(&intent.standby.primary).chain(&intent.standby.alternatives)
                {
                    keys.extend([
                        Key::ServingGroup(path.prefill.group),
                        Key::ServingGroup(path.decode.group),
                    ]);
                }
            }
            ServingAction::StageDecoderSwitch {
                model,
                paths,
                standbys,
                ..
            }
            | ServingAction::CommitDecoderSwitch {
                model,
                intent: DecoderSwitch {
                    paths, standbys, ..
                },
            } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                let previous = match &self.action {
                    ServingAction::CommitDecoderSwitch { intent, .. } => intent.previous.as_slice(),
                    _ => &[],
                };
                for path in previous.iter().chain(paths).chain(
                    standbys
                        .iter()
                        .flat_map(|standby| standby.alternatives.iter()),
                ) {
                    keys.extend([
                        Key::ServingGroup(path.prefill.group),
                        Key::ServingGroup(path.decode.group),
                    ]);
                }
            }
            ServingAction::CancelDecoderSwitch { model, .. }
            | ServingAction::ConfigureDecoderWeights { model, .. } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
            }
            ServingAction::AdmitDecoderRouted { model, routes, .. }
            | ServingAction::SubmitDecoderRouted { model, routes, .. } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                let previous = match &self.action {
                    ServingAction::SubmitDecoderRouted {
                        previous, request, ..
                    } => {
                        keys.insert(Key::ServingOperation(DecoderSubmission::work(request.id)));
                        previous.as_ref()
                    }
                    _ => None,
                };
                for path in routes.paths.iter().chain(previous) {
                    keys.extend([
                        Key::ServingGroup(path.prefill.group),
                        Key::ServingGroup(path.decode.group),
                    ]);
                }
            }
            ServingAction::PrepareDecoderSwitch {
                model,
                prepare,
                paths,
                standbys,
                ..
            }
            | ServingAction::PrepareDecoderSwitchFromPool {
                model,
                prepare,
                paths,
                standbys,
                ..
            }
            | ServingAction::PrepareWeightedDecoderSwitch {
                model,
                prepare,
                paths,
                standbys,
                ..
            } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                for placement in prepare {
                    keys.insert(Key::ServingGroup(placement.group));
                    keys.extend(placement.plan.workers.keys().copied().map(Key::Device));
                    keys.extend(placement.plan.hosts.keys().copied().map(Key::HostMemory));
                }
                for path in paths
                    .iter()
                    .chain(standbys.iter().flat_map(|rule| &rule.alternatives))
                {
                    keys.extend([
                        Key::ServingGroup(path.prefill.group),
                        Key::ServingGroup(path.decode.group),
                    ]);
                }
            }
            ServingAction::AdmitDecoder { model, path, .. }
            | ServingAction::AdmitDecoderWithRecovery { model, path, .. }
            | ServingAction::AdmitDecoderWithRetry { model, path, .. }
            | ServingAction::AdmitDecoderReplay { model, path, .. }
            | ServingAction::AdmitResumableDecoder { model, path, .. }
            | ServingAction::AppendDecoderOutput { model, path, .. }
            | ServingAction::CancelDecoderEntry { model, path, .. }
            | ServingAction::AdvanceDecoder { model, path, .. }
            | ServingAction::CancelDecoder { model, path, .. }
            | ServingAction::ExpireDecoder { model, path, .. } => {
                keys.extend([
                    Key::ServingRoutes(RouteSetId::for_model(model)),
                    Key::ServingGroup(path.prefill.group),
                    Key::ServingGroup(path.decode.group),
                ]);
            }
            ServingAction::RecoverDecoderWork {
                model,
                failed,
                replacement,
                ..
            }
            | ServingAction::RestartDecoderWork {
                model,
                failed,
                replacement,
                ..
            }
            | ServingAction::RebindDecoderEntry {
                model,
                failed,
                replacement,
                ..
            }
            | ServingAction::TakeoverDecoder {
                model,
                previous: failed,
                replacement,
                ..
            }
            | ServingAction::TakeoverDecoderWithRetry {
                model,
                previous: failed,
                replacement,
                ..
            } => {
                keys.extend([
                    Key::ServingRoutes(RouteSetId::for_model(model)),
                    Key::ServingGroup(failed.prefill.group),
                    Key::ServingGroup(failed.decode.group),
                    Key::ServingGroup(replacement.prefill.group),
                    Key::ServingGroup(replacement.decode.group),
                ]);
            }
            ServingAction::SubmitDecoder {
                model,
                path,
                previous,
                request,
                ..
            } => {
                keys.insert(Key::ServingOperation(DecoderSubmission::work(request.id)));
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
                for path in path.iter().chain(previous) {
                    keys.extend([
                        Key::ServingGroup(path.prefill.group),
                        Key::ServingGroup(path.decode.group),
                    ]);
                }
            }
            ServingAction::ObserveRoutes { model } => {
                keys.insert(Key::ServingRoutes(RouteSetId::for_model(model)));
            }
            ServingAction::ObserveGroup { group, .. } | ServingAction::AdmitW4 { group, .. } => {
                keys.insert(Key::ServingGroup(*group));
            }
        }
        keys.into_iter()
            .map(|key| Access {
                key,
                mode: Mode::Write,
            })
            .collect()
    }
}

fn hosted_pd_stage_keys(
    keys: &mut BTreeSet<Key>,
    placements: &[HostedPdPlacement],
    paths: &[PdRoute],
) {
    for placement in placements {
        keys.insert(Key::ServingGroup(placement.group));
        keys.insert(Key::HostMemory(placement.recipe.api_host));
        keys.extend(placement.plan.workers.keys().copied().map(Key::Device));
        keys.extend(placement.plan.hosts.keys().copied().map(Key::HostMemory));
    }
    for path in paths {
        keys.extend([
            Key::ServingGroup(path.prefill.group),
            Key::ServingGroup(path.decode.group),
        ]);
    }
}

fn hosted_pd_intent_keys(keys: &mut BTreeSet<Key>, intent: &HostedPdIntent) {
    keys.insert(Key::ServingRoutes(RouteSetId::for_model(&intent.model)));
    hosted_pd_stage_keys(keys, &[], &intent.paths);
    hosted_pd_stage_keys(keys, &[], &intent.previous);
    keys.extend(
        intent
            .prepared
            .iter()
            .map(|version| Key::ServingGroup(version.group)),
    );
    keys.extend(intent.hosts.iter().copied().map(Key::HostMemory));
}
