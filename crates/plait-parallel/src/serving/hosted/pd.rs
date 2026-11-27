use std::collections::{BTreeMap, BTreeSet};
use std::net::IpAddr;

use plait_core::NodeId;
use plait_core::serving::{
    GroupId, GroupPhase, GroupVersion, HostedAssignment, HostedLinkSample, HostedPdInstance,
    HostedPdKv, HostedPdPlacement, HostedPdPlanning, HostedPdRelease, HostedPdStage,
    HostedPlanning, HostedPortUse, HostedRecipe, HostedRoute, PdRoute, ServingAction,
    ServingResources, ServingRole,
};
use serde::{Deserialize, Serialize};

use super::super::{ServingCandidate, ServingForecast, SloProfile, validate_placement};
use super::pipeline::{PipelineProfile, lower_capacity};
use super::pool::{HostedPool, PoolTarget, generate_layouts};
use super::{HostedLinkPolicy, LinkEvidence, Traffic};

mod modes;
mod reconfigure;
pub use modes::{HostedModePolicy, plan_hosted_evacuation, plan_hosted_serving};
mod recovery;
use reconfigure::Reconfiguration;
pub use recovery::plan_hosted_pd_recovery;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedPdPool {
    pub kv: HostedPdKv,
    kv_bytes_per_request: u64,
    /// Aggregate throughput of the store, counting reads and writes.
    store_bytes_per_second: u64,
    store_valid_until_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_traffic: Option<CheckpointTraffic>,
    slots: Vec<RoleSlot>,
    #[serde(default)]
    allow_scale_down: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reconfigure: Option<Reconfiguration>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointTraffic {
    /// Additional page data, beyond the ordinary split prompt handoff.
    write_bytes_per_stream: u64,
    read_bytes_per_stream: u64,
    minimum_fraction_bps: u16,
    handoff_window_ms: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RoleSlot {
    role: ServingRole,
    pool: HostedPool,
}

#[derive(Serialize)]
pub struct HostedPdSelection {
    pub baseline_rps: f64,
    pub layouts_evaluated: usize,
    pub selected: Option<HostedPdDecision>,
}

pub struct HostedPdDecision {
    pub paths: Vec<PdRoute>,
    pub placements: Vec<HostedPdPlacement>,
    pub steady_rps: f64,
    pub gain_requests: f64,
    pub devices: u64,
    bridge: Option<Bridge>,
}

struct Bridge {
    stage: HostedPdStage,
    release: Vec<HostedPdRelease>,
    rate: f64,
    target_ready_after_ms: u64,
}

impl HostedPdDecision {
    pub fn into_action(
        self,
        model: String,
        expected_revision: u64,
        kv: HostedPdKv,
        request_sha256: Option<[u8; 32]>,
    ) -> ServingAction {
        match (self.bridge, request_sha256) {
            (Some(bridge), request_sha256) => ServingAction::PreparePlannedHostedPdTransition {
                model,
                expected_revision,
                kv,
                bridge: bridge.stage,
                release: bridge.release,
                target: HostedPdStage {
                    paths: self.paths,
                    placements: self.placements,
                },
                request_sha256,
            },
            (None, Some(request_sha256)) => ServingAction::PreparePlannedHostedPd {
                model,
                expected_revision,
                kv,
                paths: self.paths,
                placements: self.placements,
                request_sha256,
            },
            (None, None) => ServingAction::PrepareHostedPd {
                model,
                expected_revision,
                kv,
                paths: self.paths,
                placements: self.placements,
            },
        }
    }
}

impl Serialize for HostedPdDecision {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // GroupPlan uses compound keys. Keep the published CLI representation
        // compatible with the array representation used by prepare_hosted_pd.
        #[derive(Serialize)]
        struct Placement<'a> {
            group: GroupId,
            role: ServingRole,
            tensor_parallel: u32,
            pipeline_parallel: u32,
            workers: Vec<plait_core::DeviceId>,
            recipe: &'a plait_core::serving::HostedRecipe,
        }
        #[derive(Serialize)]
        struct Decision<'a> {
            paths: &'a [PdRoute],
            placements: Vec<Placement<'a>>,
            steady_rps: f64,
            gain_requests: f64,
            devices: u64,
            #[serde(skip_serializing_if = "Option::is_none")]
            bridge: Option<BridgeSummary<'a>>,
        }
        #[derive(Serialize)]
        struct BridgeSummary<'a> {
            paths: &'a [PdRoute],
            placements: Vec<Placement<'a>>,
            release: Vec<GroupVersion>,
            steady_rps: f64,
            target_ready_after_ms: u64,
        }
        fn placement(placement: &HostedPdPlacement) -> Placement<'_> {
            Placement {
                group: placement.group,
                role: placement.plan.role,
                tensor_parallel: placement.plan.tensor_parallel,
                pipeline_parallel: placement.plan.pipeline_parallel,
                workers: placement.plan.workers.keys().copied().collect(),
                recipe: &placement.recipe,
            }
        }
        Decision {
            paths: &self.paths,
            placements: self.placements.iter().map(placement).collect(),
            steady_rps: self.steady_rps,
            gain_requests: self.gain_requests,
            devices: self.devices,
            bridge: self.bridge.as_ref().map(|bridge| BridgeSummary {
                paths: &bridge.stage.paths,
                placements: bridge.stage.placements.iter().map(placement).collect(),
                release: bridge
                    .release
                    .iter()
                    .map(|release| release.version)
                    .collect(),
                steady_rps: bridge.rate,
                target_ready_after_ms: bridge.target_ready_after_ms,
            }),
        }
        .serialize(serializer)
    }
}

#[derive(Clone)]
struct Endpoint {
    version: GroupVersion,
    role: ServingRole,
    capacity: SloProfile,
    traffic: Traffic,
    checkpoint: Option<CheckpointIo>,
    devices: u64,
}

#[derive(Clone)]
struct CheckpointIo {
    write: Traffic,
    read: Traffic,
}

#[derive(Clone, Default)]
struct Layout {
    endpoints: Vec<Endpoint>,
    placements: Vec<HostedPdPlacement>,
    roles: Roles,
    ready_after_ms: u64,
    transition_loss_requests: f64,
    potential: f64,
}

const LAYOUTS_PER_SHAPE: usize = 8;
type Beam = BTreeMap<Roles, Vec<Layout>>;

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Roles {
    aggregated: usize,
    prefill: usize,
    decode: usize,
}

impl Roles {
    fn complete(self) -> bool {
        (self.prefill > 0 && self.decode > 0)
            || (self.aggregated > 0 && self.prefill == 0 && self.decode == 0)
    }

    fn has_split(self) -> bool {
        self.prefill > 0 && self.decode > 0
    }
}

pub fn plan_hosted_pd(
    model: &str,
    snapshot: &HostedPdPlanning,
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    policy: HostedLinkPolicy,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<HostedPdSelection, &'static str> {
    plan_hosted_pd_with_releases(model, snapshot, forecast, pool, policy, links, &[])
}

pub fn plan_hosted_pd_with_releases(
    model: &str,
    snapshot: &HostedPdPlanning,
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    policy: HostedLinkPolicy,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
    releases: &[HostedPdRelease],
) -> Result<HostedPdSelection, &'static str> {
    plan_hosted_pd_with_assignments(
        model,
        snapshot,
        forecast,
        pool,
        policy,
        links,
        HostedPdIncumbents {
            assignments: None,
            releases,
        },
    )
}

pub struct HostedPdIncumbents<'a> {
    /// Legacy TP-only callers can omit the committed launch assignments.
    pub assignments: Option<&'a BTreeMap<GroupVersion, HostedAssignment>>,
    pub releases: &'a [HostedPdRelease],
}

pub fn plan_hosted_pd_with_assignments(
    model: &str,
    snapshot: &HostedPdPlanning,
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    policy: HostedLinkPolicy,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
    incumbents: HostedPdIncumbents<'_>,
) -> Result<HostedPdSelection, &'static str> {
    forecast.validate()?;
    policy.validate()?;
    if !forecast.paths.is_empty() || !forecast.link_bytes_per_second.is_empty() {
        return Err("hosted P/D requires planner-generated link profiles");
    }
    pool.validate(model, forecast.now_ms)?;
    if !snapshot.managed {
        return Err("hosted P/D planning requires a managed route directory");
    }
    let evidence = LinkEvidence {
        policy,
        now_ms: forecast.now_ms,
        links,
    };
    let store_host = pool.store_host(&snapshot.inventory)?;
    let context = EndpointContext {
        evidence,
        store_host: Some(store_host),
        kv_bytes: pool.kv_bytes_per_request,
        checkpoint: pool.observed_checkpoint_traffic(forecast),
        aggregate_checkpoints: false,
    };
    let profiles: BTreeMap<_, _> = forecast
        .groups
        .iter()
        .map(|p| (p.version, p.capacity))
        .collect();
    if profiles.len() != forecast.groups.len() {
        return Err("duplicate hosted P/D group capacity");
    }
    if snapshot
        .active
        .iter()
        .any(|instance| instance.model != forecast.model || instance.kv != pool.kv)
    {
        return Err("active hosted P/D groups require the same model and KV contract");
    }
    let mut active = BTreeMap::new();
    for instance in &snapshot.active {
        let Some(&capacity) = profiles.get(&instance.version) else {
            continue;
        };
        let pipeline = if let Some(assignments) = incumbents.assignments {
            let assignment = assignments
                .get(&instance.version)
                .ok_or("active hosted P/D group lacks its committed launch assignment")?;
            let pp = active_pipeline_parallel(instance, assignment)
                .ok_or("active hosted P/D assignment does not match the planning snapshot")?;
            if pp > 1 {
                let calibration = pool
                    .slot(instance.version.group, instance.role)
                    .and_then(|slot| {
                        slot.pool.pipeline(
                            &assignment.recipe.profile,
                            assignment.tensor_parallel,
                            pp,
                        )
                    })
                    .ok_or("active hosted P/D pipeline lacks a matching stage calibration")?;
                Some((calibration, assignment.recipe.api_host))
            } else {
                None
            }
        } else {
            if pool.slots.iter().any(|slot| slot.pool.has_pipelines()) {
                return Err(
                    "active hosted P/D groups require launch assignments for pipeline planning",
                );
            }
            None
        };
        if let Some(endpoint) = context.incumbent(
            instance.version,
            instance.role,
            capacity,
            &instance.ranks,
            pipeline,
        ) {
            active.insert(instance.version.group, endpoint);
        }
    }
    let baseline_rps = baseline(forecast, pool, snapshot, &active)
        .ok_or("an active hosted P/D path lacks current capacity or directed links")?;
    let mut selection = HostedPdSelection {
        baseline_rps,
        layouts_evaluated: 0,
        selected: None,
    };
    if snapshot.pending.is_some() {
        return Ok(selection);
    }
    let search = Search {
        forecast,
        pool,
        context,
        aggregate: &[],
        withdrawing: None,
    };
    let layouts = search.layouts(snapshot, &active)?;

    let incumbent: BTreeSet<_> = snapshot.paths.iter().copied().collect();
    let current_devices: u64 = snapshot
        .active
        .iter()
        .flat_map(|entry| entry.ranks.values())
        .sum();
    for layout in &layouts {
        let paths = layout.paths();
        if layout.placements.is_empty()
            && paths.iter().copied().collect::<BTreeSet<_>>() == incumbent
        {
            continue;
        }
        selection.layouts_evaluated += 1;
        let Some((handoff_rate, handoff_ms)) = handoff_capacity(forecast, pool, &active, layout)
        else {
            continue;
        };
        if handoff_rate < baseline_rps {
            continue;
        }
        let gain = (layout.potential - baseline_rps)
            * (forecast.horizon_ms - layout.ready_after_ms) as f64
            / 1000.0
            - (layout.potential - handoff_rate) * handoff_ms as f64 / 1000.0
            - layout.transition_loss_requests;
        let devices = layout.devices();
        let improves = gain > 0.0 && gain >= forecast.minimum_gain_requests;
        let saves = pool.allow_scale_down
            && layout.placements.is_empty()
            && devices < current_devices
            && layout.potential >= baseline_rps
            && gain >= 0.0;
        if (improves || saves)
            && selection.selected.as_ref().is_none_or(|best| {
                gain > best.gain_requests || (gain == best.gain_requests && devices < best.devices)
            })
        {
            selection.selected = Some(HostedPdDecision {
                paths,
                placements: layout.placements.clone(),
                steady_rps: layout.potential,
                gain_requests: gain,
                devices,
                bridge: None,
            });
        }
    }
    if let Some(policy) = pool.reconfigure
        && !incumbents.releases.is_empty()
        && snapshot.revision.checked_add(2).is_some()
    {
        search.reconfigure(
            snapshot,
            &layouts,
            &active,
            incumbents.releases,
            policy,
            &mut selection,
        )?;
    }
    Ok(selection)
}

struct Search<'a> {
    forecast: &'a ServingForecast,
    pool: &'a HostedPdPool,
    context: EndpointContext<'a>,
    aggregate: &'a [HostedPool],
    withdrawing: Option<&'a BTreeSet<GroupVersion>>,
}

impl Search<'_> {
    fn layout_rate(&self, layout: &Layout, at: u64) -> Option<f64> {
        rate(self.forecast, self.pool, at, layout.shares())
    }

    fn layouts(
        &self,
        snapshot: &HostedPdPlanning,
        active: &BTreeMap<GroupId, Endpoint>,
    ) -> Result<Vec<Layout>, &'static str> {
        let Search {
            forecast,
            pool,
            context,
            aggregate,
            withdrawing,
        } = *self;
        let mut split: Vec<_> = pool.slots.iter().collect();
        split.sort_by_key(|slot| slot.role == ServingRole::Decode);
        let slots = aggregate
            .iter()
            .map(|slot| (ServingRole::Aggregated, slot))
            .chain(split.into_iter().map(|slot| (slot.role, &slot.pool)));
        let mut beam = Beam::from([(Roles::default(), vec![Layout::default()])]);
        for (role, slot) in slots {
            let group = slot.group();
            let target = match role {
                ServingRole::Aggregated => PoolTarget::RoutedAggregate,
                role => PoolTarget::Split(role, &pool.kv),
            };
            let history = snapshot
                .resources
                .groups
                .iter()
                .find(|entry| entry.latest.group == group);
            let expected_active = history
                .and_then(|entry| entry.active.as_ref())
                .map(|entry| entry.version);
            if expected_active.is_some_and(|version| {
                active
                    .get(&group)
                    .is_none_or(|entry| entry.version != version)
            }) {
                return Err("hosted P/D slot is active in another deployment");
            }
            let mut next = beam.clone();
            for layout in beam.values().flatten() {
                if layout
                    .endpoints
                    .iter()
                    .any(|entry| entry.version.group == group)
                {
                    continue;
                }
                if let Some(retained) = active.get(&group).filter(|entry| {
                    entry.role == role
                        && !withdrawing.is_some_and(|groups| groups.contains(&entry.version))
                }) {
                    let mut trial = layout.clone();
                    trial.add(retained.clone());
                    retain(&mut next, trial, forecast, pool);
                }
                let (resources, inventory) = available(snapshot, &layout.placements);
                for (candidate, recipe) in generate_layouts(
                    &resources,
                    &inventory,
                    &forecast.model,
                    slot,
                    context.evidence,
                    target,
                )? {
                    let Ok(version) = validate_placement(
                        &resources,
                        &forecast.model,
                        group,
                        candidate.expected_epoch,
                        &candidate.plan,
                    ) else {
                        continue;
                    };
                    let Some(endpoint) = context.candidate(version, &candidate, &recipe, slot)
                    else {
                        continue;
                    };
                    let mut trial = layout.clone();
                    trial.ready_after_ms = trial.ready_after_ms.max(candidate.ready_after_ms);
                    trial.transition_loss_requests += candidate.transition_loss_requests;
                    trial.placements.push(HostedPdPlacement {
                        group,
                        expected_epoch: candidate.expected_epoch,
                        expected_active: expected_active.map(|version| version.epoch),
                        plan: candidate.plan,
                        recipe,
                    });
                    trial.add(endpoint);
                    retain(&mut next, trial, forecast, pool);
                }
            }
            beam = next;
        }
        Ok(beam
            .into_iter()
            .filter(|(roles, _)| roles.complete())
            .flat_map(|(_, layouts)| layouts)
            .collect())
    }
}

impl HostedPdPool {
    fn checkpoint_fraction(&self, forecast: &ServingForecast) -> f64 {
        self.checkpoint_traffic.map_or(0.0, |profile| {
            let floor = f64::from(profile.minimum_fraction_bps) / 10_000.0;
            forecast
                .checkpoints
                .map_or(1.0, |load| load.fraction.max(floor))
        })
    }

    fn observed_checkpoint_traffic(&self, forecast: &ServingForecast) -> Option<CheckpointTraffic> {
        self.checkpoint_traffic.filter(|_| {
            self.checkpoint_fraction(forecast) > 0.0
                || forecast.checkpoints.is_some_and(|load| load.in_flight > 0)
        })
    }

    pub fn can_reconfigure(&self) -> bool {
        self.reconfigure.is_some()
    }

    pub fn calibrated_capacity(
        &self,
        group: GroupId,
        role: ServingRole,
        profile: &str,
        tensor_parallel: u32,
    ) -> Option<SloProfile> {
        let slot = self.slot(group, role)?;
        slot.pool.capacity(profile, tensor_parallel)
    }

    pub fn calibrated_assignment_capacity(
        &self,
        instance: &HostedPdInstance,
        assignment: &HostedAssignment,
    ) -> Option<SloProfile> {
        let pp = active_pipeline_parallel(instance, assignment)?;
        let name = &assignment.recipe.profile;
        let tp = assignment.tensor_parallel;
        if pp == 1 {
            return self.calibrated_capacity(instance.version.group, instance.role, name, tp);
        }
        let slot = self.slot(instance.version.group, instance.role)?;
        let profile = slot.pool.pipeline(name, tp, pp)?;
        let lanes = profile.lanes(assignment.recipe.api_host, &instance.ranks)?;
        Some(profile.capacity(&lanes))
    }

    fn slot(&self, group: GroupId, role: ServingRole) -> Option<&RoleSlot> {
        self.slots
            .iter()
            .find(|slot| slot.pool.group() == group && slot.role == role)
    }

    fn store_host(&self, inventory: &HostedPlanning) -> Result<NodeId, &'static str> {
        let address = IpAddr::V4(*self.kv.store.ip());
        let mut matches = inventory
            .hosts
            .iter()
            .filter(|entry| entry.address == address);
        let host = matches.next().filter(|_| matches.next().is_none());
        host.map(|entry| entry.host)
            .ok_or("the KV store address must match one registered host")
    }

    fn validate(&self, model: &str, now_ms: u64) -> Result<(), &'static str> {
        if model.is_empty()
            || self.kv_bytes_per_request == 0
            || self.store_bytes_per_second == 0
            || self.store_valid_until_ms <= now_ms
        {
            return Err("hosted P/D requires current store capacity");
        }
        if self.checkpoint_traffic.is_some_and(|profile| {
            (profile.write_bytes_per_stream == 0 && profile.read_bytes_per_stream == 0)
                || profile.minimum_fraction_bps > 10_000
                || profile.handoff_window_ms == 0
        }) {
            return Err("invalid checkpoint traffic profile");
        }
        let mut groups = BTreeSet::new();
        for slot in &self.slots {
            slot.pool.validate(true)?;
            if slot.role == ServingRole::Aggregated
                || slot.pool.api_model() != model
                || !groups.insert((slot.pool.group(), slot.role == ServingRole::Decode))
            {
                return Err("hosted P/D requires distinct role slots with the deployment alias");
            }
        }
        if ![ServingRole::Prefill, ServingRole::Decode]
            .iter()
            .all(|role| self.slots.iter().any(|slot| slot.role == *role))
        {
            return Err("hosted P/D requires at least one slot for each role");
        }
        Ok(())
    }
}

fn active_pipeline_parallel(
    instance: &HostedPdInstance,
    assignment: &HostedAssignment,
) -> Option<u32> {
    let tp = u64::from(assignment.tensor_parallel);
    let total = instance
        .ranks
        .values()
        .try_fold(0u64, |total, count| total.checked_add(*count))?;
    if assignment.version != instance.version
        || assignment.model != instance.model
        || assignment.phase != GroupPhase::Active
        || tp == 0
        || total == 0
        || total % tp != 0
        || instance
            .ranks
            .get(&assignment.recipe.api_host)
            .copied()
            .unwrap_or(0)
            == 0
    {
        return None;
    }
    u32::try_from(total / tp).ok()
}

#[derive(Clone, Copy)]
struct EndpointContext<'a> {
    evidence: LinkEvidence<'a>,
    store_host: Option<NodeId>,
    kv_bytes: u64,
    checkpoint: Option<CheckpointTraffic>,
    aggregate_checkpoints: bool,
}

impl EndpointContext<'_> {
    fn incumbent(
        mut self,
        version: GroupVersion,
        role: ServingRole,
        capacity: SloProfile,
        ranks: &BTreeMap<NodeId, u64>,
        pipeline: Option<(&PipelineProfile, NodeId)>,
    ) -> Option<Endpoint> {
        // Keep the measured baseline available while evaluating a replacement.
        self.evidence.policy.tp_rtt_budget = None;
        self.endpoint(version, role, capacity, ranks, pipeline)
    }

    fn candidate(
        &self,
        version: GroupVersion,
        candidate: &ServingCandidate,
        recipe: &HostedRecipe,
        pool: &HostedPool,
    ) -> Option<Endpoint> {
        let plan = &candidate.plan;
        let mut ranks = BTreeMap::new();
        for device in plan.workers.keys() {
            *ranks.entry(device.node).or_default() += 1;
        }
        let pipeline = if plan.pipeline_parallel > 1 {
            Some((
                pool.pipeline(
                    &recipe.profile,
                    plan.tensor_parallel,
                    plan.pipeline_parallel,
                )?,
                recipe.api_host,
            ))
        } else {
            None
        };
        self.endpoint(version, plan.role, candidate.capacity, &ranks, pipeline)
    }

    fn endpoint(
        &self,
        version: GroupVersion,
        role: ServingRole,
        mut capacity: SloProfile,
        ranks: &BTreeMap<NodeId, u64>,
        pipeline: Option<(&PipelineProfile, NodeId)>,
    ) -> Option<Endpoint> {
        let (mut traffic, weights) = if let Some((profile, api_host)) = pipeline {
            let lanes = profile.lanes(api_host, ranks)?;
            capacity = lower_capacity(capacity, profile.capacity(&lanes));
            (
                profile.traffic(&lanes, self.evidence)?,
                profile.kv_weights(&lanes),
            )
        } else {
            (
                self.evidence.estimate_tp(ranks).ok()?.traffic,
                ranks.clone(),
            )
        };
        match role {
            ServingRole::Prefill => {
                self.add_store_io(&mut traffic, &weights, self.kv_bytes, true)?
            }
            ServingRole::Decode => {
                self.add_store_io(&mut traffic, &weights, self.kv_bytes, false)?
            }
            ServingRole::Aggregated => {}
        }
        let checkpoint = self.checkpoint.filter(|_| {
            role == ServingRole::Decode
                || (role == ServingRole::Aggregated && self.aggregate_checkpoints)
        });
        let checkpoint = if let Some(profile) = checkpoint {
            let (mut write, mut read) = (Traffic::new(), Traffic::new());
            self.add_store_io(&mut write, &weights, profile.write_bytes_per_stream, true)?;
            self.add_store_io(&mut read, &weights, profile.read_bytes_per_stream, false)?;
            Some(CheckpointIo { write, read })
        } else {
            None
        };
        Some(Endpoint {
            version,
            role,
            capacity,
            traffic,
            checkpoint,
            devices: ranks.values().sum(),
        })
    }

    fn add_store_io(
        &self,
        traffic: &mut Traffic,
        weights: &BTreeMap<NodeId, u64>,
        bytes: u64,
        write: bool,
    ) -> Option<()> {
        if bytes == 0 {
            return Some(());
        }
        let store = self.store_host?;
        let total: u64 = weights.values().sum();
        for (&host, &weight) in weights {
            let (from, to) = if write { (host, store) } else { (store, host) };
            let proportional_bytes = bytes as f64 * weight as f64 / total as f64;
            self.evidence
                .add(traffic, from, to, proportional_bytes)
                .ok()?;
        }
        Some(())
    }
}

fn baseline(
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    snapshot: &HostedPdPlanning,
    active: &BTreeMap<GroupId, Endpoint>,
) -> Option<f64> {
    if snapshot.paths.is_empty() {
        return Some(0.0);
    }
    let mut uses: BTreeMap<GroupVersion, u32> = BTreeMap::new();
    for path in &snapshot.paths {
        for version in [path.prefill, path.decode] {
            *uses.entry(version).or_default() += 1;
        }
    }
    let endpoints: Vec<_> = uses
        .into_iter()
        .map(|(version, count)| {
            Some((
                active
                    .get(&version.group)
                    .filter(|entry| entry.version == version)?,
                f64::from(count) / snapshot.paths.len() as f64,
            ))
        })
        .collect::<Option<_>>()?;
    rate(forecast, pool, forecast.now_ms, endpoints)
}

fn rate<'a>(
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    at: u64,
    endpoints: impl IntoIterator<Item = (&'a Endpoint, f64)>,
) -> Option<f64> {
    rate_with_reservation(forecast, pool, at, endpoints, None)
}

struct CheckpointReservation {
    ports: Traffic,
    store_bytes_per_second: f64,
}

fn rate_with_reservation<'a>(
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    at: u64,
    endpoints: impl IntoIterator<Item = (&'a Endpoint, f64)>,
    reserved: Option<&CheckpointReservation>,
) -> Option<f64> {
    let mut result = forecast.demand_rps;
    let mut traffic = Traffic::new();
    let (mut writes, mut reads) = (0.0f64, 0.0f64);
    let fraction = pool.checkpoint_fraction(forecast);
    let mut checkpoint_share = 0.0;
    for (endpoint, share) in endpoints {
        if !endpoint.capacity.is_valid(at) {
            return None;
        }
        match endpoint.role {
            ServingRole::Prefill => writes += share,
            ServingRole::Decode => reads += share,
            ServingRole::Aggregated => {}
        }
        result = result.min(endpoint.capacity.rps / share);
        traffic.include(&endpoint.traffic, share);
        if let Some(io) = &endpoint.checkpoint
            && fraction > 0.0
        {
            if io.write.valid_until_ms.min(io.read.valid_until_ms) <= at {
                return None;
            }
            let share = share * fraction;
            traffic.include(&io.write, share);
            traffic.include(&io.read, share);
            checkpoint_share += share;
        }
    }
    // A partial search layout has a virtual counterpart for its P or D.
    let additional = pool.checkpoint_traffic.map_or(0.0, |profile| {
        (profile.write_bytes_per_stream as f64 + profile.read_bytes_per_stream as f64)
            * checkpoint_share
    });
    let store_bytes_per_request =
        2.0 * pool.kv_bytes_per_request as f64 * writes.max(reads) + additional;
    let reserved_store = reserved.map_or(0.0, |reserve| reserve.store_bytes_per_second);
    if store_bytes_per_request > 0.0 || reserved_store > 0.0 {
        if pool.store_valid_until_ms <= at {
            return None;
        }
        let available = pool.store_bytes_per_second as f64 - reserved_store;
        if available < 0.0 {
            return None;
        }
        if store_bytes_per_request > 0.0 {
            result = result.min(available / store_bytes_per_request);
        }
    }
    let link_limit = if let Some(reserve) = reserved {
        if reserve.ports.valid_until_ms <= at {
            return None;
        }
        traffic.rate_limit_after(&reserve.ports)?
    } else {
        traffic.rate_limit()
    };
    Some(result.min(link_limit))
}

fn handoff_capacity(
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    active: &BTreeMap<GroupId, Endpoint>,
    layout: &Layout,
) -> Option<(f64, u64)> {
    let Some((profile, load)) = pool.checkpoint_traffic.zip(forecast.checkpoints) else {
        return Some((layout.potential, 0));
    };
    if load.in_flight == 0 {
        return Some((layout.potential, 0));
    }
    let mut removed_generator = false;
    let mut source_write = Traffic::new();
    let mut target_read = Traffic::new();
    let mut target_prefill = Traffic::new();
    let (mut write_bytes, mut read_bytes) = (0, 0);
    for source in active.values() {
        if source.role == ServingRole::Prefill
            || layout
                .endpoints
                .iter()
                .any(|target| target.version == source.version)
        {
            continue;
        }
        removed_generator = true;
        if let Some(io) = &source.checkpoint {
            source_write.include_peak(&io.write);
            write_bytes = profile.write_bytes_per_stream;
        }
    }
    if !removed_generator {
        return Some((layout.potential, 0));
    }
    for target in &layout.endpoints {
        let mut inbound = Traffic::new();
        match target.role {
            ServingRole::Prefill => {
                target_prefill.include_peak(&target.traffic);
                continue;
            }
            ServingRole::Decode | ServingRole::Aggregated => inbound.include(&target.traffic, 1.0),
        }
        if let Some(io) = &target.checkpoint {
            inbound.include(&io.read, 1.0);
            read_bytes = profile.read_bytes_per_stream;
        }
        target_read.include_peak(&inbound);
    }
    // An ingress may send every resumed stream to the same ready generator.
    // Each direction reserves the largest contribution on each shared port.
    let streams_per_second = load.in_flight as f64 * 1000.0 / profile.handoff_window_ms as f64;
    let mut ports = Traffic::new();
    ports.include(&source_write, streams_per_second);
    ports.include(&target_read, streams_per_second);
    ports.include(&target_prefill, streams_per_second);
    let split_bytes = if layout.roles.has_split() {
        2.0 * pool.kv_bytes_per_request as f64
    } else {
        0.0
    };
    let reserve = CheckpointReservation {
        ports,
        store_bytes_per_second: (write_bytes as f64 + read_bytes as f64 + split_bytes)
            * streams_per_second,
    };
    let at = forecast.now_ms.checked_add(layout.ready_after_ms)?;
    let capacity = rate_with_reservation(forecast, pool, at, layout.shares(), Some(&reserve))?;
    let duration = profile
        .handoff_window_ms
        .min(forecast.horizon_ms - layout.ready_after_ms);
    Some((capacity, duration))
}

impl Layout {
    fn shares(&self) -> impl Iterator<Item = (&Endpoint, f64)> {
        let p = self.roles.prefill.max(1);
        let d = self.roles.decode.max(1);
        let split = if self.roles.prefill > 0 || self.roles.decode > 0 {
            p * d
        } else {
            0
        };
        let total = (self.roles.aggregated + split) as f64;
        self.endpoints.iter().map(move |entry| {
            let routes = match entry.role {
                ServingRole::Prefill => d,
                ServingRole::Decode => p,
                ServingRole::Aggregated => 1,
            };
            (entry, routes as f64 / total)
        })
    }

    fn add(&mut self, endpoint: Endpoint) {
        match endpoint.role {
            ServingRole::Prefill => self.roles.prefill += 1,
            ServingRole::Decode => self.roles.decode += 1,
            ServingRole::Aggregated => self.roles.aggregated += 1,
        }
        self.endpoints.push(endpoint);
    }

    fn routes(&self) -> Vec<HostedRoute> {
        self.endpoints
            .iter()
            .filter(|endpoint| endpoint.role == ServingRole::Aggregated)
            .map(|endpoint| HostedRoute::Aggregated(endpoint.version))
            .chain(self.paths().into_iter().map(HostedRoute::Split))
            .collect()
    }

    fn paths(&self) -> Vec<PdRoute> {
        self.endpoints
            .iter()
            .filter(|entry| entry.role == ServingRole::Prefill)
            .flat_map(|p| {
                self.endpoints
                    .iter()
                    .filter(|entry| entry.role == ServingRole::Decode)
                    .map(|d| PdRoute {
                        prefill: p.version,
                        decode: d.version,
                    })
            })
            .collect()
    }

    fn devices(&self) -> u64 {
        self.endpoints.iter().map(|entry| entry.devices).sum()
    }
}

fn retain(beam: &mut Beam, mut layout: Layout, forecast: &ServingForecast, pool: &HostedPdPool) {
    if layout.ready_after_ms >= forecast.horizon_ms || !layout.transition_loss_requests.is_finite()
    {
        return;
    }
    let Some(at) = forecast.now_ms.checked_add(layout.ready_after_ms) else {
        return;
    };
    let Some(potential) = rate(forecast, pool, at, layout.shares()) else {
        return;
    };
    layout.potential = potential;
    let choices = beam.entry(layout.roles).or_default();
    choices.push(layout);
    let value = |candidate: &Layout| {
        candidate.potential * (forecast.horizon_ms - candidate.ready_after_ms) as f64 / 1000.0
            - candidate.transition_loss_requests
    };
    choices.sort_by(|a, b| {
        value(b)
            .total_cmp(&value(a))
            .then_with(|| a.devices().cmp(&b.devices()))
            .then_with(|| a.placements.len().cmp(&b.placements.len()))
    });
    choices.truncate(LAYOUTS_PER_SHAPE);
}

fn available(
    snapshot: &HostedPdPlanning,
    placements: &[HostedPdPlacement],
) -> (ServingResources, HostedPlanning) {
    let mut resources = snapshot.resources.clone();
    let mut inventory = snapshot.inventory.clone();
    let mut used = BTreeSet::new();
    for placement in placements {
        used.extend(placement.plan.workers.keys().copied());
        inventory
            .occupied
            .push(HostedPortUse::from(&placement.recipe));
        for (host, demand) in &placement.plan.hosts {
            let budget = resources
                .hosts
                .iter_mut()
                .find(|entry| entry.host == *host)
                .expect("placement uses a checked host budget");
            budget.reserved.bytes += demand.bytes;
            budget.reserved.pinned_bytes += demand.pinned_bytes;
        }
    }
    resources
        .workers
        .retain(|worker| !used.contains(&worker.device));
    (resources, inventory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::GroupProfile;
    use plait_core::DeviceId;
    use plait_core::serving::{
        ActiveGroup, GroupPhase, GroupResources, HostMemory, HostResources, HostedInventory,
        HostedPdInstance, WorkerGrant, WorkerResources,
    };

    fn node(id: u8) -> NodeId {
        NodeId::from_bytes([id; 16])
    }

    fn sample(target: NodeId, bytes_per_second: u64) -> HostedLinkSample {
        HostedLinkSample {
            target,
            target_session: target,
            sampled_at_ms: 1000,
            bytes_per_second,
            rtt_ns: 10,
        }
    }

    #[test]
    fn reconfiguration_preserves_bridge_capacity_and_scores_reusing_a_live_group() {
        let [store, retiring, incoming] = [1, 2, 3].map(node);
        let roles = [ServingRole::Prefill, ServingRole::Decode];
        let version = |id| GroupVersion {
            group: GroupId(node(id)),
            epoch: 1,
        };
        let (p, d, next) = (version(7), version(8), version(9));
        let ports = |host| HostedPortUse {
            host,
            ports: [20000, 20001, 20002],
        };
        let cap = |rps| SloProfile {
            rps,
            valid_until_ms: 10_000,
        };
        let budget = HostMemory {
            bytes: 24,
            pinned_bytes: 3,
        };
        let kv = HostedPdKv {
            store: "192.0.2.1:7952".parse().unwrap(),
            namespace: "weights".into(),
            block_size: 128,
        };
        let inventory = HostedPlanning {
            hosts: [
                (store, &["prefill"][..], 1),
                (retiring, &["decode", "reclaimed"], 2),
                (incoming, &["replacement"], 1),
            ]
            .into_iter()
            .map(|(host, profiles, cards)| HostedInventory {
                host,
                address: std::net::Ipv4Addr::new(192, 0, 2, host.as_bytes()[0]).into(),
                profiles: profiles.iter().map(|profile| (*profile).into()).collect(),
                devices: (0..cards).map(|ordinal| (ordinal, 1)).collect(),
            })
            .collect(),
            occupied: vec![ports(store), ports(retiring)],
        };
        let resources = ServingResources {
            hosts: [store, retiring, incoming]
                .map(|host| HostResources {
                    host,
                    capacity: budget,
                    reserved: if host == incoming {
                        HostMemory::default()
                    } else {
                        budget
                    },
                })
                .into(),
            workers: [
                (store, 0, Some(p)),
                (retiring, 0, Some(d)),
                (retiring, 1, Some(d)),
                (incoming, 0, None),
            ]
            .into_iter()
            .map(|(host, ordinal, active)| {
                let device = DeviceId::new(host, ordinal);
                WorkerResources {
                    device,
                    incarnation: 1,
                    device_bytes: 32,
                    grant: active.map(|version| WorkerGrant {
                        version,
                        device,
                        incarnation: 1,
                        fence: 1,
                    }),
                }
            })
            .collect(),
            groups: [p, d]
                .into_iter()
                .zip(roles)
                .map(|(version, role)| GroupResources {
                    latest: version,
                    phase: GroupPhase::Active,
                    active: Some(ActiveGroup {
                        version,
                        role,
                        model: "weights".into(),
                    }),
                })
                .collect(),
        };
        let snapshot = HostedPdPlanning {
            revision: 1,
            pending: None,
            managed: true,
            active: [(p, store, 1), (d, retiring, 2)]
                .into_iter()
                .zip(roles)
                .map(|((version, host, ranks), role)| HostedPdInstance {
                    version,
                    role,
                    model: "weights".into(),
                    kv: kv.clone(),
                    ranks: BTreeMap::from([(host, ranks)]),
                })
                .collect(),
            paths: vec![PdRoute {
                prefill: p,
                decode: d,
            }],
            inventory,
            resources,
        };
        let forecast = ServingForecast {
            checkpoints: None,
            model: "weights".into(),
            demand_rps: 20.0,
            now_ms: 1000,
            horizon_ms: 2000,
            minimum_gain_requests: 1.0,
            groups: vec![
                GroupProfile {
                    version: p,
                    capacity: cap(10.0),
                },
                GroupProfile {
                    version: d,
                    capacity: cap(30.0),
                },
            ],
            paths: Vec::new(),
            link_bytes_per_second: BTreeMap::new(),
        };
        let slot = |version: GroupVersion, role, profile, tp, rps| RoleSlot {
            role,
            pool: HostedPool::fixture(version.group, profile, "api").with_capacity(tp, rps),
        };
        let mut pool = HostedPdPool {
            checkpoint_traffic: None,
            kv,
            kv_bytes_per_request: 100,
            store_bytes_per_second: 200_000,
            store_valid_until_ms: 10_000,
            slots: vec![
                slot(p, roles[0], "prefill", 1, 10.0),
                slot(d, roles[0], "reclaimed", 2, 10.0),
                slot(d, roles[1], "decode", 2, 30.0),
                slot(next, roles[1], "replacement", 1, 30.0),
            ],
            allow_scale_down: false,
            reconfigure: Some(Reconfiguration { drain_ms: 100 }),
        };
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: 10,
            max_sample_age_ms: 10_000,
            tp_rtt_budget: None,
        };
        let links = BTreeMap::from([
            ((store, retiring), sample(retiring, 100_000)),
            ((retiring, store), sample(store, 100_000)),
            ((store, incoming), sample(incoming, 100_000)),
        ]);
        let release = [HostedPdRelease {
            version: d,
            ports: ports(retiring),
            hosts: BTreeMap::from([(retiring, budget)]),
        }];
        let decide = |pool: &HostedPdPool, release: &[HostedPdRelease]| {
            plan_hosted_pd_with_releases("api", &snapshot, &forecast, pool, policy, &links, release)
                .unwrap()
        };
        assert!(decide(&pool, &[]).selected.is_none());
        let result = decide(&pool, &release);
        assert_eq!(result.baseline_rps, 10.0);
        let choice = result.selected.unwrap();
        assert_eq!(choice.steady_rps, 20.0);
        assert_eq!(choice.gain_requests, 18.8);
        let bridge = choice.bridge.unwrap();
        assert_eq!(bridge.rate, 10.0);
        assert_eq!(
            bridge.stage.paths,
            vec![PdRoute {
                prefill: p,
                decode: next
            }]
        );
        assert_eq!(bridge.release, release);
        let converted = &choice.placements[0];
        assert_eq!(converted.group, d.group);
        assert_eq!(converted.expected_epoch, d.epoch);
        assert_eq!(converted.expected_active, None);
        assert_eq!(converted.plan.role, ServingRole::Prefill);
        assert_eq!(converted.plan.tensor_parallel, 2);
        assert_eq!(converted.plan.hosts, release[0].hosts);

        pool.reconfigure = Some(Reconfiguration { drain_ms: 1999 });
        assert!(decide(&pool, &release).selected.is_none());
        pool.reconfigure = Some(Reconfiguration { drain_ms: 100 });
        pool.slots[3] = slot(next, roles[1], "replacement", 1, 9.0);
        assert!(decide(&pool, &release).selected.is_none());
    }

    #[test]
    fn joint_layout_prefers_a_fast_decode_and_rejects_missing_directional_links() {
        let [store, slow, fast] = [1, 2, 3].map(node);
        let inventory = HostedPlanning {
            hosts: [(store, "prefill"), (slow, "decode"), (fast, "decode")]
                .into_iter()
                .map(|(host, profile)| HostedInventory {
                    host,
                    address: std::net::Ipv4Addr::new(192, 0, 2, host.as_bytes()[0]).into(),
                    profiles: BTreeSet::from([profile.into()]),
                    devices: BTreeMap::from([(0, 1)]),
                })
                .collect(),
            occupied: Vec::new(),
        };
        let resources = ServingResources {
            hosts: [store, slow, fast]
                .into_iter()
                .map(|host| HostResources {
                    host,
                    capacity: HostMemory {
                        bytes: 64,
                        pinned_bytes: 8,
                    },
                    reserved: HostMemory::default(),
                })
                .collect(),
            workers: [store, slow, fast]
                .into_iter()
                .map(|host| WorkerResources {
                    device: DeviceId::new(host, 0),
                    incarnation: 1,
                    device_bytes: 32,
                    grant: None,
                })
                .collect(),
            groups: Vec::new(),
        };
        let snapshot = HostedPdPlanning {
            revision: 0,
            managed: true,
            pending: None,
            active: Vec::new(),
            paths: Vec::new(),
            resources,
            inventory,
        };
        let forecast = ServingForecast {
            checkpoints: None,
            model: "weights".into(),
            demand_rps: 20.0,
            now_ms: 1000,
            horizon_ms: 10000,
            minimum_gain_requests: 0.0,
            groups: Vec::new(),
            link_bytes_per_second: BTreeMap::new(),
            paths: Vec::new(),
        };
        let slot = |group, role, profile| RoleSlot {
            role,
            pool: HostedPool::fixture(GroupId(node(group)), profile, "api"),
        };
        let pool = HostedPdPool {
            checkpoint_traffic: None,
            kv: HostedPdKv {
                store: "192.0.2.1:7952".parse().unwrap(),
                namespace: "weights".into(),
                block_size: 128,
            },
            kv_bytes_per_request: 100,
            store_bytes_per_second: 20_000,
            store_valid_until_ms: 5000,
            slots: vec![
                slot(7, ServingRole::Prefill, "prefill"),
                slot(8, ServingRole::Decode, "decode"),
                slot(9, ServingRole::Decode, "decode"),
            ],
            allow_scale_down: false,
            reconfigure: None,
        };
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: 10,
            max_sample_age_ms: 50,
            tp_rtt_budget: None,
        };
        let links = BTreeMap::from([
            ((store, slow), sample(slow, 100)),
            ((store, fast), sample(fast, 1000)),
        ]);
        let choice = plan_hosted_pd("api", &snapshot, &forecast, &pool, policy, &links)
            .unwrap()
            .selected
            .unwrap();
        assert_eq!(choice.paths.len(), 1);
        assert_eq!(choice.steady_rps, 10.0);
        assert_eq!(
            choice
                .placements
                .iter()
                .find(|entry| entry.plan.role == ServingRole::Decode)
                .unwrap()
                .recipe
                .api_host,
            fast
        );

        let reversed = BTreeMap::from([((fast, store), sample(store, 1000))]);
        assert!(
            plan_hosted_pd("api", &snapshot, &forecast, &pool, policy, &reversed)
                .unwrap()
                .selected
                .is_none()
        );
    }

    #[test]
    fn shared_port_accounts_for_tp_and_kv_on_the_same_direction() {
        let [store, a, b] = [1, 2, 3].map(node);
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: 100,
            max_sample_age_ms: 25,
            tp_rtt_budget: None,
        };
        let links = BTreeMap::from([
            ((a, b), sample(b, 1000)),
            ((b, a), sample(a, 1000)),
            ((a, store), sample(store, 1000)),
            ((b, store), sample(store, 1000)),
        ]);
        let evidence = LinkEvidence {
            policy,
            now_ms: 1020,
            links: &links,
        };
        let context = EndpointContext {
            evidence,
            store_host: Some(store),
            kv_bytes: 200,
            checkpoint: None,
            aggregate_checkpoints: false,
        };
        let version = GroupVersion {
            group: GroupId(a),
            epoch: 1,
        };
        let role = ServingRole::Prefill;
        let capacity = SloProfile {
            rps: 100.0,
            valid_until_ms: 1100,
        };
        let ranks = BTreeMap::from([(a, 1), (b, 1)]);
        let source = context
            .endpoint(version, role, capacity, &ranks, None)
            .unwrap();
        assert_eq!(source.traffic.rate_limit(), 5.0);
        let mut shared = source.traffic.clone();
        shared.include(&source.traffic, 1.0);
        assert_eq!(shared.rate_limit(), 2.5);

        let mut constrained = context;
        constrained.evidence.policy.tp_rtt_budget = Some(crate::serving::HostedTpRttBudget {
            serial_rounds: 57,
            max_wait_ns: 100,
        });
        assert!(
            constrained
                .endpoint(version, role, capacity, &ranks, None)
                .is_none()
        );
        assert_eq!(
            constrained
                .incumbent(version, role, capacity, &ranks, None)
                .unwrap()
                .traffic
                .rate_limit(),
            5.0,
        );
    }
}
