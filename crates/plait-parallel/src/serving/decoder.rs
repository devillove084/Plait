use std::collections::{BTreeMap, BTreeSet};

use plait_core::serving::{
    DecoderRouteWeight, DecoderStandby, GroupId, GroupPlan, GroupVersion, HostMemory, PdRoute,
    ServingError, ServingResources, ServingRole,
};
use serde::Serialize;

use super::{
    CandidateForecast, KvPath, RemainingGroup, ServingForecast, ServingSelection, SloProfile,
    valid_rate, validate_placement,
};

pub(super) mod pool;

#[derive(Clone)]
pub struct DecoderPlacement {
    pub group: GroupId,
    pub expected_epoch: u64,
    pub plan: GroupPlan,
    pub capacity: SloProfile,
    pub ready_after_ms: u64,
}

pub struct DecoderLayoutCandidate {
    pub placements: Vec<DecoderPlacement>,
    pub paths: Vec<PdRoute>,
    pub standbys: Vec<DecoderStandby>,
    pub transition_loss_requests: f64,
}

#[derive(Serialize)]
pub struct DecoderSelection {
    #[serde(flatten)]
    pub summary: ServingSelection,
    pub route_weights: Vec<DecoderRouteWeight>,
}

pub fn select_decoder_layout(
    resources: &ServingResources,
    forecast: &ServingForecast,
    incumbent: &[PdRoute],
    candidates: &[DecoderLayoutCandidate],
) -> Result<DecoderSelection, &'static str> {
    forecast.validate()?;
    let profiles: BTreeMap<_, _> = forecast
        .groups
        .iter()
        .map(|profile| (profile.version, profile.capacity))
        .collect();
    let links: BTreeMap<_, _> = forecast
        .paths
        .iter()
        .map(|path| {
            (
                PdRoute {
                    prefill: path.prefill,
                    decode: path.decode,
                },
                path,
            )
        })
        .collect();
    if profiles.len() != forecast.groups.len() || links.len() != forecast.paths.len() {
        return Err("duplicate decoder group or route profile");
    }
    let active: BTreeMap<_, _> = resources
        .groups
        .iter()
        .filter_map(|entry| entry.active.as_ref())
        .filter(|group| group.model == forecast.model)
        .filter_map(|group| {
            profiles
                .get(&group.version)
                .map(|&capacity| (group.version, (group.role, capacity)))
        })
        .collect();
    let baseline_rps = allocate(
        forecast,
        incumbent,
        &active,
        &links,
        forecast.now_ms,
        |_, _| {},
    )
    .ok_or("published decoder route is missing a current compatible profile")?;
    let mut selected = None;
    let mut route_weights = Vec::new();
    let mut best = 0.0;
    let mut scores = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        if !valid_rate(candidate.transition_loss_requests)
            || !valid_rate(forecast.horizon_requests() + candidate.transition_loss_requests)
            || candidate
                .placements
                .iter()
                .any(|placement| !valid_rate(placement.capacity.rps))
        {
            return Err("invalid decoder capacity or transition cost");
        }
        let mut rates = BTreeMap::new();
        let score = candidate.evaluate(
            resources,
            forecast,
            &active,
            &links,
            baseline_rps,
            &mut rates,
        );
        if let CandidateForecast::Scored { gain_requests, .. } = score
            && gain_requests >= forecast.minimum_gain_requests
            && gain_requests > best
        {
            selected = Some(index);
            best = gain_requests;
            let largest = rates.values().copied().fold(0.0, f64::max);
            route_weights = candidate
                .paths
                .iter()
                .map(|&path| {
                    let rate = rates.get(&path).copied().unwrap_or_default();
                    let weight = if rate > 0.0 {
                        ((rate / largest) * 1_000_000.0).round().max(1.0) as u32
                    } else {
                        0
                    };
                    DecoderRouteWeight { path, weight }
                })
                .collect();
        }
        scores.push(score);
    }
    Ok(DecoderSelection {
        summary: ServingSelection {
            baseline_rps,
            selected,
            candidates: scores,
        },
        route_weights,
    })
}

type Capacity = BTreeMap<GroupVersion, (ServingRole, SloProfile)>;

fn allocate(
    forecast: &ServingForecast,
    paths: &[PdRoute],
    profiles: &Capacity,
    links: &BTreeMap<PdRoute, &KvPath>,
    at: u64,
    record_rate: impl FnMut(&KvPath, f64),
) -> Option<f64> {
    let mut groups = BTreeMap::new();
    let mut active_versions = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::with_capacity(paths.len());
    for path in paths {
        let measure = *links.get(path)?;
        if !seen.insert(path) || !measure.capacity.is_valid(at) {
            return None;
        }
        for (version, role) in [
            (path.prefill, ServingRole::Prefill),
            (path.decode, ServingRole::Decode),
        ] {
            let &(actual, capacity) = profiles.get(&version)?;
            if actual != role
                || !capacity.is_valid(at)
                || active_versions
                    .insert(version.group, version)
                    .is_some_and(|old| old != version)
            {
                return None;
            }
            groups.insert(
                version,
                RemainingGroup {
                    role,
                    rps: capacity.rps,
                },
            );
        }
        ordered.push(measure);
    }
    Some(forecast.allocate_ordered(groups, ordered, record_rate))
}

impl DecoderLayoutCandidate {
    fn evaluate(
        &self,
        resources: &ServingResources,
        forecast: &ServingForecast,
        active: &Capacity,
        links: &BTreeMap<PdRoute, &KvPath>,
        baseline_rps: f64,
        rates: &mut BTreeMap<PdRoute, f64>,
    ) -> CandidateForecast {
        let needed = self
            .placements
            .iter()
            .map(|placement| placement.ready_after_ms)
            .max()
            .unwrap_or(0);
        if needed >= forecast.horizon_ms {
            return CandidateForecast::ReadyAfterHorizon;
        }
        let mut next = active.clone();
        if let Err(error) = self.add_reservations(&mut next, resources, &forecast.model) {
            return CandidateForecast::ResourceConflict(error);
        }
        let Some(steady_rps) = forecast.now_ms.checked_add(needed).and_then(|at| {
            allocate(forecast, &self.paths, &next, links, at, |path, rate| {
                rates.insert(
                    PdRoute {
                        prefill: path.prefill,
                        decode: path.decode,
                    },
                    rate,
                );
            })
        }) else {
            return CandidateForecast::MissingProfile;
        };
        let productive_seconds = (forecast.horizon_ms - needed) as f64 / 1000.0;
        CandidateForecast::Scored {
            steady_rps,
            gain_requests: (steady_rps - baseline_rps) * productive_seconds
                - self.transition_loss_requests,
        }
    }

    fn add_reservations(
        &self,
        next: &mut Capacity,
        resources: &ServingResources,
        model: &str,
    ) -> Result<(), ServingError> {
        if self.paths.is_empty() {
            return Err(ServingError::InvalidDirectory);
        }
        let hosts: BTreeMap<_, _> = resources
            .hosts
            .iter()
            .map(|host| (host.host, host))
            .collect();
        let needed: BTreeSet<_> = self
            .paths
            .iter()
            .chain(self.standbys.iter().flat_map(|rule| &rule.alternatives))
            .flat_map(|path| [path.prefill, path.decode])
            .collect();
        let mut reserved_hosts = BTreeMap::<_, HostMemory>::new();
        let mut groups = BTreeSet::new();
        let mut devices = BTreeSet::new();
        for placement in &self.placements {
            if !groups.insert(placement.group)
                || placement.plan.workers.len() != 1
                || !matches!(
                    placement.plan.role,
                    ServingRole::Prefill | ServingRole::Decode
                )
            {
                return Err(ServingError::InvalidPlan);
            }
            let version = validate_placement(
                resources,
                model,
                placement.group,
                placement.expected_epoch,
                &placement.plan,
            )?;
            if !needed.contains(&version) {
                return Err(ServingError::InvalidPlan);
            }
            for &device in placement.plan.workers.keys() {
                if !devices.insert(device) {
                    return Err(ServingError::WorkerBusy(device));
                }
            }
            for (&id, demand) in &placement.plan.hosts {
                let host = hosts[&id];
                let total = reserved_hosts.entry(id).or_insert(host.reserved);
                total.bytes = total
                    .bytes
                    .checked_add(demand.bytes)
                    .filter(|&bytes| bytes <= host.capacity.bytes)
                    .ok_or(ServingError::HostCapacityExceeded(id))?;
                total.pinned_bytes = total
                    .pinned_bytes
                    .checked_add(demand.pinned_bytes)
                    .filter(|&bytes| bytes <= host.capacity.pinned_bytes)
                    .ok_or(ServingError::HostCapacityExceeded(id))?;
            }
            next.retain(|previous, _| previous.group != placement.group);
            next.insert(version, (placement.plan.role, placement.capacity));
        }
        Ok(())
    }
}
