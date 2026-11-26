//! Compare measured serving layouts against a committed resource snapshot.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::serving::{
    GroupId, GroupPhase, GroupPlan, GroupVersion, ServingError, ServingResources, ServingRole,
};
use serde::{Deserialize, Serialize};

mod decoder;
mod hosted;
pub use decoder::pool::{DecoderPool, generate_decoder_layouts};
pub use decoder::{
    DecoderLayoutCandidate, DecoderPlacement, DecoderSelection, select_decoder_layout,
};
pub use hosted::{
    HostedLinkLimit, HostedLinkPolicy, HostedModePolicy, HostedPdDecision, HostedPdIncumbents,
    HostedPdPool, HostedPdSelection, HostedPool, generate_hosted_candidates, limit_hosted_links,
    plan_hosted_evacuation, plan_hosted_pd, plan_hosted_pd_recovery,
    plan_hosted_pd_with_assignments, plan_hosted_pd_with_releases, plan_hosted_serving,
};

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct SloProfile {
    pub rps: f64,
    /// The caller must refresh the profile when the decision time reaches this timestamp.
    pub valid_until_ms: u64,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct GroupProfile {
    pub version: GroupVersion,
    pub capacity: SloProfile,
}

/// Capacity of a measured, compatible P→D route. Link IDs represent directional
/// shared resources, including NICs and host-staging paths.
#[derive(Clone, Deserialize, Serialize)]
pub struct KvPath {
    pub prefill: GroupVersion,
    pub decode: GroupVersion,
    pub bytes_per_request: u64,
    pub links: Vec<String>,
    pub capacity: SloProfile,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ServingForecast {
    pub model: String,
    pub demand_rps: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoints: Option<CheckpointLoad>,
    pub now_ms: u64,
    pub horizon_ms: u64,
    pub minimum_gain_requests: f64,
    pub groups: Vec<GroupProfile>,
    /// Available bytes per second after background traffic and migration reserves.
    pub link_bytes_per_second: BTreeMap<String, u64>,
    /// Routes in the caller's allocation priority, highest first.
    pub paths: Vec<KvPath>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CheckpointLoad {
    pub fraction: f64,
    pub in_flight: u64,
}

/// One additive placement or a replacement that can prepare alongside its incumbent.
pub struct ServingCandidate {
    pub group: GroupId,
    pub expected_epoch: u64,
    pub plan: GroupPlan,
    pub capacity: SloProfile,
    pub ready_after_ms: u64,
    pub transition_loss_requests: f64,
}

#[derive(Serialize)]
pub enum CandidateForecast {
    ResourceConflict(ServingError),
    MissingProfile,
    ReadyAfterHorizon,
    Scored { steady_rps: f64, gain_requests: f64 },
}

#[derive(Serialize)]
pub struct ServingSelection {
    pub baseline_rps: f64,
    /// Zero-based index in the original candidates. Exact ties keep the first.
    pub selected: Option<usize>,
    pub candidates: Vec<CandidateForecast>,
}

#[derive(Clone, Copy)]
struct RemainingGroup {
    role: ServingRole,
    rps: f64,
}

pub fn select_serving_candidate(
    resources: &ServingResources,
    input: &ServingForecast,
    candidates: &[ServingCandidate],
) -> Result<ServingSelection, &'static str> {
    input.validate()?;
    let profiles: BTreeMap<_, _> = input
        .groups
        .iter()
        .map(|profile| (profile.version, profile.capacity))
        .collect();
    if profiles.len() != input.groups.len() {
        return Err("duplicate group profile");
    }
    let mut incumbent = BTreeMap::new();
    for group in resources
        .groups
        .iter()
        .filter_map(|group| group.active.as_ref())
    {
        if group.model == input.model {
            let capacity = profiles
                .get(&group.version)
                .filter(|profile| profile.is_valid(input.now_ms))
                .ok_or("active group is missing a current profile")?;
            incumbent.insert(
                group.version,
                RemainingGroup {
                    role: group.role,
                    rps: capacity.rps,
                },
            );
        }
    }

    let baseline_rps = input.allocate(incumbent.clone());
    let mut selected = None;
    let mut best_gain = 0.0;
    let mut forecasts = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        if !valid_rate(candidate.transition_loss_requests)
            || !valid_rate(candidate.capacity.rps)
            || !valid_rate(input.horizon_requests() + candidate.transition_loss_requests)
        {
            return Err("invalid candidate capacity or transition cost");
        }
        let forecast = candidate.evaluate(resources, input, &incumbent, baseline_rps);
        if let CandidateForecast::Scored { gain_requests, .. } = forecast
            && gain_requests >= input.minimum_gain_requests
            && gain_requests > best_gain
        {
            selected = Some(index);
            best_gain = gain_requests;
        }
        forecasts.push(forecast);
    }
    Ok(ServingSelection {
        baseline_rps,
        selected,
        candidates: forecasts,
    })
}

impl SloProfile {
    fn is_valid(self, time_ms: u64) -> bool {
        valid_rate(self.rps) && self.valid_until_ms > time_ms
    }
}

impl ServingForecast {
    fn horizon_requests(&self) -> f64 {
        self.demand_rps * (self.horizon_ms as f64 / 1000.0)
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.model.is_empty()
            || !valid_rate(self.demand_rps)
            || self
                .checkpoints
                .is_some_and(|load| !(0.0..=1.0).contains(&load.fraction))
            || !valid_rate(self.minimum_gain_requests)
            || self.horizon_ms == 0
            || !valid_rate(self.horizon_requests())
        {
            return Err("invalid serving forecast policy");
        }
        for path in &self.paths {
            let links: BTreeSet<_> = path.links.iter().collect();
            if path.prefill == path.decode
                || path.bytes_per_request == 0
                || links.len() != path.links.len()
                || !valid_rate(path.capacity.rps)
                || links
                    .iter()
                    .any(|link| !self.link_bytes_per_second.contains_key(*link))
            {
                return Err("invalid KV path profile");
            }
        }
        Ok(())
    }

    fn allocate(&self, groups: BTreeMap<GroupVersion, RemainingGroup>) -> f64 {
        self.allocate_ordered(groups, &self.paths, |_, _| {})
    }

    fn allocate_ordered<'a>(
        &self,
        mut groups: BTreeMap<GroupVersion, RemainingGroup>,
        paths: impl IntoIterator<Item = &'a KvPath>,
        mut record_rate: impl FnMut(&KvPath, f64),
    ) -> f64 {
        let aggregated: f64 = groups
            .values()
            .filter(|group| group.role == ServingRole::Aggregated)
            .map(|group| group.rps)
            .sum();
        let mut remaining = (self.demand_rps - aggregated).max(0.0);
        let mut links: BTreeMap<_, _> = self
            .link_bytes_per_second
            .iter()
            .map(|(link, &capacity)| (link, capacity as f64))
            .collect();
        for path in paths {
            if remaining == 0.0 {
                break;
            }
            let (Some(prefill), Some(decode)) =
                (groups.get(&path.prefill), groups.get(&path.decode))
            else {
                continue;
            };
            if prefill.role != ServingRole::Prefill
                || decode.role != ServingRole::Decode
                || !path.capacity.is_valid(self.now_ms)
            {
                continue;
            }
            let bytes = path.bytes_per_request as f64;
            let rate = path.links.iter().fold(
                remaining
                    .min(prefill.rps)
                    .min(decode.rps)
                    .min(path.capacity.rps),
                |rate, link| rate.min(links[link] / bytes),
            );
            record_rate(path, rate);
            for link in &path.links {
                let available = links.get_mut(link).expect("validated link");
                *available = (*available - rate * bytes).max(0.0);
            }
            groups.get_mut(&path.prefill).expect("known prefill").rps -= rate;
            groups.get_mut(&path.decode).expect("known decode").rps -= rate;
            remaining -= rate;
        }
        self.demand_rps - remaining
    }
}

impl ServingCandidate {
    fn evaluate(
        &self,
        resources: &ServingResources,
        input: &ServingForecast,
        incumbent: &BTreeMap<GroupVersion, RemainingGroup>,
        baseline_rps: f64,
    ) -> CandidateForecast {
        let version = match self.validate_resources(resources, &input.model) {
            Ok(version) => version,
            Err(error) => return CandidateForecast::ResourceConflict(error),
        };
        if self.ready_after_ms >= input.horizon_ms {
            return CandidateForecast::ReadyAfterHorizon;
        }
        if !self.capacity.is_valid(input.now_ms) {
            return CandidateForecast::MissingProfile;
        }
        let mut next = incumbent.clone();
        next.retain(|previous, _| previous.group != self.group);
        next.insert(
            version,
            RemainingGroup {
                role: self.plan.role,
                rps: self.capacity.rps,
            },
        );
        let steady_rps = input.allocate(next);
        let productive_seconds = (input.horizon_ms - self.ready_after_ms) as f64 / 1000.0;
        let gain_requests =
            (steady_rps - baseline_rps) * productive_seconds - self.transition_loss_requests;
        CandidateForecast::Scored {
            steady_rps,
            gain_requests,
        }
    }

    fn validate_resources(
        &self,
        resources: &ServingResources,
        model: &str,
    ) -> Result<GroupVersion, ServingError> {
        validate_placement(
            resources,
            model,
            self.group,
            self.expected_epoch,
            &self.plan,
        )
    }
}

fn validate_placement(
    resources: &ServingResources,
    model: &str,
    group: GroupId,
    expected_epoch: u64,
    plan: &GroupPlan,
) -> Result<GroupVersion, ServingError> {
    plan.validate()?;
    let history = resources
        .groups
        .iter()
        .find(|entry| entry.latest.group == group);
    if plan.model != model
        || history
            .and_then(|entry| entry.active.as_ref())
            .is_some_and(|active| active.model != model)
    {
        return Err(ServingError::InvalidPlan);
    }
    let current = history.map_or(0, |entry| entry.latest.epoch);
    if expected_epoch != current {
        return Err(ServingError::StaleEpoch { current });
    }
    if let Some(entry) = history
        && matches!(entry.phase, GroupPhase::Preparing | GroupPhase::Ready)
    {
        return Err(ServingError::CandidateExists(entry.latest));
    }
    for (&device, demand) in &plan.workers {
        let worker = resources
            .workers
            .iter()
            .find(|worker| worker.device == device)
            .ok_or(ServingError::UnknownWorker(device))?;
        if worker.incarnation != demand.incarnation {
            return Err(ServingError::StaleIncarnation(device));
        }
        if worker.grant.is_some() {
            return Err(ServingError::WorkerBusy(device));
        }
        if worker.device_bytes < demand.device_bytes {
            return Err(ServingError::DeviceCapacityExceeded(device));
        }
    }
    for (&id, demand) in &plan.hosts {
        let host = resources
            .hosts
            .iter()
            .find(|host| host.host == id)
            .ok_or(ServingError::UnknownHost(id))?;
        if demand.bytes > host.capacity.bytes.saturating_sub(host.reserved.bytes)
            || demand.pinned_bytes
                > host
                    .capacity
                    .pinned_bytes
                    .saturating_sub(host.reserved.pinned_bytes)
        {
            return Err(ServingError::HostCapacityExceeded(id));
        }
    }
    Ok(GroupVersion {
        group,
        epoch: current
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?,
    })
}

fn valid_rate(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{
        ActiveGroup, GroupResources, HostMemory, HostResources, WorkerDemand, WorkerResources,
    };
    use plait_core::{DeviceId, NodeId};

    use super::*;

    fn version(name: &str) -> GroupVersion {
        GroupVersion {
            group: GroupId(NodeId::derive(name)),
            epoch: 1,
        }
    }

    fn profile(rps: f64) -> SloProfile {
        SloProfile {
            rps,
            valid_until_ms: 1_000,
        }
    }

    #[test]
    fn added_prefill_is_limited_by_shared_ingress_and_startup_cost() {
        let old_p = version("old-prefill");
        let old_d = version("old-decode");
        let new_p = version("new-prefill");
        let host = NodeId::derive("new-host");
        let device = DeviceId::new(host, 0);
        let resources = ServingResources {
            hosts: vec![HostResources {
                host,
                capacity: HostMemory {
                    bytes: 128,
                    pinned_bytes: 24,
                },
                reserved: HostMemory {
                    bytes: 32,
                    pinned_bytes: 6,
                },
            }],
            workers: vec![WorkerResources {
                device,
                incarnation: 1,
                device_bytes: 96,
                grant: None,
            }],
            groups: [(old_p, ServingRole::Prefill), (old_d, ServingRole::Decode)]
                .into_iter()
                .map(|(version, role)| GroupResources {
                    latest: version,
                    phase: GroupPhase::Active,
                    active: Some(ActiveGroup {
                        version,
                        role,
                        model: "test-model".into(),
                    }),
                })
                .collect(),
        };
        let mut input = ServingForecast {
            checkpoints: None,
            model: "test-model".into(),
            demand_rps: 90.0,
            now_ms: 10,
            horizon_ms: 10_000,
            minimum_gain_requests: 100.0,
            groups: vec![
                GroupProfile {
                    version: old_p,
                    capacity: profile(30.0),
                },
                GroupProfile {
                    version: old_d,
                    capacity: profile(90.0),
                },
            ],
            link_bytes_per_second: BTreeMap::from([("shared-ingress".into(), 1_600_000_000)]),
            paths: [old_p, new_p]
                .into_iter()
                .map(|prefill| KvPath {
                    prefill,
                    decode: old_d,
                    bytes_per_request: 32_000_000,
                    links: vec!["shared-ingress".into()],
                    capacity: profile(90.0),
                })
                .collect(),
        };
        let candidate = |pinned_bytes, ready_after_ms| ServingCandidate {
            group: new_p.group,
            expected_epoch: 0,
            plan: GroupPlan {
                role: ServingRole::Prefill,
                model: "test-model".into(),
                tensor_parallel: 1,
                pipeline_parallel: 1,
                launch_digest: [1; 32],
                workers: BTreeMap::from([(
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 64,
                    },
                )]),
                hosts: BTreeMap::from([(
                    host,
                    HostMemory {
                        bytes: 40,
                        pinned_bytes,
                    },
                )]),
            },
            capacity: profile(70.0),
            ready_after_ms,
            transition_loss_requests: 20.0,
        };
        let candidates = [candidate(24, 0), candidate(18, 8_000), candidate(18, 2_000)];
        let result = select_serving_candidate(&resources, &input, &candidates).unwrap();
        assert_eq!(result.baseline_rps, 30.0);
        assert_eq!(result.selected, Some(2));
        assert!(
            matches!(result.candidates[0], CandidateForecast::ResourceConflict(ServingError::HostCapacityExceeded(id)) if id == host)
        );
        assert!(matches!(
            result.candidates[1],
            CandidateForecast::Scored {
                steady_rps: 50.0,
                gain_requests: 20.0
            }
        ));
        assert!(matches!(
            result.candidates[2],
            CandidateForecast::Scored {
                steady_rps: 50.0,
                gain_requests: 140.0
            }
        ));

        input.paths[1].capacity.valid_until_ms = input.now_ms;
        assert_eq!(
            select_serving_candidate(&resources, &input, &candidates)
                .unwrap()
                .selected,
            None
        );
        input.groups[0].capacity.valid_until_ms = input.now_ms;
        assert!(select_serving_candidate(&resources, &input, &candidates).is_err());
    }
}
