//! Select a versioned aggregate, split, or mixed route directory.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::NodeId;
use plait_core::serving::{
    GroupId, GroupVersion, HostedAssignment, HostedLinkSample, HostedRoute, HostedServingPlan,
    HostedServingPlanning, ServingAction, ServingRole,
};
use serde::Deserialize;

use super::super::super::valid_rate;
use super::{
    Endpoint, EndpointContext, HostedLinkPolicy, HostedPdPool, HostedPool, LinkEvidence, Search,
    ServingForecast, rate,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostedModePolicy {
    aggregate_slots: Vec<HostedPool>,
    /// Value of reclaiming one device-second, expressed in requests.
    device_cost_requests_per_second: f64,
    mode_change_loss_requests: f64,
    /// Allowance for old workers to drain after the new directory publishes.
    drain_ms: u64,
}

impl HostedModePolicy {
    pub fn validate(&self, model: &str) -> Result<(), &'static str> {
        if self.aggregate_slots.is_empty()
            || !valid_rate(self.device_cost_requests_per_second)
            || !valid_rate(self.mode_change_loss_requests)
        {
            return Err("invalid hosted serving mode policy");
        }
        let mut groups = BTreeSet::new();
        for slot in &self.aggregate_slots {
            slot.validate(true)?;
            if slot.api_model() != model || !groups.insert(slot.group()) {
                return Err("aggregate slots must be unique and use the deployment alias");
            }
        }
        Ok(())
    }
}

pub fn plan_hosted_serving(
    model: &str,
    snapshot: &HostedServingPlanning,
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    modes: &HostedModePolicy,
    policy: HostedLinkPolicy,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<Option<ServingAction>, &'static str> {
    forecast.validate()?;
    policy.validate()?;
    modes.validate(model)?;
    // Validate the contract. Fresh store evidence is required only for split traffic.
    pool.validate(model, 0)?;
    if !snapshot.managed
        || !forecast.paths.is_empty()
        || !forecast.link_bytes_per_second.is_empty()
        || !forecast.groups.is_empty()
    {
        return Err("mode planning requires a managed directory and calibrated pool profiles");
    }
    if snapshot.split.pending.is_some() {
        return Ok(None);
    }

    let context = EndpointContext {
        evidence: LinkEvidence {
            policy,
            now_ms: forecast.now_ms,
            links,
        },
        store_host: pool.store_host(&snapshot.split.inventory).ok(),
        kv_bytes: pool.kv_bytes_per_request,
    };
    let mut active = BTreeMap::new();
    let mut routes: BTreeSet<_> = snapshot
        .split
        .paths
        .iter()
        .copied()
        .map(HostedRoute::Split)
        .collect();
    let mut ranks: BTreeMap<GroupVersion, BTreeMap<NodeId, u64>> = BTreeMap::new();
    for worker in &snapshot.split.resources.workers {
        if let Some(grant) = worker
            .grant
            .filter(|grant| snapshot.assignments.contains_key(&grant.version))
        {
            *ranks
                .entry(grant.version)
                .or_default()
                .entry(worker.device.node)
                .or_default() += 1;
        }
    }
    for &route in &snapshot.committed {
        let HostedRoute::Aggregated(version) = route else {
            continue;
        };
        let Some(assignment) = snapshot.assignments.get(&version) else {
            continue;
        };
        if assignment.model != forecast.model {
            return Err("active aggregate serving groups require the calibrated checkpoint");
        }
        let endpoint = modes
            .aggregate_slots
            .iter()
            .find(|slot| slot.group() == version.group)
            .and_then(|slot| {
                endpoint(
                    assignment,
                    ServingRole::Aggregated,
                    ranks.get(&version)?,
                    slot,
                    context,
                )
            })
            .ok_or("active aggregate route lacks current capacity or directed links")?;
        active.insert(version.group, endpoint);
        routes.insert(route);
    }
    for instance in &snapshot.split.active {
        if instance.model != forecast.model || instance.kv != pool.kv {
            return Err(
                "active split serving groups require the calibrated checkpoint and KV contract",
            );
        }
        let assignment = snapshot
            .assignments
            .get(&instance.version)
            .ok_or("active split serving group lacks its assignment")?;
        let slot = pool
            .slot(instance.version.group, instance.role)
            .ok_or("active split serving group lacks its pool slot")?;
        let endpoint = endpoint(
            assignment,
            instance.role,
            &instance.ranks,
            &slot.pool,
            context,
        )
        .ok_or("active split route lacks current capacity or directed links")?;
        active.insert(instance.version.group, endpoint);
    }

    let baseline = route_rate(forecast, pool, &routes, &active)
        .ok_or("active hosted routes require current SLO capacity")?;
    let mode = shape(&routes);
    let search = Search {
        forecast,
        pool,
        context,
        aggregate: &modes.aggregate_slots,
    };
    let mut best: Option<(f64, u64, HostedServingPlan)> = None;
    for layout in search.layouts(&snapshot.split, &active)? {
        let targets: BTreeSet<_> = layout.routes().into_iter().collect();
        if targets == routes || layout.potential < baseline {
            continue;
        }
        let moves = layout.ready_after_ms;
        let new_versions: BTreeSet<_> = targets.iter().flat_map(|route| route.groups()).collect();
        let removed: u64 = active
            .values()
            .filter(|group| !new_versions.contains(&group.version))
            .map(|group| group.devices)
            .sum();
        let added: u64 = layout
            .placements
            .iter()
            .map(|group| group.plan.workers.len() as u64)
            .sum();
        let saved_device_seconds = (removed as f64
            * forecast
                .horizon_ms
                .saturating_sub(moves.saturating_add(modes.drain_ms)) as f64
            - added as f64 * forecast.horizon_ms as f64)
            / 1000.0;
        let mut gain = (layout.potential - baseline) * (forecast.horizon_ms - moves) as f64
            / 1000.0
            + saved_device_seconds * modes.device_cost_requests_per_second
            - layout.transition_loss_requests;
        if !routes.is_empty() && mode != shape(&targets) {
            gain -= modes.mode_change_loss_requests;
        }
        let devices = layout.devices();
        if gain.is_finite()
            && gain > 0.0
            && gain >= forecast.minimum_gain_requests
            && best.as_ref().is_none_or(|(score, count, _)| {
                gain > *score || (gain == *score && devices < *count)
            })
        {
            best = Some((
                gain,
                devices,
                HostedServingPlan {
                    kv: layout.roles.has_split().then(|| pool.kv.clone()),
                    placements: layout.placements,
                    routes: targets.into_iter().collect(),
                },
            ));
        }
    }
    Ok(
        best.map(|(_, _, plan)| ServingAction::PrepareHostedServing {
            model: model.to_owned(),
            expected_revision: snapshot.split.revision,
            plan,
        }),
    )
}

fn endpoint(
    assignment: &HostedAssignment,
    role: ServingRole,
    ranks: &BTreeMap<NodeId, u64>,
    pool: &HostedPool,
    context: EndpointContext<'_>,
) -> Option<Endpoint> {
    let workers: u64 = ranks.values().sum();
    let tp = assignment.tensor_parallel;
    let name = &assignment.recipe.profile;
    if tp == 0 || workers == 0 || workers % u64::from(tp) != 0 {
        return None;
    }
    let pp = u32::try_from(workers / u64::from(tp)).ok()?;
    let api = assignment.recipe.api_host;
    let (capacity, pipeline) = if pp > 1 {
        let profile = pool.pipeline(name, tp, pp)?;
        let capacity = profile.capacity(&profile.lanes(api, ranks)?);
        (capacity, Some((profile, api)))
    } else {
        (pool.capacity(name, tp)?, None)
    };
    context.endpoint(assignment.version, role, capacity, ranks, pipeline)
}

fn route_rate(
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    routes: &BTreeSet<HostedRoute>,
    active: &BTreeMap<GroupId, Endpoint>,
) -> Option<f64> {
    if routes.is_empty() {
        return Some(0.0);
    }
    let mut uses = BTreeMap::<GroupVersion, usize>::new();
    for version in routes.iter().flat_map(|route| route.groups()) {
        *uses.entry(version).or_default() += 1;
    }
    let shares = uses
        .into_iter()
        .map(|(version, count)| {
            let group = active
                .get(&version.group)
                .filter(|group| group.version == version)?;
            Some((group, count as f64 / routes.len() as f64))
        })
        .collect::<Option<Vec<_>>>()?;
    rate(forecast, pool, forecast.now_ms, shares)
}

fn shape(routes: &BTreeSet<HostedRoute>) -> (bool, bool) {
    (
        routes
            .iter()
            .any(|route| matches!(route, HostedRoute::Aggregated(_))),
        routes
            .iter()
            .any(|route| matches!(route, HostedRoute::Split(_))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::SloProfile;
    use crate::serving::hosted::pd::{Layout, RoleSlot, Traffic};
    use plait_core::DeviceId;
    use plait_core::serving::{
        ActiveGroup, GroupPhase, GroupResources, HostMemory, HostResources, HostedAddress,
        HostedInventory, HostedPdInstance, HostedPdKv, HostedPdPlanning, HostedPlanning,
        HostedRecipe, PdRoute, ServingResources, WorkerGrant, WorkerResources,
    };

    fn node(id: u8) -> NodeId {
        NodeId::from_bytes([id; 16])
    }

    fn addr(host: NodeId) -> std::net::IpAddr {
        std::net::Ipv4Addr::new(192, 0, 2, host.as_bytes()[0]).into()
    }

    fn assignment(version: GroupVersion, host: NodeId, profile: &str) -> HostedAssignment {
        HostedAssignment {
            version,
            model: "checkpoint".into(),
            tensor_parallel: 1,
            phase: GroupPhase::Active,
            holder: None,
            recipe: HostedRecipe {
                profile: profile.into(),
                api_host: host,
                hosts: vec![HostedAddress {
                    host,
                    address: addr(host),
                }],
                master_port: 29000,
                http_port: 29001,
                backend_port: 29002,
                api_model: "model".into(),
                max_model_len: 64,
                gpu_memory_bps: 6000,
                prefix_caching: false,
            },
        }
    }

    fn fixture() -> (HostedServingPlanning, HostedPdPool, HostedModePolicy) {
        let groups = [10, 11, 12].map(|i| GroupId(node(i)));
        let hosts = [1, 2, 3].map(node);
        let memory = HostMemory {
            bytes: 64,
            pinned_bytes: 8,
        };
        let resources = ServingResources {
            groups: Vec::new(),
            hosts: hosts
                .iter()
                .map(|&host| HostResources {
                    host,
                    capacity: memory,
                    reserved: HostMemory::default(),
                })
                .collect(),
            workers: hosts
                .iter()
                .map(|&host| WorkerResources {
                    device: DeviceId::new(host, 0),
                    incarnation: 1,
                    device_bytes: 64,
                    grant: None,
                })
                .collect(),
        };
        let inventory = HostedPlanning {
            hosts: hosts
                .into_iter()
                .zip(["aggregate", "prefill", "decode"])
                .map(|(host, profile)| HostedInventory {
                    host,
                    address: addr(host),
                    profiles: BTreeSet::from([profile.into()]),
                    devices: BTreeMap::from([(0, 1)]),
                })
                .collect(),
            occupied: Vec::new(),
        };
        let snapshot = HostedServingPlanning {
            managed: true,
            split: HostedPdPlanning {
                revision: 1,
                managed: true,
                pending: None,
                active: Vec::new(),
                paths: Vec::new(),
                resources,
                inventory,
            },
            committed: Vec::new(),
            assignments: BTreeMap::new(),
        };
        let slot = |group, role, profile| RoleSlot {
            role,
            pool: HostedPool::fixture(group, profile, "model").with_capacity(1, 20.0),
        };
        let kv = HostedPdKv {
            store: "192.0.2.1:26000".parse().unwrap(),
            namespace: "kv".into(),
            block_size: 128,
        };
        let pool = HostedPdPool {
            kv,
            kv_bytes_per_request: 100,
            store_bytes_per_second: 1600,
            store_valid_until_ms: 5000,
            slots: vec![
                slot(groups[1], ServingRole::Prefill, "prefill"),
                slot(groups[2], ServingRole::Decode, "decode"),
            ],
            allow_scale_down: false,
            reconfigure: None,
        };
        let modes = HostedModePolicy {
            aggregate_slots: vec![
                HostedPool::fixture(groups[0], "aggregate", "model").with_capacity(1, 2.0),
            ],
            device_cost_requests_per_second: 0.25,
            mode_change_loss_requests: 0.1,
            drain_ms: 100,
        };
        (snapshot, pool, modes)
    }

    fn activate(
        snapshot: &mut HostedServingPlanning,
        group: GroupId,
        role: ServingRole,
        host: NodeId,
        profile: &str,
    ) -> GroupVersion {
        let version = GroupVersion { group, epoch: 1 };
        let device = DeviceId::new(host, 0);
        snapshot
            .split
            .resources
            .workers
            .iter_mut()
            .find(|worker| worker.device == device)
            .unwrap()
            .grant = Some(WorkerGrant {
            device,
            version,
            incarnation: 1,
            fence: 1,
        });
        snapshot.split.resources.groups.push(GroupResources {
            latest: version,
            phase: GroupPhase::Active,
            active: Some(ActiveGroup {
                version,
                role,
                model: "checkpoint".into(),
            }),
        });
        snapshot
            .assignments
            .insert(version, assignment(version, host, profile));
        version
    }

    fn forecast(demand_rps: f64) -> ServingForecast {
        ServingForecast {
            model: "checkpoint".into(),
            now_ms: 1000,
            demand_rps,
            horizon_ms: 10_000,
            minimum_gain_requests: 0.0,
            groups: Vec::new(),
            paths: Vec::new(),
            link_bytes_per_second: BTreeMap::new(),
        }
    }

    fn sample(target: NodeId, bytes_per_second: u64) -> HostedLinkSample {
        HostedLinkSample {
            target,
            target_session: target,
            sampled_at_ms: 995,
            bytes_per_second,
            rtt_ns: 10,
        }
    }

    #[test]
    fn selects_split_for_throughput_and_aggregate_after_load_drops() {
        let (mut aggregate, pool, mut modes) = fixture();
        let mut split = aggregate.clone();
        let [a, p, d] = [1, 2, 3].map(node);
        let agg = activate(
            &mut aggregate,
            modes.aggregate_slots[0].group(),
            ServingRole::Aggregated,
            a,
            "aggregate",
        );
        aggregate.committed.push(HostedRoute::Aggregated(agg));
        let prefill = activate(
            &mut split,
            pool.slots[0].pool.group(),
            ServingRole::Prefill,
            p,
            "prefill",
        );
        let decode = activate(
            &mut split,
            pool.slots[1].pool.group(),
            ServingRole::Decode,
            d,
            "decode",
        );
        let path = PdRoute { prefill, decode };
        split.committed.push(HostedRoute::Split(path));
        split.split.paths.push(path);
        for (version, role, host) in [
            (prefill, ServingRole::Prefill, p),
            (decode, ServingRole::Decode, d),
        ] {
            split.split.active.push(HostedPdInstance {
                version,
                role,
                model: "checkpoint".into(),
                kv: pool.kv.clone(),
                ranks: BTreeMap::from([(host, 1)]),
            });
        }
        let policy = HostedLinkPolicy {
            max_sample_age_ms: 40,
            bytes_per_rank_pair: 100,
        };
        let links = BTreeMap::from([((p, a), sample(a, 10_000)), ((a, d), sample(d, 10_000))]);
        let plan = |snap: &HostedServingPlanning,
                    demand,
                    modes: &HostedModePolicy,
                    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>| {
            plan_hosted_serving(
                "model",
                snap,
                &forecast(demand),
                &pool,
                modes,
                policy,
                links,
            )
            .unwrap()
        };
        let Some(ServingAction::PrepareHostedServing { plan: high, .. }) =
            plan(&aggregate, 8.0, &modes, &links)
        else {
            panic!("expected split capacity");
        };
        assert!(matches!(high.routes.as_slice(), [HostedRoute::Split(_)]));
        assert_eq!(high.placements.len(), 2);
        assert!(plan(&aggregate, 1.0, &modes, &links).is_none());
        assert!(plan(&aggregate, 8.0, &modes, &BTreeMap::new()).is_none());

        let Some(ServingAction::PrepareHostedServing { plan: low, .. }) =
            plan(&split, 1.0, &modes, &links)
        else {
            panic!("expected device reclaim through aggregation");
        };
        assert!(matches!(
            low.routes.as_slice(),
            [HostedRoute::Aggregated(_)]
        ));
        assert_eq!(low.placements.len(), 1);
        assert!(low.kv.is_none());
        assert!(plan(&split, 8.0, &modes, &links).is_none());

        modes.drain_ms = 9999;
        assert!(plan(&split, 1.0, &modes, &links).is_none());
    }

    #[test]
    fn mixed_round_robin_accounts_for_store_share_and_combined_nic_demand() {
        let (_, pool, _) = fixture();
        let [a, b] = [1, 2].map(node);
        let link = BTreeMap::from([((a, b), sample(b, 1000))]);
        let policy = HostedLinkPolicy {
            max_sample_age_ms: 40,
            bytes_per_rank_pair: 100,
        };
        let evidence = LinkEvidence {
            policy,
            now_ms: 1000,
            links: &link,
        };
        let load = forecast(50.0);
        let make = |i, role, bytes| {
            let mut traffic = Traffic::new();
            if bytes > 0.0 {
                evidence.add(&mut traffic, a, b, bytes).unwrap();
            }
            Endpoint {
                version: GroupVersion {
                    group: GroupId(node(i)),
                    epoch: 1,
                },
                role,
                traffic,
                capacity: SloProfile {
                    rps: 40.0,
                    valid_until_ms: 5000,
                },
                devices: 1,
            }
        };
        let mut layout = Layout::default();
        layout.add(make(11, ServingRole::Prefill, 100.0));
        layout.add(make(12, ServingRole::Decode, 0.0));
        assert_eq!(rate(&load, &pool, 1000, layout.shares()), Some(8.0));
        layout.add(make(10, ServingRole::Aggregated, 100.0));
        // Only half the requests use the store, but the aggregate and Prefill
        // consume the same outbound NIC.
        assert_eq!(rate(&load, &pool, 1000, layout.shares()), Some(10.0));
    }

    #[test]
    fn retains_an_aggregate_alongside_a_split_path_when_they_can_serve_more_together() {
        let (mut snapshot, pool, mut modes) = fixture();
        let [a, p, d] = [1, 2, 3].map(node);
        let group = modes.aggregate_slots[0].group();
        modes.aggregate_slots[0] =
            HostedPool::fixture(group, "aggregate", "model").with_capacity(1, 6.0);
        let live = HostedRoute::Aggregated(activate(
            &mut snapshot,
            group,
            ServingRole::Aggregated,
            a,
            "aggregate",
        ));
        snapshot.committed.push(live);
        let policy = HostedLinkPolicy {
            max_sample_age_ms: 40,
            bytes_per_rank_pair: 100,
        };
        let links = BTreeMap::from([((p, a), sample(a, 10_000)), ((a, d), sample(d, 10_000))]);
        let Some(ServingAction::PrepareHostedServing { plan, .. }) = plan_hosted_serving(
            "model",
            &snapshot,
            &forecast(12.0),
            &pool,
            &modes,
            policy,
            &links,
        )
        .unwrap() else {
            panic!("expected a mixed plan");
        };
        assert!(
            matches!(plan.routes.as_slice(), [retained, HostedRoute::Split(_)] if *retained == live)
        );
        assert_eq!(plan.placements.len(), 2);
        assert!(plan.placements.iter().all(|item| item.group != group));
    }
}
