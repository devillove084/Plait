use std::collections::{BTreeMap, BTreeSet};
use std::net::IpAddr;

use plait_core::NodeId;
use plait_core::serving::{
    GroupId, GroupVersion, HostedLinkSample, HostedPdKv, HostedPdPlacement, HostedPdPlanning,
    HostedPlanning, HostedPortUse, PdRoute, ServingResources, ServingRole,
};
use serde::{Deserialize, Serialize};

use super::super::{ServingForecast, SloProfile, validate_placement};
use super::pool::{HostedPool, generate_layouts};
use super::{HostedLinkPolicy, LinkEvidence, Traffic};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedPdPool {
    pub kv: HostedPdKv,
    kv_bytes_per_request: u64,
    /// Aggregate throughput of the store, counting reads and writes.
    store_bytes_per_second: u64,
    store_valid_until_ms: u64,
    slots: Vec<RoleSlot>,
    #[serde(default)]
    allow_scale_down: bool,
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
}

impl Serialize for HostedPdDecision {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // GroupPlan uses compound keys. Keep the published CLI representation
        // compatible with the array representation used by prepare_hosted_pd.
        #[derive(Serialize)]
        struct Placement<'a> {
            group: GroupId,
            role: ServingRole,
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
        }
        Decision {
            paths: &self.paths,
            placements: self
                .placements
                .iter()
                .map(|placement| Placement {
                    group: placement.group,
                    role: placement.plan.role,
                    workers: placement.plan.workers.keys().copied().collect(),
                    recipe: &placement.recipe,
                })
                .collect(),
            steady_rps: self.steady_rps,
            gain_requests: self.gain_requests,
            devices: self.devices,
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
    devices: u64,
}

#[derive(Clone, Default)]
struct Layout {
    endpoints: Vec<Endpoint>,
    placements: Vec<HostedPdPlacement>,
    roles: (usize, usize),
    ready_after_ms: u64,
    transition_loss_requests: f64,
    potential: f64,
}

const LAYOUTS_PER_SHAPE: usize = 8;
type Beam = BTreeMap<(usize, usize), Vec<Layout>>;

pub fn plan_hosted_pd(
    model: &str,
    snapshot: &HostedPdPlanning,
    forecast: &ServingForecast,
    pool: &HostedPdPool,
    policy: HostedLinkPolicy,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<HostedPdSelection, &'static str> {
    forecast.validate()?;
    policy.validate()?;
    pool.validate(model, forecast)?;
    if !snapshot.managed {
        return Err("hosted P/D planning requires a managed route directory");
    }
    let evidence = LinkEvidence {
        policy,
        now_ms: forecast.now_ms,
        links,
    };
    let store = IpAddr::V4(*pool.kv.store.ip());
    let matching: Vec<_> = snapshot
        .inventory
        .hosts
        .iter()
        .filter(|entry| entry.address == store)
        .map(|entry| entry.host)
        .collect();
    let [store_host] = matching.as_slice() else {
        return Err("the KV store address must match one registered host");
    };
    let store_host = *store_host;
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
        if let Some(&capacity) = profiles.get(&instance.version)
            && let Some(endpoint) = endpoint(
                instance.version,
                instance.role,
                capacity,
                &instance.ranks,
                evidence,
                store_host,
                pool.kv_bytes_per_request,
            )
        {
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
    let mut slots: Vec<_> = pool.slots.iter().collect();
    slots.sort_by_key(|slot| slot.role == ServingRole::Decode);
    let mut beam = Beam::from([((0, 0), vec![Layout::default()])]);
    for slot in slots {
        let group = slot.pool.group();
        let history = snapshot
            .resources
            .groups
            .iter()
            .find(|entry| entry.latest.group == group);
        let expected_active = history
            .and_then(|entry| entry.active.as_ref())
            .map(|entry| entry.version);
        if expected_active
            .is_some_and(|version| !snapshot.active.iter().any(|entry| entry.version == version))
        {
            return Err("hosted P/D slot is active in another deployment");
        }
        let mut next = beam.clone();
        for layout in beam.values().flatten() {
            if let Some(retained) = active.get(&group).filter(|entry| entry.role == slot.role) {
                let mut trial = layout.clone();
                trial.add(retained.clone());
                retain(&mut next, trial, forecast, pool);
            }
            let (resources, inventory) = available(snapshot, layout);
            for (candidate, recipe) in generate_layouts(
                &resources,
                &inventory,
                forecast,
                &slot.pool,
                policy,
                links,
                Some((slot.role, &pool.kv)),
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
                let mut ranks = BTreeMap::new();
                for device in candidate.plan.workers.keys() {
                    *ranks.entry(device.node).or_default() += 1;
                }
                let Some(endpoint) = endpoint(
                    version,
                    slot.role,
                    candidate.capacity,
                    &ranks,
                    evidence,
                    store_host,
                    pool.kv_bytes_per_request,
                ) else {
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

    let incumbent: BTreeSet<_> = snapshot.paths.iter().copied().collect();
    let current_devices: u64 = snapshot
        .active
        .iter()
        .flat_map(|entry| entry.ranks.values())
        .sum();
    for layout in beam
        .into_iter()
        .filter(|((p, d), _)| *p != 0 && *d != 0)
        .flat_map(|(_, layouts)| layouts)
    {
        let paths = layout.paths();
        if layout.placements.is_empty()
            && paths.iter().copied().collect::<BTreeSet<_>>() == incumbent
        {
            continue;
        }
        selection.layouts_evaluated += 1;
        let gain = (layout.potential - baseline_rps)
            * (forecast.horizon_ms - layout.ready_after_ms) as f64
            / 1000.0
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
                placements: layout.placements,
                steady_rps: layout.potential,
                gain_requests: gain,
                devices,
            });
        }
    }
    Ok(selection)
}

impl HostedPdPool {
    fn validate(&self, model: &str, forecast: &ServingForecast) -> Result<(), &'static str> {
        if model.is_empty()
            || self.kv_bytes_per_request == 0
            || self.store_bytes_per_second == 0
            || self.store_valid_until_ms <= forecast.now_ms
            || !forecast.paths.is_empty()
            || !forecast.link_bytes_per_second.is_empty()
        {
            return Err(
                "hosted P/D requires current store capacity and planner-generated link profiles",
            );
        }
        let mut groups = BTreeSet::new();
        for slot in &self.slots {
            slot.pool.validate(true)?;
            if slot.role == ServingRole::Aggregated
                || slot.pool.api_model() != model
                || !groups.insert(slot.pool.group())
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

fn endpoint(
    version: GroupVersion,
    role: ServingRole,
    capacity: SloProfile,
    ranks: &BTreeMap<NodeId, u64>,
    evidence: LinkEvidence<'_>,
    store: NodeId,
    kv_bytes: u64,
) -> Option<Endpoint> {
    let devices: u64 = ranks.values().sum();
    let mut traffic = evidence.tp_traffic(ranks).ok()?;
    for (&host, &rank_count) in ranks {
        let bytes = kv_bytes as f64 * rank_count as f64 / devices as f64;
        let (from, to) = if role == ServingRole::Prefill {
            (host, store)
        } else {
            (store, host)
        };
        evidence.add(&mut traffic, from, to, bytes).ok()?;
    }
    Some(Endpoint {
        version,
        role,
        capacity,
        traffic,
        devices,
    })
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
    if pool.store_valid_until_ms <= at {
        return None;
    }
    let mut result = forecast
        .demand_rps
        .min(pool.store_bytes_per_second as f64 / (2.0 * pool.kv_bytes_per_request as f64));
    let mut traffic = Traffic::new();
    for (endpoint, share) in endpoints {
        if !endpoint.capacity.is_valid(at) {
            return None;
        }
        result = result.min(endpoint.capacity.rps / share);
        traffic.include(&endpoint.traffic, share);
    }
    Some(result.min(traffic.rate_limit()))
}

impl Layout {
    fn add(&mut self, endpoint: Endpoint) {
        if endpoint.role == ServingRole::Prefill {
            self.roles.0 += 1;
        } else {
            self.roles.1 += 1;
        }
        self.endpoints.push(endpoint);
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
    let shares = layout.endpoints.iter().map(|entry| {
        let count = if entry.role == ServingRole::Prefill {
            layout.roles.0
        } else {
            layout.roles.1
        };
        (entry, 1.0 / count as f64)
    });
    let Some(at) = forecast.now_ms.checked_add(layout.ready_after_ms) else {
        return;
    };
    let Some(potential) = rate(forecast, pool, at, shares) else {
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

fn available(snapshot: &HostedPdPlanning, layout: &Layout) -> (ServingResources, HostedPlanning) {
    let mut resources = snapshot.resources.clone();
    let mut inventory = snapshot.inventory.clone();
    let mut used = BTreeSet::new();
    for placement in &layout.placements {
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
    use plait_core::DeviceId;
    use plait_core::serving::{HostMemory, HostResources, HostedInventory, WorkerResources};

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
        };
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: 10,
            max_sample_age_ms: 50,
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
        let source = endpoint(
            GroupVersion {
                group: GroupId(a),
                epoch: 1,
            },
            ServingRole::Prefill,
            SloProfile {
                rps: 100.0,
                valid_until_ms: 1100,
            },
            &BTreeMap::from([(a, 1), (b, 1)]),
            evidence,
            store,
            200,
        )
        .unwrap();
        assert_eq!(source.traffic.rate_limit(), 5.0);
        let mut shared = source.traffic.clone();
        shared.include(&source.traffic, 1.0);
        assert_eq!(shared.rate_limit(), 2.5);
    }
}
