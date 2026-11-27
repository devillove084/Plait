//! Topology-aware placement of native execution groups.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::group::{
    ComputeRate, GroupId, GroupPhase, GroupVersion, HostMemory, PlacementHost, PlacementInventory,
    PlacementWorker, ResourcePlan, WorkerDemand,
};
use plait_core::{ComputeRequirement, DeviceId, NodeId};
use serde::{Deserialize, Serialize};

const BEAM_PER_WIDTH: usize = 24;
const NANOS: u128 = 1_000_000_000;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub widths: Vec<u32>,
    #[serde(default)]
    pub requirement: ComputeRequirement,
    pub device_bytes: u64,
    pub per_worker_host: HostMemory,
    pub per_host: HostMemory,
    pub compute_work: u64,
    pub bytes_per_peer: u64,
    pub max_link_age_ms: u64,
    pub operations: u64,
    pub switch_ns: u64,
    #[serde(default)]
    pub minimum_gain_ns: u64,
    #[serde(default)]
    pub recipes: Vec<Recipe>,
}

/// Caller-defined comparable work for one operation, in stable device order.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub id: NodeId,
    pub work: Vec<u64>,
    pub compute: ComputeOrder,
    pub communication: Communication,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeOrder {
    Parallel,
    Sequential,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Communication {
    AllToAll { rounds: u64, bytes_per_peer: u64 },
    Directed { edges: Vec<Exchange> },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exchange {
    pub from: usize,
    pub to: usize,
    pub rounds: u64,
    pub bytes: u64,
}

impl Workload {
    pub fn needs_links(&self) -> bool {
        self.bytes_per_peer > 0
            || self
                .recipes
                .iter()
                .any(|recipe| match &recipe.communication {
                    Communication::AllToAll { rounds, .. } => *rounds != 0,
                    Communication::Directed { edges } => !edges.is_empty(),
                })
    }

    pub fn validate(&self, calibrated: bool) -> Result<(), &'static str> {
        if self.widths.is_empty()
            || self.widths.contains(&0)
            || self.device_bytes == 0
            || self.operations == 0
            || self.per_worker_host.pinned_bytes > self.per_worker_host.bytes
            || self.per_host.pinned_bytes > self.per_host.bytes
            || (self.compute_work == 0 && (self.bytes_per_peer == 0 || calibrated))
        {
            return Err("invalid group placement workload");
        }
        let mut recipes = BTreeSet::new();
        for recipe in &self.recipes {
            let width = recipe.work.len();
            if !recipes.insert(recipe.id)
                || !self.widths.iter().any(|&item| item as usize == width)
                || recipe.work.iter().all(|&work| work == 0)
                || matches!(
                    recipe.communication,
                    Communication::AllToAll {
                        rounds: 0,
                        bytes_per_peer: 1..
                    }
                )
                || matches!(&recipe.communication, Communication::Directed { edges } if edges.iter().any(|edge|
                    edge.from >= width || edge.to >= width || edge.from == edge.to || edge.rounds == 0))
            {
                return Err("invalid group placement recipe");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DirectedLink {
    pub source: NodeId,
    pub target: NodeId,
    pub sampled_at_ms: u64,
    pub bytes_per_second: u64,
    pub rtt_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Cost {
    pub compute_ns: u64,
    pub network_ns: u64,
}

impl Cost {
    fn operation_ns(self) -> u128 {
        u128::from(self.compute_ns) + u128::from(self.network_ns)
    }
}

#[derive(Debug)]
pub enum Decision {
    Pending(GroupVersion),
    NoPlacement,
    AwaitLinks(GroupVersion),
    AwaitComputeRates(GroupVersion),
    Keep {
        version: GroupVersion,
        cost: Cost,
    },
    Prepare {
        resources: ResourcePlan,
        cost: Cost,
        incumbent: Option<Cost>,
        recipe: Option<NodeId>,
    },
}

pub struct PlacementGroup<'a> {
    pub inventory: &'a PlacementInventory,
    pub workload: &'a Workload,
    pub rates: Option<&'a [ComputeRate]>,
}

#[derive(Clone)]
pub struct PreparationChoice {
    pub resources: ResourcePlan,
    pub cost: Cost,
    pub recipe: Option<NodeId>,
}

#[derive(Clone, Default)]
struct JointCandidate {
    choices: Vec<(usize, PreparationChoice)>,
    devices: BTreeSet<DeviceId>,
    hosts: BTreeMap<NodeId, HostMemory>,
    cost: u128,
}

/// Place a caller-selected set of groups against shared free resources.
/// Candidates are re-enumerated after reserving each preceding group.
pub fn select_set(
    groups: &[PlacementGroup<'_>],
    links: &[DirectedLink],
    now_ms: u64,
) -> Result<Option<Vec<PreparationChoice>>, &'static str> {
    let Some(first) = groups.first() else {
        return Err("empty group placement set");
    };
    let mut names = BTreeSet::<GroupId>::new();
    let mut available = free_devices(first.inventory);
    let mut hosts: BTreeMap<_, _> = first
        .inventory
        .hosts
        .iter()
        .filter(|host| host.eligible)
        .map(|host| (host.host, host_headroom(host)))
        .collect();
    let mut models = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        if !names.insert(group.inventory.group) {
            return Err("duplicate group in placement set");
        }
        let model = Model::new(group.workload, links, now_ms, group.rates)?;
        if index != 0 {
            let free = free_devices(group.inventory);
            available.retain(|id, incarnation| free.get(id) == Some(incarnation));
            let budgets: BTreeMap<_, _> = group
                .inventory
                .hosts
                .iter()
                .filter(|host| host.eligible)
                .map(|host| (host.host, host_headroom(host)))
                .collect();
            hosts.retain(|id, budget| {
                let Some(other) = budgets.get(id) else {
                    return false;
                };
                budget.bytes = budget.bytes.min(other.bytes);
                budget.pinned_bytes = budget.pinned_bytes.min(other.pinned_bytes);
                true
            });
        }
        models.push(model);
    }
    if groups.iter().any(|group| {
        matches!(
            group.inventory.latest,
            Some((_, GroupPhase::Preparing | GroupPhase::Ready))
        )
    }) {
        return Ok(None);
    }

    let mut queue: Vec<_> = groups
        .iter()
        .zip(models)
        .enumerate()
        .map(|(index, (group, model))| {
            let eligible: Vec<_> = group
                .inventory
                .workers
                .iter()
                .filter(|entry| {
                    available.contains_key(&entry.device) && hosts.contains_key(&entry.device.node)
                })
                .filter_map(|entry| {
                    model
                        .eligible(entry)
                        .filter(|worker| worker.compute_rate.is_some())
                })
                .collect();
            let minimum = *group
                .workload
                .widths
                .iter()
                .min()
                .expect("validated widths") as usize;
            (
                eligible.len().saturating_sub(minimum),
                index,
                model,
                eligible,
            )
        })
        .collect();
    queue.sort_unstable_by_key(|&(slack, index, _, _)| (slack, index));

    let mut candidates = vec![JointCandidate::default()];
    for (_, index, model, eligible) in queue {
        let mut next = Vec::new();
        for partial in candidates {
            let mut workers: BTreeMap<NodeId, Vec<Worker>> = BTreeMap::new();
            for worker in eligible
                .iter()
                .filter(|worker| !partial.devices.contains(&worker.id))
            {
                workers.entry(worker.id.node).or_default().push(*worker);
            }
            let inventory: Vec<_> = hosts
                .iter()
                .map(|(&host, &capacity)| PlacementHost {
                    host,
                    capacity,
                    reserved: partial.hosts.get(&host).copied().unwrap_or_default(),
                    eligible: true,
                })
                .collect();
            for (candidate, recipe) in model.candidates(&inventory, &workers, true) {
                let resources = model.resources(&candidate);
                let mut joint = partial.clone();
                joint.devices.extend(resources.workers.keys().copied());
                for (&host, &demand) in &resources.hosts {
                    let reserved = joint.hosts.entry(host).or_default();
                    // The candidate is fitted against this partial reservation.
                    reserved.bytes += demand.bytes;
                    reserved.pinned_bytes += demand.pinned_bytes;
                }
                joint.cost = joint.cost.saturating_add(
                    candidate
                        .cost
                        .operation_ns()
                        .saturating_mul(u128::from(model.workload.operations)),
                );
                joint.choices.push((
                    index,
                    PreparationChoice {
                        resources,
                        cost: candidate.cost,
                        recipe,
                    },
                ));
                let position =
                    next.partition_point(|prior: &JointCandidate| prior.cost <= joint.cost);
                if position < BEAM_PER_WIDTH {
                    next.insert(position, joint);
                    next.truncate(BEAM_PER_WIDTH);
                }
            }
        }
        if next.is_empty() {
            return Ok(None);
        }
        candidates = next;
    }
    Ok(candidates.into_iter().next().map(|mut candidate| {
        candidate.choices.sort_unstable_by_key(|(index, _)| *index);
        candidate
            .choices
            .into_iter()
            .map(|(_, choice)| choice)
            .collect()
    }))
}

fn free_devices(inventory: &PlacementInventory) -> BTreeMap<DeviceId, u64> {
    inventory
        .workers
        .iter()
        .filter(|entry| entry.eligible && entry.grant.is_none())
        .map(|entry| (entry.device, entry.incarnation))
        .collect()
}

fn host_headroom(entry: &PlacementHost) -> HostMemory {
    HostMemory {
        bytes: entry.capacity.bytes.saturating_sub(entry.reserved.bytes),
        pinned_bytes: entry
            .capacity
            .pinned_bytes
            .saturating_sub(entry.reserved.pinned_bytes),
    }
}

#[derive(Clone, Copy)]
struct Worker {
    id: DeviceId,
    incarnation: u64,
    compute_rate: Option<u64>,
}

#[derive(Clone)]
struct Candidate {
    workers: Vec<Worker>,
    cost: Cost,
}

struct Model<'a> {
    workload: &'a Workload,
    links: BTreeMap<(NodeId, NodeId), DirectedLink>,
    rates: Option<BTreeMap<DeviceId, ComputeRate>>,
}

pub fn select(
    inventory: &PlacementInventory,
    workload: &Workload,
    links: &[DirectedLink],
    now_ms: u64,
) -> Result<Decision, &'static str> {
    select_with_rates(inventory, workload, links, now_ms, None)
}

/// When rates are supplied, use a single calibration for the whole comparison.
pub fn select_with_rates(
    inventory: &PlacementInventory,
    workload: &Workload,
    links: &[DirectedLink],
    now_ms: u64,
    rates: Option<&[ComputeRate]>,
) -> Result<Decision, &'static str> {
    select_with_recipes(inventory, workload, links, now_ms, rates, None)
}

/// The incumbent uses its committed recipe even when another recipe would be
/// cheaper on the same devices.
pub fn select_with_recipes(
    inventory: &PlacementInventory,
    workload: &Workload,
    links: &[DirectedLink],
    now_ms: u64,
    rates: Option<&[ComputeRate]>,
    active_recipe: Option<NodeId>,
) -> Result<Decision, &'static str> {
    let widths: BTreeSet<_> = workload
        .widths
        .iter()
        .map(|&width| width as usize)
        .collect();
    let model = Model::new(workload, links, now_ms, rates)?;
    if let Some((version, GroupPhase::Preparing | GroupPhase::Ready)) = inventory.latest {
        return Ok(Decision::Pending(version));
    }

    let mut eligible = BTreeMap::new();
    for entry in &inventory.workers {
        if let Some(worker) = model.eligible(entry) {
            eligible.insert(worker.id, (worker, entry));
        }
    }
    let incumbent = if let Some((version, workers)) = model.incumbent(inventory, &eligible, &widths)
    {
        if workers.iter().any(|worker| worker.compute_rate.is_none()) {
            return Ok(Decision::AwaitComputeRates(version));
        }
        let recipe = active_recipe
            .map(|id| {
                workload
                    .recipes
                    .iter()
                    .find(|profile| profile.id == id && profile.work.len() == workers.len())
                    .ok_or("active group recipe is missing from the workload policy")
            })
            .transpose()?;
        let Some(cost) = model.cost(&workers, recipe) else {
            return Ok(Decision::AwaitLinks(version));
        };
        Some((version, cost))
    } else {
        None
    };
    let mut available: BTreeMap<NodeId, Vec<Worker>> = BTreeMap::new();
    for (worker, entry) in eligible.values() {
        if entry.grant.is_none() && worker.compute_rate.is_some() {
            available.entry(worker.id.node).or_default().push(*worker);
        }
    }

    let best = model
        .candidates(&inventory.hosts, &available, false)
        .into_iter()
        .next();

    let incumbent_cost = incumbent.map(|(_, cost)| cost);
    if let Some((version, cost)) = incumbent {
        let operations = u128::from(workload.operations);
        let ready_cost = cost.operation_ns().saturating_mul(operations);
        let use_new = best.as_ref().is_some_and(|(candidate, _)| {
            let total = candidate
                .cost
                .operation_ns()
                .saturating_mul(operations)
                .saturating_add(
                    u128::from(workload.switch_ns) + u128::from(workload.minimum_gain_ns),
                );
            total < ready_cost
        });
        if !use_new {
            return Ok(Decision::Keep { version, cost });
        }
    }
    let Some((best, recipe)) = best else {
        return Ok(Decision::NoPlacement);
    };
    let resources = model.resources(&best);
    Ok(Decision::Prepare {
        resources,
        cost: best.cost,
        incumbent: incumbent_cost,
        recipe,
    })
}

impl<'a> Model<'a> {
    fn new(
        workload: &'a Workload,
        links: &[DirectedLink],
        now_ms: u64,
        rates: Option<&[ComputeRate]>,
    ) -> Result<Self, &'static str> {
        workload.validate(rates.is_some())?;
        let mut fresh = BTreeMap::new();
        for link in links {
            if link.source == link.target || link.bytes_per_second == 0 || link.rtt_ns == 0 {
                return Err("invalid directed link sample");
            }
            if now_ms
                .checked_sub(link.sampled_at_ms)
                .is_some_and(|age| age <= workload.max_link_age_ms)
                && fresh.insert((link.source, link.target), *link).is_some()
            {
                return Err("duplicate fresh directed link sample");
            }
        }
        let rates = rates
            .map(|rates| {
                let by_device: BTreeMap<_, _> =
                    rates.iter().map(|rate| (rate.device, *rate)).collect();
                if by_device.len() != rates.len() {
                    return Err("duplicate compute rate");
                }
                Ok(by_device)
            })
            .transpose()?;
        Ok(Self {
            workload,
            links: fresh,
            rates,
        })
    }

    fn resources(&self, candidate: &Candidate) -> ResourcePlan {
        let hosts = count_hosts(&candidate.workers)
            .into_iter()
            .map(|(host, count)| (host, self.host_demand(count).expect("fitted host")))
            .collect();
        let workers = candidate
            .workers
            .iter()
            .map(|worker| {
                (
                    worker.id,
                    WorkerDemand {
                        incarnation: worker.incarnation,
                        device_bytes: self.workload.device_bytes,
                    },
                )
            })
            .collect();
        ResourcePlan { workers, hosts }
    }

    fn candidates(
        &self,
        hosts: &[PlacementHost],
        available: &BTreeMap<NodeId, Vec<Worker>>,
        alternatives: bool,
    ) -> Vec<(Candidate, Option<NodeId>)> {
        let mut candidates: Vec<_> = if self.workload.recipes.is_empty() {
            let widths = self
                .workload
                .widths
                .iter()
                .map(|&width| width as usize)
                .collect();
            self.search_candidates(hosts, available, &widths, None, alternatives)
                .into_iter()
                .map(|candidate| (candidate, None))
                .collect()
        } else {
            self.workload
                .recipes
                .iter()
                .flat_map(|recipe| {
                    let width = BTreeSet::from([recipe.work.len()]);
                    self.search_candidates(hosts, available, &width, Some(recipe), alternatives)
                        .into_iter()
                        .map(|candidate| (candidate, Some(recipe.id)))
                })
                .collect()
        };
        candidates.sort_by(|(left, left_recipe), (right, right_recipe)| {
            compare(left, right).then_with(|| left_recipe.cmp(right_recipe))
        });
        if alternatives && !self.workload.recipes.is_empty() {
            let mut seen = BTreeSet::new();
            candidates.retain(|(candidate, _)| {
                seen.insert(
                    candidate
                        .workers
                        .iter()
                        .map(|worker| worker.id)
                        .collect::<Vec<_>>(),
                )
            });
        }
        candidates
    }

    fn search_candidates(
        &self,
        hosts: &[PlacementHost],
        available: &BTreeMap<NodeId, Vec<Worker>>,
        widths: &BTreeSet<usize>,
        recipe: Option<&Recipe>,
        alternatives: bool,
    ) -> Vec<Candidate> {
        let max_width = *widths.last().expect("nonempty widths");
        let mut states = BTreeMap::from([(
            0,
            vec![Candidate {
                workers: Vec::new(),
                cost: Cost {
                    compute_ns: 0,
                    network_ns: 0,
                },
            }],
        )]);
        let mut hosts: Vec<_> = hosts.iter().filter(|host| host.eligible).collect();
        hosts.sort_unstable_by_key(|host| host.host);
        for host in hosts {
            let Some(mut workers) = available.get(&host.host).cloned() else {
                continue;
            };
            workers
                .sort_unstable_by_key(|worker| (std::cmp::Reverse(worker.compute_rate), worker.id));
            let fit = (1..=workers.len().min(max_width))
                .take_while(|&count| {
                    self.host_demand(count).is_some_and(|needed| {
                        needed.bytes <= host.capacity.bytes.saturating_sub(host.reserved.bytes)
                            && needed.pinned_bytes
                                <= host
                                    .capacity
                                    .pinned_bytes
                                    .saturating_sub(host.reserved.pinned_bytes)
                    })
                })
                .count();
            let by_count: Vec<_> = (1..=fit)
                .map(|count| worker_choices(&workers, count, alternatives))
                .collect();
            let mut next = states.clone();
            for (&width, candidates) in &states {
                for candidate in candidates {
                    for count in 1..=fit.min(max_width - width) {
                        for workers in &by_count[count - 1] {
                            let mut devices = candidate.workers.clone();
                            devices.extend_from_slice(workers);
                            if let Some(cost) = self.cost(&devices, recipe) {
                                keep_best(
                                    next.entry(width + count).or_default(),
                                    Candidate {
                                        workers: devices,
                                        cost,
                                    },
                                );
                            }
                        }
                    }
                }
            }
            states = next;
        }
        states
            .into_iter()
            .filter(|(width, _)| widths.contains(width))
            .flat_map(|(_, ranks)| {
                ranks
                    .into_iter()
                    .take(if alternatives { BEAM_PER_WIDTH } else { 1 })
            })
            .collect()
    }

    fn eligible(&self, worker: &PlacementWorker) -> Option<Worker> {
        let profile = worker.profile.as_ref()?;
        let compute_rate = match &self.rates {
            Some(rates) => rates
                .get(&worker.device)
                .filter(|rate| rate.incarnation == worker.incarnation)
                .and_then(|rate| rate.operations_per_second)
                .map(u64::from),
            None => Some(profile.flops),
        };
        (worker.eligible
            && profile.id == worker.device
            && worker.device_bytes >= self.workload.device_bytes
            && profile.memory_bytes >= self.workload.device_bytes
            && (self.workload.compute_work == 0 || compute_rate != Some(0))
            && self.workload.requirement.matches(profile))
        .then_some(Worker {
            id: worker.device,
            incarnation: worker.incarnation,
            compute_rate,
        })
    }

    fn host_demand(&self, count: usize) -> Option<HostMemory> {
        let per = self.workload.per_worker_host;
        let base = self.workload.per_host;
        Some(HostMemory {
            bytes: per
                .bytes
                .checked_mul(count as u64)?
                .checked_add(base.bytes)?,
            pinned_bytes: per
                .pinned_bytes
                .checked_mul(count as u64)?
                .checked_add(base.pinned_bytes)?,
        })
    }

    fn incumbent(
        &self,
        inventory: &PlacementInventory,
        eligible: &BTreeMap<DeviceId, (Worker, &PlacementWorker)>,
        widths: &BTreeSet<usize>,
    ) -> Option<(GroupVersion, Vec<Worker>)> {
        let active = inventory.active.as_ref()?;
        if !widths.contains(&active.workers.len()) {
            return None;
        }
        let workers = active
            .workers
            .iter()
            .map(|(id, demand)| {
                let (worker, entry) = eligible.get(id)?;
                (demand.device_bytes >= self.workload.device_bytes
                    && demand.incarnation == worker.incarnation
                    && entry
                        .grant
                        .is_some_and(|grant| grant.version == active.version))
                .then_some(*worker)
            })
            .collect::<Option<Vec<_>>>()?;
        for (host, count) in count_hosts(&workers) {
            let budget = active.hosts.iter().find(|(id, _)| *id == host)?.1;
            let needed = self.host_demand(count)?;
            if needed.bytes > budget.bytes
                || needed.pinned_bytes > budget.pinned_bytes
                || !inventory
                    .hosts
                    .iter()
                    .any(|entry| entry.host == host && entry.eligible)
            {
                return None;
            }
        }
        Some((active.version, workers))
    }

    fn cost(&self, workers: &[Worker], recipe: Option<&Recipe>) -> Option<Cost> {
        if let Some(recipe) = recipe {
            return self.recipe_cost(workers, recipe);
        }
        let compute_ns = if self.workload.compute_work == 0 {
            0
        } else {
            let floor_rate = workers
                .iter()
                .map(|worker| worker.compute_rate)
                .min()
                .flatten()?;
            ns(
                u128::from(self.workload.compute_work) * NANOS,
                u128::from(floor_rate) * workers.len() as u128,
            )
        };
        let network_ns = if self.workload.bytes_per_peer > 0 {
            self.all_to_all(workers, self.workload.bytes_per_peer, 1)?
        } else {
            0
        };
        Some(Cost {
            compute_ns,
            network_ns,
        })
    }

    fn recipe_cost(&self, workers: &[Worker], recipe: &Recipe) -> Option<Cost> {
        let mut ordered = workers.to_vec();
        ordered.sort_unstable_by_key(|worker| worker.id);
        let mut compute_ns = 0u64;
        for (worker, &work) in ordered.iter().zip(&recipe.work) {
            if work == 0 {
                continue;
            }
            let rate = worker.compute_rate.filter(|&rate| rate != 0)?;
            let time = ns(u128::from(work) * NANOS, u128::from(rate));
            compute_ns = match recipe.compute {
                ComputeOrder::Parallel => compute_ns.max(time),
                ComputeOrder::Sequential => compute_ns.saturating_add(time),
            };
        }
        let network_ns = match &recipe.communication {
            Communication::AllToAll {
                rounds,
                bytes_per_peer,
            } => self.all_to_all(workers, *bytes_per_peer, *rounds)?,
            Communication::Directed { edges } => {
                let mut total = 0u64;
                for edge in edges {
                    if let (Some(from), Some(to)) = (ordered.get(edge.from), ordered.get(edge.to))
                        && from.id.node != to.id.node
                    {
                        total = total.saturating_add(self.link_time(
                            from.id.node,
                            to.id.node,
                            u128::from(edge.bytes),
                            edge.rounds,
                        )?);
                    }
                }
                total
            }
        };
        Some(Cost {
            compute_ns,
            network_ns,
        })
    }

    fn all_to_all(&self, workers: &[Worker], bytes_per_peer: u64, rounds: u64) -> Option<u64> {
        let mut network_ns = 0;
        if rounds > 0 {
            let counts = count_hosts(workers);
            for (&src, &left) in &counts {
                let mut incoming = 0u64;
                let mut outgoing = 0u64;
                for (&dst, &right) in counts.iter().filter(|&(&host, _)| host != src) {
                    let traffic = u128::from(bytes_per_peer)
                        .saturating_mul(left as u128)
                        .saturating_mul(right as u128);
                    outgoing = outgoing.saturating_add(self.link_time(src, dst, traffic, rounds)?);
                    incoming = incoming.saturating_add(self.link_time(dst, src, traffic, rounds)?);
                }
                network_ns = network_ns.max(incoming).max(outgoing);
            }
        }
        Some(network_ns)
    }

    fn link_time(&self, source: NodeId, target: NodeId, bytes: u128, rounds: u64) -> Option<u64> {
        let link = self.links.get(&(source, target))?;
        Some(
            ns(
                bytes
                    .saturating_mul(u128::from(rounds))
                    .saturating_mul(NANOS),
                u128::from(link.bytes_per_second),
            )
            .saturating_add(link.rtt_ns.saturating_mul(rounds)),
        )
    }
}

fn worker_choices(workers: &[Worker], count: usize, alternatives: bool) -> Vec<Vec<Worker>> {
    if !alternatives {
        return vec![workers[..count].to_vec()];
    }
    let mut indices: Vec<_> = (0..count).collect();
    let mut choices = Vec::new();
    while choices.len() < BEAM_PER_WIDTH {
        choices.push(indices.iter().map(|&index| workers[index]).collect());
        let Some(position) = (0..count)
            .rev()
            .find(|&position| indices[position] < workers.len() - count + position)
        else {
            break;
        };
        let next = indices[position] + 1;
        for (offset, index) in indices[position..].iter_mut().enumerate() {
            *index = next + offset;
        }
    }
    choices
}

fn ns(numerator: u128, denominator: u128) -> u64 {
    u64::try_from(numerator.div_ceil(denominator)).unwrap_or(u64::MAX)
}

fn count_hosts(workers: &[Worker]) -> BTreeMap<NodeId, usize> {
    let mut counts = BTreeMap::new();
    for worker in workers {
        *counts.entry(worker.id.node).or_insert(0) += 1;
    }
    counts
}

fn compare(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    a.cost
        .operation_ns()
        .cmp(&b.cost.operation_ns())
        .then_with(|| a.workers.len().cmp(&b.workers.len()))
        .then_with(|| {
            a.workers
                .iter()
                .map(|worker| worker.id)
                .cmp(b.workers.iter().map(|worker| worker.id))
        })
}

fn keep_best(candidates: &mut Vec<Candidate>, candidate: Candidate) {
    let index = candidates.partition_point(|prior| compare(prior, &candidate).is_le());
    if index < BEAM_PER_WIDTH {
        candidates.insert(index, candidate);
        candidates.truncate(BEAM_PER_WIDTH);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::group::{ActivePlacement, GroupId, PlacementHost, WorkerGrant};
    use plait_core::{AcceleratorKind, DeviceProfile};

    fn test_group() -> GroupId {
        GroupId(NodeId::from_bytes([9; 16]))
    }

    fn host(index: u8) -> NodeId {
        NodeId::from_bytes([index; 16])
    }

    fn memory(bytes: u64, pinned_bytes: u64) -> HostMemory {
        HostMemory {
            bytes,
            pinned_bytes,
        }
    }

    fn worker(host: NodeId, ordinal: u32, flops: u64) -> PlacementWorker {
        let device = DeviceId::new(host, ordinal);
        PlacementWorker {
            device,
            incarnation: 1,
            device_bytes: 32,
            grant: None,
            eligible: true,
            profile: Some(DeviceProfile {
                id: device,
                name: "test".into(),
                arch: AcceleratorKind::Nvidia,
                memory_bytes: 32,
                flops,
                compute_units: 1,
                supports_fp16: true,
                supports_bf16: true,
                supports_fp64: false,
                rdma_capable: false,
                numa_node: None,
            }),
        }
    }

    fn workload() -> Workload {
        Workload {
            widths: vec![2],
            requirement: ComputeRequirement::default(),
            device_bytes: 16,
            per_worker_host: memory(4, 2),
            per_host: memory(1, 1),
            compute_work: 1_000_000,
            bytes_per_peer: 8,
            max_link_age_ms: 10,
            operations: 5,
            switch_ns: 0,
            minimum_gain_ns: 0,
            recipes: Vec::new(),
        }
    }

    fn links_between(
        a: NodeId,
        b: NodeId,
        bytes_per_second: u64,
        sampled_at_ms: u64,
    ) -> [DirectedLink; 2] {
        let link = |source, target| DirectedLink {
            source,
            target,
            bytes_per_second,
            rtt_ns: 100,
            sampled_at_ms,
        };
        [link(a, b), link(b, a)]
    }

    #[test]
    fn joint_placement_uses_disjoint_devices_and_combined_host_headroom() {
        let a = host(1);
        let mut flexible = PlacementInventory {
            group: test_group(),
            latest: None,
            active: None,
            hosts: vec![PlacementHost {
                host: a,
                capacity: memory(80, 34),
                reserved: memory(0, 0),
                eligible: true,
            }],
            workers: (0..16)
                .map(|ordinal| worker(a, ordinal, if ordinal < 8 { 2000 } else { 1000 }))
                .collect(),
        };
        let mut restricted = flexible.clone();
        restricted.group = GroupId(host(2));
        let mut either = workload();
        either.widths = vec![8];
        either.bytes_per_peer = 0;
        let mut fast = either.clone();
        fast.requirement.min_flops = 2000;
        let select = |flexible: &PlacementInventory, restricted: &PlacementInventory| {
            select_set(
                &[
                    PlacementGroup {
                        inventory: flexible,
                        workload: &either,
                        rates: None,
                    },
                    PlacementGroup {
                        inventory: restricted,
                        workload: &fast,
                        rates: None,
                    },
                ],
                &[],
                0,
            )
            .unwrap()
        };
        let choices = select(&flexible, &restricted).unwrap();
        assert!(
            choices[0]
                .resources
                .workers
                .keys()
                .copied()
                .eq((8..16).map(|ordinal| DeviceId::new(a, ordinal)))
        );
        assert!(
            choices[1]
                .resources
                .workers
                .keys()
                .copied()
                .eq((0..8).map(|ordinal| DeviceId::new(a, ordinal)))
        );
        flexible.hosts[0].capacity.pinned_bytes -= 1;
        assert!(select(&flexible, &restricted).is_none());
    }

    #[test]
    fn asymmetric_inventory_filters_pinning_and_requires_both_fresh_directions() {
        let [a, b, c] = [1, 2, 3].map(host);
        let host_view = |host, pinned_bytes| PlacementHost {
            host,
            capacity: memory(32, pinned_bytes),
            reserved: memory(0, 0),
            eligible: true,
        };
        let mut weak = worker(a, 1, 10_000);
        weak.device_bytes = 8;
        let mut inventory = PlacementInventory {
            group: test_group(),
            latest: None,
            active: None,
            hosts: vec![host_view(c, 2), host_view(b, 3), host_view(a, 3)],
            workers: vec![
                worker(a, 0, 1000),
                weak,
                worker(b, 0, 2000),
                worker(c, 0, 9000),
            ],
        };
        let links = links_between(a, b, 1000, 51);
        assert!(matches!(
            select(&inventory, &workload(), &links[..1], 55).unwrap(),
            Decision::NoPlacement
        ));
        assert!(matches!(
            select(&inventory, &workload(), &links, 62).unwrap(),
            Decision::NoPlacement
        ));
        let Decision::Prepare { resources, .. } =
            select(&inventory, &workload(), &links, 55).unwrap()
        else {
            panic!("two eligible workers")
        };
        assert_eq!(
            resources.workers.keys().copied().collect::<Vec<_>>(),
            [DeviceId::new(a, 0), DeviceId::new(b, 0)]
        );
        assert_eq!(
            resources.hosts,
            BTreeMap::from([(a, memory(5, 3)), (b, memory(5, 3))])
        );
        inventory.hosts.reverse();
        inventory.workers.reverse();
        let Decision::Prepare {
            resources: reversed,
            ..
        } = select(&inventory, &workload(), &links, 55).unwrap()
        else {
            panic!("same placement")
        };
        assert_eq!(resources, reversed);
    }

    #[test]
    fn a_new_layout_must_pay_its_switch_cost_and_unknown_links_preserve_the_incumbent() {
        let [a, b, c] = [1, 2, 3].map(host);
        let mut workers = vec![worker(a, 0, 100_000_000), worker(b, 0, 100_000_000)];
        let version = GroupVersion {
            group: test_group(),
            epoch: 1,
        };
        for worker in &mut workers {
            worker.grant = Some(WorkerGrant {
                version,
                device: worker.device,
                incarnation: 1,
                fence: 1,
            });
        }
        let use_bytes = WorkerDemand {
            incarnation: 1,
            device_bytes: 16,
        };
        let held = memory(5, 3);
        let active = ActivePlacement {
            version,
            workers: workers
                .iter()
                .map(|worker| (worker.device, use_bytes))
                .collect(),
            hosts: vec![(a, held), (b, held)],
        };
        workers.extend([worker(c, 0, 100_000_000), worker(c, 1, 100_000_000)]);
        let inventory = PlacementInventory {
            group: test_group(),
            latest: Some((version, GroupPhase::Active)),
            active: Some(active),
            hosts: [a, b, c]
                .into_iter()
                .map(|host| PlacementHost {
                    host,
                    capacity: memory(32, 16),
                    reserved: if host == c { memory(0, 0) } else { held },
                    eligible: true,
                })
                .collect(),
            workers,
        };
        let mut workload = workload();
        assert!(matches!(
            select(&inventory, &workload, &[], 55).unwrap(),
            Decision::AwaitLinks(_)
        ));
        let slow = links_between(a, b, 1000, 54);
        workload.switch_ns = u64::MAX;
        assert!(matches!(
            select(&inventory, &workload, &slow, 55).unwrap(),
            Decision::Keep { .. }
        ));
        workload.switch_ns = 100_000;
        let Decision::Prepare {
            resources,
            cost,
            incumbent,
            ..
        } = select(&inventory, &workload, &slow, 55).unwrap()
        else {
            panic!("cheaper placement")
        };
        assert_eq!(resources.hosts, BTreeMap::from([(c, memory(9, 5))]));
        assert_eq!(cost.network_ns, 0);
        assert!(incumbent.unwrap().network_ns > 0);
    }

    #[test]
    fn link_cost_selects_different_recipes_on_the_same_devices() {
        let [a, b] = [1, 2].map(host);
        let inventory = PlacementInventory {
            group: test_group(),
            latest: None,
            active: None,
            hosts: [a, b]
                .into_iter()
                .map(|host| PlacementHost {
                    host,
                    capacity: memory(32, 16),
                    reserved: memory(0, 0),
                    eligible: true,
                })
                .collect(),
            workers: vec![worker(b, 0, 1_000_000_000), worker(a, 0, 2_000_000_000)],
        };
        let parallel = NodeId::derive("all-to-all");
        let sequential = NodeId::derive("directed");
        let mut workload = workload();
        workload.recipes = vec![
            Recipe {
                id: parallel,
                work: vec![100_000_000; 2],
                compute: ComputeOrder::Parallel,
                communication: Communication::AllToAll {
                    rounds: 57,
                    bytes_per_peer: 1024,
                },
            },
            Recipe {
                id: sequential,
                work: vec![150_000_000, 50_000_000],
                compute: ComputeOrder::Sequential,
                communication: Communication::Directed {
                    edges: vec![Exchange {
                        from: 0,
                        to: 1,
                        rounds: 1,
                        bytes: 1024,
                    }],
                },
            },
        ];
        let mut links = links_between(a, b, 1_000_000_000, 51);
        let Decision::Prepare {
            resources,
            recipe: Some(id),
            ..
        } = select(&inventory, &workload, &links, 55).unwrap()
        else {
            panic!("the measured link fits both recipes");
        };
        assert_eq!(id, parallel);
        for link in &mut links {
            link.rtt_ns = 41_000_000;
        }
        let Decision::Prepare {
            resources: selected,
            recipe: Some(id),
            cost,
            ..
        } = select(&inventory, &workload, &links, 55).unwrap()
        else {
            panic!("the directed recipe fits the slower link");
        };
        assert_eq!(id, sequential);
        assert_eq!(selected, resources);
        assert_eq!(cost.compute_ns, 125_000_000);
        assert_eq!(cost.network_ns, 41_001_024);
    }

    #[test]
    fn rates_rank_slow_and_fast_devices_within_one_calibration() {
        let host = host(1);
        let mut workers = vec![
            worker(host, 0, 100),
            worker(host, 1, 10),
            worker(host, 2, 20),
        ];
        let version = GroupVersion {
            group: test_group(),
            epoch: 1,
        };
        let device = workers[0].device;
        workers[0].grant = Some(WorkerGrant {
            version,
            device,
            incarnation: 1,
            fence: 1,
        });
        let held = memory(5, 3);
        let inventory = PlacementInventory {
            group: test_group(),
            latest: Some((version, GroupPhase::Active)),
            active: Some(ActivePlacement {
                version,
                workers: vec![(
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 16,
                    },
                )],
                hosts: vec![(host, held)],
            }),
            hosts: vec![PlacementHost {
                host,
                capacity: memory(32, 16),
                reserved: held,
                eligible: true,
            }],
            workers,
        };
        let mut workload = workload();
        workload.widths = vec![1];
        let rate = |ordinal, speed| ComputeRate {
            device: DeviceId::new(host, ordinal),
            incarnation: 1,
            revision: 1,
            operations_per_second: std::num::NonZeroU64::new(speed),
        };
        let mut rates = [rate(0, 20), rate(1, 80), rate(2, 50)];
        assert!(matches!(
            select(&inventory, &workload, &[], 0).unwrap(),
            Decision::Keep { .. }
        ));
        assert!(matches!(
            select_with_rates(&inventory, &workload, &[], 0, Some(&rates[1..])).unwrap(),
            Decision::AwaitComputeRates(waiting) if waiting == version
        ));
        let Decision::Prepare {
            resources,
            incumbent,
            cost,
            ..
        } = select_with_rates(&inventory, &workload, &[], 0, Some(&rates)).unwrap()
        else {
            panic!("expected a faster candidate");
        };
        assert_eq!(
            resources.workers.keys().copied().collect::<Vec<_>>(),
            [rates[1].device]
        );
        assert_eq!(incumbent.unwrap().compute_ns, 50_000_000_000_000);
        assert_eq!(cost.compute_ns, 12_500_000_000_000);
        rates[0] = rate(0, 120);
        assert!(matches!(
            select_with_rates(&inventory, &workload, &[], 0, Some(&rates)).unwrap(),
            Decision::Keep { .. }
        ));
    }
}
