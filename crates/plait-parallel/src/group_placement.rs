//! Topology-aware placement of native execution groups.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

use plait_core::collective::{
    PeerAllReducePlan, bind_peer_all_reduce_recipe, bind_tensor_pipeline_recipe,
    peer_all_reduce_plan, peer_all_reduce_plan_with_egress, peer_shard_elements,
};
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
    /// Per-slot reservation in the same device order as `work`.
    pub device_bytes_by_slot: Option<Vec<u64>>,
    pub compute: ComputeOrder,
    pub communication: Communication,
}

impl Recipe {
    pub fn binding_id(&self) -> NodeId {
        match &self.communication {
            Communication::TensorPipeline {
                tensor_parallel,
                all_reduce_by_stage,
                replicated_egress_bytes,
                ..
            } => bind_tensor_pipeline_recipe(
                self.id,
                tensor_parallel.get() as u64,
                all_reduce_by_stage.len() as u64,
                *replicated_egress_bytes,
            ),
            _ => self.legacy_binding_id(),
        }
    }

    fn legacy_binding_id(&self) -> NodeId {
        match self.communication {
            Communication::PeerAllReduce {
                replicated_egress_bytes: Some(bytes),
                ..
            }
            | Communication::TensorPipeline {
                replicated_egress_bytes: Some(bytes),
                ..
            } => bind_peer_all_reduce_recipe(self.id, self.work.len() as u64, bytes),
            _ => self.id,
        }
    }

    fn matches_binding(&self, id: NodeId) -> bool {
        self.binding_id() == id || self.legacy_binding_id() == id
    }

    fn compute_ns(&self, workers: &[Worker], first_slot: usize) -> Option<u64> {
        let mut times = workers
            .iter()
            .zip(&self.work[first_slot..])
            .map(|(worker, &work)| worker.work_ns(work));
        match self.compute {
            ComputeOrder::Parallel => times.try_fold(0u64, |max, time| Some(max.max(time?))),
            ComputeOrder::Sequential => {
                times.try_fold(0u64, |sum, time| Some(sum.saturating_add(time?)))
            }
            ComputeOrder::Pipeline => {
                let Communication::TensorPipeline {
                    tensor_parallel, ..
                } = &self.communication
                else {
                    return None;
                };
                let mut completed = 0u64;
                let mut stage = 0u64;
                for (offset, time) in times.enumerate() {
                    if (first_slot + offset) % tensor_parallel.get() == 0 {
                        completed = completed.saturating_add(stage);
                        stage = 0;
                    }
                    stage = stage.max(time?);
                }
                Some(completed.saturating_add(stage))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeOrder {
    Parallel,
    Sequential,
    Pipeline,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Communication {
    AllToAll {
        rounds: u64,
        bytes_per_peer: u64,
    },
    PeerAllReduce {
        calls: Vec<AllReduceCalls>,
        #[serde(default)]
        replicated_egress_bytes: Option<u64>,
    },
    Directed {
        edges: Vec<Exchange>,
    },
    TensorPipeline {
        tensor_parallel: NonZeroUsize,
        all_reduce_by_stage: Vec<Vec<AllReduceCalls>>,
        transfers: Vec<PipelineTransfer>,
        #[serde(default)]
        replicated_egress_bytes: Option<u64>,
    },
}

/// A class of serial host-staged peer AllReduce calls for one operation.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllReduceCalls {
    pub count: u64,
    pub bytes: u64,
    pub element_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exchange {
    pub from: usize,
    pub to: usize,
    pub rounds: u64,
    pub bytes: u64,
}

/// One serial transfer phase between pipeline stages, with equal-sized TP lanes.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineTransfer {
    pub from_stage: usize,
    pub to_stage: usize,
    pub bytes_per_lane: u64,
    pub rounds: u64,
}

impl Communication {
    fn valid(&self, width: usize) -> bool {
        match self {
            Self::AllToAll {
                rounds,
                bytes_per_peer,
            } => *rounds > 0 || *bytes_per_peer == 0,
            Self::PeerAllReduce { calls, .. } => !calls.is_empty() && valid_all_reduce(calls),
            Self::Directed { edges } => edges.iter().all(|edge| {
                edge.from < width && edge.to < width && edge.from != edge.to && edge.rounds > 0
            }),
            Self::TensorPipeline {
                tensor_parallel,
                all_reduce_by_stage,
                transfers,
                ..
            } => {
                let tensor = tensor_parallel.get();
                let stages = width / tensor;
                tensor > 1
                    && stages > 1
                    && width % tensor == 0
                    && all_reduce_by_stage.len() == stages
                    && all_reduce_by_stage
                        .iter()
                        .all(|calls| valid_all_reduce(calls))
                    && transfers.iter().all(|transfer| {
                        transfer.from_stage < stages
                            && transfer.to_stage < stages
                            && transfer.from_stage != transfer.to_stage
                            && transfer.rounds > 0
                    })
            }
        }
    }
}

fn valid_all_reduce(calls: &[AllReduceCalls]) -> bool {
    calls.iter().all(|call| {
        call.count > 0
            && call.bytes > 0
            && matches!(call.element_bytes, 1 | 2 | 4 | 8)
            && call.bytes % call.element_bytes == 0
    })
}

impl Workload {
    fn device_demand(&self, recipe: Option<&Recipe>, slot: usize) -> u64 {
        recipe
            .and_then(|recipe| recipe.device_bytes_by_slot.as_deref())
            .map_or(self.device_bytes, |bytes| bytes[slot])
    }

    pub fn needs_links(&self) -> bool {
        self.bytes_per_peer > 0
            || self
                .recipes
                .iter()
                .any(|recipe| match &recipe.communication {
                    Communication::AllToAll { rounds, .. } => *rounds != 0,
                    Communication::PeerAllReduce { calls, .. } => !calls.is_empty(),
                    Communication::Directed { edges } => !edges.is_empty(),
                    Communication::TensorPipeline {
                        all_reduce_by_stage,
                        transfers,
                        ..
                    } => {
                        !transfers.is_empty()
                            || all_reduce_by_stage.iter().any(|calls| !calls.is_empty())
                    }
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
        let mut bindings = BTreeSet::new();
        for recipe in &self.recipes {
            let width = recipe.work.len();
            let binding = recipe.binding_id();
            let legacy = recipe.legacy_binding_id();
            if !recipes.insert(recipe.id)
                || !bindings.insert(binding)
                || (legacy != binding && !bindings.insert(legacy))
                || !self.widths.iter().any(|&item| item as usize == width)
                || recipe.work.iter().all(|&work| work == 0)
                || recipe
                    .device_bytes_by_slot
                    .as_ref()
                    .is_some_and(|bytes| bytes.len() != width || bytes.contains(&0))
                || matches!(recipe.compute, ComputeOrder::Pipeline)
                    != matches!(recipe.communication, Communication::TensorPipeline { .. })
                || !recipe.communication.valid(width)
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
        /// Directed samples used to price the selected and compared layouts.
        link_samples: Vec<DirectedLink>,
    },
}

pub struct PlacementGroup<'a> {
    pub inventory: &'a PlacementInventory,
    pub workload: &'a Workload,
    pub rates: Option<&'a [ComputeRate]>,
    pub hosts: Option<&'a BTreeSet<NodeId>>,
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
                    available.contains_key(&entry.device)
                        && hosts.contains_key(&entry.device.node)
                        && group
                            .hosts
                            .is_none_or(|allowed| allowed.contains(&entry.device.node))
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
                let resources = model.resources(&candidate, recipe);
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
                        recipe: recipe.map(Recipe::binding_id),
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
    device_bytes: u64,
}

impl Worker {
    fn work_ns(self, work: u64) -> Option<u64> {
        if work == 0 {
            return Some(0);
        }
        let rate = self.compute_rate.filter(|&rate| rate != 0)?;
        Some(ns(u128::from(work) * NANOS, u128::from(rate)))
    }
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
    minimum_device_bytes: u64,
}

struct LinkCosts<'a> {
    samples: &'a BTreeMap<(NodeId, NodeId), DirectedLink>,
    observed: Option<&'a mut BTreeMap<(NodeId, NodeId), DirectedLink>>,
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
    let mut observed = BTreeMap::new();
    let mut network = LinkCosts {
        samples: &model.links,
        observed: Some(&mut observed),
    };
    let incumbent = if let Some((version, workers)) = model.incumbent(inventory, &eligible, &widths)
    {
        let recipe = active_recipe
            .map(|id| {
                workload
                    .recipes
                    .iter()
                    .find(|profile| {
                        profile.matches_binding(id) && profile.work.len() == workers.len()
                    })
                    .ok_or("active group recipe is missing from the workload policy")
            })
            .transpose()?;
        if workers
            .iter()
            .enumerate()
            .all(|(slot, worker)| worker.device_bytes >= workload.device_demand(recipe, slot))
        {
            if workers.iter().any(|worker| worker.compute_rate.is_none()) {
                return Ok(Decision::AwaitComputeRates(version));
            }
            let Some(cost) = model.cost_using(&workers, recipe, &mut network) else {
                return Ok(Decision::AwaitLinks(version));
            };
            Some((version, cost))
        } else {
            None
        }
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
    let resources = model.resources(&best, recipe);
    let cost = model
        .cost_using(&best.workers, recipe, &mut network)
        .expect("selected layout was priced from the same samples");
    Ok(Decision::Prepare {
        resources,
        cost,
        incumbent: incumbent_cost,
        recipe: recipe.map(Recipe::binding_id),
        link_samples: observed.into_values().collect(),
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
        let minimum_device_bytes = workload
            .recipes
            .iter()
            .filter_map(|recipe| recipe.device_bytes_by_slot.as_deref())
            .flatten()
            .copied()
            .fold(workload.device_bytes, u64::min);
        Ok(Self {
            workload,
            links: fresh,
            rates,
            minimum_device_bytes,
        })
    }

    fn resources(&self, candidate: &Candidate, recipe: Option<&Recipe>) -> ResourcePlan {
        let hosts = count_hosts(&candidate.workers)
            .into_iter()
            .map(|(host, count)| (host, self.host_demand(count).expect("fitted host")))
            .collect();
        debug_assert!(
            recipe.is_none()
                || candidate
                    .workers
                    .windows(2)
                    .all(|pair| pair[0].id < pair[1].id)
        );
        let workers = candidate
            .workers
            .iter()
            .enumerate()
            .map(|(slot, worker)| {
                (
                    worker.id,
                    WorkerDemand {
                        incarnation: worker.incarnation,
                        device_bytes: self.workload.device_demand(recipe, slot),
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
    ) -> Vec<(Candidate, Option<&'a Recipe>)> {
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
                        .map(move |candidate| (candidate, Some(recipe)))
                })
                .collect()
        };
        candidates.sort_by(|(left, left_recipe), (right, right_recipe)| {
            compare(left, right).then_with(|| {
                left_recipe
                    .map(Recipe::binding_id)
                    .cmp(&right_recipe.map(Recipe::binding_id))
            })
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
            if recipe.is_some() {
                workers.sort_unstable_by_key(|worker| worker.id);
            } else {
                workers.sort_unstable_by_key(|worker| {
                    (std::cmp::Reverse(worker.compute_rate), worker.id)
                });
            }
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
            let uniform_choices: Vec<_> = if recipe.is_none() {
                (1..=fit)
                    .map(|count| worker_choices(&workers, count, alternatives))
                    .collect()
            } else {
                Vec::new()
            };
            let mut next = states.clone();
            for (&width, candidates) in &states {
                let slots = fit.min(max_width - width);
                if slots == 0 {
                    continue;
                }
                let recipe_choices;
                let by_count = if let Some(recipe) = recipe {
                    let demands: Vec<_> = (width..width + slots)
                        .map(|slot| {
                            (
                                recipe.work[slot],
                                self.workload.device_demand(Some(recipe), slot),
                            )
                        })
                        .collect();
                    recipe_choices =
                        recipe_worker_choices(&workers, &demands, recipe, width, alternatives);
                    &recipe_choices[..]
                } else {
                    &uniform_choices[..slots]
                };
                for candidate in candidates {
                    for choices in by_count {
                        for workers in choices {
                            let mut devices = candidate.workers.clone();
                            devices.extend_from_slice(workers);
                            if let Some(cost) = self.cost(&devices, recipe) {
                                keep_best(
                                    next.entry(devices.len()).or_default(),
                                    Candidate {
                                        workers: devices,
                                        cost,
                                    },
                                    BEAM_PER_WIDTH,
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
        let device_bytes = worker.device_bytes.min(profile.memory_bytes);
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
            && device_bytes >= self.minimum_device_bytes
            && (self.workload.compute_work == 0 || compute_rate != Some(0))
            && self.workload.requirement.matches(profile))
        .then_some(Worker {
            id: worker.device,
            incarnation: worker.incarnation,
            compute_rate,
            device_bytes,
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
        let mut workers = active
            .workers
            .iter()
            .map(|(id, demand)| {
                let (worker, entry) = eligible.get(id)?;
                (demand.incarnation == worker.incarnation
                    && entry
                        .grant
                        .is_some_and(|grant| grant.version == active.version))
                .then_some(Worker {
                    device_bytes: worker.device_bytes.min(demand.device_bytes),
                    ..*worker
                })
            })
            .collect::<Option<Vec<_>>>()?;
        workers.sort_unstable_by_key(|worker| worker.id);
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
        self.cost_using(workers, recipe, &mut self.link_costs())
    }

    fn link_costs(&self) -> LinkCosts<'_> {
        LinkCosts {
            samples: &self.links,
            observed: None,
        }
    }

    fn cost_using(
        &self,
        workers: &[Worker],
        recipe: Option<&Recipe>,
        network: &mut LinkCosts<'_>,
    ) -> Option<Cost> {
        if let Some(recipe) = recipe {
            return Self::recipe_cost(workers, recipe, network);
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
            network.all_to_all(workers, self.workload.bytes_per_peer, 1)?
        } else {
            0
        };
        Some(Cost {
            compute_ns,
            network_ns,
        })
    }

    fn recipe_cost(
        workers: &[Worker],
        recipe: &Recipe,
        network: &mut LinkCosts<'_>,
    ) -> Option<Cost> {
        let mut ordered = workers.to_vec();
        ordered.sort_unstable_by_key(|worker| worker.id);
        let compute_ns = recipe.compute_ns(&ordered, 0)?;
        let network_ns = match &recipe.communication {
            Communication::AllToAll {
                rounds,
                bytes_per_peer,
            } => network.all_to_all(workers, *bytes_per_peer, *rounds)?,
            Communication::PeerAllReduce {
                calls,
                replicated_egress_bytes,
            } => network.peer_all_reduce(&ordered, calls, *replicated_egress_bytes)?,
            Communication::Directed { edges } => {
                let mut total = 0u64;
                for edge in edges {
                    if let (Some(from), Some(to)) = (ordered.get(edge.from), ordered.get(edge.to))
                        && from.id.node != to.id.node
                    {
                        total = total.saturating_add(network.link_time(
                            from.id.node,
                            to.id.node,
                            u128::from(edge.bytes),
                            edge.rounds,
                        )?);
                    }
                }
                total
            }
            Communication::TensorPipeline {
                tensor_parallel,
                all_reduce_by_stage,
                transfers,
                replicated_egress_bytes,
            } => network.tensor_pipeline(
                &ordered,
                tensor_parallel.get(),
                all_reduce_by_stage,
                transfers,
                *replicated_egress_bytes,
            )?,
        };
        Some(Cost {
            compute_ns,
            network_ns,
        })
    }
}

impl LinkCosts<'_> {
    fn read(&mut self, source: NodeId, target: NodeId) -> Option<DirectedLink> {
        let key = (source, target);
        let sample = *self.samples.get(&key)?;
        if let Some(observed) = &mut self.observed {
            observed.insert(key, sample);
        }
        Some(sample)
    }

    fn all_to_all(&mut self, workers: &[Worker], bytes_per_peer: u64, rounds: u64) -> Option<u64> {
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

    fn peer_all_reduce(
        &mut self,
        ordered: &[Worker],
        calls: &[AllReduceCalls],
        replicated_egress_bytes: Option<u64>,
    ) -> Option<u64> {
        let hosts = count_hosts(ordered);
        if hosts.len() < 2 {
            return Some(0);
        }
        let mut total = 0u64;
        for call in calls {
            let plan = match replicated_egress_bytes {
                Some(bytes) => peer_all_reduce_plan_with_egress(call.bytes, ordered.len(), bytes),
                None => peer_all_reduce_plan(call.bytes, ordered.len()),
            };
            let elapsed = match plan {
                PeerAllReducePlan::Replicated => {
                    self.exchange_phase(&hosts, |source, target| {
                        u128::from(call.bytes) * hosts[&source] as u128 * hosts[&target] as u128
                    })?
                }
                PeerAllReducePlan::ScatterGather => {
                    let elements = call.bytes / call.element_bytes;
                    let mut shards = BTreeMap::new();
                    for (slot, worker) in ordered.iter().enumerate() {
                        let count = peer_shard_elements(elements, ordered.len(), slot);
                        *shards.entry(worker.id.node).or_insert(0u64) += count * call.element_bytes;
                    }
                    let scatter =
                        |source, target| hosts[&source] as u128 * u128::from(shards[&target]);
                    let scatter_ns = self.exchange_phase(&hosts, scatter)?;
                    let gather_ns =
                        self.exchange_phase(&hosts, |source, target| scatter(target, source))?;
                    scatter_ns.saturating_add(gather_ns)
                }
            };
            total = total.saturating_add(elapsed.saturating_mul(call.count));
        }
        Some(total)
    }

    fn exchange_phase(
        &mut self,
        hosts: &BTreeMap<NodeId, usize>,
        traffic: impl Fn(NodeId, NodeId) -> u128,
    ) -> Option<u64> {
        let pairs = hosts.keys().flat_map(|&source| {
            let traffic = &traffic;
            hosts
                .keys()
                .filter(move |&&target| target != source)
                .map(move |&target| ((source, target), traffic(source, target)))
        });
        self.exchange_links(pairs)
    }

    fn tensor_pipeline(
        &mut self,
        workers: &[Worker],
        tensor: usize,
        all_reduce_by_stage: &[Vec<AllReduceCalls>],
        transfers: &[PipelineTransfer],
        replicated_egress_bytes: Option<u64>,
    ) -> Option<u64> {
        let mut total = 0u64;
        // An unfinished row is a placement prefix. Tensor shard sizes
        // use the committed row width once all its slots have devices.
        for (stage, calls) in workers.chunks_exact(tensor).zip(all_reduce_by_stage) {
            total = total.saturating_add(self.peer_all_reduce(
                stage,
                calls,
                replicated_egress_bytes,
            )?);
        }
        for transfer in transfers {
            let from_slot = transfer.from_stage * tensor;
            let to_slot = transfer.to_stage * tensor;
            let mut traffic = BTreeMap::new();
            for lane in 0..tensor {
                if let (Some(from), Some(to)) =
                    (workers.get(from_slot + lane), workers.get(to_slot + lane))
                    && from.id.node != to.id.node
                {
                    *traffic.entry((from.id.node, to.id.node)).or_insert(0u128) +=
                        u128::from(transfer.bytes_per_lane);
                }
            }
            let elapsed = self.exchange_links(traffic)?;
            total = total.saturating_add(elapsed.saturating_mul(transfer.rounds));
        }
        Some(total)
    }

    fn exchange_links(
        &mut self,
        pairs: impl IntoIterator<Item = ((NodeId, NodeId), u128)>,
    ) -> Option<u64> {
        let mut peak_rtt = 0;
        let mut peak_serialization = 0u64;
        let mut inbound = BTreeMap::new();
        let mut outbound = BTreeMap::new();
        for ((source, target), bytes) in pairs {
            let link = self.read(source, target)?;
            peak_rtt = peak_rtt.max(link.rtt_ns);
            let time = ns(
                bytes.saturating_mul(NANOS),
                u128::from(link.bytes_per_second),
            );
            let incoming = inbound.entry(target).or_insert(0u64);
            let outgoing = outbound.entry(source).or_insert(0u64);
            *incoming = incoming.saturating_add(time);
            *outgoing = outgoing.saturating_add(time);
            peak_serialization = peak_serialization.max(*incoming).max(*outgoing);
        }
        Some(peak_rtt.saturating_add(peak_serialization))
    }

    fn link_time(
        &mut self,
        source: NodeId,
        target: NodeId,
        bytes: u128,
        rounds: u64,
    ) -> Option<u64> {
        let link = self.read(source, target)?;
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

fn recipe_worker_choices(
    workers: &[Worker],
    demands: &[(u64, u64)],
    recipe: &Recipe,
    first_slot: usize,
    alternatives: bool,
) -> Vec<Vec<Vec<Worker>>> {
    // A host can cross a PP stage boundary. Keep alternatives whose partial
    // stage time may be masked by workers on another host.
    let limit = if alternatives || matches!(recipe.compute, ComputeOrder::Pipeline) {
        BEAM_PER_WIDTH
    } else {
        1
    };
    let mut choices = vec![Vec::new(); demands.len() + 1];
    choices[0].push(Candidate {
        workers: Vec::new(),
        cost: Cost {
            compute_ns: 0,
            network_ns: 0,
        },
    });
    // Hosts are visited in ID order, so this host fills a contiguous slot range.
    // For a fixed range any device subset on the host has the same link cost.
    for &worker in workers {
        for (slot, &(work, device_bytes)) in demands.iter().enumerate().rev() {
            if worker.device_bytes < device_bytes {
                continue;
            }
            let Some(device_ns) = worker.work_ns(work) else {
                continue;
            };
            // Extend the previous prefix before it can pick this device itself.
            let (prefixes, extensions) = choices.split_at_mut(slot + 1);
            for prefix in &prefixes[slot] {
                let mut slice = prefix.clone();
                slice.workers.push(worker);
                slice.cost.compute_ns = match recipe.compute {
                    ComputeOrder::Parallel => slice.cost.compute_ns.max(device_ns),
                    ComputeOrder::Sequential => slice.cost.compute_ns.saturating_add(device_ns),
                    ComputeOrder::Pipeline => recipe
                        .compute_ns(&slice.workers, first_slot)
                        .expect("validated pipeline with known worker compute rates"),
                };
                keep_best(&mut extensions[0], slice, limit);
            }
        }
    }
    choices
        .into_iter()
        .skip(1)
        .map(|choices| choices.into_iter().map(|choice| choice.workers).collect())
        .collect()
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

fn keep_best(candidates: &mut Vec<Candidate>, candidate: Candidate, limit: usize) {
    let index = candidates.partition_point(|prior| compare(prior, &candidate).is_le());
    if index < limit {
        candidates.insert(index, candidate);
        candidates.truncate(limit);
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

    fn pipeline_workload() -> Workload {
        let mut workload = workload();
        workload.widths = vec![4];
        workload.bytes_per_peer = 0;
        workload.recipes = vec![Recipe {
            id: NodeId::derive("two tensor stages"),
            work: vec![1_000_000; 4],
            device_bytes_by_slot: None,
            compute: ComputeOrder::Pipeline,
            communication: Communication::TensorPipeline {
                tensor_parallel: NonZeroUsize::new(2).unwrap(),
                all_reduce_by_stage: vec![
                    vec![AllReduceCalls {
                        count: 2,
                        bytes: 512,
                        element_bytes: 2,
                    }],
                    vec![AllReduceCalls {
                        count: 3,
                        bytes: 256,
                        element_bytes: 2,
                    }],
                ],
                transfers: vec![
                    PipelineTransfer {
                        from_stage: 0,
                        to_stage: 1,
                        bytes_per_lane: 4096,
                        rounds: 2,
                    },
                    PipelineTransfer {
                        from_stage: 1,
                        to_stage: 0,
                        bytes_per_lane: 64,
                        rounds: 1,
                    },
                ],
                replicated_egress_bytes: Some(1024),
            },
        }];
        workload
    }

    fn pipeline_inventory(hosts: &[NodeId], workers: Vec<PlacementWorker>) -> PlacementInventory {
        PlacementInventory {
            group: test_group(),
            latest: None,
            active: None,
            hosts: hosts
                .iter()
                .map(|&host| PlacementHost {
                    host,
                    capacity: memory(32, 16),
                    reserved: memory(0, 0),
                    eligible: true,
                })
                .collect(),
            workers,
        }
    }

    #[test]
    fn mixed_layout_requires_samples_only_for_tensor_rows_and_pipeline_columns() {
        let [a, b, c, d] = [1, 2, 3, 4].map(host);
        let workload = pipeline_workload();
        let inventory = pipeline_inventory(
            &[a, b, c, d],
            vec![
                worker(d, 0, 200_000_000),
                worker(c, 0, 400_000_000),
                worker(b, 0, 100_000_000),
                worker(a, 0, 200_000_000),
            ],
        );
        let mut links = [
            (a, b, 10),
            (b, a, 10),
            (c, d, 20),
            (d, c, 20),
            (a, c, 40),
            (b, d, 60),
            (c, a, 30),
            (d, b, 50),
        ]
        .map(|(source, target, rtt_ns)| DirectedLink {
            source,
            target,
            rtt_ns,
            bytes_per_second: 1_000_000_000,
            sampled_at_ms: 50,
        });
        let Decision::Prepare {
            cost,
            recipe,
            link_samples,
            ..
        } = select(&inventory, &workload, &links, 55).unwrap()
        else {
            panic!("two tensor rows and two pipeline columns are sufficient");
        };
        assert_eq!(
            recipe,
            Some(bind_tensor_pipeline_recipe(
                workload.recipes[0].id,
                2,
                2,
                Some(1024),
            ))
        );
        assert_eq!(
            cost,
            Cost {
                compute_ns: 15_000_000,
                network_ns: 10_298
            }
        );
        let affected = |links: &[DirectedLink]| {
            links
                .iter()
                .map(|link| (link.source, link.target))
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(affected(&link_samples), affected(&links));

        links[7].sampled_at_ms = 40;
        assert!(matches!(
            select(&inventory, &workload, &links, 55).unwrap(),
            Decision::NoPlacement
        ));
    }

    #[test]
    fn mixed_pipeline_columns_share_host_bandwidth() {
        let [a, b] = [1, 2].map(host);
        let workload = pipeline_workload();
        let inventory = pipeline_inventory(
            &[a, b],
            vec![
                worker(a, 0, 200_000_000),
                worker(a, 1, 100_000_000),
                worker(b, 0, 400_000_000),
                worker(b, 1, 200_000_000),
            ],
        );
        let mut links = links_between(a, b, 1_000_000_000, 50);
        links[1].bytes_per_second = 500_000_000;
        let Decision::Prepare { cost, .. } = select(&inventory, &workload, &links, 55).unwrap()
        else {
            panic!("local TP rows use only the two measured host directions");
        };
        assert_eq!(cost.compute_ns, 15_000_000);
        assert_eq!(cost.network_ns, 2 * (100 + 2 * 4096) + 100 + 4 * 64);
    }

    #[test]
    fn mixed_recipe_rejects_inconsistent_compute_and_geometry() {
        let mut workload = pipeline_workload();
        assert!(workload.needs_links());
        assert!(workload.validate(false).is_ok());
        workload.recipes[0].compute = ComputeOrder::Parallel;
        assert!(workload.validate(false).is_err());
        workload.recipes[0].compute = ComputeOrder::Pipeline;
        let Communication::TensorPipeline { transfers, .. } =
            &mut workload.recipes[0].communication
        else {
            unreachable!()
        };
        transfers[1].to_stage = 2;
        assert!(workload.validate(false).is_err());
    }

    #[test]
    fn improved_pipeline_links_can_pay_for_a_new_placement() {
        let [a, b, c, d] = [1, 2, 3, 4].map(host);
        let mut workload = pipeline_workload();
        workload.switch_ns = 10_000;
        let recipe = workload.recipes[0].binding_id();
        let legacy = bind_peer_all_reduce_recipe(workload.recipes[0].id, 4, 1024);
        let mut inventory = pipeline_inventory(
            &[a, b, c, d],
            [a, b, c, d]
                .into_iter()
                .flat_map(|host| (0..2).map(move |slot| worker(host, slot, 1_000_000_000)))
                .collect(),
        );
        let version = GroupVersion {
            group: inventory.group,
            epoch: 1,
        };
        for worker in &mut inventory.workers[..4] {
            worker.grant = Some(WorkerGrant {
                version,
                device: worker.device,
                incarnation: 1,
                fence: 1,
            });
        }
        for host in &mut inventory.hosts[..2] {
            host.reserved = memory(9, 5);
        }
        inventory.latest = Some((version, GroupPhase::Active));
        inventory.active = Some(ActivePlacement {
            version,
            workers: inventory.workers[..4]
                .iter()
                .map(|worker| {
                    (
                        worker.device,
                        WorkerDemand {
                            incarnation: 1,
                            device_bytes: workload.device_bytes,
                        },
                    )
                })
                .collect(),
            hosts: vec![(a, memory(9, 5)), (b, memory(9, 5))],
        });
        let old = links_between(a, b, 1_000_000_000, 50);
        let replan = |spare_bandwidth, active_recipe| {
            let mut links = old.to_vec();
            links.extend(links_between(c, d, spare_bandwidth, 50));
            select_with_recipes(&inventory, &workload, &links, 55, None, Some(active_recipe))
                .unwrap()
        };
        assert!(matches!(replan(100_000_000, recipe), Decision::Keep { .. }));
        assert!(matches!(replan(100_000_000, legacy), Decision::Keep { .. }));
        let Decision::Prepare {
            resources,
            recipe: Some(chosen),
            ..
        } = replan(10_000_000_000, legacy)
        else {
            panic!("faster spare links cover the switch cost");
        };
        assert_eq!(chosen, recipe);
        assert!(resources.hosts.keys().copied().eq([c, d]));
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
                        hosts: None,
                    },
                    PlacementGroup {
                        inventory: restricted,
                        workload: &fast,
                        rates: None,
                        hosts: None,
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
            link_samples,
            ..
        } = select(&inventory, &workload, &slow, 55).unwrap()
        else {
            panic!("cheaper placement")
        };
        assert_eq!(resources.hosts, BTreeMap::from([(c, memory(9, 5))]));
        assert_eq!(cost.network_ns, 0);
        assert!(incumbent.unwrap().network_ns > 0);
        assert!(
            link_samples
                .iter()
                .map(|sample| (sample.source, sample.target))
                .eq([(a, b), (b, a)])
        );
    }

    #[test]
    fn slot_memory_is_bound_to_the_selected_and_incumbent_devices() {
        let [a, b] = [1, 2].map(host);
        let mut light = worker(a, 0, 2_000_000_000);
        light.device_bytes = 8;
        let large = worker(a, 1, 100_000_000_000);
        let mut reserved_for_other_use = worker(a, 2, 300_000_000_000);
        reserved_for_other_use.device_bytes = 23;
        let mut small_profile = worker(a, 3, 200_000_000_000);
        small_profile.profile.as_mut().unwrap().memory_bytes = 16;
        let group = test_group();
        let mut inventory = PlacementInventory {
            group,
            latest: None,
            active: None,
            hosts: [a, b]
                .into_iter()
                .map(|host| PlacementHost {
                    host,
                    capacity: memory(32, 16),
                    reserved: HostMemory::default(),
                    eligible: true,
                })
                .collect(),
            workers: vec![reserved_for_other_use, large, small_profile, light],
        };
        let recipe = NodeId::derive("slot-memory");
        let mut workload = workload();
        workload.switch_ns = u64::MAX;
        workload.recipes = vec![Recipe {
            id: recipe,
            work: vec![1_000_000, 100_000_000],
            device_bytes_by_slot: Some(vec![8, 24]),
            compute: ComputeOrder::Parallel,
            communication: Communication::Directed { edges: Vec::new() },
        }];
        let Decision::Prepare { resources, .. } = select(&inventory, &workload, &[], 55).unwrap()
        else {
            panic!("the small device fits the first slot");
        };
        assert!(
            resources
                .workers
                .iter()
                .map(|(id, demand)| (*id, demand.device_bytes))
                .eq([(DeviceId::new(a, 0), 8), (DeviceId::new(a, 1), 24)])
        );
        let set = select_set(
            &[PlacementGroup {
                inventory: &inventory,
                workload: &workload,
                rates: None,
                hosts: None,
            }],
            &[],
            55,
        )
        .unwrap()
        .unwrap();
        assert_eq!(set[0].resources, resources);
        assert_eq!(set[0].recipe, Some(recipe));

        let version = GroupVersion { group, epoch: 1 };
        inventory.latest = Some((version, GroupPhase::Active));
        inventory.hosts[0].reserved = resources.hosts[&a];
        inventory.active = Some(ActivePlacement {
            version,
            workers: resources
                .workers
                .iter()
                .rev()
                .map(|(id, demand)| (*id, *demand))
                .collect(),
            hosts: resources
                .hosts
                .iter()
                .map(|(id, demand)| (*id, *demand))
                .collect(),
        });
        for worker in &mut inventory.workers {
            if resources.workers.contains_key(&worker.device) {
                worker.grant = Some(WorkerGrant {
                    version,
                    device: worker.device,
                    incarnation: 1,
                    fence: 1,
                });
            }
        }
        let mut other_light = worker(b, 0, 4_000_000_000);
        other_light.device_bytes = 8;
        inventory
            .workers
            .extend([other_light, worker(b, 1, 200_000_000_000)]);
        assert!(matches!(
            select_with_recipes(&inventory, &workload, &[], 55, None, Some(recipe)).unwrap(),
            Decision::Keep { .. }
        ));

        workload.recipes[0].device_bytes_by_slot = Some(vec![8, 25]);
        let Decision::Prepare {
            resources,
            incumbent: None,
            ..
        } = select_with_recipes(&inventory, &workload, &[], 55, None, Some(recipe)).unwrap()
        else {
            panic!("the old heavy-slot reservation is too small");
        };
        assert_eq!(resources.workers[&DeviceId::new(b, 1)].device_bytes, 25);
        for invalid in [vec![8], vec![8, 0]] {
            workload.recipes[0].device_bytes_by_slot = Some(invalid);
            assert!(workload.validate(false).is_err());
        }
    }

    #[test]
    fn asymmetric_slots_choose_devices_under_online_resource_and_link_limits() {
        let [a, b, c] = [1, 2, 3].map(host);
        let version = GroupVersion {
            group: test_group(),
            epoch: 1,
        };
        let mut incumbent = [worker(a, 0, 5_000_000_000), worker(a, 1, 95_000_000_000)];
        for worker in &mut incumbent {
            worker.grant = Some(WorkerGrant {
                version,
                device: worker.device,
                incarnation: 1,
                fence: 1,
            });
        }
        let mut undersized = worker(b, 3, 300_000_000_000);
        undersized.device_bytes = 15;
        let held = memory(9, 5);
        let mut inventory = PlacementInventory {
            group: version.group,
            latest: Some((version, GroupPhase::Active)),
            active: Some(ActivePlacement {
                version,
                workers: incumbent
                    .iter()
                    .map(|worker| {
                        (
                            worker.device,
                            WorkerDemand {
                                device_bytes: 16,
                                incarnation: 1,
                            },
                        )
                    })
                    .collect(),
                hosts: vec![(a, held)],
            }),
            hosts: [a, b, c]
                .into_iter()
                .map(|host| PlacementHost {
                    host,
                    capacity: if host == c { memory(5, 3) } else { held },
                    reserved: if host == a {
                        held
                    } else {
                        HostMemory::default()
                    },
                    eligible: true,
                })
                .collect(),
            workers: incumbent
                .into_iter()
                .chain([
                    worker(b, 2, 90_000_000_000),
                    worker(c, 0, 200_000_000_000),
                    worker(b, 1, 100_000_000_000),
                    undersized,
                    worker(b, 0, 1_000_000_000),
                ])
                .collect(),
        };
        let recipe = NodeId::derive("asymmetric");
        let mut workload = workload();
        workload.switch_ns = 150_000;
        workload.recipes = vec![Recipe {
            id: recipe,
            work: vec![1_000_000, 100_000_000],
            device_bytes_by_slot: None,
            compute: ComputeOrder::Parallel,
            communication: Communication::PeerAllReduce {
                replicated_egress_bytes: None,
                calls: vec![AllReduceCalls {
                    count: 1,
                    bytes: 8,
                    element_bytes: 2,
                }],
            },
        }];
        let select = |inventory: &PlacementInventory, links: &[DirectedLink]| {
            select_with_recipes(inventory, &workload, links, 55, None, Some(recipe)).unwrap()
        };
        let slow = links_between(b, c, 1_000, 54);
        let Decision::Prepare {
            resources, cost, ..
        } = select(&inventory, &slow)
        else {
            panic!("the best local slot mapping pays for the switch");
        };
        assert!(
            resources
                .workers
                .keys()
                .copied()
                .eq([DeviceId::new(b, 0), DeviceId::new(b, 1)])
        );
        assert_eq!(
            cost,
            Cost {
                compute_ns: 1_000_000,
                network_ns: 0
            }
        );
        inventory.hosts[1].capacity.pinned_bytes -= 1;
        assert!(matches!(select(&inventory, &slow), Decision::Keep { .. }));
        let fast = links_between(b, c, 1_000_000_000, 54);
        let Decision::Prepare {
            resources, cost, ..
        } = select(&inventory, &fast)
        else {
            panic!("a faster link makes the split placement cheaper");
        };
        assert!(
            resources
                .workers
                .keys()
                .copied()
                .eq([DeviceId::new(b, 1), DeviceId::new(c, 0)])
        );
        assert_eq!(
            cost,
            Cost {
                compute_ns: 500_000,
                network_ns: 108
            }
        );
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
                device_bytes_by_slot: None,
                compute: ComputeOrder::Parallel,
                communication: Communication::PeerAllReduce {
                    replicated_egress_bytes: None,
                    calls: vec![AllReduceCalls {
                        count: 57,
                        bytes: 1024,
                        element_bytes: 2,
                    }],
                },
            },
            Recipe {
                id: sequential,
                work: vec![150_000_000, 50_000_000],
                device_bytes_by_slot: None,
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
            link_samples,
            ..
        } = select(&inventory, &workload, &links, 55).unwrap()
        else {
            panic!("the directed recipe fits the slower link");
        };
        assert_eq!(id, sequential);
        assert_eq!(selected, resources);
        assert_eq!(cost.compute_ns, 125_000_000);
        assert_eq!(cost.network_ns, 41_001_024);
        assert!(
            link_samples
                .iter()
                .map(|sample| (sample.source, sample.target))
                .eq([(a, b)])
        );
    }

    #[test]
    fn peer_all_reduce_charges_each_call_and_overlaps_peer_rtt() {
        let [a, b, c] = [1, 2, 3].map(host);
        let ranks = [
            worker(a, 0, 1),
            worker(a, 1, 1),
            worker(b, 0, 1),
            worker(c, 0, 1),
        ];
        let links: Vec<_> = [(a, b), (a, c), (b, c)]
            .into_iter()
            .flat_map(|(left, right)| links_between(left, right, 1_000_000_000, 50))
            .collect();
        let workload = workload();
        let model = Model::new(&workload, &links, 55, None).unwrap();
        let workers: Vec<_> = ranks
            .iter()
            .map(|rank| model.eligible(rank).unwrap())
            .collect();
        let calls = [
            AllReduceCalls {
                count: 399,
                bytes: 2048,
                element_bytes: 2,
            },
            AllReduceCalls {
                count: 57,
                bytes: 10240,
                element_bytes: 2,
            },
        ];
        // The two-worker host services four cross-host worker pairs per direction.
        assert_eq!(
            model.link_costs().peer_all_reduce(&workers, &calls, None),
            Some(399 * (100 + 4 * 2048) + 57 * (100 + 4 * 10240))
        );
        assert_eq!(
            model
                .link_costs()
                .peer_all_reduce(&workers[..2], &calls, None),
            Some(0)
        );
        let incomplete = Model::new(&workload, &links[..5], 55, None).unwrap();
        assert_eq!(
            incomplete
                .link_costs()
                .peer_all_reduce(&workers, &calls, None),
            None
        );
    }

    #[test]
    fn peer_all_reduce_prices_uneven_shards_in_both_directions() {
        let [a, b] = [1, 2].map(host);
        let ranks = [worker(a, 0, 1), worker(a, 1, 1), worker(b, 0, 1)];
        let mut links = links_between(a, b, 1_000_000_000, 50);
        links[1].bytes_per_second = 500_000_000;
        let workload = workload();
        let model = Model::new(&workload, &links, 55, None).unwrap();
        let workers: Vec<_> = ranks
            .iter()
            .map(|rank| model.eligible(rank).unwrap())
            .collect();
        let calls = [AllReduceCalls {
            count: 1,
            bytes: 65_540,
            element_bytes: 2,
        }];
        // A owns 43,694 bytes and B owns 21,846. B's slower egress is the
        // bottleneck in both sequential phases, with different payload sizes.
        assert_eq!(
            model.link_costs().peer_all_reduce(&workers, &calls, None),
            Some(2 * 43_694 + 2 * 43_692 + 2 * 100)
        );
    }

    #[test]
    fn profile_selection_switches_all_reduce_recipe_with_link_latency() {
        let [a, b, c] = [1, 2, 3].map(host);
        let inventory = PlacementInventory {
            group: test_group(),
            latest: None,
            active: None,
            hosts: [a, b, c]
                .into_iter()
                .map(|host| PlacementHost {
                    host,
                    capacity: memory(32, 16),
                    reserved: memory(0, 0),
                    eligible: true,
                })
                .collect(),
            workers: [a, b, c].map(|host| worker(host, 0, 1_000_000_000)).into(),
        };
        let mut workload = workload();
        workload.widths = vec![3];
        let default = NodeId::from_bytes([3; 16]);
        let tuned = NodeId::from_bytes([4; 16]);
        workload.recipes = [(default, None), (tuned, Some(256 * 1024))]
            .into_iter()
            .map(|(id, replicated_egress_bytes)| Recipe {
                id,
                work: vec![10; 3],
                device_bytes_by_slot: None,
                compute: ComputeOrder::Parallel,
                communication: Communication::PeerAllReduce {
                    calls: vec![AllReduceCalls {
                        count: 1,
                        bytes: 128 * 1024,
                        element_bytes: 2,
                    }],
                    replicated_egress_bytes,
                },
            })
            .collect();
        let mut links: Vec<_> = [(a, b), (a, c), (b, c)]
            .into_iter()
            .flat_map(|(left, right)| links_between(left, right, 1_000_000_000, 50))
            .collect();
        let quote = |links: &[DirectedLink]| {
            let Decision::Prepare {
                resources,
                recipe: Some(id),
                cost,
                ..
            } = select(&inventory, &workload, links, 55).unwrap()
            else {
                panic!("expected a priced TP recipe");
            };
            (resources, id, cost.network_ns)
        };
        let (before, low_rtt, low_cost) = quote(&links);
        for link in &mut links {
            link.rtt_ns = 1_000_000;
        }
        let (after, high_rtt, high_cost) = quote(&links);
        assert_eq!(before, after);
        assert_eq!(low_rtt, default);
        assert_eq!(low_cost, 174_968);
        assert_eq!(high_rtt, bind_peer_all_reduce_recipe(tuned, 3, 256 * 1024));
        assert_ne!(high_rtt, tuned);
        assert_eq!(high_rtt.to_hex(), "a04f7569b5d79140032b71bdf79dd49a");
        assert_eq!(high_cost, 1_262_144);
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
