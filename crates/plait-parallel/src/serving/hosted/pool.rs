use std::collections::{BTreeMap, BTreeSet};

use plait_core::NodeId;
use plait_core::serving::{
    GroupId, GroupPlan, HostMemory, HostedAddress, HostedLinkSample, HostedPdKv, HostedPdLaunch,
    HostedPlanning, HostedPortUse, HostedRecipe, ServingResources, ServingRole, WorkerDemand,
    WorkerResources, hosted_all_reduce_profile,
};
use serde::{Deserialize, Serialize};

use super::super::{ServingCandidate, ServingForecast, SloProfile, valid_rate};
use super::pipeline::{PipelineProfile, search_pipeline};
use super::{
    HostedLinkLimit, HostedLinkPolicy, HostedTpLinkProfile, LinkEvidence, deserialize_tp_profiles,
    limit_with_evidence,
};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedPool {
    group: GroupId,
    profile: String,
    api_model: String,
    max_model_len: u32,
    gpu_memory_bps: u16,
    prefix_caching: bool,
    ports: PortRange,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replicated_egress_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    all_reduce_profiles: Vec<AllReduceProfile>,
    #[serde(default)]
    sizes: Vec<SizeProfile>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pipelines: Vec<PipelineProfile>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AllReduceProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replicated_egress_bytes: Option<u64>,
    #[serde(deserialize_with = "deserialize_tp_profiles")]
    tp_profiles: BTreeMap<u32, HostedTpLinkProfile>,
}

#[derive(Clone, Copy)]
struct AllReduceChoice<'a> {
    bytes: Option<u64>,
    profiles: Option<&'a BTreeMap<u32, HostedTpLinkProfile>>,
}

impl AllReduceChoice<'_> {
    fn name(self, profile: &str) -> Option<String> {
        match self.bytes {
            Some(bytes) => hosted_all_reduce_profile(profile, bytes),
            None => Some(profile.to_owned()),
        }
    }

    fn supports(self, width: u32) -> bool {
        self.profiles
            .is_none_or(|profiles| profiles.contains_key(&width))
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortRange {
    first: u16,
    /// Number of consecutive ports in the managed range.
    count: u16,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SizeProfile {
    tensor_parallel: u32,
    device_bytes: u64,
    host_base: HostMemory,
    host_per_worker: HostMemory,
    capacity: SloProfile,
    ready_after_ms: u64,
    transition_loss_requests: f64,
}

struct AvailableHost {
    address: HostedAddress,
    workers: Vec<WorkerResources>,
    memory: Vec<HostMemory>,
    ports: Option<[u16; 3]>,
}

pub(super) struct InventoryHost {
    pub address: HostedAddress,
    pub workers: Vec<WorkerResources>,
    pub memory: HostMemory,
    pub ports: Option<[u16; 3]>,
}

#[derive(Clone)]
struct Layout {
    ranks: BTreeMap<NodeId, u64>,
    api_host: Option<NodeId>,
    link_rps: f64,
}

// Keep alternatives with different progress, host counts and ability to host
// the API. A wider search improves coverage at the cost of planning time.
pub(super) const LAYOUTS_PER_BUCKET: usize = 16;
type Search = BTreeMap<(u32, usize, bool), Vec<Layout>>;

#[derive(Clone, Copy)]
pub(super) enum PoolTarget<'a> {
    LegacyAggregate,
    RoutedAggregate,
    Split(ServingRole, &'a HostedPdKv),
}

impl PoolTarget<'_> {
    fn role(self) -> ServingRole {
        match self {
            Self::LegacyAggregate | Self::RoutedAggregate => ServingRole::Aggregated,
            Self::Split(role, _) => role,
        }
    }
}

pub fn generate_hosted_candidates(
    resources: &ServingResources,
    planning: &HostedPlanning,
    forecast: &ServingForecast,
    pool: &HostedPool,
    policy: &HostedLinkPolicy,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<Vec<(ServingCandidate, HostedRecipe)>, &'static str> {
    let evidence = LinkEvidence::new(policy, forecast.now_ms, links);
    generate_layouts(
        resources,
        planning,
        &forecast.model,
        pool,
        evidence,
        PoolTarget::LegacyAggregate,
    )
}

pub fn limit_hosted_pool_tp_links(
    plan: &GroupPlan,
    recipe: &HostedRecipe,
    pool: &HostedPool,
    policy: &HostedLinkPolicy,
    now_ms: u64,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<HostedLinkLimit, &'static str> {
    if plan.pipeline_parallel != 1 {
        return Err("direct hosted link quotes require a TP-only group");
    }
    let profiles = pool
        .recipe_link_profiles(&recipe.profile, plan.tensor_parallel, 1)
        .ok_or("hosted candidate lacks its AllReduce link profile")?;
    limit_with_evidence(
        plan,
        LinkEvidence::new(policy, now_ms, links).with_tp_profiles(profiles),
    )
}

pub(super) fn generate_layouts(
    resources: &ServingResources,
    planning: &HostedPlanning,
    checkpoint: &str,
    pool: &HostedPool,
    evidence: LinkEvidence<'_>,
    target: PoolTarget<'_>,
) -> Result<Vec<(ServingCandidate, HostedRecipe)>, &'static str> {
    evidence.policy.validate()?;
    let role = target.role();
    let allow_local = !matches!(target, PoolTarget::LegacyAggregate);
    pool.validate(allow_local)?;
    let mut candidates = Vec::new();
    for choice in pool.all_reduce_choices() {
        let context = evidence.for_role(role).with_tp_profiles(choice.profiles);
        candidates.extend(generate_for_choice(
            resources, planning, checkpoint, pool, context, target, choice,
        )?);
    }
    Ok(candidates)
}

fn generate_for_choice(
    resources: &ServingResources,
    planning: &HostedPlanning,
    checkpoint: &str,
    pool: &HostedPool,
    evidence: LinkEvidence<'_>,
    target: PoolTarget<'_>,
    choice: AllReduceChoice<'_>,
) -> Result<Vec<(ServingCandidate, HostedRecipe)>, &'static str> {
    let role = target.role();
    let allow_local = !matches!(target, PoolTarget::LegacyAggregate);
    let expected_epoch = resources
        .groups
        .iter()
        .find(|group| group.latest.group == pool.group)
        .map_or(0, |group| group.latest.epoch);
    let mut candidates = Vec::new();
    let profile = choice
        .name(&pool.profile)
        .expect("validated hosted profile");
    for size in pool
        .sizes
        .iter()
        .filter(|size| choice.supports(size.tensor_parallel))
    {
        let hosts = available_hosts(resources, planning, &profile, &pool.ports, size);
        for layout in search_layouts(&hosts, size.tensor_parallel, evidence, allow_local) {
            let api_host = layout.api_host.expect("layout has a host with free ports");
            let [master_port, http_port, backend_port] = hosts[&api_host]
                .ports
                .expect("ports were selected for API host");
            let recipe = HostedRecipe {
                profile: profile.clone(),
                api_host,
                hosts: layout
                    .ranks
                    .keys()
                    .map(|host| hosts[host].address.clone())
                    .collect(),
                master_port,
                http_port,
                backend_port,
                api_model: pool.api_model.clone(),
                max_model_len: pool.max_model_len,
                gpu_memory_bps: pool.gpu_memory_bps,
                prefix_caching: pool.prefix_caching,
            };
            let mut plan = GroupPlan {
                role,
                model: checkpoint.to_owned(),
                tensor_parallel: size.tensor_parallel,
                pipeline_parallel: 1,
                launch_digest: recipe.launch_digest(),
                workers: BTreeMap::new(),
                hosts: BTreeMap::new(),
            };
            for (host, count) in layout.ranks {
                let count = count as usize;
                let available = &hosts[&host];
                plan.hosts.insert(host, available.memory[count - 1]);
                plan.workers
                    .extend(available.workers[..count].iter().map(|worker| {
                        (
                            worker.device,
                            WorkerDemand {
                                incarnation: worker.incarnation,
                                device_bytes: size.device_bytes,
                            },
                        )
                    }));
            }
            bind_launch(&mut plan, &recipe, target)?;
            candidates.push((
                ServingCandidate {
                    group: pool.group,
                    expected_epoch,
                    plan,
                    capacity: size.capacity,
                    ready_after_ms: size.ready_after_ms,
                    transition_loss_requests: size.transition_loss_requests,
                },
                recipe,
            ));
        }
    }
    for pipeline in pool
        .pipelines
        .iter()
        .filter(|entry| choice.supports(entry.tensor_parallel))
    {
        let profile = choice
            .name(
                &pipeline
                    .qualified_name()
                    .expect("validated pipeline profile"),
            )
            .expect("validated hosted AllReduce profile");
        let hosts = inventory_hosts(resources, planning, &profile, &pool.ports);
        for layout in search_pipeline(pipeline, &hosts, evidence) {
            let [master_port, http_port, backend_port] = hosts[&layout.api_host]
                .ports
                .expect("pipeline API host has free ports");
            let recipe = HostedRecipe {
                profile: profile.clone(),
                api_host: layout.api_host,
                hosts: layout
                    .hosts
                    .keys()
                    .map(|host| hosts[host].address.clone())
                    .collect(),
                master_port,
                http_port,
                backend_port,
                api_model: pool.api_model.clone(),
                max_model_len: pool.max_model_len,
                gpu_memory_bps: pool.gpu_memory_bps,
                prefix_caching: pool.prefix_caching,
            };
            let mut plan = GroupPlan {
                role,
                model: checkpoint.to_owned(),
                tensor_parallel: pipeline.tensor_parallel,
                pipeline_parallel: pipeline.stages.len() as u32,
                launch_digest: [0; 32],
                workers: layout.workers,
                hosts: layout.hosts,
            };
            bind_launch(&mut plan, &recipe, target)?;
            candidates.push((
                ServingCandidate {
                    group: pool.group,
                    expected_epoch,
                    plan,
                    capacity: layout.capacity,
                    ready_after_ms: pipeline.ready_after_ms,
                    transition_loss_requests: pipeline.transition_loss_requests,
                },
                recipe,
            ));
        }
    }
    Ok(candidates)
}

fn bind_launch(
    plan: &mut GroupPlan,
    recipe: &HostedRecipe,
    target: PoolTarget<'_>,
) -> Result<(), &'static str> {
    if let PoolTarget::Split(role, kv) = target {
        let launch = HostedPdLaunch {
            role,
            kv: kv.clone(),
        };
        plan.launch_digest = launch.launch_digest(recipe);
        launch
            .validate(recipe, plan)
            .map_err(|_| "invalid hosted P/D pool recipe")
    } else {
        plan.launch_digest = recipe.launch_digest();
        if matches!(target, PoolTarget::RoutedAggregate) {
            recipe.validate_aggregated_route(plan)
        } else {
            recipe.validate(plan)
        }
        .map_err(|_| "invalid hosted pool recipe")
    }
}

impl HostedPool {
    fn all_reduce_choices(&self) -> impl Iterator<Item = AllReduceChoice<'_>> {
        let fixed = self
            .all_reduce_profiles
            .is_empty()
            .then_some(AllReduceChoice {
                bytes: self.replicated_egress_bytes,
                profiles: None,
            });
        fixed
            .into_iter()
            .chain(self.all_reduce_profiles.iter().map(|row| AllReduceChoice {
                bytes: row.replicated_egress_bytes,
                profiles: Some(&row.tp_profiles),
            }))
    }

    pub(super) fn recipe_link_profiles(
        &self,
        profile: &str,
        tp: u32,
        pp: u32,
    ) -> Option<Option<&BTreeMap<u32, HostedTpLinkProfile>>> {
        if pp == 1 {
            self.choice_for(&self.profile, profile, tp)
                .map(|choice| choice.profiles)
        } else {
            let entry = self.pipeline(profile, tp, pp)?;
            self.choice_for(&entry.qualified_name()?, profile, tp)
                .map(|choice| choice.profiles)
        }
    }

    fn choice_for(&self, base: &str, profile: &str, tp: u32) -> Option<AllReduceChoice<'_>> {
        self.all_reduce_choices()
            .find(|choice| choice.supports(tp) && choice.name(base).as_deref() == Some(profile))
    }

    pub(super) fn group(&self) -> GroupId {
        self.group
    }

    pub(super) fn api_model(&self) -> &str {
        &self.api_model
    }

    pub(super) fn capacity(&self, profile: &str, tensor_parallel: u32) -> Option<SloProfile> {
        self.choice_for(&self.profile, profile, tensor_parallel)?;
        self.sizes
            .iter()
            .find(|size| size.tensor_parallel == tensor_parallel)
            .map(|size| size.capacity)
    }

    pub(super) fn pipeline(
        &self,
        profile: &str,
        tensor_parallel: u32,
        pipeline_parallel: u32,
    ) -> Option<&PipelineProfile> {
        self.pipelines.iter().find(|entry| {
            entry.tensor_parallel == tensor_parallel
                && entry.stages.len() == pipeline_parallel as usize
                && entry
                    .qualified_name()
                    .is_some_and(|base| self.choice_for(&base, profile, tensor_parallel).is_some())
        })
    }

    pub(super) fn has_pipelines(&self) -> bool {
        !self.pipelines.is_empty()
    }

    pub(super) fn has_all_reduce_profiles(&self) -> bool {
        !self.all_reduce_profiles.is_empty()
    }

    pub(super) fn validate(&self, allow_local: bool) -> Result<(), &'static str> {
        let end = u32::from(self.ports.first) + u32::from(self.ports.count);
        if self.ports.first == 0 || self.ports.count < 3 || end > u32::from(u16::MAX) + 1 {
            return Err("hosted pool requires a port range with at least three valid ports");
        }
        let mut degrees = BTreeSet::new();
        if (self.sizes.is_empty() && self.pipelines.is_empty())
            || self.sizes.iter().any(|size| {
                size.tensor_parallel < if allow_local { 1 } else { 2 }
                    || !degrees.insert(size.tensor_parallel)
                    || size.device_bytes == 0
                    || size.host_base.pinned_bytes > size.host_base.bytes
                    || size.host_per_worker.pinned_bytes > size.host_per_worker.bytes
                    || !valid_rate(size.capacity.rps)
                    || !valid_rate(size.transition_loss_requests)
            })
        {
            return Err("hosted pool requires distinct supported TP sizes and valid budgets");
        }
        let mut shapes = BTreeSet::new();
        for pipeline in &self.pipelines {
            let name = pipeline.qualified_name();
            if !allow_local
                || !pipeline.validate()
                || name.is_none()
                || !shapes.insert((name, pipeline.tensor_parallel))
            {
                return Err(
                    "hosted P/D pool requires distinct supported pipelines and valid stage budgets",
                );
            }
        }
        self.validate_all_reduce_profiles(&degrees)?;
        Ok(())
    }

    fn validate_all_reduce_profiles(&self, widths: &BTreeSet<u32>) -> Result<(), &'static str> {
        if self.replicated_egress_bytes.is_some() && self.has_all_reduce_profiles() {
            return Err("hosted all_reduce_profiles cannot be combined with a fixed cap");
        }
        let mut caps = BTreeSet::new();
        for choice in self.all_reduce_choices() {
            if !caps.insert(choice.bytes)
                || choice.name(&self.profile).is_none()
                || self.pipelines.iter().any(|entry| {
                    entry
                        .qualified_name()
                        .and_then(|base| choice.name(&base))
                        .is_none()
                })
            {
                return Err("hosted AllReduce profiles require distinct caps and valid names");
            }
            if let Some(profiles) = choice.profiles {
                if profiles.is_empty() {
                    return Err("hosted AllReduce profiles require at least one TP width");
                }
                for (&width, profile) in profiles {
                    if !widths.contains(&width)
                        && !self
                            .pipelines
                            .iter()
                            .any(|entry| entry.tensor_parallel == width)
                    {
                        return Err("hosted AllReduce profile has no matching TP width");
                    }
                    profile.validate()?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl HostedPool {
    pub(super) fn fixture(group: GroupId, profile: &str, api_model: &str) -> Self {
        let memory = HostMemory {
            bytes: 8,
            pinned_bytes: 1,
        };
        Self {
            group,
            profile: profile.into(),
            api_model: api_model.into(),
            max_model_len: 1024,
            gpu_memory_bps: 8000,
            prefix_caching: false,
            ports: PortRange {
                first: 20000,
                count: 6,
            },
            replicated_egress_bytes: None,
            all_reduce_profiles: Vec::new(),
            sizes: vec![SizeProfile {
                tensor_parallel: 1,
                device_bytes: 8,
                host_base: memory,
                host_per_worker: memory,
                capacity: SloProfile {
                    rps: 50.0,
                    valid_until_ms: 5000,
                },
                ready_after_ms: 10,
                transition_loss_requests: 0.0,
            }],
            pipelines: Vec::new(),
        }
    }

    pub(super) fn with_capacity(mut self, tp: u32, rps: f64) -> Self {
        self.sizes[0].tensor_parallel = tp;
        self.sizes[0].capacity.rps = rps;
        self
    }

    pub(super) fn with_all_reduce_profiles(
        mut self,
        width: u32,
        choices: impl IntoIterator<Item = (Option<u64>, HostedTpLinkProfile)>,
    ) -> Self {
        self.all_reduce_profiles = choices
            .into_iter()
            .map(|(replicated_egress_bytes, profile)| AllReduceProfile {
                replicated_egress_bytes,
                tp_profiles: BTreeMap::from([(width, profile)]),
            })
            .collect();
        self
    }
}

fn available_hosts(
    resources: &ServingResources,
    planning: &HostedPlanning,
    profile: &str,
    ports: &PortRange,
    size: &SizeProfile,
) -> BTreeMap<NodeId, AvailableHost> {
    inventory_hosts(resources, planning, profile, ports)
        .into_iter()
        .filter_map(|(host, available)| {
            let mut workers: Vec<_> = available
                .workers
                .into_iter()
                .filter(|worker| worker.device_bytes >= size.device_bytes)
                .collect();
            let max_ranks = workers.len().min(size.tensor_parallel as usize);
            let memory: Vec<_> = (1..=max_ranks as u64)
                .map_while(|count| {
                    Some(HostMemory {
                        bytes: size
                            .host_base
                            .bytes
                            .checked_add(size.host_per_worker.bytes.checked_mul(count)?)?,
                        pinned_bytes: size
                            .host_base
                            .pinned_bytes
                            .checked_add(size.host_per_worker.pinned_bytes.checked_mul(count)?)?,
                    })
                })
                .take_while(|memory| {
                    memory.bytes <= available.memory.bytes
                        && memory.pinned_bytes <= available.memory.pinned_bytes
                })
                .collect();
            if memory.is_empty() {
                return None;
            }
            workers.truncate(memory.len());
            Some((
                host,
                AvailableHost {
                    address: available.address,
                    workers,
                    memory,
                    ports: available.ports,
                },
            ))
        })
        .collect()
}

fn inventory_hosts(
    resources: &ServingResources,
    planning: &HostedPlanning,
    profile: &str,
    ports: &PortRange,
) -> BTreeMap<NodeId, InventoryHost> {
    let memory: BTreeMap<_, _> = resources
        .hosts
        .iter()
        .map(|host| (host.host, host))
        .collect();
    let inventories: BTreeMap<_, _> = planning
        .hosts
        .iter()
        .filter(|inventory| inventory.profiles.contains(profile))
        .map(|inventory| (inventory.host, inventory))
        .collect();
    let mut workers: BTreeMap<NodeId, Vec<_>> = BTreeMap::new();
    for &worker in &resources.workers {
        if worker.grant.is_none()
            && inventories
                .get(&worker.device.node)
                .and_then(|inventory| inventory.devices.get(&worker.device.ordinal))
                == Some(&worker.incarnation)
        {
            workers.entry(worker.device.node).or_default().push(worker);
        }
    }
    workers
        .into_iter()
        .filter_map(|(host, mut workers)| {
            let limits = memory.get(&host)?;
            let bytes = limits.capacity.bytes.saturating_sub(limits.reserved.bytes);
            let pinned = limits
                .capacity
                .pinned_bytes
                .saturating_sub(limits.reserved.pinned_bytes);
            workers.sort_by_key(|worker| worker.device.ordinal);
            Some((
                host,
                InventoryHost {
                    address: HostedAddress {
                        host,
                        address: inventories[&host].address,
                    },
                    workers,
                    memory: HostMemory {
                        bytes,
                        pinned_bytes: pinned,
                    },
                    ports: free_ports(host, ports, &planning.occupied),
                },
            ))
        })
        .collect()
}

fn free_ports(host: NodeId, range: &PortRange, occupied: &[HostedPortUse]) -> Option<[u16; 3]> {
    let occupied: BTreeSet<_> = occupied
        .iter()
        .filter(|entry| entry.host == host)
        .flat_map(|entry| entry.ports)
        .collect();
    (0..range.count / 3)
        .map(|slot| {
            let start = range.first + 3 * slot;
            [start, start + 1, start + 2]
        })
        .find(|ports| ports.iter().all(|port| !occupied.contains(port)))
}

fn search_layouts(
    hosts: &BTreeMap<NodeId, AvailableHost>,
    tensor_parallel: u32,
    evidence: LinkEvidence<'_>,
    allow_local: bool,
) -> Vec<Layout> {
    let mut search = Search::from([(
        (0, 0, false),
        vec![Layout {
            ranks: BTreeMap::new(),
            api_host: None,
            link_rps: f64::INFINITY,
        }],
    )]);
    for (&host, available) in hosts {
        let mut next = search.clone();
        for (&(used, ..), layouts) in &search {
            let remaining = (tensor_parallel - used).min(available.workers.len() as u32);
            for count in 1..=remaining {
                for layout in layouts {
                    let mut layout = layout.clone();
                    layout.ranks.insert(host, u64::from(count));
                    layout.link_rps = match evidence.limit_tp(tensor_parallel, &layout.ranks) {
                        HostedLinkLimit::Local => f64::INFINITY,
                        HostedLinkLimit::MissingTpProfile { .. }
                        | HostedLinkLimit::MissingLink { .. }
                        | HostedLinkLimit::ExceedsTpRttBudget { .. } => continue,
                        HostedLinkLimit::Measured { rps, .. } => rps,
                    };
                    if layout.api_host.is_none() && available.ports.is_some() {
                        layout.api_host = Some(host);
                    }
                    let bucket = (
                        used + count,
                        layout.ranks.len().min(2),
                        layout.api_host.is_some(),
                    );
                    next.entry(bucket).or_default().push(layout);
                }
            }
        }
        for layouts in next.values_mut() {
            layouts.sort_by(|left, right| {
                right
                    .link_rps
                    .total_cmp(&left.link_rps)
                    .then_with(|| left.ranks.len().cmp(&right.ranks.len()))
                    .then_with(|| left.ranks.cmp(&right.ranks))
            });
            layouts.truncate(LAYOUTS_PER_BUCKET);
        }
        search = next;
    }
    let mut layouts = search
        .remove(&(tensor_parallel, 2, true))
        .unwrap_or_default();
    if allow_local {
        layouts.extend(
            search
                .remove(&(tensor_parallel, 1, true))
                .unwrap_or_default(),
        );
    }
    layouts
}

#[cfg(test)]
mod tests {
    use plait_core::DeviceId;
    use plait_core::serving::{GroupVersion, HostResources, HostedInventory, WorkerGrant};

    use super::*;
    use crate::serving::HostedTpRttBudget;

    #[test]
    fn pool_combines_asymmetric_hosts_and_skips_stale_devices_and_held_ports() {
        let [a, b, c] = [1, 2, 3].map(|n| NodeId::from_bytes([n; 16]));
        let group = GroupId(NodeId::derive("hosted-pool"));
        let version = GroupVersion { group, epoch: 1 };
        let worker = |node, ordinal, incarnation, device_bytes, busy: bool| {
            let device = DeviceId::new(node, ordinal);
            WorkerResources {
                device,
                incarnation,
                device_bytes,
                grant: busy.then_some(WorkerGrant {
                    device,
                    version,
                    incarnation,
                    fence: 1,
                }),
            }
        };
        let resources = ServingResources {
            hosts: [(a, 8), (b, 2), (c, 8)]
                .into_iter()
                .map(|(host, pinned_bytes)| HostResources {
                    host,
                    capacity: HostMemory {
                        bytes: 100,
                        pinned_bytes,
                    },
                    reserved: HostMemory::default(),
                })
                .collect(),
            workers: vec![
                worker(a, 0, 2, 16, false),
                worker(a, 1, 1, 16, true),
                worker(a, 2, 1, 16, false),
                worker(a, 3, 1, 16, false),
                worker(a, 4, 1, 4, false),
                worker(b, 0, 1, 16, false),
                worker(b, 1, 1, 16, false),
                worker(c, 0, 1, 16, false),
            ],
            groups: Vec::new(),
        };
        let planning = HostedPlanning {
            hosts: [(a, 5), (b, 2), (c, 1)]
                .into_iter()
                .map(|(host, devices)| HostedInventory {
                    host,
                    address: std::net::Ipv4Addr::new(192, 0, 2, host.as_bytes()[0]).into(),
                    profiles: BTreeSet::from(["weights".into()]),
                    devices: (0..devices).map(|ordinal| (ordinal, 1)).collect(),
                })
                .collect(),
            occupied: vec![HostedPortUse {
                host: a,
                ports: [20001, 20020, 20021],
            }],
        };
        let forecast = ServingForecast {
            checkpoints: None,
            model: "model".into(),
            demand_rps: 8.0,
            now_ms: 10,
            horizon_ms: 60000,
            minimum_gain_requests: 0.0,
            groups: Vec::new(),
            link_bytes_per_second: BTreeMap::new(),
            paths: Vec::new(),
        };
        let pool = HostedPool {
            group,
            profile: "weights".into(),
            replicated_egress_bytes: None,
            all_reduce_profiles: Vec::new(),
            api_model: "model".into(),
            max_model_len: 1024,
            gpu_memory_bps: 8000,
            prefix_caching: false,
            ports: PortRange {
                first: 20000,
                count: 6,
            },
            sizes: vec![SizeProfile {
                tensor_parallel: 4,
                device_bytes: 8,
                host_base: HostMemory {
                    bytes: 5,
                    pinned_bytes: 1,
                },
                host_per_worker: HostMemory {
                    bytes: 10,
                    pinned_bytes: 1,
                },
                capacity: SloProfile {
                    rps: 8.0,
                    valid_until_ms: 100,
                },
                ready_after_ms: 10,
                transition_loss_requests: 0.0,
            }],
            pipelines: Vec::new(),
        };
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: Some(100),
            max_sample_age_ms: 20,
            tp_rtt_budget: None,
            tp_profiles: BTreeMap::new(),
            pd_tp_profiles: Default::default(),
        };
        let mut links = BTreeMap::new();
        for from in [a, b, c] {
            for to in [a, b, c].into_iter().filter(|&to| to != from) {
                links.insert(
                    (from, to),
                    HostedLinkSample {
                        target: to,
                        target_session: to,
                        sampled_at_ms: 1,
                        bytes_per_second: 1_000_000,
                        rtt_ns: 100,
                    },
                );
            }
        }
        let layouts =
            generate_hosted_candidates(&resources, &planning, &forecast, &pool, &policy, &links)
                .unwrap();
        assert_eq!(layouts.len(), 1);
        let (candidate, recipe) = &layouts[0];
        assert_eq!(recipe.api_host, a);
        assert_eq!(HostedPortUse::from(recipe).ports, [20003, 20004, 20005]);
        assert_eq!(
            candidate.plan.workers.keys().copied().collect::<Vec<_>>(),
            [
                DeviceId::new(a, 2),
                DeviceId::new(a, 3),
                DeviceId::new(b, 0),
                DeviceId::new(c, 0),
            ]
        );
        assert_eq!(
            candidate.plan.hosts[&b],
            HostMemory {
                bytes: 15,
                pinned_bytes: 2
            }
        );
        let mut capped = HostedPool::fixture(group, "weights", "model").with_capacity(4, 8.0);
        capped.replicated_egress_bytes = Some(0);
        let qualified = hosted_all_reduce_profile("weights", 0).unwrap();
        let mut inventory = planning.clone();
        for host in inventory.hosts.iter_mut().filter(|host| host.host != c) {
            host.profiles.insert(qualified.clone());
        }
        let candidates = |inventory: &HostedPlanning| {
            generate_hosted_candidates(&resources, inventory, &forecast, &capped, &policy, &links)
                .unwrap()
        };
        // The four-worker layout requires all three hosts, including C.
        assert!(candidates(&inventory).is_empty());
        inventory
            .hosts
            .iter_mut()
            .find(|host| host.host == c)
            .unwrap()
            .profiles
            .insert(qualified.clone());
        let candidates = candidates(&inventory);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].1.profile, qualified);
        assert_ne!(
            candidates[0].0.plan.launch_digest,
            candidate.plan.launch_digest
        );
        let budget = HostedLinkPolicy {
            tp_rtt_budget: Some(HostedTpRttBudget {
                serial_rounds: 57,
                max_wait_ns: 6_000,
            }),
            ..policy.clone()
        };
        assert_eq!(
            generate_hosted_candidates(&resources, &planning, &forecast, &pool, &budget, &links)
                .unwrap()
                .len(),
            1,
        );
        let calibrated = HostedLinkPolicy {
            bytes_per_rank_pair: None,
            max_sample_age_ms: 20,
            tp_rtt_budget: None,
            tp_profiles: BTreeMap::from([(
                4,
                crate::serving::HostedTpLinkProfile {
                    bytes_per_rank_pair: 100,
                    tp_rtt_budget: budget.tp_rtt_budget,
                },
            )]),
            pd_tp_profiles: Default::default(),
        };
        assert_eq!(
            generate_hosted_candidates(
                &resources,
                &planning,
                &forecast,
                &pool,
                &calibrated,
                &links
            )
            .unwrap()
            .len(),
            1,
        );
        links.get_mut(&(b, a)).unwrap().rtt_ns = 120;
        assert!(
            generate_hosted_candidates(&resources, &planning, &forecast, &pool, &budget, &links)
                .unwrap()
                .is_empty()
        );
        links.remove(&(b, a));
        assert!(
            generate_hosted_candidates(&resources, &planning, &forecast, &pool, &policy, &links)
                .unwrap()
                .is_empty()
        );
    }
}
