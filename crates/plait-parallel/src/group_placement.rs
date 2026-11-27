//! Topology-aware placement of native execution groups.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::group::{
    ComputeRate, GroupPhase, GroupVersion, HostMemory, PlacementInventory, PlacementWorker,
    ResourcePlan, WorkerDemand,
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
    },
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
    let widths: BTreeSet<_> = workload
        .widths
        .iter()
        .map(|&width| width as usize)
        .collect();
    if widths.is_empty()
        || widths.contains(&0)
        || workload.device_bytes == 0
        || workload.operations == 0
        || workload.per_worker_host.pinned_bytes > workload.per_worker_host.bytes
        || workload.per_host.pinned_bytes > workload.per_host.bytes
        || (workload.compute_work == 0 && workload.bytes_per_peer == 0)
        || (rates.is_some() && workload.compute_work == 0)
    {
        return Err("invalid group placement workload");
    }

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
    if let Some((version, GroupPhase::Preparing | GroupPhase::Ready)) = inventory.latest {
        return Ok(Decision::Pending(version));
    }

    let model = Model {
        workload,
        links: fresh,
        rates: rates
            .map(|rates| {
                let rates_by_device: BTreeMap<_, _> =
                    rates.iter().map(|rate| (rate.device, *rate)).collect();
                if rates_by_device.len() != rates.len() {
                    return Err("duplicate compute rate");
                }
                Ok(rates_by_device)
            })
            .transpose()?,
    };
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
        let Some(cost) = model.cost(&workers) else {
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
    let mut hosts: Vec<_> = inventory
        .hosts
        .iter()
        .filter(|host| host.eligible)
        .collect();
    hosts.sort_unstable_by_key(|host| host.host);
    for host in hosts {
        let Some(mut workers) = available.remove(&host.host) else {
            continue;
        };
        workers.sort_unstable_by_key(|worker| (std::cmp::Reverse(worker.compute_rate), worker.id));
        let fit = (1..=workers.len().min(max_width))
            .take_while(|&count| {
                model.host_demand(count).is_some_and(|needed| {
                    needed.bytes <= host.capacity.bytes.saturating_sub(host.reserved.bytes)
                        && needed.pinned_bytes
                            <= host
                                .capacity
                                .pinned_bytes
                                .saturating_sub(host.reserved.pinned_bytes)
                })
            })
            .count();
        let mut next = states.clone();
        for (&width, candidates) in &states {
            for candidate in candidates {
                for count in 1..=fit.min(max_width - width) {
                    let mut devices = candidate.workers.clone();
                    devices.extend_from_slice(&workers[..count]);
                    if let Some(cost) = model.cost(&devices) {
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
        states = next;
    }
    let best = states
        .into_iter()
        .filter(|(width, _)| widths.contains(width))
        .flat_map(|(_, ranks)| ranks.into_iter().take(1))
        .min_by(compare);

    let incumbent_cost = incumbent.map(|(_, cost)| cost);
    if let Some((version, cost)) = incumbent {
        let operations = u128::from(workload.operations);
        let ready_cost = cost.operation_ns().saturating_mul(operations);
        let use_new = best.as_ref().is_some_and(|candidate| {
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
    let Some(best) = best else {
        return Ok(Decision::NoPlacement);
    };
    let hosts = count_hosts(&best.workers)
        .into_iter()
        .map(|(host, count)| (host, model.host_demand(count).expect("fitted host")))
        .collect();
    let workers = best
        .workers
        .into_iter()
        .map(|worker| {
            (
                worker.id,
                WorkerDemand {
                    incarnation: worker.incarnation,
                    device_bytes: workload.device_bytes,
                },
            )
        })
        .collect();
    Ok(Decision::Prepare {
        resources: ResourcePlan { workers, hosts },
        cost: best.cost,
        incumbent: incumbent_cost,
    })
}

impl Model<'_> {
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

    fn cost(&self, workers: &[Worker]) -> Option<Cost> {
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
        let mut network_ns = 0;
        if self.workload.bytes_per_peer > 0 {
            let counts = count_hosts(workers);
            for (&src, &left) in &counts {
                let mut incoming = 0u64;
                let mut outgoing = 0u64;
                for (&dst, &right) in counts.iter().filter(|&(&host, _)| host != src) {
                    let traffic = u128::from(self.workload.bytes_per_peer)
                        .saturating_mul(left as u128)
                        .saturating_mul(right as u128);
                    outgoing = outgoing.saturating_add(self.link_time(src, dst, traffic)?);
                    incoming = incoming.saturating_add(self.link_time(dst, src, traffic)?);
                }
                network_ns = network_ns.max(incoming).max(outgoing);
            }
        }
        Some(Cost {
            compute_ns,
            network_ns,
        })
    }

    fn link_time(&self, source: NodeId, target: NodeId, bytes: u128) -> Option<u64> {
        let link = self.links.get(&(source, target))?;
        Some(
            ns(
                bytes.saturating_mul(NANOS),
                u128::from(link.bytes_per_second),
            )
            .saturating_add(link.rtt_ns),
        )
    }
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
        } = select(&inventory, &workload, &slow, 55).unwrap()
        else {
            panic!("cheaper placement")
        };
        assert_eq!(resources.hosts, BTreeMap::from([(c, memory(9, 5))]));
        assert_eq!(cost.network_ns, 0);
        assert!(incumbent.unwrap().network_ns > 0);
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
