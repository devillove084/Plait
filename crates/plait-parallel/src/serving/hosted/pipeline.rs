//! Placement and traffic for the hosted vLLM worker order.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::serving::{HostMemory, WorkerDemand, hosted_pipeline_profile};
use plait_core::{DeviceId, NodeId};
use serde::{Deserialize, Serialize};

use super::super::{SloProfile, valid_rate};
use super::pool::{InventoryHost, LAYOUTS_PER_BUCKET};
use super::{LinkEvidence, Traffic};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PipelineProfile {
    profile: String,
    pub tensor_parallel: u32,
    host_base: HostMemory,
    pub stages: Vec<StageProfile>,
    activation_bytes_per_rank: u64,
    feedback_bytes_per_rank: u64,
    pub ready_after_ms: u64,
    pub transition_loss_requests: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StageProfile {
    layers: u32,
    device_bytes: u64,
    host_per_worker: HostMemory,
    capacity: SloProfile,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    host_capacities: Vec<HostCapacity>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HostCapacity {
    host: NodeId,
    capacity: SloProfile,
}

impl PipelineProfile {
    pub(super) fn qualified_name(&self) -> Option<String> {
        let layers: Vec<_> = self.stages.iter().map(|stage| stage.layers).collect();
        hosted_pipeline_profile(&self.profile, &layers)
    }

    pub(super) fn validate(&self) -> bool {
        self.qualified_name().is_some()
            && self.tensor_parallel > 0
            && u32::try_from(self.stages.len())
                .ok()
                .and_then(|stages| stages.checked_mul(self.tensor_parallel))
                .is_some()
            && valid_memory(self.host_base)
            && self.activation_bytes_per_rank > 0
            && self.feedback_bytes_per_rank > 0
            && valid_rate(self.transition_loss_requests)
            && self.stages.iter().all(|stage| {
                let mut hosts = BTreeSet::new();
                stage.device_bytes > 0
                    && valid_memory(stage.host_per_worker)
                    && valid_rate(stage.capacity.rps)
                    && stage
                        .host_capacities
                        .iter()
                        .all(|host| hosts.insert(host.host) && valid_rate(host.capacity.rps))
            })
    }

    fn workers(&self) -> usize {
        self.tensor_parallel as usize * self.stages.len()
    }

    /// Match the launcher: API host first, remaining hosts in NodeId order,
    /// then local devices by ordinal. Stages contain consecutive TP slots.
    pub(super) fn lanes(
        &self,
        api_host: NodeId,
        ranks: &BTreeMap<NodeId, u64>,
    ) -> Option<Vec<NodeId>> {
        let workers = self.workers();
        let total = ranks
            .values()
            .try_fold(0u64, |total, count| total.checked_add(*count))?;
        if total != workers as u64 || ranks.get(&api_host).copied().unwrap_or(0) == 0 {
            return None;
        }
        Some(
            std::iter::once((&api_host, &ranks[&api_host]))
                .chain(ranks.iter().filter(|(host, _)| **host != api_host))
                .flat_map(|(&host, &count)| std::iter::repeat_n(host, count as usize))
                .collect(),
        )
    }

    pub(super) fn capacity(&self, lanes: &[NodeId]) -> SloProfile {
        lanes
            .chunks(self.tensor_parallel as usize)
            .zip(&self.stages)
            .flat_map(|(hosts, stage)| hosts.iter().map(|host| stage.capacity_on(*host)))
            .fold(
                SloProfile {
                    rps: f64::INFINITY,
                    valid_until_ms: u64::MAX,
                },
                lower_capacity,
            )
    }

    pub(super) fn traffic(&self, lanes: &[NodeId], evidence: LinkEvidence<'_>) -> Option<Traffic> {
        let mut traffic = Traffic::new();
        let stages = lanes.chunks(self.tensor_parallel as usize);
        for stage in stages.clone() {
            let mut ranks = BTreeMap::new();
            for &host in stage {
                *ranks.entry(host).or_default() += 1;
            }
            traffic.include(&evidence.estimate_tp(&ranks).ok()?.traffic, 1.0);
        }
        for (sender, receiver) in stages.clone().zip(stages.clone().skip(1)) {
            for (&from, &to) in sender.iter().zip(receiver) {
                evidence
                    .add(
                        &mut traffic,
                        from,
                        to,
                        self.activation_bytes_per_rank as f64,
                    )
                    .ok()?;
            }
        }
        // Only the last stage sends sampling feedback. In a prefix search
        // it can be partially assigned, so zip its present lanes.
        if let Some(last) = stages.clone().nth(self.stages.len() - 1) {
            for prior in stages.take(self.stages.len() - 1) {
                for (&from, &to) in last.iter().zip(prior) {
                    evidence
                        .add(&mut traffic, from, to, self.feedback_bytes_per_rank as f64)
                        .ok()?;
                }
            }
        }
        Some(traffic)
    }

    pub(super) fn kv_weights(&self, lanes: &[NodeId]) -> BTreeMap<NodeId, u64> {
        let mut weights = BTreeMap::new();
        for (hosts, stage) in lanes
            .chunks(self.tensor_parallel as usize)
            .zip(&self.stages)
        {
            for &host in hosts {
                *weights.entry(host).or_default() += u64::from(stage.layers);
            }
        }
        weights
    }
}

impl StageProfile {
    fn capacity_on(&self, host: NodeId) -> SloProfile {
        self.host_capacities
            .iter()
            .find(|entry| entry.host == host)
            .map_or(self.capacity, |entry| entry.capacity)
    }
}

pub(super) fn lower_capacity(left: SloProfile, right: SloProfile) -> SloProfile {
    SloProfile {
        rps: left.rps.min(right.rps),
        valid_until_ms: left.valid_until_ms.min(right.valid_until_ms),
    }
}

fn valid_memory(memory: HostMemory) -> bool {
    memory.pinned_bytes <= memory.bytes
}

#[derive(Clone)]
pub(super) struct PipelineLayout {
    pub api_host: NodeId,
    pub hosts: BTreeMap<NodeId, HostMemory>,
    pub workers: BTreeMap<DeviceId, WorkerDemand>,
    pub capacity: SloProfile,
    lanes: Vec<NodeId>,
    potential: f64,
}

pub(super) fn search_pipeline(
    profile: &PipelineProfile,
    hosts: &BTreeMap<NodeId, InventoryHost>,
    evidence: LinkEvidence<'_>,
) -> Vec<PipelineLayout> {
    let mut complete = Vec::new();
    for (&api_host, api) in hosts.iter().filter(|(_, host)| host.ports.is_some()) {
        let mut search = BTreeMap::new();
        let empty = PipelineLayout {
            api_host,
            hosts: BTreeMap::new(),
            workers: BTreeMap::new(),
            capacity: SloProfile {
                rps: f64::INFINITY,
                valid_until_ms: u64::MAX,
            },
            lanes: Vec::new(),
            potential: f64::INFINITY,
        };
        extend_host(profile, evidence, &empty, api_host, api, &mut search);
        for (&host, inventory) in hosts.iter().filter(|(host, _)| **host != api_host) {
            let mut next = search.clone();
            for layout in search.values().flatten() {
                extend_host(profile, evidence, layout, host, inventory, &mut next);
            }
            for layouts in next.values_mut() {
                layouts.sort_by(compare_layout);
                layouts.truncate(LAYOUTS_PER_BUCKET);
            }
            search = next;
        }
        complete.extend(search.remove(&profile.workers()).unwrap_or_default());
    }
    complete.sort_by(compare_layout);
    complete
}

fn extend_host(
    profile: &PipelineProfile,
    evidence: LinkEvidence<'_>,
    prefix: &PipelineLayout,
    host: NodeId,
    inventory: &InventoryHost,
    layouts: &mut BTreeMap<usize, Vec<PipelineLayout>>,
) {
    let mut trial = prefix.clone();
    let mut memory = profile.host_base;
    let mut devices = inventory.workers.iter();
    while trial.lanes.len() < profile.workers() {
        let stage = &profile.stages[trial.lanes.len() / profile.tensor_parallel as usize];
        let Some(worker) = devices.find(|worker| worker.device_bytes >= stage.device_bytes) else {
            break;
        };
        let (Some(bytes), Some(pinned_bytes)) = (
            memory.bytes.checked_add(stage.host_per_worker.bytes),
            memory
                .pinned_bytes
                .checked_add(stage.host_per_worker.pinned_bytes),
        ) else {
            break;
        };
        if bytes > inventory.memory.bytes || pinned_bytes > inventory.memory.pinned_bytes {
            break;
        }
        memory = HostMemory {
            bytes,
            pinned_bytes,
        };
        trial.hosts.insert(host, memory);
        trial.workers.insert(
            worker.device,
            WorkerDemand {
                incarnation: worker.incarnation,
                device_bytes: stage.device_bytes,
            },
        );
        trial.lanes.push(host);
        trial.capacity = lower_capacity(trial.capacity, stage.capacity_on(host));
        let Some(traffic) = profile.traffic(&trial.lanes, evidence) else {
            break;
        };
        if !trial.capacity.is_valid(evidence.now_ms) || trial.capacity.rps == 0.0 {
            break;
        }
        trial.potential = trial.capacity.rps.min(traffic.rate_limit());
        layouts
            .entry(trial.lanes.len())
            .or_default()
            .push(trial.clone());
    }
}

fn compare_layout(left: &PipelineLayout, right: &PipelineLayout) -> std::cmp::Ordering {
    right
        .potential
        .total_cmp(&left.potential)
        .then_with(|| left.hosts.len().cmp(&right.hosts.len()))
        .then_with(|| left.lanes.cmp(&right.lanes))
        .then_with(|| left.workers.keys().cmp(right.workers.keys()))
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{HostedAddress, HostedLinkSample, WorkerResources};

    use super::*;
    use crate::serving::HostedLinkPolicy;

    fn profile() -> PipelineProfile {
        PipelineProfile {
            profile: "weights".into(),
            tensor_parallel: 1,
            host_base: HostMemory {
                bytes: 4,
                pinned_bytes: 1,
            },
            stages: [(4, 32, 2), (5, 8, 3), (7, 16, 4)]
                .map(|(layers, device_bytes, bytes)| StageProfile {
                    layers,
                    device_bytes,
                    host_per_worker: HostMemory {
                        bytes,
                        pinned_bytes: bytes - 1,
                    },
                    capacity: SloProfile {
                        rps: 40.0,
                        valid_until_ms: 100,
                    },
                    host_capacities: Vec::new(),
                })
                .into(),
            activation_bytes_per_rank: 20,
            feedback_bytes_per_rank: 5,
            ready_after_ms: 10,
            transition_loss_requests: 0.0,
        }
    }

    fn node(byte: u8) -> NodeId {
        NodeId::from_bytes([byte; 16])
    }

    fn sample(target: NodeId, bandwidth: u64) -> HostedLinkSample {
        HostedLinkSample {
            target,
            target_session: target,
            sampled_at_ms: 5,
            bytes_per_second: bandwidth,
            rtt_ns: 10,
        }
    }

    #[test]
    fn placement_follows_api_first_order_and_accumulates_memory_across_stages() {
        let [earlier, api] = [1, 2].map(node);
        let mut profile = profile();
        profile.stages[2].host_capacities.push(HostCapacity {
            host: earlier,
            capacity: SloProfile {
                rps: 15.0,
                valid_until_ms: 90,
            },
        });
        let available = |node, sizes: &[u64], has_ports: bool| InventoryHost {
            address: HostedAddress {
                host: node,
                address: std::net::Ipv4Addr::LOCALHOST.into(),
            },
            workers: sizes
                .iter()
                .enumerate()
                .map(|(ordinal, &bytes)| WorkerResources {
                    device: DeviceId::new(node, ordinal as u32),
                    incarnation: 1,
                    device_bytes: bytes,
                    grant: None,
                })
                .collect(),
            memory: HostMemory {
                bytes: 16,
                pinned_bytes: 4,
            },
            ports: has_ports.then_some([20000, 20001, 20002]),
        };
        let mut hosts = BTreeMap::from([
            (api, available(api, &[8, 40, 8], true)),
            (earlier, available(earlier, &[16], false)),
        ]);
        let links = BTreeMap::from([
            ((api, earlier), sample(earlier, 1000)),
            ((earlier, api), sample(api, 1000)),
        ]);
        let evidence = LinkEvidence {
            policy: HostedLinkPolicy {
                bytes_per_rank_pair: 100,
                max_sample_age_ms: 10,
                tp_rtt_budget: None,
            },
            now_ms: 10,
            links: &links,
        };
        let layouts = search_pipeline(&profile, &hosts, evidence);
        let [layout] = layouts.as_slice() else {
            panic!("expected one feasible pipeline")
        };
        assert_eq!(layout.lanes, [api, api, earlier]);
        assert_eq!(layout.capacity.rps, 15.0);
        assert_eq!(layout.capacity.valid_until_ms, 90);
        assert!(!layout.workers.contains_key(&DeviceId::new(api, 0)));
        assert_eq!(layout.workers[&DeviceId::new(api, 1)].device_bytes, 32);
        assert_eq!(layout.workers[&DeviceId::new(api, 2)].device_bytes, 8);
        assert_eq!(
            layout.hosts[&api],
            HostMemory {
                bytes: 9,
                pinned_bytes: 4
            }
        );
        hosts.get_mut(&api).unwrap().memory.pinned_bytes -= 1;
        assert!(search_pipeline(&profile, &hosts, evidence).is_empty());
    }

    #[test]
    fn last_stage_needs_a_direct_feedback_link_to_each_prior_stage() {
        let [a, b, c] = [1, 2, 3].map(node);
        let profile = profile();
        let mut links = BTreeMap::from([
            ((a, b), sample(b, 1000)),
            ((b, c), sample(c, 1000)),
            ((c, b), sample(b, 1000)),
        ]);
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: 100,
            max_sample_age_ms: 10,
            tp_rtt_budget: None,
        };
        assert!(
            profile
                .traffic(
                    &[a, b, c],
                    LinkEvidence {
                        policy,
                        now_ms: 10,
                        links: &links
                    }
                )
                .is_none()
        );
        links.insert((c, a), sample(a, 300));
        let traffic = profile
            .traffic(
                &[a, b, c],
                LinkEvidence {
                    policy,
                    now_ms: 10,
                    links: &links,
                },
            )
            .unwrap();
        assert_eq!(traffic.rate_limit(), 30.0);
        assert_eq!(traffic.valid_until_ms, 15);
        assert_eq!(
            profile.kv_weights(&[a, b, c]),
            BTreeMap::from([(a, 4), (b, 5), (c, 7)])
        );
    }
}
