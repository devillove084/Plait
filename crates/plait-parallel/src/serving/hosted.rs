use std::collections::BTreeMap;

use plait_core::NodeId;
use plait_core::serving::{GroupPlan, HostedLinkSample};
use serde::{Deserialize, Serialize};

mod pd;
mod pipeline;
mod pool;
pub use pd::{
    HostedModePolicy, HostedPdDecision, HostedPdIncumbents, HostedPdPool, HostedPdSelection,
    plan_hosted_evacuation, plan_hosted_pd, plan_hosted_pd_recovery,
    plan_hosted_pd_with_assignments, plan_hosted_pd_with_releases, plan_hosted_serving,
};
pub use pool::{HostedPool, generate_hosted_candidates};

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct HostedLinkPolicy {
    /// Estimated wire bytes per request for each directed pair of TP workers.
    pub bytes_per_rank_pair: u64,
    pub max_sample_age_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp_rtt_budget: Option<HostedTpRttBudget>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedTpRttBudget {
    /// Serial peer exchanges within one TP group during the profiled window.
    pub serial_rounds: u64,
    pub max_wait_ns: u64,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum HostedLinkLimit {
    Local,
    MissingLink {
        source: NodeId,
        target: NodeId,
    },
    ExceedsTpRttBudget {
        peak_rtt_ns: u64,
        serial_rounds: u64,
        max_wait_ns: u64,
    },
    Measured {
        rps: f64,
        valid_until_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        estimated_tp_wait_ns: Option<u64>,
    },
}

pub fn limit_hosted_links(
    plan: &GroupPlan,
    policy: HostedLinkPolicy,
    now_ms: u64,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<HostedLinkLimit, &'static str> {
    policy.validate()?;
    let mut ranks = BTreeMap::<NodeId, u64>::new();
    for device in plan.workers.keys() {
        *ranks.entry(device.node).or_default() += 1;
    }
    Ok(limit_hosted_ranks(&ranks, policy, now_ms, links))
}

impl HostedLinkPolicy {
    fn validate(self) -> Result<(), &'static str> {
        if self.bytes_per_rank_pair == 0 || self.max_sample_age_ms == 0 {
            return Err("hosted link policy requires positive traffic and sample age");
        }
        if self
            .tp_rtt_budget
            .is_some_and(|budget| budget.serial_rounds == 0 || budget.max_wait_ns == 0)
        {
            return Err("hosted TP RTT budget requires positive rounds and wait");
        }
        Ok(())
    }
}

fn limit_hosted_ranks(
    ranks: &BTreeMap<NodeId, u64>,
    policy: HostedLinkPolicy,
    now_ms: u64,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> HostedLinkLimit {
    if ranks.len() < 2 {
        return HostedLinkLimit::Local;
    }
    match (LinkEvidence {
        policy,
        now_ms,
        links,
    })
    .estimate_tp(ranks)
    {
        Ok(estimate) => HostedLinkLimit::Measured {
            rps: estimate.traffic.rate_limit(),
            valid_until_ms: estimate.traffic.valid_until_ms,
            estimated_tp_wait_ns: estimate.wait_ns,
        },
        Err(limit) => limit,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Direction {
    Out,
    In,
}

#[derive(Clone, Copy)]
struct LinkEvidence<'a> {
    policy: HostedLinkPolicy,
    now_ms: u64,
    links: &'a BTreeMap<(NodeId, NodeId), HostedLinkSample>,
}

struct TpEstimate {
    traffic: Traffic,
    wait_ns: Option<u64>,
}

impl LinkEvidence<'_> {
    fn estimate_tp(&self, ranks: &BTreeMap<NodeId, u64>) -> Result<TpEstimate, HostedLinkLimit> {
        let mut traffic = Traffic::new();
        let mut peak_rtt_ns = 0;
        for (&source, &senders) in ranks {
            for (&target, &receivers) in ranks.iter().filter(|&(&target, _)| target != source) {
                let bytes =
                    self.policy.bytes_per_rank_pair as f64 * senders as f64 * receivers as f64;
                let rtt_ns = self
                    .add(&mut traffic, source, target, bytes)
                    .map_err(|(source, target)| HostedLinkLimit::MissingLink { source, target })?;
                if self.policy.tp_rtt_budget.is_some() && rtt_ns == 0 {
                    return Err(HostedLinkLimit::MissingLink { source, target });
                }
                peak_rtt_ns = peak_rtt_ns.max(rtt_ns);
            }
        }
        let wait_ns = if let Some(budget) = self.policy.tp_rtt_budget {
            Some(
                peak_rtt_ns
                    .checked_mul(budget.serial_rounds)
                    .filter(|&wait| wait <= budget.max_wait_ns)
                    .ok_or(HostedLinkLimit::ExceedsTpRttBudget {
                        peak_rtt_ns,
                        serial_rounds: budget.serial_rounds,
                        max_wait_ns: budget.max_wait_ns,
                    })?,
            )
        } else {
            None
        };
        Ok(TpEstimate { traffic, wait_ns })
    }

    fn add(
        &self,
        traffic: &mut Traffic,
        source: NodeId,
        target: NodeId,
        bytes: f64,
    ) -> Result<u64, (NodeId, NodeId)> {
        if source == target {
            return Ok(0);
        }
        let sample = self
            .links
            .get(&(source, target))
            .filter(|sample| {
                sample.sampled_at_ms <= self.now_ms
                    && self.now_ms - sample.sampled_at_ms < self.policy.max_sample_age_ms
                    && sample.bytes_per_second > 0
            })
            .ok_or((source, target))?;
        traffic.valid_until_ms = traffic.valid_until_ms.min(
            sample
                .sampled_at_ms
                .saturating_add(self.policy.max_sample_age_ms),
        );
        for key in [(source, Direction::Out), (target, Direction::In)] {
            traffic
                .ports
                .entry(key)
                .or_default()
                .add(sample.bytes_per_second, bytes);
        }
        Ok(sample.rtt_ns)
    }
}

#[derive(Clone)]
struct Traffic {
    ports: BTreeMap<(NodeId, Direction), PortBudget>,
    valid_until_ms: u64,
}

impl Traffic {
    fn new() -> Self {
        Self {
            ports: BTreeMap::new(),
            valid_until_ms: u64::MAX,
        }
    }

    fn rate_limit(&self) -> f64 {
        self.ports
            .values()
            .map(|port| port.bytes_per_second as f64 / port.bytes_per_request)
            .fold(f64::INFINITY, f64::min)
    }

    fn include(&mut self, other: &Self, share: f64) {
        self.valid_until_ms = self.valid_until_ms.min(other.valid_until_ms);
        for (&port, budget) in &other.ports {
            self.ports
                .entry(port)
                .or_default()
                .add(budget.bytes_per_second, budget.bytes_per_request * share);
        }
    }

    fn include_peak(&mut self, other: &Self) {
        self.valid_until_ms = self.valid_until_ms.min(other.valid_until_ms);
        for (&key, incoming) in &other.ports {
            self.ports
                .entry(key)
                .and_modify(|port| {
                    port.bytes_per_second = port.bytes_per_second.min(incoming.bytes_per_second);
                    port.bytes_per_request = port.bytes_per_request.max(incoming.bytes_per_request);
                })
                .or_insert_with(|| incoming.clone());
        }
    }

    fn rate_limit_after(&self, reserved: &Self) -> Option<f64> {
        // Reserved stream bytes have already been scaled to bytes per second.
        for (key, port) in &reserved.ports {
            let capacity = self.ports.get(key).map_or(port.bytes_per_second, |steady| {
                steady.bytes_per_second.min(port.bytes_per_second)
            });
            if port.bytes_per_request > capacity as f64 {
                return None;
            }
        }
        Some(
            self.ports
                .iter()
                .map(|(key, port)| {
                    let available =
                        reserved
                            .ports
                            .get(key)
                            .map_or(port.bytes_per_second as f64, |fixed| {
                                port.bytes_per_second.min(fixed.bytes_per_second) as f64
                                    - fixed.bytes_per_request
                            });
                    available / port.bytes_per_request
                })
                .fold(f64::INFINITY, f64::min),
        )
    }
}

#[derive(Clone, Default)]
struct PortBudget {
    bytes_per_second: u64,
    bytes_per_request: f64,
}

impl PortBudget {
    fn add(&mut self, rate: u64, demand: f64) {
        if self.bytes_per_request == 0.0 {
            self.bytes_per_second = rate;
        } else {
            self.bytes_per_second = self.bytes_per_second.min(rate);
        }
        self.bytes_per_request += demand;
    }
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{HostedLinkSample, ServingRole, WorkerDemand};
    use plait_core::{DeviceId, NodeId};

    use super::*;

    #[test]
    fn shared_ports_include_every_rank_pair_and_reject_expired_samples() {
        let [a, b, c] = ["source", "receiver", "spare"].map(NodeId::derive);
        let plan = GroupPlan {
            role: ServingRole::Aggregated,
            model: "model".into(),
            tensor_parallel: 4,
            pipeline_parallel: 1,
            launch_digest: [0; 32],
            workers: [(a, 0), (a, 1), (b, 0), (c, 0)]
                .into_iter()
                .map(|(host, ordinal)| {
                    (
                        DeviceId::new(host, ordinal),
                        WorkerDemand {
                            incarnation: 1,
                            device_bytes: 1,
                        },
                    )
                })
                .collect(),
            hosts: [a, b, c]
                .into_iter()
                .map(|host| (host, Default::default()))
                .collect(),
        };
        let mut links = BTreeMap::new();
        for from in [a, b, c] {
            for to in [a, b, c].into_iter().filter(|&to| to != from) {
                links.insert(
                    (from, to),
                    HostedLinkSample {
                        target: to,
                        target_session: to,
                        sampled_at_ms: 1000,
                        bytes_per_second: 100_000_000,
                        rtt_ns: 10_000,
                    },
                );
            }
        }
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: 1_000_000,
            max_sample_age_ms: 50,
            tp_rtt_budget: None,
        };
        assert_eq!(
            limit_hosted_links(&plan, policy, 1049, &links).unwrap(),
            HostedLinkLimit::Measured {
                rps: 25.0,
                valid_until_ms: 1050,
                estimated_tp_wait_ns: None,
            }
        );
        links.get_mut(&(c, a)).unwrap().sampled_at_ms = 999;
        assert_eq!(
            limit_hosted_links(&plan, policy, 1049, &links).unwrap(),
            HostedLinkLimit::MissingLink {
                source: c,
                target: a
            }
        );
    }

    #[test]
    fn tp_wait_uses_serial_rounds_and_the_slower_direction() {
        let [a, b] = ["left", "right"].map(NodeId::derive);
        let ranks = BTreeMap::from([(a, 2), (b, 4)]);
        let sample = |target, rtt_ns| HostedLinkSample {
            target,
            target_session: target,
            sampled_at_ms: 10,
            bytes_per_second: 1_000_000,
            rtt_ns,
        };
        let mut links = BTreeMap::from([((a, b), sample(b, 5)), ((b, a), sample(a, 9))]);
        let policy = HostedLinkPolicy {
            bytes_per_rank_pair: 100,
            max_sample_age_ms: 40,
            tp_rtt_budget: Some(HostedTpRttBudget {
                serial_rounds: 3,
                max_wait_ns: 27,
            }),
        };
        assert_eq!(
            limit_hosted_ranks(&ranks, policy, 11, &links),
            HostedLinkLimit::Measured {
                rps: 1250.0,
                valid_until_ms: 50,
                estimated_tp_wait_ns: Some(27),
            },
        );
        links.get_mut(&(b, a)).unwrap().rtt_ns = 10;
        assert_eq!(
            limit_hosted_ranks(&ranks, policy, 11, &links),
            HostedLinkLimit::ExceedsTpRttBudget {
                peak_rtt_ns: 10,
                serial_rounds: 3,
                max_wait_ns: 27,
            },
        );
        links.get_mut(&(b, a)).unwrap().rtt_ns = 0;
        assert_eq!(
            limit_hosted_ranks(&ranks, policy, 11, &links),
            HostedLinkLimit::MissingLink {
                source: b,
                target: a
            },
        );
        assert_eq!(
            limit_hosted_ranks(&BTreeMap::from([(a, 6)]), policy, 11, &links),
            HostedLinkLimit::Local,
        );
    }
}
