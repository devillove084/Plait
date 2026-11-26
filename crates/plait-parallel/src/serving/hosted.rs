use std::collections::BTreeMap;

use plait_core::NodeId;
use plait_core::serving::{GroupPlan, HostedLinkSample};
use serde::{Deserialize, Serialize};

mod pd;
mod pool;
pub use pd::{
    HostedPdDecision, HostedPdPool, HostedPdSelection, plan_hosted_pd, plan_hosted_pd_recovery,
    plan_hosted_pd_with_releases,
};
pub use pool::{HostedPool, generate_hosted_candidates};

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct HostedLinkPolicy {
    /// Estimated wire bytes per request for each directed pair of TP workers.
    pub bytes_per_rank_pair: u64,
    pub max_sample_age_ms: u64,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum HostedLinkLimit {
    Local,
    MissingLink { source: NodeId, target: NodeId },
    Measured { rps: f64, valid_until_ms: u64 },
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
    .tp_traffic(ranks)
    {
        Ok(traffic) => HostedLinkLimit::Measured {
            rps: traffic.rate_limit(),
            valid_until_ms: traffic.valid_until_ms,
        },
        Err((source, target)) => HostedLinkLimit::MissingLink { source, target },
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

impl LinkEvidence<'_> {
    fn tp_traffic(&self, ranks: &BTreeMap<NodeId, u64>) -> Result<Traffic, (NodeId, NodeId)> {
        let mut result = Traffic::new();
        for (&source, &senders) in ranks {
            for (&target, &receivers) in ranks {
                let bytes =
                    self.policy.bytes_per_rank_pair as f64 * senders as f64 * receivers as f64;
                self.add(&mut result, source, target, bytes)?;
            }
        }
        Ok(result)
    }

    fn add(
        &self,
        traffic: &mut Traffic,
        source: NodeId,
        target: NodeId,
        bytes: f64,
    ) -> Result<(), (NodeId, NodeId)> {
        if source == target {
            return Ok(());
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
        Ok(())
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
        };
        assert_eq!(
            limit_hosted_links(&plan, policy, 1049, &links).unwrap(),
            HostedLinkLimit::Measured {
                rps: 25.0,
                valid_until_ms: 1050
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
}
