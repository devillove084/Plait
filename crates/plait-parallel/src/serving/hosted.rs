use std::collections::BTreeMap;

use plait_core::NodeId;
use plait_core::serving::{GroupPlan, HostedLinkSample};
use serde::{Deserialize, Serialize};

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
    if policy.bytes_per_rank_pair == 0 || policy.max_sample_age_ms == 0 {
        return Err("hosted link policy requires positive traffic and sample age");
    }
    let mut ranks = BTreeMap::<NodeId, u64>::new();
    for device in plan.workers.keys() {
        *ranks.entry(device.node).or_default() += 1;
    }
    if ranks.len() < 2 {
        return Ok(HostedLinkLimit::Local);
    }
    let mut outgoing = BTreeMap::<NodeId, PortBudget>::new();
    let mut incoming = BTreeMap::<NodeId, PortBudget>::new();
    let mut valid_until_ms = u64::MAX;
    for (&source, &senders) in &ranks {
        for (&target, &receivers) in &ranks {
            if source == target {
                continue;
            }
            let Some(sample) = links.get(&(source, target)).filter(|sample| {
                sample.sampled_at_ms <= now_ms
                    && now_ms - sample.sampled_at_ms < policy.max_sample_age_ms
                    && sample.bytes_per_second > 0
            }) else {
                return Ok(HostedLinkLimit::MissingLink { source, target });
            };
            valid_until_ms = valid_until_ms.min(
                sample
                    .sampled_at_ms
                    .saturating_add(policy.max_sample_age_ms),
            );
            let traffic = policy.bytes_per_rank_pair as f64 * senders as f64 * receivers as f64;
            outgoing
                .entry(source)
                .or_default()
                .add(sample.bytes_per_second, traffic);
            incoming
                .entry(target)
                .or_default()
                .add(sample.bytes_per_second, traffic);
        }
    }
    let rps = outgoing
        .values()
        .chain(incoming.values())
        .map(|port| port.bytes_per_second as f64 / port.bytes_per_request)
        .fold(f64::INFINITY, f64::min);
    Ok(HostedLinkLimit::Measured {
        rps,
        valid_until_ms,
    })
}

#[derive(Default)]
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
