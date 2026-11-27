use std::collections::BTreeMap;

use plait_core::NodeId;
use plait_core::serving::{GroupPlan, HostedLinkSample, ServingRole};
use serde::{Deserialize, Deserializer, Serialize};

mod pd;
mod pipeline;
mod pool;
pub use pd::{
    HostedModePolicy, HostedPdDecision, HostedPdIncumbents, HostedPdPool, HostedPdSelection,
    plan_hosted_evacuation, plan_hosted_pd, plan_hosted_pd_recovery,
    plan_hosted_pd_with_assignments, plan_hosted_pd_with_releases, plan_hosted_serving,
};
pub use pool::{HostedPool, generate_hosted_candidates, limit_hosted_pool_tp_links};

#[derive(Clone, Deserialize, Serialize)]
pub struct HostedLinkPolicy {
    /// Default bytes per directed pair when a TP width has no explicit profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_per_rank_pair: Option<u64>,
    pub max_sample_age_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp_rtt_budget: Option<HostedTpRttBudget>,
    #[serde(
        default,
        skip_serializing_if = "BTreeMap::is_empty",
        deserialize_with = "deserialize_tp_profiles"
    )]
    pub tp_profiles: BTreeMap<u32, HostedTpLinkProfile>,
    #[serde(default, skip_serializing_if = "HostedPdTpProfiles::is_empty")]
    pub pd_tp_profiles: HostedPdTpProfiles,
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedPdTpProfiles {
    #[serde(
        default,
        skip_serializing_if = "BTreeMap::is_empty",
        deserialize_with = "deserialize_tp_profiles"
    )]
    pub prefill: BTreeMap<u32, HostedTpLinkProfile>,
    #[serde(
        default,
        skip_serializing_if = "BTreeMap::is_empty",
        deserialize_with = "deserialize_tp_profiles"
    )]
    pub decode: BTreeMap<u32, HostedTpLinkProfile>,
}

impl HostedPdTpProfiles {
    fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}

fn deserialize_tp_profiles<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<u32, HostedTpLinkProfile>, D::Error>
where
    D: Deserializer<'de>,
{
    BTreeMap::<String, HostedTpLinkProfile>::deserialize(deserializer)?
        .into_iter()
        .map(|(key, profile)| {
            let width = key.parse::<u32>().map_err(serde::de::Error::custom)?;
            if width.to_string() != key {
                return Err(serde::de::Error::custom("expected a decimal TP width key"));
            }
            Ok((width, profile))
        })
        .collect()
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostedTpLinkProfile {
    /// Estimated wire bytes per request for each directed pair of TP workers.
    pub bytes_per_rank_pair: u64,
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
    MissingTpProfile {
        tensor_parallel: u32,
    },
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
    policy: &HostedLinkPolicy,
    now_ms: u64,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<HostedLinkLimit, &'static str> {
    limit_with_evidence(plan, LinkEvidence::new(policy, now_ms, links))
}

fn limit_with_evidence(
    plan: &GroupPlan,
    evidence: LinkEvidence<'_>,
) -> Result<HostedLinkLimit, &'static str> {
    evidence.policy.validate()?;
    let mut ranks = BTreeMap::<NodeId, u64>::new();
    for device in plan.workers.keys() {
        *ranks.entry(device.node).or_default() += 1;
    }
    Ok(evidence
        .for_role(plan.role)
        .limit_tp(plan.tensor_parallel, &ranks))
}

impl HostedLinkPolicy {
    fn validate(&self) -> Result<(), &'static str> {
        if self.max_sample_age_ms == 0 {
            return Err("hosted link policy requires a positive sample age");
        }
        if let Some(default) = self.default_profile() {
            default.validate()?;
        } else if self.tp_rtt_budget.is_some()
            || (self.tp_profiles.is_empty() && self.pd_tp_profiles.is_empty())
        {
            return Err("hosted link policy requires default traffic or per-TP profiles");
        }
        for (&width, profile) in self
            .tp_profiles
            .iter()
            .chain(&self.pd_tp_profiles.prefill)
            .chain(&self.pd_tp_profiles.decode)
        {
            if width == 0 {
                return Err("hosted TP link profile requires a positive TP width");
            }
            profile.validate()?;
        }
        Ok(())
    }

    fn default_profile(&self) -> Option<HostedTpLinkProfile> {
        self.bytes_per_rank_pair
            .map(|bytes_per_rank_pair| HostedTpLinkProfile {
                bytes_per_rank_pair,
                tp_rtt_budget: self.tp_rtt_budget,
            })
    }

    fn profile(&self, role: ServingRole, tensor_parallel: u32) -> Option<HostedTpLinkProfile> {
        let role_profile = match role {
            ServingRole::Prefill => self.pd_tp_profiles.prefill.get(&tensor_parallel),
            ServingRole::Decode => self.pd_tp_profiles.decode.get(&tensor_parallel),
            ServingRole::Aggregated => None,
        };
        role_profile
            .or_else(|| self.tp_profiles.get(&tensor_parallel))
            .copied()
            .or_else(|| self.default_profile())
    }
}

impl HostedTpLinkProfile {
    fn validate(self) -> Result<(), &'static str> {
        if self.bytes_per_rank_pair == 0 {
            return Err("hosted TP link profile requires positive traffic");
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Direction {
    Out,
    In,
}

#[derive(Clone, Copy)]
struct LinkEvidence<'a> {
    policy: &'a HostedLinkPolicy,
    now_ms: u64,
    links: &'a BTreeMap<(NodeId, NodeId), HostedLinkSample>,
    enforce_tp_wait: bool,
    role: ServingRole,
    tp_profiles: Option<&'a BTreeMap<u32, HostedTpLinkProfile>>,
}

struct TpEstimate {
    traffic: Traffic,
    wait_ns: Option<u64>,
}

impl<'a> LinkEvidence<'a> {
    fn new(
        policy: &'a HostedLinkPolicy,
        now_ms: u64,
        links: &'a BTreeMap<(NodeId, NodeId), HostedLinkSample>,
    ) -> Self {
        Self {
            policy,
            now_ms,
            links,
            enforce_tp_wait: true,
            role: ServingRole::Aggregated,
            tp_profiles: None,
        }
    }

    fn for_role(self, role: ServingRole) -> Self {
        Self { role, ..self }
    }

    fn with_tp_profiles<'b>(
        self,
        tp_profiles: Option<&'b BTreeMap<u32, HostedTpLinkProfile>>,
    ) -> LinkEvidence<'b>
    where
        'a: 'b,
    {
        LinkEvidence {
            tp_profiles,
            ..self
        }
    }

    fn without_tp_wait(self) -> Self {
        Self {
            enforce_tp_wait: false,
            ..self
        }
    }

    fn limit_tp(&self, tensor_parallel: u32, ranks: &BTreeMap<NodeId, u64>) -> HostedLinkLimit {
        if ranks.len() < 2 {
            return HostedLinkLimit::Local;
        }
        match self.estimate_tp(tensor_parallel, ranks) {
            Ok(estimate) => HostedLinkLimit::Measured {
                rps: estimate.traffic.rate_limit(),
                valid_until_ms: estimate.traffic.valid_until_ms,
                estimated_tp_wait_ns: estimate.wait_ns,
            },
            Err(limit) => limit,
        }
    }

    fn estimate_tp(
        &self,
        tensor_parallel: u32,
        ranks: &BTreeMap<NodeId, u64>,
    ) -> Result<TpEstimate, HostedLinkLimit> {
        let mut traffic = Traffic::new();
        if ranks.len() < 2 {
            return Ok(TpEstimate {
                traffic,
                wait_ns: None,
            });
        }
        let profile = match self.tp_profiles {
            Some(profiles) => profiles.get(&tensor_parallel).copied(),
            None => self.policy.profile(self.role, tensor_parallel),
        }
        .ok_or(HostedLinkLimit::MissingTpProfile { tensor_parallel })?;
        let tp_rtt_budget = profile.tp_rtt_budget.filter(|_| self.enforce_tp_wait);
        let mut peak_rtt_ns = 0;
        for (&source, &senders) in ranks {
            for (&target, &receivers) in ranks.iter().filter(|&(&target, _)| target != source) {
                let bytes = profile.bytes_per_rank_pair as f64 * senders as f64 * receivers as f64;
                let rtt_ns = self
                    .add(&mut traffic, source, target, bytes)
                    .map_err(|(source, target)| HostedLinkLimit::MissingLink { source, target })?;
                if tp_rtt_budget.is_some() && rtt_ns == 0 {
                    return Err(HostedLinkLimit::MissingLink { source, target });
                }
                peak_rtt_ns = peak_rtt_ns.max(rtt_ns);
            }
        }
        let wait_ns = if let Some(budget) = tp_rtt_budget {
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
            bytes_per_rank_pair: Some(1_000_000),
            max_sample_age_ms: 50,
            tp_rtt_budget: None,
            tp_profiles: BTreeMap::new(),
            pd_tp_profiles: Default::default(),
        };
        assert_eq!(
            limit_hosted_links(&plan, &policy, 1049, &links).unwrap(),
            HostedLinkLimit::Measured {
                rps: 25.0,
                valid_until_ms: 1050,
                estimated_tp_wait_ns: None,
            }
        );
        links.get_mut(&(c, a)).unwrap().sampled_at_ms = 999;
        assert_eq!(
            limit_hosted_links(&plan, &policy, 1049, &links).unwrap(),
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
            bytes_per_rank_pair: Some(100),
            max_sample_age_ms: 40,
            tp_rtt_budget: Some(HostedTpRttBudget {
                serial_rounds: 3,
                max_wait_ns: 27,
            }),
            tp_profiles: BTreeMap::new(),
            pd_tp_profiles: Default::default(),
        };
        assert_eq!(
            LinkEvidence::new(&policy, 11, &links).limit_tp(6, &ranks),
            HostedLinkLimit::Measured {
                rps: 1250.0,
                valid_until_ms: 50,
                estimated_tp_wait_ns: Some(27),
            },
        );
        links.get_mut(&(b, a)).unwrap().rtt_ns = 10;
        assert_eq!(
            LinkEvidence::new(&policy, 11, &links).limit_tp(6, &ranks),
            HostedLinkLimit::ExceedsTpRttBudget {
                peak_rtt_ns: 10,
                serial_rounds: 3,
                max_wait_ns: 27,
            },
        );
        links.get_mut(&(b, a)).unwrap().rtt_ns = 0;
        assert_eq!(
            LinkEvidence::new(&policy, 11, &links).limit_tp(6, &ranks),
            HostedLinkLimit::MissingLink {
                source: b,
                target: a
            },
        );
        assert_eq!(
            LinkEvidence::new(&policy, 11, &links).limit_tp(6, &BTreeMap::from([(a, 6)])),
            HostedLinkLimit::Local,
        );
    }

    #[test]
    fn a_width_profile_replaces_both_parts_of_the_default_budget() {
        let [a, b] = ["source", "target"].map(NodeId::derive);
        let ranks = BTreeMap::from([(a, 1), (b, 1)]);
        let links = [(a, b), (b, a)]
            .into_iter()
            .map(|(from, target)| {
                (
                    (from, target),
                    HostedLinkSample {
                        target,
                        target_session: target,
                        sampled_at_ms: 1,
                        bytes_per_second: 12_000,
                        rtt_ns: 10,
                    },
                )
            })
            .collect();
        let mut policy = HostedLinkPolicy {
            bytes_per_rank_pair: Some(500),
            max_sample_age_ms: 20,
            tp_rtt_budget: Some(HostedTpRttBudget {
                serial_rounds: 3,
                max_wait_ns: 20,
            }),
            tp_profiles: BTreeMap::from([(
                2,
                HostedTpLinkProfile {
                    bytes_per_rank_pair: 100,
                    tp_rtt_budget: None,
                },
            )]),
            pd_tp_profiles: Default::default(),
        };
        let evidence = LinkEvidence::new(&policy, 10, &links);
        assert_eq!(
            evidence.limit_tp(2, &ranks),
            HostedLinkLimit::Measured {
                rps: 120.0,
                valid_until_ms: 21,
                estimated_tp_wait_ns: None,
            }
        );
        assert_eq!(
            evidence.limit_tp(4, &ranks),
            HostedLinkLimit::ExceedsTpRttBudget {
                peak_rtt_ns: 10,
                serial_rounds: 3,
                max_wait_ns: 20,
            }
        );
        policy.bytes_per_rank_pair = None;
        policy.tp_rtt_budget = None;
        assert!(policy.validate().is_ok());
        let evidence = LinkEvidence::new(&policy, 10, &links);
        assert_eq!(
            evidence.limit_tp(4, &ranks),
            HostedLinkLimit::MissingTpProfile { tensor_parallel: 4 }
        );
        assert_eq!(
            evidence.limit_tp(4, &BTreeMap::from([(a, 4)])),
            HostedLinkLimit::Local
        );
    }

    #[test]
    fn a_missing_role_profile_uses_the_shared_width_before_the_default() {
        let prefill = HostedTpLinkProfile {
            bytes_per_rank_pair: 12,
            tp_rtt_budget: None,
        };
        let mut policy = HostedLinkPolicy {
            bytes_per_rank_pair: None,
            max_sample_age_ms: 10,
            tp_rtt_budget: None,
            tp_profiles: BTreeMap::new(),
            pd_tp_profiles: HostedPdTpProfiles {
                prefill: BTreeMap::from([(2, prefill)]),
                ..HostedPdTpProfiles::default()
            },
        };
        assert!(policy.validate().is_ok());
        assert!(policy.profile(ServingRole::Decode, 2).is_none());
        let shared = HostedTpLinkProfile {
            bytes_per_rank_pair: 5,
            ..prefill
        };
        policy.tp_profiles.insert(2, shared);
        policy.bytes_per_rank_pair = Some(3);
        assert_eq!(
            policy
                .profile(ServingRole::Prefill, 2)
                .unwrap()
                .bytes_per_rank_pair,
            12
        );
        assert_eq!(
            policy
                .profile(ServingRole::Decode, 2)
                .unwrap()
                .bytes_per_rank_pair,
            5
        );
        assert_eq!(
            policy
                .profile(ServingRole::Decode, 4)
                .unwrap()
                .bytes_per_rank_pair,
            3
        );
    }
}
