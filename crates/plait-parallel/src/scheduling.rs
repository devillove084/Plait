//! Scheduling plans that combine the ordering log with heterogeneous ranks.
//!
//! The ordering log decides membership and the epoch. A plan is derived from
//! the committed epoch, never from a mutable "current leader" field, so every
//! replica computes the same host leaders and chunk bounds.

use plait_core::NodeId;

use crate::{HeterogeneousLayout, RankProfile};

#[derive(Clone, Debug, PartialEq)]
pub struct HostPlan {
    pub host: usize,
    pub ranks: Vec<usize>,
    pub leader: NodeId,
    pub share: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkPlan {
    pub host: usize,
    pub leader: NodeId,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectivePlan {
    /// Committed ordering epoch that produced this plan.
    pub epoch: u64,
    pub bytes: usize,
    pub hosts: Vec<HostPlan>,
    pub chunks: Vec<ChunkPlan>,
}

impl CollectivePlan {
    /// Build one plan from the committed membership epoch and local profiles.
    /// Hosts are explicit groups; rank 0 of each group is not assumed to be
    /// the leader. The leader is the fastest rank in the group, with NodeId as
    /// a deterministic tie-breaker.
    pub fn build(
        profiles: &[RankProfile],
        hosts: &[Vec<usize>],
        bytes: usize,
        epoch: u64,
        message_bytes: usize,
    ) -> Result<Self, String> {
        if profiles.is_empty() || hosts.is_empty() {
            return Err("collective plan needs profiles and hosts".to_owned());
        }
        if hosts.iter().flatten().any(|rank| *rank >= profiles.len()) {
            return Err("host group references an unknown rank".to_owned());
        }
        let layout = HeterogeneousLayout::plan_detailed(profiles, message_bytes)?;
        let shares: Vec<(NodeId, f64)> = layout.shares;
        let mut host_plans = Vec::with_capacity(hosts.len());
        for (host, ranks) in hosts.iter().enumerate() {
            if ranks.is_empty() {
                return Err("host group cannot be empty".to_owned());
            }
            let share: f64 = ranks
                .iter()
                .map(|rank| {
                    shares
                        .iter()
                        .find(|(node, _)| *node == profiles[*rank].node)
                        .map_or(0.0, |(_, share)| *share)
                })
                .sum();
            let leader = ranks
                .iter()
                .copied()
                .max_by(|left, right| {
                    let left_profile = &profiles[*left];
                    let right_profile = &profiles[*right];
                    left_profile
                        .compute_units
                        .total_cmp(&right_profile.compute_units)
                        .then_with(|| right_profile.node.cmp(&left_profile.node))
                })
                .map(|rank| profiles[rank].node)
                .ok_or_else(|| "host group has no eligible leader".to_owned())?;
            host_plans.push(HostPlan {
                host,
                ranks: ranks.clone(),
                leader,
                share,
            });
        }

        let total_share: f64 = host_plans.iter().map(|host| host.share).sum();
        if total_share <= 0.0 {
            return Err("collective plan has no host share".to_owned());
        }
        let mut chunks = Vec::with_capacity(host_plans.len());
        let mut cumulative = 0.0;
        for (index, host) in host_plans.iter().enumerate() {
            let start = ((cumulative / total_share) * bytes as f64).round() as usize;
            cumulative += host.share;
            let end = if index + 1 == host_plans.len() {
                bytes
            } else {
                ((cumulative / total_share) * bytes as f64).round() as usize
            };
            chunks.push(ChunkPlan {
                host: host.host,
                leader: host.leader,
                start: start.min(bytes),
                end: end.min(bytes),
            });
        }
        Ok(Self {
            epoch,
            bytes,
            hosts: host_plans,
            chunks,
        })
    }

    pub fn is_current(&self, epoch: u64) -> bool {
        self.epoch == epoch
    }

    pub fn leader(&self, host: usize) -> Option<NodeId> {
        self.hosts
            .iter()
            .find(|plan| plan.host == host)
            .map(|plan| plan.leader)
    }
}

/// A leader epoch is part of the ordered state. A membership or failure
/// command produces a new epoch; no replica changes leaders out of band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaderEpoch {
    pub epoch: u64,
    pub host: usize,
    pub leader: NodeId,
}

impl LeaderEpoch {
    pub fn from_plan(plan: &CollectivePlan) -> Vec<Self> {
        plan.hosts
            .iter()
            .map(|host| Self {
                epoch: plan.epoch,
                host: host.host,
                leader: host.leader,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinkKind;

    fn profile(node: &str, compute: f64, link: LinkKind, bandwidth: f64) -> RankProfile {
        RankProfile {
            node: NodeId::derive(node),
            gpus: 1,
            compute_units: compute,
            link,
            bandwidth_gbps: Some(bandwidth),
            latency_us: Some(2.0),
        }
    }

    #[test]
    fn four_ampere_and_eight_blackwell_choose_one_leader_per_host() {
        let mut profiles = Vec::new();
        for index in 0..4 {
            profiles.push(profile(
                &format!("local-{index}"),
                1.0,
                LinkKind::Pcie,
                25.0,
            ));
        }
        for index in 0..8 {
            profiles.push(profile(
                &format!("remote-{index}"),
                4.0,
                LinkKind::Ub,
                200.0,
            ));
        }
        let hosts = vec![(0..4).collect::<Vec<_>>(), (4..12).collect::<Vec<_>>()];
        let plan = CollectivePlan::build(&profiles, &hosts, 64, 7, 1 << 20).expect("plan");
        assert_eq!(plan.epoch, 7);
        assert_eq!(plan.hosts.len(), 2);
        assert!(plan.hosts[0].share < plan.hosts[1].share);
        assert!(
            plan.hosts[0]
                .ranks
                .iter()
                .any(|rank| profiles[*rank].node == plan.hosts[0].leader)
        );
        assert!(
            plan.hosts[1]
                .ranks
                .iter()
                .any(|rank| profiles[*rank].node == plan.hosts[1].leader)
        );
        assert_eq!(plan.chunks.first().unwrap().start, 0);
        assert_eq!(plan.chunks.last().unwrap().end, 64);
        assert_eq!(LeaderEpoch::from_plan(&plan).len(), 2);
    }

    #[test]
    fn stale_epoch_is_rejected_by_current_check() {
        let profiles = vec![
            profile("a", 1.0, LinkKind::Pcie, 25.0),
            profile("b", 1.0, LinkKind::Pcie, 25.0),
        ];
        let plan =
            CollectivePlan::build(&profiles, &[vec![0], vec![1]], 16, 2, 1024).expect("plan");
        assert!(plan.is_current(2));
        assert!(!plan.is_current(3));
    }
}
