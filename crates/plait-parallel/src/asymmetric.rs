//! Asymmetric rank-group work partitioning.
//!
//! A cluster can contain groups with different accelerator counts (for
//! example 4 local GPUs and 8 remote GPUs). Plait ordering quorum membership stays
//! one-vote-per-node; only data-parallel work is weighted. Weighted row ranges
//! are contiguous and cover the matrix exactly once.

use std::ops::Range;

use plait_core::NodeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankWeight {
    pub node: NodeId,
    /// Advertised accelerator count for this rank. Used only for work share.
    pub gpus: u32,
    /// Relative capacity; a Blackwell rank may be heavier than a 3090 rank.
    pub weight: u32,
}

#[derive(Clone, Debug)]
pub struct AsymmetricLayout {
    ranks: Vec<RankWeight>,
    total_weight: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightedRowRange {
    pub node: NodeId,
    pub range: Range<usize>,
}

impl AsymmetricLayout {
    pub fn new(ranks: Vec<RankWeight>) -> Result<Self, String> {
        if ranks.is_empty() {
            return Err("asymmetric layout needs at least one rank".to_owned());
        }
        if ranks.iter().any(|rank| rank.weight == 0) {
            return Err("rank weight must be positive".to_owned());
        }
        let total_weight = ranks.iter().map(|rank| rank.weight as u64).sum();
        Ok(Self {
            ranks,
            total_weight,
        })
    }

    pub fn ranks(&self) -> &[RankWeight] {
        &self.ranks
    }

    /// Partition `rows` into contiguous weighted ranges. The union is exactly
    /// `0..rows`; ranges may be empty for very small matrices.
    pub fn row_ranges(&self, rows: usize) -> Vec<WeightedRowRange> {
        let rows = rows as u64;
        let mut result = Vec::with_capacity(self.ranks.len());
        let mut cumulative = 0u64;
        for rank in &self.ranks {
            let start = cumulative * rows / self.total_weight;
            cumulative += rank.weight as u64;
            let end = cumulative * rows / self.total_weight;
            result.push(WeightedRowRange {
                node: rank.node,
                range: start as usize..end as usize,
            });
        }
        result
    }

    /// Work share in parts per million, useful for scheduler telemetry.
    pub fn share_ppm(&self, node: NodeId) -> Option<u64> {
        self.ranks
            .iter()
            .find(|rank| rank.node == node)
            .map(|rank| rank.weight as u64 * 1_000_000 / self.total_weight)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkKind {
    NvLink,
    Pcie,
    Ub,
    Network,
}

impl LinkKind {
    pub fn nominal_bandwidth_gbps(self) -> f64 {
        match self {
            Self::NvLink => 300.0,
            Self::Ub => 200.0,
            Self::Pcie => 50.0,
            Self::Network => 25.0,
        }
    }

    pub fn nominal_latency_us(self) -> f64 {
        match self {
            Self::NvLink => 1.0,
            Self::Ub => 2.0,
            Self::Pcie => 5.0,
            Self::Network => 20.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RankProfile {
    pub node: NodeId,
    pub gpus: u32,
    /// Relative compute rate. A value of 2 means twice as many rows per unit time.
    pub compute_units: f64,
    pub link: LinkKind,
    /// Measured effective bandwidth; falls back to the link class nominal value.
    pub bandwidth_gbps: Option<f64>,
    /// Measured collective latency; falls back to the link class nominal value.
    pub latency_us: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct HeterogeneousPlan {
    pub layout: AsymmetricLayout,
    /// Fractional row share chosen by the LP, before integer quantization.
    pub shares: Vec<(NodeId, f64)>,
    /// LP optimum: the slowest rank's modeled time.
    pub makespan: f64,
}

pub struct HeterogeneousLayout;

impl HeterogeneousLayout {
    /// Plan row weights from compute rate and effective collective bandwidth.
    ///
    /// The linear model minimizes the maximum per-rank time:
    /// `cost_i = 1 / compute_i + message_bytes / bandwidth_i + latency_i`.
    /// The share is proportional to `1 / cost_i`, which is the max-min
    /// allocation for a sum of independent costs. This is deliberately a
    /// small LP-style planner, not a global MILP over every link.
    pub fn plan(
        profiles: &[RankProfile],
        message_bytes: usize,
    ) -> Result<AsymmetricLayout, String> {
        Self::plan_detailed(profiles, message_bytes).map(|plan| plan.layout)
    }

    /// Solve the max-min shard allocation LP:
    ///
    ///   minimize T
    ///   subject to  a_i * x_i + b_i <= T   for every rank
    ///               sum_i x_i = 1, x_i >= 0
    ///
    /// `a_i` is variable compute/communication cost per unit share and `b_i`
    /// is fixed collective latency. This is a one-constraint LP; its optimum
    /// is found by parametric search on T, so no external solver is needed.
    pub fn plan_detailed(
        profiles: &[RankProfile],
        message_bytes: usize,
    ) -> Result<HeterogeneousPlan, String> {
        if profiles.is_empty() {
            return Err("heterogeneous layout needs at least one rank".to_owned());
        }
        let mut slopes = Vec::with_capacity(profiles.len());
        let mut intercepts = Vec::with_capacity(profiles.len());
        for profile in profiles {
            if !profile.compute_units.is_finite() || profile.compute_units <= 0.0 {
                return Err(format!("rank {} has invalid compute rate", profile.node));
            }
            let bandwidth = profile
                .bandwidth_gbps
                .unwrap_or_else(|| profile.link.nominal_bandwidth_gbps());
            if !bandwidth.is_finite() || bandwidth <= 0.0 {
                return Err(format!("rank {} has invalid bandwidth", profile.node));
            }
            let latency_us = profile
                .latency_us
                .unwrap_or_else(|| profile.link.nominal_latency_us());
            let a = 1.0 / profile.compute_units + message_bytes as f64 / (bandwidth * 1e9);
            let b = latency_us / 1e6;
            if !a.is_finite() || a <= 0.0 || !b.is_finite() || b < 0.0 {
                return Err(format!("rank {} has invalid cost model", profile.node));
            }
            slopes.push(a);
            intercepts.push(b);
        }

        let lower = intercepts.iter().copied().fold(0.0_f64, f64::max);
        let upper = slopes
            .iter()
            .zip(&intercepts)
            .map(|(a, b)| a + b)
            .fold(0.0_f64, f64::max);
        let mut low = lower;
        let mut high = upper;
        for _ in 0..100 {
            let mid = (low + high) * 0.5;
            let capacity: f64 = slopes
                .iter()
                .zip(&intercepts)
                .map(|(a, b)| ((mid - b) / a).max(0.0))
                .sum();
            if capacity >= 1.0 {
                high = mid;
            } else {
                low = mid;
            }
        }
        let makespan = high;
        let mut shares: Vec<f64> = slopes
            .iter()
            .zip(&intercepts)
            .map(|(a, b)| ((makespan - b) / a).max(0.0))
            .collect();
        let total: f64 = shares.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err("heterogeneous LP produced no feasible allocation".to_owned());
        }
        for share in &mut shares {
            *share /= total;
        }

        let mut ranks = Vec::with_capacity(profiles.len());
        for (profile, share) in profiles.iter().zip(&shares) {
            let weight = (share * 1_000_000.0).round().clamp(1.0, u32::MAX as f64) as u32;
            ranks.push(RankWeight {
                node: profile.node,
                gpus: profile.gpus,
                weight,
            });
        }
        let layout = AsymmetricLayout::new(ranks)?;
        Ok(HeterogeneousPlan {
            layout,
            shares: profiles
                .iter()
                .zip(shares)
                .map(|(profile, share)| (profile.node, share))
                .collect(),
            makespan,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_ranges_cover_each_row_once() {
        let ranks = vec![
            RankWeight {
                node: NodeId::derive("local0"),
                gpus: 1,
                weight: 1,
            },
            RankWeight {
                node: NodeId::derive("local1"),
                gpus: 1,
                weight: 1,
            },
            RankWeight {
                node: NodeId::derive("remote0"),
                gpus: 1,
                weight: 2,
            },
            RankWeight {
                node: NodeId::derive("remote1"),
                gpus: 1,
                weight: 2,
            },
        ];
        let layout = AsymmetricLayout::new(ranks).expect("layout");
        let ranges = layout.row_ranges(17);
        let mut expected = 0usize;
        for range in &ranges {
            assert_eq!(range.range.start, expected);
            expected = range.range.end;
        }
        assert_eq!(expected, 17);
        assert_eq!(ranges[0].range, 0..2);
        assert_eq!(ranges[2].range, 5..11);
    }

    #[test]
    fn local_four_remote_eight_assigns_two_thirds_to_remote() {
        let mut ranks = Vec::new();
        for index in 0..4 {
            ranks.push(RankWeight {
                node: NodeId::derive(&format!("local-{index}")),
                gpus: 1,
                weight: 1,
            });
        }
        for index in 0..8 {
            ranks.push(RankWeight {
                node: NodeId::derive(&format!("remote-{index}")),
                gpus: 1,
                weight: 1,
            });
        }
        let layout = AsymmetricLayout::new(ranks).expect("layout");
        let ranges = layout.row_ranges(120);
        let local: usize = ranges[..4].iter().map(|r| r.range.len()).sum();
        let remote: usize = ranges[4..].iter().map(|r| r.range.len()).sum();
        assert_eq!(local, 40);
        assert_eq!(remote, 80);
    }
    #[test]
    fn four_3090s_get_less_than_eight_pro5000s() {
        let mut profiles = Vec::new();
        for index in 0..4 {
            profiles.push(RankProfile {
                node: NodeId::derive(&format!("3090-{index}")),
                gpus: 1,
                compute_units: 1.0,
                link: LinkKind::Pcie,
                bandwidth_gbps: Some(25.0),
                latency_us: Some(5.0),
            });
        }
        for index in 0..8 {
            profiles.push(RankProfile {
                node: NodeId::derive(&format!("pro5000-{index}")),
                gpus: 1,
                compute_units: 4.0,
                link: LinkKind::Ub,
                bandwidth_gbps: Some(200.0),
                latency_us: Some(2.0),
            });
        }
        let plan = HeterogeneousLayout::plan_detailed(&profiles, 1 << 20).expect("plan");
        let ranges = plan.layout.row_ranges(120);
        let local: usize = ranges[..4].iter().map(|r| r.range.len()).sum();
        let remote: usize = ranges[4..].iter().map(|r| r.range.len()).sum();
        assert_eq!(local + remote, 120);
        assert!(
            remote > local,
            "eight Pro5000s must own more rows than four 3090s"
        );
        assert!(ranges[4].range.len() > ranges[0].range.len());
        let local_share: f64 = plan.shares[..4].iter().map(|(_, share)| share).sum();
        let remote_share: f64 = plan.shares[4..].iter().map(|(_, share)| share).sum();
        assert!(remote_share > 4.0 * local_share);
    }
    #[test]
    fn lp_matches_closed_form_for_two_ranks_without_latency() {
        let profiles = vec![
            RankProfile {
                node: NodeId::derive("fast"),
                gpus: 1,
                compute_units: 4.0,
                link: LinkKind::NvLink,
                bandwidth_gbps: Some(1e9),
                latency_us: Some(0.0),
            },
            RankProfile {
                node: NodeId::derive("slow"),
                gpus: 1,
                compute_units: 1.0,
                link: LinkKind::Pcie,
                bandwidth_gbps: Some(1e9),
                latency_us: Some(0.0),
            },
        ];
        let plan = HeterogeneousLayout::plan_detailed(&profiles, 0).expect("plan");
        assert!((plan.shares[0].1 - 0.8).abs() < 1e-9);
        assert!((plan.shares[1].1 - 0.2).abs() < 1e-9);
        assert!((plan.makespan - 0.2).abs() < 1e-9);
    }
}
