//! Asymmetric rank-group work partitioning.
//!
//! A cluster can contain groups with different accelerator counts (for
//! example 4 local GPUs and 8 remote GPUs). EPaxos quorum membership stays
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
}
