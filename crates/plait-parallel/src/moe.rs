//! Expert-parallel MoE reference execution.
//!
//! Routing and gating are replicated; expert compute is sharded by EP
//! coordinate. Each expert contributes a full token-major delta with zeros for
//! tokens it did not serve, and a fused sum combines the experts. There is no
//! global all-to-all barrier: experts that finish early do not wait for the
//! slowest expert of an unrelated layer.

use std::collections::BTreeMap;

use plait_core::{NodeId, OpId};

use crate::cluster::LocalCluster;
use crate::plan::{ParallelOp, SyncKind};
use crate::topology::{Axis, Coord, Topology};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeConfig {
    pub tokens: u32,
    pub hidden: u32,
    pub experts: u32,
    pub ffn: u32,
    pub top_k: u32,
}

impl MoeConfig {
    pub fn small() -> Self {
        Self {
            tokens: 4,
            hidden: 4,
            experts: 3,
            ffn: 8,
            top_k: 1,
        }
    }

    fn validate(&self) -> Result<(), MoeError> {
        if self.tokens == 0
            || self.hidden == 0
            || self.experts == 0
            || self.ffn == 0
            || self.top_k == 0
        {
            return Err(MoeError::Config(
                "all MoE dimensions must be non-zero".into(),
            ));
        }
        if self.top_k > self.experts {
            return Err(MoeError::Config("top_k exceeds expert count".into()));
        }
        Ok(())
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let value = (self.0 >> 11) as f64 / ((1u64 << 53) as f64);
        value * 2.0 - 1.0
    }

    fn tensor(&mut self, shape: &[usize]) -> Vec<f64> {
        let count: usize = shape.iter().product();
        (0..count).map(|_| self.unit() * 0.5).collect()
    }
}

pub struct GlobalMoe {
    pub input: Vec<Vec<f64>>,
    pub gate: Vec<Vec<f64>>,
    pub w1: Vec<Vec<Vec<f64>>>,
    pub w2: Vec<Vec<Vec<f64>>>,
}

impl GlobalMoe {
    pub fn new(config: &MoeConfig) -> Self {
        let mut lcg = Lcg::new(0x4d4f_455f_2026_0001);
        let input = (0..config.tokens as usize)
            .map(|_| lcg.tensor(&[config.hidden as usize]))
            .collect();
        let gate = (0..config.hidden as usize)
            .map(|_| lcg.tensor(&[config.experts as usize]))
            .collect();
        let w1 = (0..config.experts as usize)
            .map(|_| {
                (0..config.hidden as usize)
                    .map(|_| lcg.tensor(&[config.ffn as usize]))
                    .collect()
            })
            .collect();
        let w2 = (0..config.experts as usize)
            .map(|_| {
                (0..config.ffn as usize)
                    .map(|_| lcg.tensor(&[config.hidden as usize]))
                    .collect()
            })
            .collect();
        Self {
            input,
            gate,
            w1,
            w2,
        }
    }

    fn route(&self, config: &MoeConfig, token: usize) -> Vec<(usize, f64)> {
        let mut logits: Vec<(f64, usize)> = (0..config.experts as usize)
            .map(|expert| {
                let mut score = 0.0;
                for d in 0..config.hidden as usize {
                    score += self.input[token][d] * self.gate[d][expert];
                }
                (score, expert)
            })
            .collect();
        logits.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let selected: Vec<(f64, usize)> = logits.into_iter().take(config.top_k as usize).collect();
        let max = selected
            .iter()
            .map(|(score, _)| *score)
            .fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = selected.iter().map(|(score, _)| (score - max).exp()).sum();
        selected
            .into_iter()
            .map(|(score, expert)| (expert, (score - max).exp() / denominator))
            .collect()
    }

    #[allow(clippy::needless_range_loop)]
    fn expert_delta(&self, config: &MoeConfig, expert: usize, token: usize) -> Vec<f64> {
        let mut hidden = vec![0.0; config.ffn as usize];
        for f in 0..config.ffn as usize {
            let mut sum = 0.0;
            for d in 0..config.hidden as usize {
                sum += self.input[token][d] * self.w1[expert][d][f];
            }
            hidden[f] = sum.max(0.0);
        }
        let mut out = vec![0.0; config.hidden as usize];
        for d in 0..config.hidden as usize {
            let mut sum = 0.0;
            for f in 0..config.ffn as usize {
                sum += hidden[f] * self.w2[expert][f][d];
            }
            out[d] = sum;
        }
        out
    }

    #[allow(clippy::needless_range_loop)]
    pub fn reference(&self, config: &MoeConfig) -> Vec<Vec<f64>> {
        let mut output = self.input.clone();
        for token in 0..config.tokens as usize {
            for (expert, weight) in self.route(config, token) {
                let delta = self.expert_delta(config, expert, token);
                for d in 0..config.hidden as usize {
                    output[token][d] += weight * delta[d];
                }
            }
        }
        output
    }
}

pub struct MoeOutput {
    pub values: Vec<Vec<f64>>,
    pub instances: usize,
}

pub struct MoeRunner {
    config: MoeConfig,
    topology: Topology,
    cluster: LocalCluster,
    plan_op: ParallelOp,
}

impl MoeRunner {
    pub fn new(topology: Topology, config: MoeConfig) -> Result<Self, MoeError> {
        config.validate()?;
        let sizes = topology.sizes();
        if sizes.data != 1
            || sizes.pipeline != 1
            || sizes.tensor != 1
            || sizes.context != 1
            || sizes.sequence != 1
        {
            return Err(MoeError::Config(
                "the MoE reference runner expects one DP/PP/TP/CP/SP coordinate".into(),
            ));
        }
        if sizes.expert != config.experts {
            return Err(MoeError::Config(
                "one expert must map to one EP coordinate".into(),
            ));
        }
        let participants = topology.group(Axis::Expert, Coord::default());
        let op_id = OpId::new(NodeId::derive("moe-combine"), 1);
        let plan_op = ParallelOp {
            name: "ep.combine:0".into(),
            op_id,
            axis: Axis::Expert,
            sync: SyncKind::Combine,
            participants: participants.clone(),
            conflict_key: "ep:0:combine".into(),
            transfer: crate::plan::TransferPolicy::Inline,
        };
        let cluster = LocalCluster::new(topology.nodes().iter().copied());
        Ok(Self {
            config,
            topology,
            cluster,
            plan_op,
        })
    }

    pub fn run(mut self, global: &GlobalMoe) -> Result<MoeOutput, MoeError> {
        let tokens = self.config.tokens as usize;
        let hidden = self.config.hidden as usize;
        let mut contributions = BTreeMap::new();
        for node in &self.plan_op.participants {
            let coord = self
                .topology
                .coord(*node)
                .ok_or_else(|| MoeError::Plan("node missing from topology".into()))?;
            let expert = coord.expert as usize;
            let mut values = vec![0.0; tokens * hidden];
            for token in 0..tokens {
                let routed = global
                    .route(&self.config, token)
                    .into_iter()
                    .find(|(candidate, _)| *candidate == expert);
                if let Some((_, weight)) = routed {
                    let delta = global.expert_delta(&self.config, expert, token);
                    for d in 0..hidden {
                        values[token * hidden + d] = weight * delta[d];
                    }
                }
            }
            contributions.insert(*node, values);
        }
        let combined = self
            .cluster
            .run_op(&self.plan_op, contributions)
            .map_err(|error| MoeError::Plan(error.to_string()))?;
        let mut output = global.input.clone();
        for token in 0..tokens {
            for d in 0..hidden {
                output[token][d] += combined[token * hidden + d];
            }
        }
        let instances = self
            .topology
            .nodes()
            .first()
            .map(|node| self.cluster.executed(*node))
            .unwrap_or(0);
        Ok(MoeOutput {
            values: output,
            instances,
        })
    }
}

#[derive(Debug)]
pub enum MoeError {
    Config(String),
    Plan(String),
}

impl std::fmt::Display for MoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(message) => write!(f, "configuration: {message}"),
            Self::Plan(message) => write!(f, "plan: {message}"),
        }
    }
}

impl std::error::Error for MoeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::AxisSizes;

    fn compare(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (token, (left, right)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(left.len(), right.len());
            for (column, (a, b)) in left.iter().zip(right).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "mismatch at [{token}][{column}]: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn expert_parallel_moe_matches_reference() {
        let config = MoeConfig::small();
        let global = GlobalMoe::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(1, 1, 1, 3, 1, 1), "moe-ep").unwrap();
        let runner = MoeRunner::new(topology, config.clone()).unwrap();
        let output = runner.run(&global).unwrap();
        compare(&output.values, &global.reference(&config));
        assert!(output.instances > 0);
    }

    #[test]
    fn top_two_experts_are_summed() {
        let mut config = MoeConfig::small();
        config.top_k = 2;
        let global = GlobalMoe::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(1, 1, 1, 3, 1, 1), "moe-top2").unwrap();
        let output = MoeRunner::new(topology, config.clone())
            .unwrap()
            .run(&global)
            .unwrap();
        compare(&output.values, &global.reference(&config));
    }
}
