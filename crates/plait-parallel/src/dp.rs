//! Data-parallel gradient accumulation through fused consensus operations.
//!
//! Each replica computes a local gradient for its batch. Gradients are split
//! into parameter shards; each shard is a separate fused-sum operation with the
//! DP group as contributors. A shard can materialize without waiting for
//! unrelated shards, and no rank-zero optimizer process is involved.

use std::collections::BTreeMap;

use plait_core::{NodeId, OpId};

use crate::cluster::LocalCluster;
use crate::plan::{ParallelOp, SyncKind, TransferPolicy};
use crate::topology::{Axis, Coord, Topology};

#[derive(Clone, Debug, PartialEq)]
pub struct DpConfig {
    pub replicas: u32,
    pub samples_per_replica: u32,
    pub features: u32,
    pub shards: u32,
    pub steps: u32,
    pub learning_rate: f64,
}

impl DpConfig {
    pub fn small() -> Self {
        Self {
            replicas: 3,
            samples_per_replica: 2,
            features: 4,
            shards: 2,
            steps: 3,
            learning_rate: 0.1,
        }
    }

    fn validate(&self) -> Result<(), DpError> {
        if self.replicas == 0
            || self.samples_per_replica == 0
            || self.features == 0
            || self.shards == 0
            || self.steps == 0
        {
            return Err(DpError::Config(
                "all DP dimensions and steps must be non-zero".into(),
            ));
        }
        if self.shards > self.features {
            return Err(DpError::Config("shards exceed feature count".into()));
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
}

pub struct GlobalDp {
    pub weights: Vec<f64>,
    /// Per-replica batches: `[replica][sample][feature]`.
    pub inputs: Vec<Vec<Vec<f64>>>,
    /// Per-replica targets: `[replica][sample]`.
    pub targets: Vec<Vec<f64>>,
}

impl GlobalDp {
    pub fn new(config: &DpConfig) -> Self {
        let mut lcg = Lcg::new(0x4450_5f32_3032_3601);
        let weights = (0..config.features).map(|_| lcg.unit() * 0.2).collect();
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for _ in 0..config.replicas {
            let mut batch = Vec::new();
            let mut target = Vec::new();
            for _ in 0..config.samples_per_replica {
                let x: Vec<f64> = (0..config.features).map(|_| lcg.unit()).collect();
                let y: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(index, value)| value * (index as f64 + 1.0) * 0.1)
                    .sum();
                batch.push(x);
                target.push(y);
            }
            inputs.push(batch);
            targets.push(target);
        }
        Self {
            weights,
            inputs,
            targets,
        }
    }

    fn global_gradient(&self, config: &DpConfig, weights: &[f64]) -> Vec<f64> {
        let total = (config.replicas * config.samples_per_replica) as f64;
        let mut gradient = vec![0.0; config.features as usize];
        for replica in 0..config.replicas as usize {
            for sample in 0..config.samples_per_replica as usize {
                let prediction: f64 = self.inputs[replica][sample]
                    .iter()
                    .zip(weights)
                    .map(|(x, w)| x * w)
                    .sum();
                let error = prediction - self.targets[replica][sample];
                for (feature, x) in self.inputs[replica][sample].iter().enumerate() {
                    gradient[feature] += 2.0 * error * x / total;
                }
            }
        }
        gradient
    }

    pub fn reference(&self, config: &DpConfig) -> (Vec<f64>, Vec<f64>) {
        let mut weights = self.weights.clone();
        let mut losses = Vec::new();
        let total = (config.replicas * config.samples_per_replica) as f64;
        for _ in 0..config.steps {
            let gradient = self.global_gradient(config, &weights);
            let mut loss = 0.0;
            for replica in 0..config.replicas as usize {
                for sample in 0..config.samples_per_replica as usize {
                    let prediction: f64 = self.inputs[replica][sample]
                        .iter()
                        .zip(&weights)
                        .map(|(x, w)| x * w)
                        .sum();
                    loss += (prediction - self.targets[replica][sample]).powi(2) / total;
                }
            }
            losses.push(loss);
            for (weight, grad) in weights.iter_mut().zip(gradient) {
                *weight -= config.learning_rate * grad;
            }
        }
        (weights, losses)
    }
}

pub struct DpOutput {
    pub weights: Vec<f64>,
    pub losses: Vec<f64>,
    pub instances: usize,
}

pub struct DpRunner {
    config: DpConfig,
    topology: Topology,
    cluster: LocalCluster,
    participants: Vec<NodeId>,
}

impl DpRunner {
    pub fn new(topology: Topology, config: DpConfig) -> Result<Self, DpError> {
        config.validate()?;
        let sizes = topology.sizes();
        if sizes.pipeline != 1
            || sizes.tensor != 1
            || sizes.expert != 1
            || sizes.context != 1
            || sizes.sequence != 1
        {
            return Err(DpError::Config(
                "the DP reference runner expects one PP/TP/EP/CP/SP coordinate".into(),
            ));
        }
        if sizes.data != config.replicas {
            return Err(DpError::Config(
                "DP replication factor must match the topology".into(),
            ));
        }
        let participants = topology.group(Axis::Data, Coord::default());
        let cluster = LocalCluster::new(topology.nodes().iter().copied());
        Ok(Self {
            config,
            topology,
            cluster,
            participants,
        })
    }

    pub fn run(mut self, global: &GlobalDp) -> Result<DpOutput, DpError> {
        let mut weights = global.weights.clone();
        let mut losses = Vec::new();
        let total = (self.config.replicas * self.config.samples_per_replica) as f64;
        let features = self.config.features as usize;
        let shard_width = features.div_ceil(self.config.shards as usize);

        for _step in 0..self.config.steps {
            let step_weights = weights.clone();
            let mut loss = 0.0;
            for replica in 0..self.config.replicas as usize {
                for sample in 0..self.config.samples_per_replica as usize {
                    let prediction: f64 = global.inputs[replica][sample]
                        .iter()
                        .zip(&step_weights)
                        .map(|(x, w)| x * w)
                        .sum();
                    loss += (prediction - global.targets[replica][sample]).powi(2) / total;
                }
            }
            losses.push(loss);

            for shard_index in 0..self.config.shards as usize {
                let start = shard_index * shard_width;
                let end = (start + shard_width).min(features);
                let name = format!("dp.gradient:{_step}:{shard_index}");
                let op = ParallelOp {
                    op_id: OpId::new(NodeId::derive(&name), 1),
                    name,
                    axis: Axis::Data,
                    sync: SyncKind::FusedSum,
                    participants: self.participants.clone(),
                    conflict_key: format!("dp:{_step}:gradient:{shard_index}"),
                    transfer: TransferPolicy::Registered,
                };
                let mut contributions = BTreeMap::new();
                for node in &op.participants {
                    let coord = self
                        .topology
                        .coord(*node)
                        .ok_or_else(|| DpError::Plan("node missing from topology".into()))?;
                    let replica = coord.data as usize;
                    let mut gradient = vec![0.0; end - start];
                    for sample in 0..self.config.samples_per_replica as usize {
                        let prediction: f64 = global.inputs[replica][sample]
                            .iter()
                            .zip(&step_weights)
                            .map(|(x, w)| x * w)
                            .sum();
                        let error = prediction - global.targets[replica][sample];
                        for (offset, feature) in (start..end).enumerate() {
                            gradient[offset] +=
                                2.0 * error * global.inputs[replica][sample][feature] / total;
                        }
                    }
                    contributions.insert(*node, gradient);
                }
                let fused = self
                    .cluster
                    .run_op(&op, contributions)
                    .map_err(|error| DpError::Plan(error.to_string()))?;
                if fused.len() != end - start {
                    return Err(DpError::Plan("gradient shard width mismatch".into()));
                }
                for (offset, feature) in (start..end).enumerate() {
                    weights[feature] -= self.config.learning_rate * fused[offset];
                }
            }
        }

        let instances = self
            .topology
            .nodes()
            .first()
            .map(|node| self.cluster.executed(*node))
            .unwrap_or(0);
        Ok(DpOutput {
            weights,
            losses,
            instances,
        })
    }
}

#[derive(Debug)]
pub enum DpError {
    Config(String),
    Plan(String),
}

impl std::fmt::Display for DpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(message) => write!(f, "configuration: {message}"),
            Self::Plan(message) => write!(f, "plan: {message}"),
        }
    }
}

impl std::error::Error for DpError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::AxisSizes;

    #[test]
    fn data_parallel_gradient_matches_single_optimizer() {
        let config = DpConfig::small();
        let global = GlobalDp::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(3, 1, 1, 1, 1, 1), "dp").unwrap();
        let output = DpRunner::new(topology, config.clone())
            .unwrap()
            .run(&global)
            .unwrap();
        let (weights, losses) = global.reference(&config);
        assert_eq!(output.weights.len(), weights.len());
        for (index, (actual, expected)) in output.weights.iter().zip(&weights).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-9,
                "weight {index}: {actual} vs {expected}"
            );
        }
        for (actual, expected) in output.losses.iter().zip(&losses) {
            assert!((actual - expected).abs() < 1e-9);
        }
        assert!(output.instances > 0);
    }

    #[test]
    fn more_shards_than_one_still_reduces_exactly() {
        let mut config = DpConfig::small();
        config.shards = 4;
        config.replicas = 2;
        let global = GlobalDp::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(2, 1, 1, 1, 1, 1), "dp4").unwrap();
        let output = DpRunner::new(topology, config.clone())
            .unwrap()
            .run(&global)
            .unwrap();
        let (weights, _) = global.reference(&config);
        for (actual, expected) in output.weights.iter().zip(&weights) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }
}
