//! Pipeline-parallel execution through ordered stage handoffs.
//!
//! Each stage owns its weights and emits a `Stage` command that reads the
//! previous stage's activation and writes the next one. EPaxos orders a stage
//! after its input but leaves different microbatches independent, so the
//! runner can put several microbatches in flight without a per-stage barrier.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::{Command, ShardId, TensorId, TensorKey};

use crate::cluster::LocalCluster;
use crate::topology::{Coord, Topology};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PpConfig {
    pub stages: u32,
    pub microbatches: u32,
    pub hidden: u32,
    pub layers_per_stage: u32,
}

impl PpConfig {
    pub fn small() -> Self {
        Self {
            stages: 3,
            microbatches: 3,
            hidden: 4,
            layers_per_stage: 1,
        }
    }

    fn validate(&self) -> Result<(), PpError> {
        if self.stages == 0
            || self.microbatches == 0
            || self.hidden == 0
            || self.layers_per_stage == 0
        {
            return Err(PpError::Config("all PP dimensions must be non-zero".into()));
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

pub struct GlobalPp {
    /// `[microbatch][hidden]`
    pub input: Vec<Vec<f64>>,
    /// `[stage][layer][dst][src]`
    pub weights: Vec<Vec<Vec<Vec<f64>>>>,
    /// `[stage][layer][hidden]`
    pub biases: Vec<Vec<Vec<f64>>>,
}

impl GlobalPp {
    pub fn new(config: &PpConfig) -> Self {
        let mut lcg = Lcg::new(0x5050_5f32_3032_3602);
        let input = (0..config.microbatches as usize)
            .map(|_| lcg.tensor(&[config.hidden as usize]))
            .collect();
        let weights = (0..config.stages as usize)
            .map(|_| {
                (0..config.layers_per_stage as usize)
                    .map(|_| {
                        (0..config.hidden as usize)
                            .map(|_| lcg.tensor(&[config.hidden as usize]))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let biases = (0..config.stages as usize)
            .map(|_| {
                (0..config.layers_per_stage as usize)
                    .map(|_| lcg.tensor(&[config.hidden as usize]))
                    .collect()
            })
            .collect();
        Self {
            input,
            weights,
            biases,
        }
    }

    fn forward_stage(&self, config: &PpConfig, stage: usize, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for layer in 0..config.layers_per_stage as usize {
            let mut y = vec![0.0; config.hidden as usize];
            let weight = &self.weights[stage][layer];
            let bias = &self.biases[stage][layer];
            for dst in 0..config.hidden as usize {
                for src in 0..config.hidden as usize {
                    y[dst] += weight[dst][src] * x[src];
                }
                y[dst] += bias[dst];
                let final_layer = stage == config.stages as usize - 1
                    && layer == config.layers_per_stage as usize - 1;
                if !final_layer {
                    y[dst] = y[dst].max(0.0);
                }
            }
            x = y;
        }
        x
    }

    pub fn reference(&self, config: &PpConfig) -> Vec<Vec<f64>> {
        (0..config.microbatches as usize)
            .map(|microbatch| {
                let mut activation = self.input[microbatch].clone();
                for stage in 0..config.stages as usize {
                    activation = self.forward_stage(config, stage, &activation);
                }
                activation
            })
            .collect()
    }
}

pub struct PpOutput {
    pub values: Vec<Vec<f64>>,
    pub instances: usize,
    pub max_in_flight: usize,
}

pub struct PpRunner {
    config: PpConfig,
    topology: Topology,
    cluster: LocalCluster,
}

impl PpRunner {
    pub fn new(topology: Topology, config: PpConfig) -> Result<Self, PpError> {
        config.validate()?;
        let sizes = topology.sizes();
        if sizes.data != 1
            || sizes.tensor != 1
            || sizes.expert != 1
            || sizes.context != 1
            || sizes.sequence != 1
        {
            return Err(PpError::Config(
                "the PP reference runner expects one DP/TP/EP/CP/SP coordinate".into(),
            ));
        }
        if sizes.pipeline != config.stages {
            return Err(PpError::Config(
                "pipeline stage count must match the topology".into(),
            ));
        }
        let cluster = LocalCluster::new(topology.nodes().iter().copied());
        Ok(Self {
            config,
            topology,
            cluster,
        })
    }

    pub fn run(mut self, global: &GlobalPp) -> Result<PpOutput, PpError> {
        let stages = self.config.stages as usize;
        let microbatches = self.config.microbatches as usize;
        let hidden = self.config.hidden as usize;
        let mut done: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut activations: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
        let mut max_in_flight = 0usize;
        let coordinator = *self
            .topology
            .nodes()
            .first()
            .ok_or_else(|| PpError::Plan("topology has no nodes".into()))?;

        while done.len() < stages * microbatches {
            let mut ready = Vec::new();
            for stage in 0..stages {
                for microbatch in 0..microbatches {
                    if done.contains(&(stage, microbatch)) {
                        continue;
                    }
                    if stage == 0 || done.contains(&(stage - 1, microbatch)) {
                        ready.push((stage, microbatch));
                    }
                }
            }
            if ready.is_empty() {
                return Err(PpError::Plan("no runnable pipeline stage".into()));
            }
            max_in_flight = max_in_flight.max(ready.len());

            for (stage, microbatch) in &ready {
                let input = if *stage == 0 {
                    global.input[*microbatch].clone()
                } else {
                    activations
                        .get(&(*stage - 1, *microbatch))
                        .cloned()
                        .ok_or_else(|| PpError::Plan("missing predecessor activation".into()))?
                };
                let values = global.forward_stage(&self.config, *stage, &input);
                let input_key = if *stage == 0 {
                    None
                } else {
                    Some(tensor_key(*stage - 1, *microbatch))
                };
                let output_key = tensor_key(*stage, *microbatch);
                let node = self
                    .topology
                    .node_at(Coord {
                        pipeline: *stage as u32,
                        ..Coord::default()
                    })
                    .ok_or_else(|| PpError::Plan("stage node missing".into()))?;
                self.cluster
                    .propose(
                        node,
                        Command::Stage {
                            input: input_key,
                            output: output_key,
                            version: *microbatch as u64,
                            values,
                        },
                    )
                    .map_err(|error| PpError::Plan(error.to_string()))?;
            }
            self.cluster.run();

            for (stage, microbatch) in ready {
                let key = tensor_key(stage, microbatch);
                let view = self
                    .cluster
                    .runtime(coordinator)
                    .shard(key)
                    .ok_or_else(|| PpError::Plan("stage output was not materialized".into()))?;
                if view.values.len() != hidden {
                    return Err(PpError::Plan("stage output width mismatch".into()));
                }
                activations.insert((stage, microbatch), view.values.clone());
                done.insert((stage, microbatch));
            }
        }

        let values: Vec<Vec<f64>> = (0..microbatches)
            .map(|microbatch| {
                activations
                    .get(&(stages - 1, microbatch))
                    .cloned()
                    .ok_or_else(|| PpError::Plan("missing final activation".into()))
            })
            .collect::<Result<_, _>>()?;
        let instances = self
            .topology
            .nodes()
            .first()
            .map(|node| self.cluster.executed(*node))
            .unwrap_or(0);
        Ok(PpOutput {
            values,
            instances,
            max_in_flight,
        })
    }
}

fn tensor_key(stage: usize, microbatch: usize) -> TensorKey {
    TensorKey::new(
        TensorId(10_000 + stage as u128 * 1_000 + microbatch as u128),
        ShardId(0),
    )
}

#[derive(Debug)]
pub enum PpError {
    Config(String),
    Plan(String),
}

impl std::fmt::Display for PpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(message) => write!(f, "configuration: {message}"),
            Self::Plan(message) => write!(f, "plan: {message}"),
        }
    }
}

impl std::error::Error for PpError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::AxisSizes;

    #[test]
    fn pipeline_parallel_matches_single_node_and_overlaps_microbatches() {
        let config = PpConfig::small();
        let global = GlobalPp::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(1, 3, 1, 1, 1, 1), "pp").unwrap();
        let output = PpRunner::new(topology, config.clone())
            .unwrap()
            .run(&global)
            .unwrap();
        let expected = global.reference(&config);
        assert_eq!(output.values.len(), expected.len());
        for (microbatch, (actual, expected)) in output.values.iter().zip(&expected).enumerate() {
            for (column, (a, b)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "microbatch {microbatch} column {column}: {a} vs {b}"
                );
            }
        }
        assert!(output.instances > 0);
        assert!(
            output.max_in_flight > 1,
            "pipeline did not overlap microbatches"
        );
    }

    #[test]
    fn pipeline_stage_handoff_enforces_dependencies() {
        let config = PpConfig {
            stages: 2,
            microbatches: 2,
            hidden: 3,
            layers_per_stage: 2,
        };
        let global = GlobalPp::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(1, 2, 1, 1, 1, 1), "pp2").unwrap();
        let output = PpRunner::new(topology, config.clone())
            .unwrap()
            .run(&global)
            .unwrap();
        let expected = global.reference(&config);
        for (actual, expected) in output.values.iter().zip(&expected) {
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    }
}
