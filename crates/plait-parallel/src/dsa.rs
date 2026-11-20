use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use plait_consensus::ProposeError;
use plait_core::{Command, NodeId, OpId, OpSpec, ShardId, TensorId, TensorKey};
use plait_mem::{BufferPool, LoopbackRdma, MemoryKind, PoolStats, RdmaPlane, RegionHandle};

use crate::cluster::LocalCluster;
use crate::plan::{DsaPlan, ParallelOp, dsa_plan, fuse_kind, seq_range};
use crate::topology::{Axis, Topology, TopologyError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsaConfig {
    pub seq: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub index_dim: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub experts: u32,
    pub ffn: u32,
}

impl DsaConfig {
    pub fn small() -> Self {
        Self {
            seq: 6,
            heads: 2,
            head_dim: 4,
            index_dim: 8,
            top_k: 2,
            hidden: 8,
            experts: 2,
            ffn: 16,
        }
    }

    fn validate(&self) -> Result<(), DsaError> {
        if self.seq == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.index_dim == 0
            || self.top_k == 0
        {
            return Err(DsaError::Config(
                "all DSA dimensions must be non-zero".into(),
            ));
        }
        if self.hidden != self.heads * self.head_dim {
            return Err(DsaError::Config(
                "hidden must equal heads * head_dim".into(),
            ));
        }
        if self.top_k > self.seq {
            return Err(DsaError::Config("top_k exceeds seq".into()));
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

/// Deterministic reference tensors shared by the reference implementation and
/// every shard. In a real model these come from the checkpoint and local
/// compute; for the demo they make distributed/reference comparison exact.
pub struct GlobalDsa {
    pub q: Vec<Vec<Vec<f64>>>,
    pub k: Vec<Vec<Vec<f64>>>,
    pub v: Vec<Vec<Vec<f64>>>,
    pub qi: Vec<Vec<Vec<f64>>>,
    pub ki: Vec<Vec<Vec<f64>>>,
    /// Output projection, indexed `[dst][src]`.
    pub wo: Vec<Vec<f64>>,
}

impl GlobalDsa {
    pub fn new(config: &DsaConfig) -> Self {
        let mut lcg = Lcg::new(0x5eed_1234_abcd_0001);
        let heads = config.heads as usize;
        let seq = config.seq as usize;
        let head_dim = config.head_dim as usize;
        let index_dim = config.index_dim as usize;
        let hidden = config.hidden as usize;
        let tensor = |lcg: &mut Lcg, dim: usize| -> Vec<Vec<Vec<f64>>> {
            (0..heads)
                .map(|_| (0..seq).map(|_| lcg.tensor(&[dim])).collect())
                .collect()
        };
        let q = tensor(&mut lcg, head_dim);
        let k = tensor(&mut lcg, head_dim);
        let v = tensor(&mut lcg, head_dim);
        let qi = tensor(&mut lcg, index_dim);
        let ki = tensor(&mut lcg, index_dim);
        let wo = (0..hidden).map(|_| lcg.tensor(&[hidden])).collect();
        Self {
            q,
            k,
            v,
            qi,
            ki,
            wo,
        }
    }

    /// Single-node exact DSA for this reference configuration. The
    /// distributed path must reproduce this within floating-point tolerance.
    #[allow(clippy::needless_range_loop)]
    pub fn reference(&self, config: &DsaConfig) -> Vec<Vec<f64>> {
        let heads = config.heads as usize;
        let seq = config.seq as usize;
        let head_dim = config.head_dim as usize;
        let index_dim = config.index_dim as usize;
        let hidden = config.hidden as usize;
        let scale_index = 1.0 / (index_dim as f64).sqrt();
        let scale_head = 1.0 / (head_dim as f64).sqrt();

        let mut attention = vec![vec![vec![0.0; head_dim]; seq]; heads];
        for head in 0..heads {
            for query in 0..seq {
                let mut scores: Vec<(f64, usize)> = (0..=query)
                    .map(|key| {
                        let mut score = 0.0;
                        for d in 0..index_dim {
                            score += self.qi[head][query][d] * self.ki[head][key][d];
                        }
                        (score * scale_index, key)
                    })
                    .collect();
                scores.sort_by(|left, right| {
                    right
                        .0
                        .total_cmp(&left.0)
                        .then_with(|| left.1.cmp(&right.1))
                });
                let selected: Vec<usize> = scores
                    .iter()
                    .take(config.top_k as usize)
                    .map(|(_, key)| *key)
                    .collect();

                let mut logits: Vec<(f64, usize)> = selected
                    .iter()
                    .map(|key| {
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += self.q[head][query][d] * self.k[head][*key][d];
                        }
                        (score * scale_head, *key)
                    })
                    .collect();
                let max = logits
                    .iter()
                    .map(|(score, _)| *score)
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut denominator = 0.0;
                for (score, _) in &mut logits {
                    *score = (*score - max).exp();
                    denominator += *score;
                }
                let mut out = vec![0.0; head_dim];
                for (weight, key) in &logits {
                    let weight = *weight / denominator;
                    for d in 0..head_dim {
                        out[d] += weight * self.v[head][*key][d];
                    }
                }
                attention[head][query] = out;
            }
        }

        let mut output = vec![vec![0.0; hidden]; seq];
        for query in 0..seq {
            for dst in 0..hidden {
                let mut sum = 0.0;
                for head in 0..heads {
                    for d in 0..head_dim {
                        sum += attention[head][query][d] * self.wo[dst][head * head_dim + d];
                    }
                }
                output[query][dst] = sum;
            }
        }
        output
    }
}

#[derive(Clone, Copy, Debug)]
enum Matrix {
    Q,
}

struct NodeRegions {
    q: RegionHandle,
}

struct DsaNode {
    cp: u32,
    head_start: u32,
    head_count: u32,
    seq_start: u32,
    seq_len: u32,
    /// `[local_head][local_seq][head_dim]`
    q: Vec<Vec<Vec<f64>>>,
    k: Vec<Vec<Vec<f64>>>,
    v: Vec<Vec<Vec<f64>>>,
    /// `[local_head][local_seq][index_dim]`
    qi: Vec<Vec<Vec<f64>>>,
    ki: Vec<Vec<Vec<f64>>>,
    /// `[hidden][local_heads * head_dim]`
    wo: Vec<Vec<f64>>,
    regions: NodeRegions,
}

impl DsaNode {
    fn local_head(&self, head: u32) -> Option<usize> {
        let offset = head.checked_sub(self.head_start)?;
        (offset < self.head_count).then_some(offset as usize)
    }

    fn head_dim(&self) -> u32 {
        // The global shape is uniform; infer from a local tensor.
        self.q
            .first()
            .and_then(|head| head.first())
            .map(|row| row.len() as u32)
            .unwrap_or(0)
    }

    fn index_dim(&self) -> u32 {
        self.qi
            .first()
            .and_then(|head| head.first())
            .map(|row| row.len() as u32)
            .unwrap_or(0)
    }
}

pub struct DsaOutput {
    pub values: Vec<Vec<f64>>,
    pub instances: usize,
}

pub struct DsaRunner {
    config: DsaConfig,
    topology: Topology,
    plan: DsaPlan,
    cluster: LocalCluster,
    plane: Arc<LoopbackRdma>,
    pool: BufferPool,
    nodes: BTreeMap<NodeId, DsaNode>,
    declared: BTreeSet<OpId>,
    gathered_q: BTreeMap<u32, Vec<Vec<f64>>>,
    gathered_qi: BTreeMap<u32, Vec<Vec<f64>>>,
    selected: BTreeMap<(u32, u32), Vec<u32>>,
    attention: BTreeMap<(u32, u32), Vec<f64>>,
}

impl DsaRunner {
    pub fn new(
        topology: Topology,
        config: DsaConfig,
        global: &GlobalDsa,
    ) -> Result<Self, DsaError> {
        config.validate()?;
        let sizes = topology.sizes();
        if sizes.data != 1 || sizes.pipeline != 1 || sizes.expert != 1 {
            return Err(DsaError::Config(
                "the reference DSA runner executes one DP/PP/EP coordinate; use the planner for larger axes"
                    .into(),
            ));
        }
        if sizes.sequence != 1 {
            return Err(DsaError::Config(
                "DSA execution aliases SP to the CP shard group; set sequence=1".into(),
            ));
        }
        if sizes.tensor > config.heads {
            return Err(DsaError::Config("TP size exceeds head count".into()));
        }
        let plan = dsa_plan(&topology, &config);
        let cluster = LocalCluster::new(topology.nodes().iter().copied());
        let plane = Arc::new(LoopbackRdma::new());
        let pool = BufferPool::with_limits(4096, 64);
        let mut nodes = BTreeMap::new();
        for node in topology.nodes() {
            let coord = topology
                .coord(*node)
                .ok_or_else(|| DsaError::Config("node missing from topology".into()))?;
            let head_range = crate::plan::head_range(coord.tensor, &config, &topology);
            let local_seq = seq_range(coord.context, &config, &topology);
            let q: Vec<Vec<Vec<f64>>> = head_range
                .clone()
                .map(|head| {
                    local_seq
                        .clone()
                        .map(|query| global.q[head as usize][query as usize].clone())
                        .collect()
                })
                .collect();
            let k: Vec<Vec<Vec<f64>>> = head_range
                .clone()
                .map(|head| {
                    local_seq
                        .clone()
                        .map(|key| global.k[head as usize][key as usize].clone())
                        .collect()
                })
                .collect();
            let v: Vec<Vec<Vec<f64>>> = head_range
                .clone()
                .map(|head| {
                    local_seq
                        .clone()
                        .map(|key| global.v[head as usize][key as usize].clone())
                        .collect()
                })
                .collect();
            let qi: Vec<Vec<Vec<f64>>> = head_range
                .clone()
                .map(|head| {
                    local_seq
                        .clone()
                        .map(|query| global.qi[head as usize][query as usize].clone())
                        .collect()
                })
                .collect();
            let ki: Vec<Vec<Vec<f64>>> = head_range
                .clone()
                .map(|head| {
                    local_seq
                        .clone()
                        .map(|key| global.ki[head as usize][key as usize].clone())
                        .collect()
                })
                .collect();
            let wo: Vec<Vec<f64>> = (0..config.hidden as usize)
                .map(|dst| {
                    let mut row = Vec::with_capacity(head_range.len() * config.head_dim as usize);
                    for head in head_range.clone() {
                        let src = head as usize * config.head_dim as usize;
                        row.extend_from_slice(&global.wo[dst][src..src + config.head_dim as usize]);
                    }
                    row
                })
                .collect();
            let seq_len = local_seq.len() as u32;
            let regions = NodeRegions {
                q: register_tensor(&plane, &pool, &flatten(&q))?,
            };
            nodes.insert(
                *node,
                DsaNode {
                    cp: coord.context,
                    head_start: head_range.start,
                    head_count: head_range.len() as u32,
                    seq_start: local_seq.start,
                    seq_len,
                    q,
                    k,
                    v,
                    qi,
                    ki,
                    wo,
                    regions,
                },
            );
        }
        Ok(Self {
            config,
            topology,
            plan,
            cluster,
            plane,
            pool,
            nodes,
            declared: BTreeSet::new(),
            gathered_q: BTreeMap::new(),
            gathered_qi: BTreeMap::new(),
            selected: BTreeMap::new(),
            attention: BTreeMap::new(),
        })
    }

    pub fn plan(&self) -> &DsaPlan {
        &self.plan
    }

    pub fn buffer_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    pub fn run(mut self) -> Result<DsaOutput, DsaError> {
        self.gather_q_and_index()?;
        self.select_candidates()?;
        self.attention()?;
        let values = self.output_projection()?;
        let instances = self
            .topology
            .nodes()
            .first()
            .map(|node| self.cluster.executed(*node))
            .unwrap_or(0);
        let violations = self.cluster.violations();
        if !violations.is_empty() {
            return Err(DsaError::Invariant(format!(
                "replica invariant violations: {violations:?}"
            )));
        }
        Ok(DsaOutput { values, instances })
    }

    fn gather_q_and_index(&mut self) -> Result<(), DsaError> {
        for head in 0..self.config.heads {
            let tp = crate::plan::tp_for_head(head, &self.config, &self.topology);
            let base = crate::topology::Coord {
                tensor: tp,
                ..Default::default()
            };
            let cp_participants = self.topology.group(Axis::Context, base);
            let op_id = *self
                .plan
                .q_index
                .get(&head)
                .ok_or_else(|| DsaError::Plan(format!("no q_index op for head {head}")))?;
            let op = self
                .plan
                .plan
                .ops
                .iter()
                .find(|op| op.op_id == op_id)
                .ok_or_else(|| DsaError::Plan("q_index op missing".into()))?
                .clone();
            let mut contributions = BTreeMap::new();
            for node_id in &cp_participants {
                let node = self
                    .nodes
                    .get(node_id)
                    .ok_or_else(|| DsaError::Plan("participant missing".into()))?;
                let local_head = node
                    .local_head(head)
                    .ok_or_else(|| DsaError::Plan("head not owned by participant".into()))?;
                contributions.insert(
                    *node_id,
                    flatten_rows(&node.qi[local_head], node.index_dim() as usize),
                );
            }
            let gathered = self.run_op(&op, contributions)?;
            let mut global =
                vec![vec![0.0; self.config.index_dim as usize]; self.config.seq as usize];
            let contributors: BTreeSet<NodeId> = op.participants.iter().copied().collect();
            let mut offset = 0usize;
            for node_id in contributors {
                let node = self
                    .nodes
                    .get(&node_id)
                    .ok_or_else(|| DsaError::Plan("participant missing".into()))?;
                let chunk =
                    &gathered[offset..offset + node.seq_len as usize * node.index_dim() as usize];
                for local in 0..node.seq_len as usize {
                    let query = node.seq_start as usize + local;
                    let start = local * node.index_dim() as usize;
                    global[query].copy_from_slice(&chunk[start..start + node.index_dim() as usize]);
                }
                offset += node.seq_len as usize * node.index_dim() as usize;
            }
            if offset != gathered.len() {
                return Err(DsaError::Plan("q_index gather length mismatch".into()));
            }
            self.gathered_qi.insert(head, global);

            let mut global_q =
                vec![vec![0.0; self.config.head_dim as usize]; self.config.seq as usize];
            for node_id in cp_participants {
                let node = self
                    .nodes
                    .get(&node_id)
                    .ok_or_else(|| DsaError::Plan("participant missing".into()))?;
                let local_head = node
                    .local_head(head)
                    .ok_or_else(|| DsaError::Plan("head not owned by participant".into()))?;
                let rows = self.read_matrix(&node_id, Matrix::Q, local_head)?;
                for local in 0..node.seq_len as usize {
                    let query = node.seq_start as usize + local;
                    let start = local * self.config.head_dim as usize;
                    global_q[query]
                        .copy_from_slice(&rows[start..start + self.config.head_dim as usize]);
                }
            }
            self.gathered_q.insert(head, global_q);
        }
        Ok(())
    }

    fn select_candidates(&mut self) -> Result<(), DsaError> {
        for head in 0..self.config.heads {
            for query in 0..self.config.seq {
                let op_id = *self
                    .plan
                    .candidates
                    .get(&(head, query))
                    .ok_or_else(|| DsaError::Plan("candidate op missing".into()))?;
                let op = self
                    .plan
                    .plan
                    .ops
                    .iter()
                    .find(|op| op.op_id == op_id)
                    .ok_or_else(|| DsaError::Plan("candidate op missing".into()))?
                    .clone();
                let q_index = self
                    .gathered_qi
                    .get(&head)
                    .ok_or_else(|| DsaError::Plan("q_index not gathered".into()))?;
                let mut contributions = BTreeMap::new();
                for node_id in &op.participants {
                    let node = self
                        .nodes
                        .get(node_id)
                        .ok_or_else(|| DsaError::Plan("participant missing".into()))?;
                    let local_head = node
                        .local_head(head)
                        .ok_or_else(|| DsaError::Plan("head not owned by participant".into()))?;
                    let mut candidates: Vec<(f64, u32)> = Vec::new();
                    let end = (query + 1).min(node.seq_start + node.seq_len);
                    if end > node.seq_start {
                        let q = &q_index[query as usize];
                        let scale = 1.0 / (self.config.index_dim as f64).sqrt();
                        for key in node.seq_start..end {
                            let local = (key - node.seq_start) as usize;
                            let k = &node.ki[local_head][local];
                            let mut score = 0.0;
                            for d in 0..k.len() {
                                score += q[d] * k[d];
                            }
                            candidates.push((score * scale, key));
                        }
                    }
                    candidates.sort_by(|left, right| {
                        right
                            .0
                            .total_cmp(&left.0)
                            .then_with(|| left.1.cmp(&right.1))
                    });
                    candidates.truncate(self.config.top_k as usize);
                    let mut values = Vec::with_capacity(candidates.len() * 2);
                    for (score, key) in candidates {
                        values.push(score);
                        values.push(key as f64);
                    }
                    contributions.insert(*node_id, values);
                }
                let selected = self.run_op(&op, contributions)?;
                let selected: Vec<u32> = selected.into_iter().map(|value| value as u32).collect();
                self.selected.insert((head, query), selected);
            }
        }
        Ok(())
    }

    fn attention(&mut self) -> Result<(), DsaError> {
        for head in 0..self.config.heads {
            for query in 0..self.config.seq {
                let op_id = *self
                    .plan
                    .attention
                    .get(&(head, query))
                    .ok_or_else(|| DsaError::Plan("attention op missing".into()))?;
                let op = self
                    .plan
                    .plan
                    .ops
                    .iter()
                    .find(|op| op.op_id == op_id)
                    .ok_or_else(|| DsaError::Plan("attention op missing".into()))?
                    .clone();
                let gathered_q = self
                    .gathered_q
                    .get(&head)
                    .ok_or_else(|| DsaError::Plan("q not gathered".into()))?;
                let selected = self
                    .selected
                    .get(&(head, query))
                    .ok_or_else(|| DsaError::Plan("selection missing".into()))?;
                let scale = 1.0 / (self.config.head_dim as f64).sqrt();
                let mut contributions = BTreeMap::new();
                for node_id in &op.participants {
                    let node = self
                        .nodes
                        .get(node_id)
                        .ok_or_else(|| DsaError::Plan("participant missing".into()))?;
                    let local_head = node
                        .local_head(head)
                        .ok_or_else(|| DsaError::Plan("head not owned by participant".into()))?;
                    let mut local_keys: Vec<u32> = selected
                        .iter()
                        .copied()
                        .filter(|key| {
                            *key >= node.seq_start && *key < node.seq_start + node.seq_len
                        })
                        .collect();
                    local_keys.sort_unstable();
                    let q = &gathered_q[query as usize];
                    let mut logits: Vec<(f64, u32)> = local_keys
                        .iter()
                        .map(|key| {
                            let k = &node.k[local_head][(key - node.seq_start) as usize];
                            let mut score = 0.0;
                            for d in 0..k.len() {
                                score += q[d] * k[d];
                            }
                            (score * scale, *key)
                        })
                        .collect();
                    let max = logits
                        .iter()
                        .map(|(score, _)| *score)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0.0;
                    let mut acc = vec![0.0; self.config.head_dim as usize];
                    for (score, key) in &mut logits {
                        let weight = (*score - max).exp();
                        *score = weight;
                        sum += weight;
                        let v = &node.v[local_head][(*key - node.seq_start) as usize];
                        for d in 0..acc.len() {
                            acc[d] += weight * v[d];
                        }
                    }
                    let mut values = Vec::with_capacity(2 + acc.len());
                    values.push(max);
                    values.push(sum);
                    values.extend(acc);
                    contributions.insert(*node_id, values);
                }
                let combined = self.run_op(&op, contributions)?;
                let sum = combined[1];
                let mut output = vec![0.0; self.config.head_dim as usize];
                if sum > 0.0 {
                    for d in 0..output.len() {
                        output[d] = combined[2 + d] / sum;
                    }
                }
                self.attention.insert((head, query), output);
            }
        }
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn output_projection(&mut self) -> Result<Vec<Vec<f64>>, DsaError> {
        let mut output = vec![vec![0.0; self.config.hidden as usize]; self.config.seq as usize];
        for cp in 0..self.topology.sizes().context {
            let op_id = *self
                .plan
                .output
                .get(&cp)
                .ok_or_else(|| DsaError::Plan("output op missing".into()))?;
            let op = self
                .plan
                .plan
                .ops
                .iter()
                .find(|op| op.op_id == op_id)
                .ok_or_else(|| DsaError::Plan("output op missing".into()))?
                .clone();
            let mut contributions = BTreeMap::new();
            for node_id in &op.participants {
                let node = self
                    .nodes
                    .get(node_id)
                    .ok_or_else(|| DsaError::Plan("participant missing".into()))?;
                let mut values = vec![0.0; node.seq_len as usize * self.config.hidden as usize];
                for local in 0..node.seq_len as usize {
                    let query = node.seq_start as usize + local;
                    for dst in 0..self.config.hidden as usize {
                        let mut sum = 0.0;
                        for local_head in 0..node.head_count as usize {
                            let head = node.head_start + local_head as u32;
                            let attn = self
                                .attention
                                .get(&(head, query as u32))
                                .ok_or_else(|| DsaError::Plan("attention missing".into()))?;
                            for d in 0..self.config.head_dim as usize {
                                sum += attn[d]
                                    * node.wo[dst][local_head * self.config.head_dim as usize + d];
                            }
                        }
                        values[local * self.config.hidden as usize + dst] = sum;
                    }
                }
                contributions.insert(*node_id, values);
            }
            let projected = self.run_op(&op, contributions)?;
            let seq_start = self
                .nodes
                .values()
                .find(|node| node.cp == cp)
                .map(|node| node.seq_start)
                .ok_or_else(|| DsaError::Plan("cp coordinate missing".into()))?;
            let seq_len = self
                .nodes
                .values()
                .find(|node| node.cp == cp)
                .map(|node| node.seq_len)
                .ok_or_else(|| DsaError::Plan("cp coordinate missing".into()))?;
            for local in 0..seq_len as usize {
                let query = seq_start as usize + local;
                let start = local * self.config.hidden as usize;
                output[query]
                    .copy_from_slice(&projected[start..start + self.config.hidden as usize]);
            }
        }
        Ok(output)
    }

    fn run_op(
        &mut self,
        op: &ParallelOp,
        contributions: BTreeMap<NodeId, Vec<f64>>,
    ) -> Result<Vec<f64>, DsaError> {
        let coordinator = *op
            .participants
            .iter()
            .min()
            .ok_or_else(|| DsaError::Plan("operation has no participants".into()))?;
        if self.declared.insert(op.op_id) {
            let kind = fuse_kind(op.sync)
                .ok_or_else(|| DsaError::Plan("operation is not a fused reduction".into()))?;
            let spec = OpSpec {
                id: op.op_id,
                kind,
                contributors: op.participants.iter().copied().collect(),
                tensor: TensorKey::new(tensor_id(op.op_id), ShardId(0)),
                epoch: 0,
            };
            self.cluster
                .propose(coordinator, Command::DeclareOp(spec))
                .map_err(DsaError::Consensus)?;
            self.cluster.run();
        }
        for (node, values) in contributions {
            self.cluster
                .propose(
                    node,
                    Command::Contribute {
                        op: op.op_id,
                        contributor: node,
                        values,
                    },
                )
                .map_err(DsaError::Consensus)?;
        }
        self.cluster.run();
        self.cluster
            .result(coordinator, op.op_id)
            .ok_or_else(|| DsaError::Plan(format!("no fused result for {}", op.name)))
    }

    fn read_matrix(
        &self,
        node_id: &NodeId,
        matrix: Matrix,
        local_head: usize,
    ) -> Result<Vec<f64>, DsaError> {
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| DsaError::Plan("peer missing".into()))?;
        let (region, dim) = match matrix {
            Matrix::Q => (&node.regions.q, node.head_dim()),
        };
        let len = node.seq_len as u64 * dim as u64 * 8;
        let offset = local_head as u64 * node.seq_len as u64 * dim as u64 * 8;
        let descriptor = region.descriptor(*node_id, offset, len)?;
        let mut bytes = vec![0u8; len as usize];
        self.plane.read(&descriptor, &mut bytes)?;
        Ok(unpack(&bytes))
    }
}

fn tensor_id(op: OpId) -> TensorId {
    TensorId(u128::from_be_bytes(*op.origin.as_bytes()) ^ u128::from(op.counter))
}

fn register_tensor(
    plane: &LoopbackRdma,
    pool: &BufferPool,
    values: &[f64],
) -> Result<RegionHandle, DsaError> {
    let bytes = pack(values);
    let mut buffer = pool.acquire(bytes.len().max(8))?;
    buffer[..bytes.len()].copy_from_slice(&bytes);
    plane
        .register(buffer, MemoryKind::Host)
        .map_err(DsaError::Io)
}

fn flatten(heads: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let mut out = Vec::new();
    for head in heads {
        for row in head {
            out.extend_from_slice(row);
        }
    }
    out
}

fn flatten_rows(rows: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * dim);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn pack(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn unpack(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
        .collect()
}

#[derive(Debug)]
pub enum DsaError {
    Config(String),
    Plan(String),
    Consensus(ProposeError),
    Invariant(String),
    Topology(TopologyError),
    Io(std::io::Error),
}

impl std::fmt::Display for DsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(message) => write!(f, "configuration: {message}"),
            Self::Plan(message) => write!(f, "plan: {message}"),
            Self::Consensus(error) => write!(f, "consensus: {error}"),
            Self::Invariant(message) => write!(f, "invariant: {message}"),
            Self::Topology(error) => write!(f, "topology: {error}"),
            Self::Io(error) => write!(f, "io: {error}"),
        }
    }
}

impl std::error::Error for DsaError {}

impl From<std::io::Error> for DsaError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{AxisSizes, Topology};

    fn compare(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (row, (left, right)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(left.len(), right.len());
            for (column, (a, b)) in left.iter().zip(right).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "mismatch at [{row}][{column}]: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn context_parallel_dsa_matches_reference() {
        let config = DsaConfig::small();
        let global = GlobalDsa::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(1, 1, 1, 1, 3, 1), "dsa-cp").unwrap();
        let runner = DsaRunner::new(topology, config.clone(), &global).unwrap();
        let output = runner.run().unwrap();
        let expected = global.reference(&config);
        compare(&output.values, &expected);
        assert!(output.instances > 0);
    }

    #[test]
    fn tensor_parallel_dsa_matches_reference() {
        let config = DsaConfig::small();
        let global = GlobalDsa::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(1, 1, 2, 1, 1, 1), "dsa-tp").unwrap();
        let runner = DsaRunner::new(topology, config.clone(), &global).unwrap();
        let output = runner.run().unwrap();
        let expected = global.reference(&config);
        compare(&output.values, &expected);
    }

    #[test]
    fn rejects_execution_for_unimplemented_axes() {
        let config = DsaConfig::small();
        let global = GlobalDsa::new(&config);
        let topology = Topology::with_prefix(AxisSizes::new(2, 1, 1, 1, 1, 1), "dsa-dp").unwrap();
        assert!(DsaRunner::new(topology, config, &global).is_err());
    }
}
