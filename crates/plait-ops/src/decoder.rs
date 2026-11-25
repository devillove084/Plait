//! Causal decoder layers with pluggable linear projections and portable KV.

use half::f16;

use crate::OperatorError;

mod model;
pub use model::TokenHead;

#[derive(Clone, Copy, Debug)]
pub struct DecoderConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
}

impl DecoderConfig {
    pub fn projection_shapes(self) -> Result<[(u32, u32); 7], OperatorError> {
        let h = self.hidden_size;
        let q = self.attention_heads;
        let kv = self.kv_heads;
        let ffn = self.intermediate_size;
        if h == 0
            || q == 0
            || kv == 0
            || ffn == 0
            || h % q != 0
            || (h / q) % 2 != 0
            || q % kv != 0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_epsilon.is_finite()
            || self.rms_epsilon <= 0.0
        {
            return Err(OperatorError::Shape(
                "invalid dense decoder dimensions or constants",
            ));
        }
        let kv_width = (h / q) * kv;
        // Q, K, V, output, gate, up, down. Shapes are (input, output).
        Ok([
            (h, h),
            (h, kv_width),
            (h, kv_width),
            (h, h),
            (h, ffn),
            (h, ffn),
            (ffn, h),
        ])
    }
}

pub trait LinearProjection {
    fn shape(&self) -> (u32, u32);
    fn forward(&self, input_f16: &[u16]) -> Result<Vec<f32>, OperatorError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvSnapshot {
    pub layer: [u8; 32],
    /// Token-major rows of rotary-transformed keys, stored as binary16 bits.
    pub keys_f16: Vec<u16>,
    /// Token-major rows of values, stored as binary16 bits.
    pub values_f16: Vec<u16>,
}

pub struct DecoderKv {
    state: KvSnapshot,
    width: usize,
    max_tokens: u32,
}

impl DecoderKv {
    pub fn snapshot(&self) -> &KvSnapshot {
        &self.state
    }
}

pub struct DecoderBlock<P> {
    layer: [u8; 32],
    config: DecoderConfig,
    kv_width: usize,
    rope: Vec<f32>,
    attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    projections: [P; 7],
}

pub struct DecoderStack<P> {
    blocks: Vec<DecoderBlock<P>>,
}

pub struct DecoderStackKv {
    layers: Vec<DecoderKv>,
    tokens: usize,
}

impl DecoderStackKv {
    pub fn snapshots(&self) -> impl ExactSizeIterator<Item = &KvSnapshot> {
        self.layers.iter().map(DecoderKv::snapshot)
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    fn truncate(&mut self, tokens: usize) {
        for layer in &mut self.layers {
            let len = tokens * layer.width;
            layer.state.keys_f16.truncate(len);
            layer.state.values_f16.truncate(len);
        }
        self.tokens = tokens;
    }
}

impl<P: LinearProjection> DecoderStack<P> {
    pub fn new(blocks: Vec<DecoderBlock<P>>) -> Result<Self, OperatorError> {
        let first = blocks
            .first()
            .ok_or(OperatorError::Shape("decoder stack needs a layer"))?;
        if blocks
            .iter()
            .any(|block| block.config.hidden_size != first.config.hidden_size)
        {
            return Err(OperatorError::Shape(
                "decoder stack layers need a common hidden size",
            ));
        }
        Ok(Self { blocks })
    }

    pub fn empty_cache(&self, max_tokens: u32) -> Result<DecoderStackKv, OperatorError> {
        let layers = self
            .blocks
            .iter()
            .map(|block| block.empty_cache(max_tokens))
            .collect::<Result<_, _>>()?;
        Ok(DecoderStackKv { layers, tokens: 0 })
    }

    pub fn restore_cache(
        &self,
        snapshots: Vec<KvSnapshot>,
        max_tokens: u32,
    ) -> Result<DecoderStackKv, OperatorError> {
        if snapshots.len() != self.blocks.len() {
            return Err(OperatorError::Shape(
                "KV snapshot has a different decoder layer count",
            ));
        }
        let layers = self
            .blocks
            .iter()
            .zip(snapshots)
            .map(|(block, snapshot)| block.restore_cache(snapshot, max_tokens))
            .collect::<Result<Vec<_>, _>>()?;
        let tokens = layers[0].state.keys_f16.len() / layers[0].width;
        if layers
            .iter()
            .any(|layer| layer.state.keys_f16.len() / layer.width != tokens)
        {
            return Err(OperatorError::Shape(
                "decoder KV layers have different token counts",
            ));
        }
        Ok(DecoderStackKv { layers, tokens })
    }

    /// Commit a token batch for the entire stack. Only the just-appended rows
    /// need to be rolled back when a later layer fails.
    pub fn forward(
        &self,
        hidden: &[f32],
        cache: &mut DecoderStackKv,
    ) -> Result<Vec<f32>, OperatorError> {
        if self.blocks.len() != cache.layers.len() {
            return Err(OperatorError::Shape(
                "KV cache has a different decoder layer count",
            ));
        }
        let tokens = cache.tokens;
        let mut output = Vec::new();
        for (index, (block, layer)) in self.blocks.iter().zip(&mut cache.layers).enumerate() {
            let input = if index == 0 { hidden } else { &output };
            match block.forward(input, layer) {
                Ok(next) => output = next,
                Err(error) => {
                    cache.truncate(tokens);
                    return Err(error);
                }
            }
        }
        cache.tokens += hidden.len() / self.blocks[0].config.hidden_size as usize;
        Ok(output)
    }
}

impl<P: LinearProjection> DecoderBlock<P> {
    pub fn new(
        layer: [u8; 32],
        config: DecoderConfig,
        attention_norm_f16: &[u16],
        ffn_norm_f16: &[u16],
        projections: [P; 7],
    ) -> Result<Self, OperatorError> {
        let shapes = config.projection_shapes()?;
        if projections
            .iter()
            .zip(shapes)
            .any(|(p, shape)| p.shape() != shape)
        {
            return Err(OperatorError::Shape(
                "decoder projection shapes do not match the layer",
            ));
        }
        let norm = |weights: &[u16]| {
            if weights.len() != config.hidden_size as usize || !finite_half(weights) {
                return Err(OperatorError::Shape(
                    "decoder normalization needs finite weights for each hidden channel",
                ));
            }
            Ok(weights.iter().map(|&v| decode(v)).collect())
        };
        let head_dim = config.hidden_size / config.attention_heads;
        let rope = (0..head_dim / 2)
            .map(|i| config.rope_theta.powf(-2.0 * i as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            layer,
            config,
            kv_width: shapes[1].1 as usize,
            rope,
            attention_norm: norm(attention_norm_f16)?,
            ffn_norm: norm(ffn_norm_f16)?,
            projections,
        })
    }

    pub fn empty_cache(&self, max_tokens: u32) -> Result<DecoderKv, OperatorError> {
        self.restore_cache(
            KvSnapshot {
                layer: self.layer,
                keys_f16: Vec::new(),
                values_f16: Vec::new(),
            },
            max_tokens,
        )
    }

    pub fn replace_projection(&mut self, index: usize, projection: P) -> Result<P, OperatorError> {
        let current = self.projections.get_mut(index).ok_or(OperatorError::Shape(
            "decoder projection index is out of range",
        ))?;
        if current.shape() != projection.shape() {
            return Err(OperatorError::Shape(
                "replacement decoder projection has a different shape",
            ));
        }
        Ok(std::mem::replace(current, projection))
    }

    pub fn restore_cache(
        &self,
        state: KvSnapshot,
        max_tokens: u32,
    ) -> Result<DecoderKv, OperatorError> {
        let len = state.keys_f16.len();
        if max_tokens == 0
            || state.layer != self.layer
            || len != state.values_f16.len()
            || len % self.kv_width != 0
            || len / self.kv_width > max_tokens as usize
            || !finite_half(&state.keys_f16)
            || !finite_half(&state.values_f16)
        {
            return Err(OperatorError::Shape(
                "KV snapshot does not fit the decoder layer or capacity",
            ));
        }
        Ok(DecoderKv {
            state,
            width: self.kv_width,
            max_tokens,
        })
    }

    /// Append a prompt chunk or decode tokens. A failed forward leaves the KV
    /// contents unchanged.
    pub fn forward(
        &self,
        hidden: &[f32],
        cache: &mut DecoderKv,
    ) -> Result<Vec<f32>, OperatorError> {
        let width = self.config.hidden_size as usize;
        let rows = hidden.len() / width;
        let past = cache.state.keys_f16.len() / self.kv_width;
        if cache.state.layer != self.layer
            || cache.width != self.kv_width
            || rows == 0
            || hidden.len() % width != 0
            || past > cache.max_tokens as usize
            || rows > cache.max_tokens as usize - past
            || hidden.iter().any(|v| !v.is_finite())
        {
            return Err(OperatorError::Shape(
                "decoder input or KV capacity is invalid",
            ));
        }
        let [q, k, v, o, gate, up, down] = &self.projections;
        let input = rms_norm(hidden, &self.attention_norm, self.config.rms_epsilon)?;
        let mut query = project(q, &input, rows)?;
        let mut keys = project(k, &input, rows)?;
        let values = encode(project(v, &input, rows)?)?;
        self.rotate(&mut query, &mut keys, past);
        let keys = encode(keys)?;
        let attention = encode(self.attend(&query, cache.snapshot(), &keys, &values))?;
        let mut output = project(o, &attention, rows)?;
        for (value, &input) in output.iter_mut().zip(hidden) {
            *value += input;
        }
        let residual = rms_norm(&output, &self.ffn_norm, self.config.rms_epsilon)?;
        let mut gated = project(gate, &residual, rows)?;
        for (value, up) in gated.iter_mut().zip(project(up, &residual, rows)?) {
            *value = (*value / (1.0 + (-*value).exp())) * up;
        }
        for (value, delta) in output.iter_mut().zip(project(down, &encode(gated)?, rows)?) {
            *value += delta;
        }
        checked_decoder_result(&output)?;
        cache
            .state
            .keys_f16
            .try_reserve(keys.len())
            .map_err(|error| OperatorError::Backend(error.to_string()))?;
        cache
            .state
            .values_f16
            .try_reserve(values.len())
            .map_err(|error| OperatorError::Backend(error.to_string()))?;
        cache.state.keys_f16.extend(keys);
        cache.state.values_f16.extend(values);
        Ok(output)
    }

    fn rotate(&self, q: &mut [f32], k: &mut [f32], past: usize) {
        let dim = self.rope.len() * 2;
        let mut angles = vec![(0.0, 0.0); self.rope.len()];
        for (row, (q, k)) in q
            .chunks_exact_mut(self.config.hidden_size as usize)
            .zip(k.chunks_exact_mut(self.kv_width))
            .enumerate()
        {
            for (angle, frequency) in angles.iter_mut().zip(&self.rope) {
                *angle = (((past + row) as f32) * frequency).sin_cos();
            }
            for head in q.chunks_exact_mut(dim).chain(k.chunks_exact_mut(dim)) {
                let (first, second) = head.split_at_mut(dim / 2);
                for ((a, b), &(sin, cos)) in first.iter_mut().zip(second).zip(&angles) {
                    (*a, *b) = (*a * cos - *b * sin, *b * cos + *a * sin);
                }
            }
        }
    }

    fn attend(&self, query: &[f32], cache: &KvSnapshot, keys: &[u16], values: &[u16]) -> Vec<f32> {
        let h = self.config.hidden_size as usize;
        let dim = self.rope.len() * 2;
        let query_per_kv = (self.config.attention_heads / self.config.kv_heads) as usize;
        let past = cache.keys_f16.len() / self.kv_width;
        let scale = (dim as f32).sqrt().recip();
        let mut output = vec![0.0; query.len()];
        let kv = cache
            .keys_f16
            .chunks_exact(self.kv_width)
            .chain(keys.chunks_exact(self.kv_width))
            .zip(
                cache
                    .values_f16
                    .chunks_exact(self.kv_width)
                    .chain(values.chunks_exact(self.kv_width)),
            );
        for (row, (query, out)) in query
            .chunks_exact(h)
            .zip(output.chunks_exact_mut(h))
            .enumerate()
        {
            for (head, (q, out)) in query
                .chunks_exact(dim)
                .zip(out.chunks_exact_mut(dim))
                .enumerate()
            {
                let start = (head / query_per_kv) * dim;
                let mut max = f32::NEG_INFINITY;
                let mut denominator = 0.0f32;
                for (k, v) in kv.clone().take(past + row + 1) {
                    let score = q
                        .iter()
                        .zip(&k[start..start + dim])
                        .fold(0.0, |sum, (&q, &k)| q.mul_add(decode(k), sum))
                        * scale;
                    let next_max = max.max(score);
                    let old = (max - next_max).exp();
                    let new = (score - next_max).exp();
                    denominator = denominator.mul_add(old, new);
                    max = next_max;
                    for (acc, &v) in out.iter_mut().zip(&v[start..start + dim]) {
                        *acc = new.mul_add(decode(v), old * *acc);
                    }
                }
                for acc in out {
                    *acc /= denominator;
                }
            }
        }
        output
    }
}

fn project(
    layer: &impl LinearProjection,
    input: &[u16],
    rows: usize,
) -> Result<Vec<f32>, OperatorError> {
    let values = layer.forward(input)?;
    if rows.checked_mul(layer.shape().1 as usize) != Some(values.len()) {
        return Err(OperatorError::Shape(
            "wrong number of values returned by decoder projection",
        ));
    }
    checked_decoder_result(&values)?;
    Ok(values)
}

fn checked_decoder_result(values: &[f32]) -> Result<(), OperatorError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(OperatorError::Shape("decoder produced a nonfinite value"));
    }
    Ok(())
}

fn rms_norm(input: &[f32], scale: &[f32], epsilon: f32) -> Result<Vec<u16>, OperatorError> {
    encode(input.chunks_exact(scale.len()).flat_map(|row| {
        let mean = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / row.len() as f64;
        let inverse = (mean + f64::from(epsilon)).sqrt().recip() as f32;
        row.iter()
            .zip(scale)
            .map(move |(&x, &weight)| (x * inverse) * weight)
    }))
}

fn decode(value: u16) -> f32 {
    f16::from_bits(value).to_f32()
}

fn finite_half(values: &[u16]) -> bool {
    values.iter().all(|&v| f16::from_bits(v).is_finite())
}

fn encode(values: impl IntoIterator<Item = f32>) -> Result<Vec<u16>, OperatorError> {
    values
        .into_iter()
        .map(|v| {
            let half = f16::from_f32(v);
            if !half.is_finite() {
                return Err(OperatorError::Shape(
                    "decoder intermediate cannot be represented as finite FP16",
                ));
            }
            Ok(half.to_bits())
        })
        .collect()
}
