use std::io;
use std::path::PathBuf;

use half::f16;
use plait_ops::decoder::DecoderConfig;
use serde::Deserialize;
use serde_json::Value;

use super::{LayerInput, ProjectionExecution, ProjectionInput, W4WeightInput, model};

mod safetensors;
use safetensors::{Safetensors, Tensor, invalid, read_json};

#[derive(Deserialize)]
pub(super) struct Config {
    path: PathBuf,
    #[serde(flatten)]
    execution: ProjectionExecution,
    head: Option<ProjectionExecution>,
}

#[derive(Deserialize)]
struct Qwen2 {
    model_type: String,
    hidden_act: String,
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    head_dim: Option<u32>,
    num_key_value_heads: u32,
    num_hidden_layers: u32,
    max_position_embeddings: u32,
    vocab_size: u32,
    tie_word_embeddings: bool,
    rope_theta: f32,
    rms_norm_eps: f32,
    #[serde(default)]
    use_sliding_window: bool,
    layer_types: Option<Vec<String>>,
    rope_scaling: Option<Value>,
    rope_parameters: Option<Value>,
    quantization_config: Option<Value>,
}

pub(super) struct Checkpoint {
    tensors: Safetensors,
    config: Qwen2,
    layers: ProjectionExecution,
    head: ProjectionExecution,
    group_size: u32,
}

impl Checkpoint {
    pub(super) fn open(source: Config, group_size: u32, max_tokens: u32) -> io::Result<Self> {
        let config: Qwen2 = read_json(&source.path.join("config.json"), 1 << 20)?;
        if config.model_type != "qwen2"
            || config.hidden_act != "silu"
            || config.use_sliding_window
            || config.layer_types.as_ref().is_some_and(|layers| {
                layers.len() != config.num_hidden_layers as usize
                    || layers.iter().any(|layer| layer != "full_attention")
            })
            || config.rope_scaling.is_some()
            || config.rope_parameters.is_some()
            || config.head_dim.is_some_and(|head_dim| {
                config.hidden_size.checked_div(config.num_attention_heads) != Some(head_dim)
            })
            || config.quantization_config.is_some()
        {
            return Err(invalid(
                "checkpoint must be floating point Qwen2 with full attention and unscaled RoPE",
            ));
        }
        if group_size == 0
            || config.num_hidden_layers == 0
            || config.vocab_size == 0
            || max_tokens == 0
            || max_tokens > config.max_position_embeddings
        {
            return Err(invalid(
                "invalid checkpoint layer, group, vocabulary or context limit",
            ));
        }
        Ok(Self {
            tensors: Safetensors::open(&source.path)?,
            config,
            layers: source.execution,
            head: source.head.unwrap_or(source.execution),
            group_size,
        })
    }

    pub(super) fn decoder_config(&self) -> DecoderConfig {
        DecoderConfig {
            hidden_size: self.config.hidden_size,
            intermediate_size: self.config.intermediate_size,
            attention_heads: self.config.num_attention_heads,
            kv_heads: self.config.num_key_value_heads,
            rope_theta: self.config.rope_theta,
            rms_epsilon: self.config.rms_norm_eps,
        }
    }

    pub(super) fn layer_count(&self) -> u32 {
        self.config.num_hidden_layers
    }

    pub(super) fn head_execution(&self) -> ProjectionExecution {
        self.head
    }

    pub(super) fn load_head(&mut self) -> io::Result<model::Weights> {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let shape = [vocab, h];
        let (embeddings, weights) = if self.config.tie_word_embeddings {
            let embeddings = self.tensors.read("model.embed_tokens.weight", &shape)?;
            let weights = quantize(&embeddings, h, vocab, self.group_size)?;
            (into_f16(embeddings)?, weights)
        } else {
            let weights = quantize(
                &self.tensors.read("lm_head.weight", &shape)?,
                h,
                vocab,
                self.group_size,
            )?;
            let embeddings = into_f16(self.tensors.read("model.embed_tokens.weight", &shape)?)?;
            (embeddings, weights)
        };
        let norm = self.read_f16("model.norm.weight", h)?;
        Ok(model::Weights {
            vocab_size: vocab,
            embeddings,
            norm,
            weights,
            execution: self.head,
        })
    }

    pub(super) fn load_layer(
        &mut self,
        position: u32,
        shapes: [(u32, u32); 7],
    ) -> io::Result<LayerInput> {
        let prefix = format!("model.layers.{position}");
        let h = self.config.hidden_size;
        let attention_norm_f16 = self.read_f16(&format!("{prefix}.input_layernorm.weight"), h)?;
        let ffn_norm_f16 =
            self.read_f16(&format!("{prefix}.post_attention_layernorm.weight"), h)?;
        let names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ];
        let mut projections = Vec::with_capacity(names.len());
        for (index, (name, (k, n))) in names.into_iter().zip(shapes).enumerate() {
            let weight = self
                .tensors
                .read(&format!("{prefix}.{name}.weight"), &[n, k])?;
            let weights = quantize(&weight, k, n, self.group_size)?;
            let bias_name = format!("{prefix}.{name}.bias");
            let bias_f16 = if index < 3 {
                Some(self.read_f16(&bias_name, n)?)
            } else if self.tensors.contains(&bias_name) {
                return Err(invalid(format!("unexpected Qwen2 bias tensor {bias_name}")));
            } else {
                None
            };
            projections.push(ProjectionInput {
                weights,
                bias_f16,
                execution: self.layers,
            });
        }
        Ok(LayerInput {
            attention_norm_f16,
            ffn_norm_f16,
            projections: projections
                .try_into()
                .map_err(|_| invalid("expected seven Qwen2 projections"))?,
            tune: None,
        })
    }

    fn read_f16(&mut self, name: &str, width: u32) -> io::Result<Vec<u16>> {
        let bytes = into_f16(self.tensors.read(name, &[width])?)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes([word[0], word[1]]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, f16, into_f16, quantize, safetensors::FloatFormat};

    #[test]
    fn quantization_preserves_signed_codes_and_zero_groups() {
        let tensor = Tensor {
            bytes: [7.0_f32, -7.0, 0.0, 0.0, 0.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            format: FloatFormat::F32,
        };
        let weights = quantize(&tensor, 5, 1, 2).unwrap();
        assert_eq!(weights.packed, [0x1f, 0x88, 0x88]);
        assert_eq!(weights.scales_f16, [f16::ONE.to_bits(), 0, 0]);
        assert!(weights.zero_points.is_none());
    }

    #[test]
    fn converts_bf16_and_rejects_non_finite_values() {
        let bf16 = Tensor {
            bytes: half::bf16::from_f32(-1.5).to_le_bytes().to_vec(),
            format: FloatFormat::Bf16,
        };
        assert_eq!(into_f16(bf16).unwrap(), f16::from_f32(-1.5).to_le_bytes());
        let non_finite = Tensor {
            bytes: f32::INFINITY.to_le_bytes().to_vec(),
            format: FloatFormat::F32,
        };
        assert!(quantize(&non_finite, 1, 1, 1).is_err());
        assert!(into_f16(non_finite).is_err());
    }
}

fn into_f16(mut tensor: Tensor) -> io::Result<Vec<u8>> {
    let width = tensor.format.word_bytes();
    let count = tensor.bytes.len() / width;
    for index in 0..count {
        let start = index * width;
        let value = f16::from_f32(tensor.format.decode(&tensor.bytes[start..start + width]));
        if !value.is_finite() {
            return Err(invalid(
                "checkpoint input cannot be represented as finite FP16",
            ));
        }
        // F32 converts toward the beginning of its own allocation.
        tensor.bytes[2 * index..2 * index + 2].copy_from_slice(&value.to_le_bytes());
    }
    tensor.bytes.truncate(2 * count);
    tensor.bytes.shrink_to_fit();
    Ok(tensor.bytes)
}

fn quantize(tensor: &Tensor, k: u32, n: u32, group_size: u32) -> io::Result<W4WeightInput> {
    let (k, n, group_size) = (k as usize, n as usize, group_size as usize);
    let stride = k.div_ceil(2);
    let width = tensor.format.word_bytes();
    let mut packed = vec![0x88; n * stride];
    let mut scales_f16 = Vec::with_capacity(n * k.div_ceil(group_size));
    let mut values = Vec::with_capacity(group_size.min(k));
    for (row, weights) in tensor.bytes.chunks_exact(k * width).enumerate() {
        for (group, raw) in weights.chunks(group_size.min(k) * width).enumerate() {
            values.clear();
            let mut maximum = 0.0f32;
            for word in raw.chunks_exact(width) {
                let value = tensor.format.decode(word);
                if !value.is_finite() {
                    return Err(invalid("checkpoint weight contains a non-finite value"));
                }
                maximum = maximum.max(value.abs());
                values.push(value);
            }
            let scale = f16::from_f32(maximum / 7.0);
            if !scale.is_finite() {
                return Err(invalid(
                    "checkpoint weight requires a scale outside the FP16 range",
                ));
            }
            scales_f16.push(scale.to_bits());
            if scale == f16::ZERO {
                continue;
            }
            let scale = scale.to_f32();
            for (index, &value) in values.iter().enumerate() {
                let col = group * group_size + index;
                let code = ((value / scale).round_ties_even().clamp(-7.0, 7.0) as i8 + 8) as u8;
                let shift = (col % 2) * 4;
                let cell = &mut packed[row * stride + col / 2];
                *cell = (*cell & !(15 << shift)) | (code << shift);
            }
        }
    }
    Ok(W4WeightInput {
        packed,
        scales_f16,
        zero_points: None,
    })
}
