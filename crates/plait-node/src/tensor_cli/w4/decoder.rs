use std::io::{self, BufRead};
use std::net::SocketAddr;
use std::time::Duration;

use plait_core::NodeId;
use plait_ops::OperatorError;
use plait_ops::decoder::{DecoderBlock, DecoderConfig, DecoderStack, KvSnapshot, LinearProjection};
use plait_ops::native::{DeviceSession, NativeW4A16, PreparedW4Projection, W4ProjectionKernel};
use plait_parallel::precision::W4Quantization;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{W4Device, W4WeightInput, activation_format, write_json};

mod calibration;
mod checkpoint;
mod model;
mod plan;
mod service;
mod transfer;
pub(in crate::tensor_cli) use service::replay_pd;
pub(in crate::tensor_cli) use service::resume_pd;
pub(in crate::tensor_cli) use service::submit as generate_pd;
use transfer::{Publication, Source};

#[derive(Deserialize)]
pub(in crate::tensor_cli) struct DecoderCommand {
    #[serde(flatten)]
    device: W4Device,
    #[serde(flatten)]
    legacy: LegacyModel,
    checkpoint: Option<checkpoint::Config>,
    max_tokens: u32,
    tune_model: Option<calibration::ModelConfig>,
    plan: Option<plan::Config>,
    service: Option<service::Config>,
}

#[derive(Deserialize)]
struct LegacyModel {
    hidden_size: Option<u32>,
    intermediate_size: Option<u32>,
    attention_heads: Option<u32>,
    kv_heads: Option<u32>,
    rope_theta: Option<f32>,
    rms_epsilon: Option<f32>,
    layer_count: Option<u32>,
    model: Option<model::Config>,
}

impl LegacyModel {
    fn is_configured(&self) -> bool {
        self.hidden_size.is_some()
            || self.intermediate_size.is_some()
            || self.attention_heads.is_some()
            || self.kv_heads.is_some()
            || self.rope_theta.is_some()
            || self.rms_epsilon.is_some()
            || self.layer_count.is_some()
            || self.model.is_some()
    }

    fn decoder_config(&self) -> io::Result<DecoderConfig> {
        let (
            Some(hidden_size),
            Some(intermediate_size),
            Some(attention_heads),
            Some(kv_heads),
            Some(rope_theta),
            Some(rms_epsilon),
        ) = (
            self.hidden_size,
            self.intermediate_size,
            self.attention_heads,
            self.kv_heads,
            self.rope_theta,
            self.rms_epsilon,
        )
        else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "decoder-w4 needs dimensions for layer records or a checkpoint",
            ));
        };
        Ok(DecoderConfig {
            hidden_size,
            intermediate_size,
            attention_heads,
            kv_heads,
            rope_theta,
            rms_epsilon,
        })
    }
}

#[derive(Deserialize)]
struct LayerInput {
    attention_norm_f16: Vec<u16>,
    ffn_norm_f16: Vec<u16>,
    // Ordered Q, K, V, O, gate, up, down.
    projections: [ProjectionInput; 7],
    tune: Option<calibration::Config>,
}

#[derive(Deserialize)]
struct ProjectionInput {
    #[serde(flatten)]
    weights: W4WeightInput,
    bias_f16: Option<Vec<u16>>,
    #[serde(flatten)]
    execution: ProjectionExecution,
}

#[derive(Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
struct ProjectionExecution {
    activation_bits: Option<u32>,
    quantization: Option<W4Quantization>,
    #[serde(default)]
    kernel: ProjectionKernel,
}

#[derive(Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ProjectionKernel {
    #[default]
    Native,
    Humming,
    Marlin,
}

impl ProjectionExecution {
    fn prepare<'a, 'lib>(
        &self,
        device: &'a DeviceSession<'lib>,
        weights: &W4WeightInput,
        bias_f16: Option<&[u16]>,
        shape: (u32, u32),
        group_size: u32,
    ) -> Result<DeviceProjection<'a, 'lib>, OperatorError> {
        let (k, n) = shape;
        let mode = activation_format(
            self.activation_bits.unwrap_or(16),
            self.quantization,
            device.supports_f16_quantization(),
        )?;
        let kernel = match self.kernel {
            ProjectionKernel::Native => W4ProjectionKernel::Native,
            ProjectionKernel::Humming => W4ProjectionKernel::Humming,
            ProjectionKernel::Marlin => W4ProjectionKernel::Marlin,
        };
        let linear = device.prepare_w4_projection(
            weights.layout(k, n, group_size)?,
            bias_f16,
            mode,
            kernel,
        )?;
        Ok(DeviceProjection {
            device,
            linear,
            shape,
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum Request {
    Forward {
        hidden: Vec<f32>,
    },
    Generate {
        tokens: Vec<u32>,
        max_new_tokens: u32,
        #[serde(default)]
        stop_tokens: Vec<u32>,
    },
    Snapshot,
    Restore {
        kv: SnapshotInput,
    },
    PublishKv {
        listen: SocketAddr,
        lifetime_ms: u64,
    },
    PullKv {
        source: Source,
    },
    ReleaseKv {
        transfer: NodeId,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SnapshotInput {
    Layer(KvInput),
    Stack(Vec<KvInput>),
}

#[derive(Deserialize)]
struct KvInput {
    layer: [u8; 32],
    keys_f16: Vec<u16>,
    values_f16: Vec<u16>,
}

impl From<KvInput> for KvSnapshot {
    fn from(value: KvInput) -> Self {
        Self {
            layer: value.layer,
            keys_f16: value.keys_f16,
            values_f16: value.values_f16,
        }
    }
}

struct DeviceProjection<'a, 'lib> {
    device: &'a DeviceSession<'lib>,
    linear: PreparedW4Projection<'lib>,
    shape: (u32, u32),
}

impl LinearProjection for DeviceProjection<'_, '_> {
    fn shape(&self) -> (u32, u32) {
        self.shape
    }

    fn forward(&self, input: &[u16]) -> Result<Vec<f32>, OperatorError> {
        self.device.project_w4(&self.linear, input)
    }
}

pub(in crate::tensor_cli) async fn run(
    command: DecoderCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    if command.checkpoint.is_some() && command.legacy.is_configured() {
        return Err(
            "provide either a checkpoint or explicit decoder dimensions and weights".into(),
        );
    }
    let group_size = command.device.group_size;
    let mut checkpoint = command
        .checkpoint
        .map(|source| checkpoint::Checkpoint::open(source, group_size, command.max_tokens))
        .transpose()?;
    let (config, layer_count, head_execution) = match &checkpoint {
        Some(checkpoint) => (
            checkpoint.decoder_config(),
            checkpoint.layer_count(),
            Some(checkpoint.head_execution()),
        ),
        None => (
            command.legacy.decoder_config()?,
            command.legacy.layer_count.unwrap_or(1),
            command.legacy.model.as_ref().map(|model| model.execution),
        ),
    };
    let shapes = config.projection_shapes()?;
    let mut lines = io::stdin().lock().lines();
    let max_snapshot_bytes = transfer::snapshot_size(layer_count, command.max_tokens, shapes[1].1)?;
    let _lock = command
        .service
        .as_ref()
        .map(|service| service.lock(command.device.ordinal))
        .transpose()?;
    let mut plan = command
        .plan
        .map(|plan| {
            let head = head_execution.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "decoder plans require token model weights",
                )
            })?;
            plan::Deployment::open(
                plan,
                command.service.as_ref().map(service::Config::grant),
                command.device.ordinal,
                &command.device.library,
                head,
                layer_count,
                command.tune_model.is_some(),
            )
        })
        .transpose()?;
    // The local deployment configuration selects a trusted device library.
    let library = unsafe { NativeW4A16::load(&command.device.library) }?;
    let device = library.open_device(command.device.ordinal)?;
    let (head, model_id) = match &mut checkpoint {
        Some(checkpoint) => Some(model::prepare(
            checkpoint.load_head()?,
            &device,
            config,
            group_size,
        )?),
        None => command
            .legacy
            .model
            .map(|model| model::load(model, &device, config, group_size))
            .transpose()?,
    }
    .unzip();
    let mut model_tuning = if let Some(tuning) = command.tune_model {
        let head = head
            .as_ref()
            .ok_or("model tuning requires token model weights")?;
        Some(calibration::ModelCalibration::new(
            tuning,
            head,
            command.max_tokens,
        )?)
    } else {
        None
    };
    let mut blocks = Vec::new();
    let mut layers = Vec::new();
    for position in 0..layer_count {
        let mut input: LayerInput = match &mut checkpoint {
            Some(checkpoint) => checkpoint.load_layer(position, shapes)?,
            None => serde_json::from_str(&lines.next().ok_or("decoder-w4 needs layer weights")??)?,
        };
        let mut layer = layer_digest(config, group_size, &input, &shapes);
        if layer_count > 1 || model_id.is_some() {
            let mut digest = Sha256::new();
            digest.update(b"plait-decoder-stack-layer-v1");
            digest.update(layer_count.to_le_bytes());
            digest.update(position.to_le_bytes());
            digest.update(layer);
            if let Some(model) = model_id {
                digest.update(model);
            }
            layer = digest.finalize().into();
        }
        if let Some(plan) = &plan {
            if input.tune.is_some() {
                return Err("decoder plans and per-layer tuning cannot be combined".into());
            }
            plan.layer(position, layer, &mut input.projections)?;
        }
        let mut block = prepare_block(&device, layer, config, &input, shapes, group_size)?;
        let report = match (&input.tune, &mut model_tuning) {
            (Some(_), Some(_)) => {
                return Err("model tuning and per-layer tuning cannot be combined".into());
            }
            (None, Some(tuning)) => Some(tuning.layer(
                &mut block,
                &device,
                &input.projections,
                shapes,
                group_size,
                command.max_tokens,
            )?),
            (Some(tuning), None) => Some(calibration::tune(
                &mut block,
                &device,
                &input.projections,
                shapes,
                group_size,
                command.max_tokens,
                tuning,
            )?),
            (None, None) => None,
        };
        if let Some(result) = report {
            if let Some(plan) = &mut plan {
                plan.record(layer, result.projections);
            }
            eprintln!(
                "{}",
                json!({"event":"decoder_calibrated", "layer":position, "result":result})
            );
        }
        blocks.push(block);
        layers.push(layer);
    }
    if let Some(tuning) = model_tuning {
        let comparison = tuning.compare(head.as_ref().expect("model tuning has a token head"))?;
        eprintln!(
            "{}",
            json!({"event":"decoder_model_calibrated", "tokens":comparison})
        );
        if comparison.baseline != comparison.tuned {
            return Err("model tuning changed a calibration token".into());
        }
    }
    let stack = DecoderStack::new(blocks)?;
    if let Some(plan) = plan {
        eprintln!(
            "{}",
            json!({"event":"decoder_plan_ready", "sha256":plan.finish()?})
        );
    }
    if let Some(service) = command.service {
        let head = head
            .as_ref()
            .ok_or("decoder service needs token model weights")?;
        let mut digest = Sha256::new();
        digest.update(b"plait-decoder-service-v1");
        for layer in layers {
            digest.update(layer);
        }
        drop(lines);
        return service::serve(
            service,
            &stack,
            head,
            digest.finalize().into(),
            service::KvLimits {
                layers: layer_count,
                bytes: max_snapshot_bytes,
                tokens: command.max_tokens,
            },
        )
        .await;
    }
    let mut cache = stack.empty_cache(command.max_tokens)?;
    let ready = if layers.len() == 1 {
        json!(layers[0])
    } else {
        json!(layers)
    };
    if let Some(id) = model_id {
        write_json(&json!({ "ready": ready, "model": id }))?;
    } else {
        write_json(&json!({ "ready": ready }))?;
    }
    let mut publication = Publication::default();
    for line in lines {
        let response: Result<Value, Box<dyn std::error::Error>> = async {
            match serde_json::from_str::<Request>(&line?)? {
                Request::Forward { hidden } => Ok(json!({ "hidden": stack.forward(&hidden, &mut cache)? })),
                Request::Generate { tokens, max_new_tokens, stop_tokens } => {
                    let head = head.as_ref().ok_or("decoder-w4 generate needs model embeddings and output weights")?;
                    let generated = head.generate(&stack, &mut cache, &tokens, max_new_tokens, &stop_tokens)?;
                    Ok(json!({ "tokens": generated, "cache_tokens": cache.tokens() }))
                }
                Request::Snapshot => {
                    let layers: Vec<_> = cache.snapshots().map(|kv| {
                        json!({ "layer": kv.layer, "keys_f16": kv.keys_f16, "values_f16": kv.values_f16 })
                    }).collect();
                    let kv = if layers.len() == 1 { layers.into_iter().next().unwrap() } else { json!(layers) };
                    Ok(json!({ "kv": kv }))
                }
                Request::Restore { kv } => {
                    let layers = match kv {
                        SnapshotInput::Layer(layer) => vec![layer.into()],
                        SnapshotInput::Stack(layers) => layers.into_iter().map(Into::into).collect(),
                    };
                    cache = stack.restore_cache(layers, command.max_tokens)?;
                    Ok(json!({ "restored": true }))
                }
                Request::PublishKv { listen, lifetime_ms } => {
                    let source = publication.publish(listen, lifetime_ms, &cache).await?;
                    Ok(json!({ "source": source }))
                }
                Request::PullKv { source } => {
                    let snapshot = transfer::pull(source, max_snapshot_bytes, layer_count, Duration::from_secs(15)).await?;
                    let restored = stack.restore_cache(snapshot.layers, command.max_tokens)?;
                    if restored.tokens() != snapshot.tokens as usize {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "KV header has a different token count").into());
                    }
                    cache = restored;
                    Ok(json!({ "restored": true, "tokens": cache.tokens() }))
                }
                Request::ReleaseKv { transfer } => Ok(json!({ "released": publication.release(transfer) })),
            }
        }.await;
        write_json(&response.unwrap_or_else(|error| json!({ "error": error.to_string() })))?;
    }
    Ok(())
}

fn prepare_block<'a, 'lib>(
    device: &'a DeviceSession<'lib>,
    layer: [u8; 32],
    config: DecoderConfig,
    input: &LayerInput,
    shapes: [(u32, u32); 7],
    group_size: u32,
) -> Result<DecoderBlock<DeviceProjection<'a, 'lib>>, OperatorError> {
    let mut projections = Vec::with_capacity(shapes.len());
    for (shape, source) in shapes.into_iter().zip(&input.projections) {
        projections.push(source.execution.prepare(
            device,
            &source.weights,
            source.bias_f16.as_deref(),
            shape,
            group_size,
        )?);
    }
    DecoderBlock::new(
        layer,
        config,
        &input.attention_norm_f16,
        &input.ffn_norm_f16,
        projections
            .try_into()
            .ok()
            .expect("one projection per decoder shape"),
    )
}

fn layer_digest(
    config: DecoderConfig,
    group_size: u32,
    input: &LayerInput,
    shapes: &[(u32, u32); 7],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"plait-decoder-fp16-kv-v1");
    for v in [
        config.hidden_size,
        config.intermediate_size,
        config.attention_heads,
        config.kv_heads,
        config.rope_theta.to_bits(),
        config.rms_epsilon.to_bits(),
        group_size,
    ] {
        digest.update(v.to_le_bytes());
    }
    for norm in [&input.attention_norm_f16, &input.ffn_norm_f16] {
        digest.update((norm.len() as u64).to_le_bytes());
        for value in norm {
            digest.update(value.to_le_bytes());
        }
    }
    for (&(k, n), projection) in shapes.iter().zip(&input.projections) {
        digest.update(projection.weights.digest(k, n, group_size));
        digest.update([u8::from(projection.bias_f16.is_some())]);
        if let Some(bias) = &projection.bias_f16 {
            digest.update((bias.len() as u64).to_le_bytes());
            for value in bias {
                digest.update(value.to_le_bytes());
            }
        }
    }
    digest.finalize().into()
}
