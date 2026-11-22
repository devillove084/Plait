//! Quantization and parallelism co-design.
//!
//! The planner does not implement GEMM kernels. It picks the checkpoint
//! format and the vLLM kernel backend per device, then prefers data
//! parallelism whenever every rank can hold one replica. Tensor parallelism
//! is only selected when the model does not fit any single device; that keeps
//! the expensive per-layer activation allreduce off the WAN.

use plait_core::DeviceProfile;
use plait_device::kernel_caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantFormat {
    Bf16,
    Fp8,
    NvFp4,
    W4A16,
}

impl QuantFormat {
    /// Approximate bytes per BF16-weight byte, including quant scales and
    /// alignment. These are planning constants, not tensor sizes.
    pub const fn bytes_ratio(self) -> f64 {
        match self {
            Self::Bf16 => 1.0,
            Self::Fp8 => 0.50,
            Self::NvFp4 => 0.33,
            Self::W4A16 => 0.30,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelBackend {
    Native,
    Cutlass,
    Humming,
    Marlin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelMode {
    DataParallel,
    TensorParallel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceQuantPlan {
    pub format: QuantFormat,
    pub backend: KernelBackend,
    pub weight_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CodesignPlan {
    pub mode: ParallelMode,
    pub format: QuantFormat,
    pub backend: KernelBackend,
    pub tp_size: u32,
    pub replicas: u32,
    pub per_device: Vec<DeviceQuantPlan>,
}

pub struct CodesignInput {
    /// BF16 weight footprint of the full model.
    pub weight_bytes_bf16: u64,
    /// KV-cache and runtime reserve that must coexist with one shard.
    pub kv_bytes: u64,
    /// Fraction of device memory the planner may use.
    pub memory_fraction: f64,
}

impl Default for CodesignInput {
    fn default() -> Self {
        Self {
            weight_bytes_bf16: 0,
            kv_bytes: 0,
            memory_fraction: 0.90,
        }
    }
}

fn backend(profile: &DeviceProfile, format: QuantFormat) -> Option<KernelBackend> {
    let caps = kernel_caps(profile);
    match format {
        QuantFormat::Bf16 => Some(KernelBackend::Native),
        QuantFormat::Fp8 if caps.fp8 => Some(if caps.humming {
            KernelBackend::Humming
        } else {
            KernelBackend::Cutlass
        }),
        QuantFormat::NvFp4 if caps.nvfp4 => Some(KernelBackend::Humming),
        QuantFormat::W4A16 if caps.marlin => Some(KernelBackend::Marlin),
        _ => None,
    }
}

fn fits(profile: &DeviceProfile, input: &CodesignInput, weight_bytes: u64) -> bool {
    let budget = (profile.memory_bytes as f64 * input.memory_fraction) as u64;
    weight_bytes.saturating_add(input.kv_bytes) <= budget
}

/// Choose one format for a whole replica. A mixed-format TP group is not
/// representable by a single vLLM checkpoint, so the common format must be
/// supported by every rank. Data parallel is returned whenever each device
/// can hold a full replica; otherwise the same format is sharded across all
/// devices (tensor parallel).
pub fn choose_codesign(
    profiles: &[DeviceProfile],
    input: &CodesignInput,
) -> Result<CodesignPlan, String> {
    if profiles.is_empty() {
        return Err("codesign needs at least one device".to_owned());
    }
    if input.weight_bytes_bf16 == 0 {
        return Err("codesign needs a non-zero weight footprint".to_owned());
    }
    let n = profiles.len() as u64;
    for format in [
        QuantFormat::Bf16,
        QuantFormat::Fp8,
        QuantFormat::NvFp4,
        QuantFormat::W4A16,
    ] {
        let mut backends = Vec::with_capacity(profiles.len());
        for profile in profiles {
            let Some(backend) = backend(profile, format) else {
                backends.clear();
                break;
            };
            backends.push(backend);
        }
        if backends.is_empty() {
            continue;
        }
        let full_bytes = (input.weight_bytes_bf16 as f64 * format.bytes_ratio()) as u64;
        if profiles
            .iter()
            .all(|profile| fits(profile, input, full_bytes))
        {
            return Ok(CodesignPlan {
                mode: ParallelMode::DataParallel,
                format,
                backend: backends[0],
                tp_size: 1,
                replicas: profiles.len() as u32,
                per_device: backends
                    .iter()
                    .map(|backend| DeviceQuantPlan {
                        format,
                        backend: *backend,
                        weight_bytes: full_bytes,
                    })
                    .collect(),
            });
        }
        let shard_bytes = full_bytes / n;
        if profiles
            .iter()
            .all(|profile| fits(profile, input, shard_bytes))
        {
            return Ok(CodesignPlan {
                mode: ParallelMode::TensorParallel,
                format,
                backend: backends[0],
                tp_size: profiles.len() as u32,
                replicas: 1,
                per_device: backends
                    .iter()
                    .map(|backend| DeviceQuantPlan {
                        format,
                        backend: *backend,
                        weight_bytes: shard_bytes,
                    })
                    .collect(),
            });
        }
    }
    Err("model does not fit the selected device group".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::{AcceleratorKind, DeviceId, NodeId};

    fn device(name: &str, memory_gib: u64) -> DeviceProfile {
        DeviceProfile {
            id: DeviceId::new(NodeId::derive(name), 0),
            name: name.to_owned(),
            arch: AcceleratorKind::Nvidia,
            memory_bytes: memory_gib << 30,
            flops: 1,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: false,
            numa_node: None,
        }
    }

    #[test]
    fn small_model_uses_data_parallel_everywhere() {
        let devices = vec![
            device("NVIDIA GeForce RTX 3090 [cc 8.6, 82 SM]", 24),
            device("NVIDIA RTX PRO 5000 72GB Blackwell [cc 12.0, 110 SM]", 72),
        ];
        let plan = choose_codesign(
            &devices,
            &CodesignInput {
                weight_bytes_bf16: 8 << 30,
                kv_bytes: 2 << 30,
                ..Default::default()
            },
        )
        .expect("plan");
        assert_eq!(plan.mode, ParallelMode::DataParallel);
        assert_eq!(plan.tp_size, 1);
        assert_eq!(plan.replicas, 2);
        assert_eq!(plan.format, QuantFormat::Bf16);
    }

    #[test]
    fn ampere_drops_to_w4a16_for_a_9b_model() {
        let devices = vec![device("NVIDIA GeForce RTX 3090 [cc 8.6, 82 SM]", 24)];
        let plan = choose_codesign(
            &devices,
            &CodesignInput {
                weight_bytes_bf16: 19 << 30,
                kv_bytes: 4 << 30,
                ..Default::default()
            },
        )
        .expect("plan");
        assert_eq!(plan.mode, ParallelMode::DataParallel);
        assert_eq!(plan.format, QuantFormat::W4A16);
        assert_eq!(plan.backend, KernelBackend::Marlin);
    }

    #[test]
    fn two_sm120_cards_shard_122b_nvfp4() {
        let devices = vec![
            device("NVIDIA RTX PRO 5000 72GB Blackwell [cc 12.0, 110 SM]", 72),
            device("NVIDIA RTX PRO 5000 72GB Blackwell [cc 12.0, 110 SM]", 72),
        ];
        let plan = choose_codesign(
            &devices,
            &CodesignInput {
                weight_bytes_bf16: 244 << 30,
                kv_bytes: 4 << 30,
                memory_fraction: 0.85,
            },
        )
        .expect("plan");
        assert_eq!(plan.mode, ParallelMode::TensorParallel);
        assert_eq!(plan.format, QuantFormat::NvFp4);
        assert_eq!(plan.backend, KernelBackend::Humming);
        assert_eq!(plan.tp_size, 2);
    }
}
