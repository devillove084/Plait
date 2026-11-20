use serde::{Deserialize, Serialize};

use crate::identity::NodeId;
use crate::tensor::DType;

/// Accelerator families Plait can schedule onto. `Other` is deliberate: adding a
/// vendor must not require changing the consensus wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AcceleratorKind {
    Cpu,
    Nvidia,
    Amd,
    Huawei,
    Ppu,
    Other,
}

impl AcceleratorKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Nvidia => "nvidia",
            Self::Amd => "amd",
            Self::Huawei => "huawei",
            Self::Ppu => "ppu",
            Self::Other => "other",
        }
    }
}

impl std::fmt::Display for AcceleratorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Stable device identity. The node identity is separate from the local ordinal
/// so a device can move between hosts only through an explicit re-registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeviceId {
    pub node: NodeId,
    pub ordinal: u32,
}

impl DeviceId {
    pub const fn new(node: NodeId, ordinal: u32) -> Self {
        Self { node, ordinal }
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}#{}", self.node.to_hex(), self.ordinal)
    }
}

/// Advertised capabilities. These are replicated as data, not inferred from a
/// rank or a backend name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub id: DeviceId,
    pub name: String,
    pub arch: AcceleratorKind,
    pub memory_bytes: u64,
    /// Relative advertised compute score in operations/second terms. It is a
    /// scheduling hint derived from vendor data, not a measured benchmark.
    pub flops: u64,
    pub compute_units: u32,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp64: bool,
    pub rdma_capable: bool,
    pub numa_node: Option<u32>,
}

impl DeviceProfile {
    pub fn supports(&self, dtype: DType) -> bool {
        match dtype {
            DType::F16 => self.supports_fp16,
            DType::BF16 => self.supports_bf16,
            DType::F32 => true,
            DType::F64 => self.supports_fp64,
        }
    }
}

/// A task's eligibility and placement constraints. Empty/minimum values mean
/// "no constraint"; an explicit architecture pins the task to that family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeRequirement {
    pub arch: Option<AcceleratorKind>,
    pub min_memory_bytes: u64,
    pub min_flops: u64,
    pub dtype: DType,
    pub prefer_rdma: bool,
}

impl Default for ComputeRequirement {
    fn default() -> Self {
        Self {
            arch: None,
            min_memory_bytes: 0,
            min_flops: 0,
            dtype: DType::F32,
            prefer_rdma: false,
        }
    }
}

impl ComputeRequirement {
    pub fn matches(&self, profile: &DeviceProfile) -> bool {
        if let Some(arch) = self.arch {
            if profile.arch != arch {
                return false;
            }
        }
        profile.memory_bytes >= self.min_memory_bytes
            && profile.flops >= self.min_flops
            && profile.supports(self.dtype)
            && (!self.prefer_rdma || profile.rdma_capable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(arch: AcceleratorKind, memory: u64, flops: u64) -> DeviceProfile {
        DeviceProfile {
            id: DeviceId::new(NodeId::derive("dev"), 0),
            name: "test".into(),
            arch,
            memory_bytes: memory,
            flops,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: arch == AcceleratorKind::Cpu,
            rdma_capable: true,
            numa_node: None,
        }
    }

    #[test]
    fn requirements_match_architecture_and_capacity() {
        let requirement = ComputeRequirement {
            arch: Some(AcceleratorKind::Huawei),
            min_memory_bytes: 1024,
            min_flops: 100,
            dtype: DType::F16,
            prefer_rdma: true,
        };
        assert!(requirement.matches(&profile(AcceleratorKind::Huawei, 2048, 200)));
        assert!(!requirement.matches(&profile(AcceleratorKind::Nvidia, 4096, 400)));
        assert!(!requirement.matches(&profile(AcceleratorKind::Huawei, 512, 200)));
        assert!(!requirement.matches(&profile(AcceleratorKind::Huawei, 2048, 50)));
    }
}
