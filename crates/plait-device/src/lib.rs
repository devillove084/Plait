//! Capability-aware accelerator discovery and scheduling.
//!
//! Discovery is deliberately vendor-neutral: a profile is advertised as
//! replicated data (`DeviceProfile`), never inferred from a process rank or a
//! hard-coded backend. The scheduler then chooses among eligible devices using
//! load, advertised peak, memory, and stable identity as deterministic
//! tie-breakers.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::process::Command;

use plait_core::{AcceleratorKind, ComputeRequirement, DeviceId, DeviceProfile, NodeId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    LeastLoaded,
    Fastest,
    MostMemory,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceRegistry {
    profiles: BTreeMap<DeviceId, DeviceProfile>,
    active: BTreeMap<DeviceId, u32>,
}

impl DeviceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, profile: DeviceProfile) {
        self.profiles.insert(profile.id, profile);
    }

    pub fn profiles(&self) -> &BTreeMap<DeviceId, DeviceProfile> {
        &self.profiles
    }

    pub fn active(&self, id: DeviceId) -> u32 {
        self.active.get(&id).copied().unwrap_or(0)
    }

    pub fn eligible(&self, requirement: &ComputeRequirement) -> Vec<&DeviceProfile> {
        self.profiles
            .values()
            .filter(|profile| requirement.matches(profile))
            .collect()
    }

    pub fn select(
        &self,
        requirement: &ComputeRequirement,
        policy: SchedulingPolicy,
    ) -> Option<DeviceId> {
        self.eligible(requirement)
            .into_iter()
            .min_by_key(|profile| {
                let load = u64::from(self.active(profile.id));
                let fastest = u64::MAX - profile.flops;
                let memory = u64::MAX - profile.memory_bytes;
                let primary = match policy {
                    SchedulingPolicy::LeastLoaded => (load, fastest, memory),
                    SchedulingPolicy::Fastest => (fastest, load, memory),
                    SchedulingPolicy::MostMemory => (memory, load, fastest),
                };
                (primary, profile.id)
            })
            .map(|profile| profile.id)
    }

    pub fn reserve(&mut self, id: DeviceId) -> bool {
        if !self.profiles.contains_key(&id) {
            return false;
        }
        *self.active.entry(id).or_default() += 1;
        true
    }

    pub fn release(&mut self, id: DeviceId) {
        if let Some(load) = self.active.get_mut(&id) {
            *load = load.saturating_sub(1);
        }
    }

    /// Assign a set of requirements, reserving each chosen device. This is the
    /// planner-to-scheduler contract; callers can release reservations when the
    /// replicated compute command executes.
    pub fn assign<'a>(
        &mut self,
        requirements: impl IntoIterator<Item = &'a ComputeRequirement>,
        policy: SchedulingPolicy,
    ) -> Vec<Option<DeviceId>> {
        let mut assignments = Vec::new();
        for requirement in requirements {
            let selected = self.select(requirement, policy);
            if let Some(device) = selected {
                self.reserve(device);
            }
            assignments.push(selected);
        }
        assignments
    }
}

/// Discover the devices visible on this host. Missing vendor tools are not an
/// error: the CPU profile is always available and other vendors are optional.
pub fn discover(node: NodeId) -> Vec<DeviceProfile> {
    let mut profiles = Vec::new();
    profiles.extend(discover_nvidia(node));
    profiles.extend(discover_amd(node));
    profiles.extend(discover_huawei(node));
    profiles.extend(discover_ppu(node));
    if let Some(cpu) = cpu_profile(node) {
        profiles.push(cpu);
    }
    let mut seen = BTreeSet::new();
    profiles.retain(|profile| seen.insert(profile.id));
    profiles
}

/// Quantization GEMM kernels that are both present in vLLM and useful for a
/// device. `marlin` is the W4A16 path that works on Ampere; `humming` and
/// `b12x` are the Blackwell mixed-precision/NVFP4 paths. Capability is read
/// from the `[cc X.Y, N SM]` suffix emitted by discovery, so this remains a
/// pure profile query with no driver call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelCaps {
    pub marlin: bool,
    pub humming: bool,
    pub b12x: bool,
    pub fp8: bool,
    pub nvfp4: bool,
}

pub fn compute_capability(profile: &DeviceProfile) -> Option<(u8, u8)> {
    let start = profile.name.find("[cc ")? + 4;
    let rest = &profile.name[start..];
    let version = rest.split(',').next()?.trim();
    let (major, minor) = version.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

pub fn kernel_caps(profile: &DeviceProfile) -> KernelCaps {
    let (major, minor) = compute_capability(profile).unwrap_or((0, 0));
    let sm = major * 10 + minor;
    KernelCaps {
        marlin: sm >= 75,
        // vLLM's Humming mixed-precision path targets Blackwell; Ampere and
        // Ada keep Marlin so the same checkpoint can run on every rank.
        humming: sm >= 100,
        b12x: sm >= 120,
        fp8: sm >= 89,
        nvfp4: sm >= 100,
    }
}

fn cpu_profile(node: NodeId) -> Option<DeviceProfile> {
    let memory = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("MemTotal:")
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|kb| kb.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
        })
        .unwrap_or(0);
    let compute_units = std::thread::available_parallelism()
        .map(|count| count.get() as u32)
        .unwrap_or(1);
    Some(DeviceProfile {
        id: DeviceId::new(node, u32::MAX),
        name: format!("cpu-{compute_units}"),
        arch: AcceleratorKind::Cpu,
        memory_bytes: memory,
        flops: 0,
        compute_units,
        supports_fp16: true,
        supports_bf16: true,
        supports_fp64: true,
        rdma_capable: rdma_visible(),
        numa_node: None,
    })
}

fn discover_nvidia(node: NodeId) -> Vec<DeviceProfile> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,clocks.max.sm,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let driver = CudaDriver::open();
    parse_nvidia(&String::from_utf8_lossy(&output.stdout), node, |index| {
        driver.as_ref()?.sm_count(index)
    })
}

struct CudaDriver {
    _library: libloading::Library,
    device_get: unsafe extern "system" fn(*mut i32, i32) -> i32,
    attribute: unsafe extern "system" fn(*mut i32, i32, i32) -> i32,
}

impl CudaDriver {
    fn open() -> Option<Self> {
        let name = match std::env::consts::OS {
            "linux" => "libcuda.so.1",
            "windows" => "nvcuda.dll",
            "macos" => "libcuda.dylib",
            _ => return None,
        };
        // These are the device query entry points of the system CUDA driver.
        // Discovery keeps using the management tool if the driver is unavailable.
        unsafe {
            let library = libloading::Library::new(name).ok()?;
            let init = library
                .get::<unsafe extern "system" fn(u32) -> i32>(b"cuInit")
                .ok()?;
            if init(0) != 0 {
                return None;
            }
            Some(Self {
                device_get: *library.get(b"cuDeviceGet").ok()?,
                attribute: *library.get(b"cuDeviceGetAttribute").ok()?,
                _library: library,
            })
        }
    }

    fn sm_count(&self, index: u32) -> Option<u32> {
        const MULTIPROCESSOR_COUNT: i32 = 16;
        let ordinal = i32::try_from(index).ok()?;
        let mut device = 0;
        let mut count = 0;
        unsafe {
            if (self.device_get)(&mut device, ordinal) != 0
                || (self.attribute)(&mut count, MULTIPROCESSOR_COUNT, device) != 0
            {
                return None;
            }
        }
        (count > 0).then_some(count as u32)
    }
}

fn parse_nvidia(
    text: &str,
    node: NodeId,
    sm_count: impl Fn(u32) -> Option<u32>,
) -> Vec<DeviceProfile> {
    let mut profiles = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 5 {
            continue;
        }
        let Ok(index) = fields[0].parse::<u32>() else {
            continue;
        };
        let Ok(memory_mib) = fields[2].parse::<u64>() else {
            continue;
        };
        let clock_mhz = fields[3].parse::<u64>().unwrap_or(0);
        let name = fields[1];
        let capability = fields[4].trim();
        let sm_count = sm_count(index).unwrap_or(1);
        let tensor_ops_per_sm = match capability {
            "12.0" => 256,
            "9.0" => 256,
            "8.9" => 128,
            "8.6" => 128,
            _ => 128,
        };
        let flops = (sm_count as u64)
            .saturating_mul(clock_mhz)
            .saturating_mul(1_000_000)
            .saturating_mul(tensor_ops_per_sm);
        profiles.push(DeviceProfile {
            id: DeviceId::new(node, index),
            name: format!("{name} [cc {capability}, {sm_count} SM]"),
            arch: AcceleratorKind::Nvidia,
            memory_bytes: memory_mib.saturating_mul(1024 * 1024),
            flops,
            compute_units: sm_count,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: rdma_visible(),
            numa_node: None,
        });
    }
    profiles
}

fn discover_amd(node: NodeId) -> Vec<DeviceProfile> {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };
    let mut profiles = Vec::new();
    let mut ordinal = 1_000u32;
    for entry in entries.flatten() {
        let path = entry.path().join("device");
        let vendor = fs::read_to_string(path.join("vendor")).unwrap_or_default();
        if !vendor.trim().eq_ignore_ascii_case("0x1002") {
            continue;
        }
        let memory = fs::read_to_string(path.join("mem_info_vram_total"))
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .unwrap_or(0);
        let name = fs::read_to_string(path.join("product_name"))
            .unwrap_or_else(|_| "AMD GPU".to_owned())
            .trim()
            .to_owned();
        profiles.push(DeviceProfile {
            id: DeviceId::new(node, ordinal),
            name,
            arch: AcceleratorKind::Amd,
            memory_bytes: memory,
            flops: 0,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: rdma_visible(),
            numa_node: None,
        });
        ordinal += 1;
    }
    profiles
}

fn discover_huawei(node: NodeId) -> Vec<DeviceProfile> {
    let Ok(output) = Command::new("npu-smi").arg("info").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let count = text
        .lines()
        .filter(|line| line.contains("Ascend") || line.contains("910"))
        .count()
        .max(1);
    (0..count as u32)
        .map(|index| DeviceProfile {
            id: DeviceId::new(node, 2_000 + index),
            name: format!("Ascend NPU {index}"),
            arch: AcceleratorKind::Huawei,
            memory_bytes: 0,
            flops: 0,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: false,
            rdma_capable: rdma_visible(),
            numa_node: None,
        })
        .collect()
}

fn discover_ppu(node: NodeId) -> Vec<DeviceProfile> {
    // PPU vendors expose different management tools. Accept an explicit profile
    // string instead of guessing: `name:memory_gb:count` separated by commas.
    let Ok(spec) = std::env::var("PLAIT_PPU_DEVICES") else {
        return Vec::new();
    };
    spec.split(',')
        .filter_map(|entry| {
            let mut parts = entry.split(':');
            let name = parts.next()?.trim().to_owned();
            let memory_gb = parts.next()?.trim().parse::<u64>().ok()?;
            let count = parts.next()?.trim().parse::<u32>().ok()?;
            Some((name, memory_gb, count))
        })
        .flat_map(|(name, memory_gb, count)| {
            (0..count).map(move |index| DeviceProfile {
                id: DeviceId::new(node, 3_000 + index),
                name: format!("{name} {index}"),
                arch: AcceleratorKind::Ppu,
                memory_bytes: memory_gb.saturating_mul(1024 * 1024 * 1024),
                flops: 0,
                compute_units: 1,
                supports_fp16: true,
                supports_bf16: true,
                supports_fp64: false,
                rdma_capable: rdma_visible(),
                numa_node: None,
            })
        })
        .collect()
}

fn rdma_visible() -> bool {
    fs::read_dir("/sys/class/infiniband")
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}

/// A deterministic synthetic registry for tests and single-host demos.
pub fn synthetic_registry(node: NodeId) -> DeviceRegistry {
    let profiles = [
        DeviceProfile {
            id: DeviceId::new(node, 0),
            name: "synthetic-nvidia".into(),
            arch: AcceleratorKind::Nvidia,
            memory_bytes: 72 * 1024 * 1024 * 1024,
            flops: 100,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: true,
            numa_node: None,
        },
        DeviceProfile {
            id: DeviceId::new(node, 1),
            name: "synthetic-amd".into(),
            arch: AcceleratorKind::Amd,
            memory_bytes: 64 * 1024 * 1024 * 1024,
            flops: 90,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: true,
            numa_node: None,
        },
        DeviceProfile {
            id: DeviceId::new(node, 2),
            name: "synthetic-huawei".into(),
            arch: AcceleratorKind::Huawei,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            flops: 80,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: false,
            rdma_capable: true,
            numa_node: None,
        },
        DeviceProfile {
            id: DeviceId::new(node, 3),
            name: "synthetic-ppu".into(),
            arch: AcceleratorKind::Ppu,
            memory_bytes: 48 * 1024 * 1024 * 1024,
            flops: 70,
            compute_units: 1,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: false,
            rdma_capable: true,
            numa_node: None,
        },
        DeviceProfile {
            id: DeviceId::new(node, 4),
            name: "synthetic-cpu".into(),
            arch: AcceleratorKind::Cpu,
            memory_bytes: 512 * 1024 * 1024 * 1024,
            flops: 10,
            compute_units: 64,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: true,
            numa_node: None,
        },
    ];
    let mut registry = DeviceRegistry::new();
    for profile in profiles {
        registry.register(profile);
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::DType;

    #[test]
    fn nvidia_csv_parses_into_a_profile() {
        let node = NodeId::derive("dev-parse");
        let profiles = parse_nvidia("0, NVIDIA Blackwell, 73415, 2600, 12.0\n", node, |index| {
            assert_eq!(index, 0);
            Some(110)
        });
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].arch, AcceleratorKind::Nvidia);
        assert_eq!(profiles[0].memory_bytes, 73415 * 1024 * 1024);
        assert_eq!(profiles[0].compute_units, 110);
        assert!(profiles[0].flops > 0);
    }

    #[test]
    fn scheduler_respects_architecture_and_capability() {
        let node = NodeId::derive("sched");
        let mut registry = synthetic_registry(node);
        let huawei = ComputeRequirement {
            arch: Some(AcceleratorKind::Huawei),
            min_memory_bytes: 16 * 1024 * 1024 * 1024,
            min_flops: 50,
            dtype: DType::F16,
            prefer_rdma: true,
        };
        let selected = registry
            .select(&huawei, SchedulingPolicy::LeastLoaded)
            .unwrap();
        assert_eq!(registry.profiles()[&selected].arch, AcceleratorKind::Huawei);
        registry.reserve(selected);
        let amd = ComputeRequirement {
            arch: Some(AcceleratorKind::Amd),
            ..ComputeRequirement::default()
        };
        let selected = registry
            .select(&amd, SchedulingPolicy::LeastLoaded)
            .unwrap();
        assert_eq!(registry.profiles()[&selected].arch, AcceleratorKind::Amd);
        let fp64_huawei = ComputeRequirement {
            arch: Some(AcceleratorKind::Huawei),
            dtype: DType::F64,
            ..ComputeRequirement::default()
        };
        assert!(
            registry
                .select(&fp64_huawei, SchedulingPolicy::LeastLoaded)
                .is_none()
        );
    }

    #[test]
    fn least_loaded_is_deterministic() {
        let node = NodeId::derive("load");
        let mut registry = synthetic_registry(node);
        let any = ComputeRequirement::default();
        let first = registry
            .select(&any, SchedulingPolicy::LeastLoaded)
            .unwrap();
        registry.reserve(first);
        let second = registry
            .select(&any, SchedulingPolicy::LeastLoaded)
            .unwrap();
        assert_ne!(first, second);
        registry.release(first);
        assert_eq!(
            registry.select(&any, SchedulingPolicy::LeastLoaded),
            Some(first)
        );
    }

    fn profile_with_name(name: &str) -> DeviceProfile {
        DeviceProfile {
            id: DeviceId::new(NodeId::derive(name), 0),
            name: name.to_owned(),
            arch: AcceleratorKind::Nvidia,
            memory_bytes: 24 << 30,
            flops: 1,
            compute_units: 82,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: true,
            rdma_capable: false,
            numa_node: None,
        }
    }

    #[test]
    fn kernel_caps_split_ampere_from_blackwell() {
        let ampere = profile_with_name("NVIDIA Ampere [cc 8.6, 82 SM]");
        assert_eq!(compute_capability(&ampere), Some((8, 6)));
        let caps = kernel_caps(&ampere);
        assert!(caps.marlin);
        assert!(!caps.humming);
        assert!(!caps.fp8);
        assert!(!caps.nvfp4);

        let blackwell = profile_with_name("NVIDIA Blackwell [cc 12.0, 110 SM]");
        assert_eq!(compute_capability(&blackwell), Some((12, 0)));
        let caps = kernel_caps(&blackwell);
        assert!(caps.marlin);
        assert!(caps.humming);
        assert!(caps.b12x);
        assert!(caps.fp8);
        assert!(caps.nvfp4);
    }
}
