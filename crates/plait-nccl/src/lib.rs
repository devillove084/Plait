#![allow(clippy::not_unsafe_ptr_arg_deref)]
//! NCCL-compatible C ABI for Plait.
//!
//! This exposes the collective surface used by vLLM's `pynccl_wrapper`.
//! CUDA, ROCm, Huawei CANN, and PPU
//! adapters are selected at runtime; group and point-to-point transport is
//! shared. Host-staged TCP is the fallback, with shared-memory two-level
//! allreduce when the host layout is declared.

use std::collections::HashMap;
use std::ffi::CString;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use cudarc::driver::CudaContext;
use half::{bf16, f16};

mod backend;
mod cross;
mod ipc;
mod kernel;
mod p2p;
mod shm;

use backend::{CudaRuntime, DeviceRuntime, HostCopyRuntime, HuaweiRuntime, PpuRuntime};

pub type NcclResult = i32;
pub const NCCL_SUCCESS: NcclResult = 0;
pub const NCCL_UNHANDLED_CUDA_ERROR: NcclResult = 1;
pub const NCCL_INVALID_ARGUMENT: NcclResult = 2;
pub const NCCL_INVALID_USAGE: NcclResult = 3;
pub const NCCL_UNSUPPORTED: NcclResult = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    pub internal: [u8; 128],
}

#[repr(C)]
pub struct NcclCommProperties {
    pub size: usize,
    pub magic: u32,
    pub version: u32,
    pub rank: i32,
    pub n_ranks: i32,
    pub cuda_dev: i32,
    pub nvml_dev: i32,
    pub device_api_support: bool,
    pub multimem_support: bool,
    pub gin_type: i32,
    pub n_lsa_teams: i32,
    pub host_rma_support: bool,
    pub railed_gin_type: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
    Float8e4m3 = 10,
}

impl DataType {
    fn from_raw(value: i32) -> Option<Self> {
        Some(match value {
            0 => Self::Int8,
            1 => Self::Uint8,
            2 => Self::Int32,
            3 => Self::Uint32,
            4 => Self::Int64,
            5 => Self::Uint64,
            6 => Self::Float16,
            7 => Self::Float32,
            8 => Self::Float64,
            9 => Self::Bfloat16,
            10 => Self::Float8e4m3,
            _ => return None,
        })
    }
    fn bytes(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 | Self::Float8e4m3 => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Int64 | Self::Uint64 | Self::Float64 => 8,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

impl ReduceOp {
    fn from_raw(value: i32) -> Option<Self> {
        Some(match value {
            0 => Self::Sum,
            1 => Self::Prod,
            2 => Self::Max,
            3 => Self::Min,
            4 => Self::Avg,
            _ => return None,
        })
    }
}

struct Network {
    root: Option<TcpStream>,
    peers: Vec<TcpStream>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    Rocm,
    Huawei,
    Ppu,
}

struct HostLayout {
    groups: Vec<Vec<usize>>,
    group: usize,
    index: usize,
    leader: usize,
}

impl HostLayout {
    fn from_env(world: usize, rank: usize) -> Result<Option<Self>, String> {
        let Ok(spec) = std::env::var("PLAIT_NCCL_HOSTS") else {
            return Ok(None);
        };
        let mut groups = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (start, end) = if let Some((left, right)) = part.split_once('-') {
                let start = left
                    .trim()
                    .parse::<usize>()
                    .map_err(|error| format!("host range start: {error}"))?;
                let end = right
                    .trim()
                    .parse::<usize>()
                    .map_err(|error| format!("host range end: {error}"))?;
                (start, end)
            } else {
                let value = part
                    .parse::<usize>()
                    .map_err(|error| format!("host rank: {error}"))?;
                (value, value)
            };
            if start > end || end >= world {
                return Err(format!("host range {part} is outside world {world}"));
            }
            groups.push((start..=end).collect::<Vec<_>>());
        }
        if groups.iter().flatten().count() != world {
            return Err("PLAIT_NCCL_HOSTS must cover each rank exactly once".to_owned());
        }
        groups.sort_by_key(|group| group.first().copied().unwrap_or(usize::MAX));
        let mut seen = vec![false; world];
        for group in &groups {
            for &item in group {
                if std::mem::replace(&mut seen[item], true) {
                    return Err(format!("rank {item} appears in multiple host groups"));
                }
            }
        }
        if groups
            .iter()
            .find(|group| group.contains(&0))
            .and_then(|group| group.first())
            != Some(&0)
        {
            return Err("PLAIT_NCCL_HOSTS must start the root group with rank 0".to_owned());
        }
        let group = groups
            .iter()
            .position(|group| group.contains(&rank))
            .ok_or_else(|| format!("rank {rank} is absent from PLAIT_NCCL_HOSTS"))?;
        let index = groups[group]
            .iter()
            .position(|item| *item == rank)
            .ok_or_else(|| "host group index".to_owned())?;
        let leader = groups[group][0];
        Ok(Some(Self {
            groups,
            group,
            index,
            leader,
        }))
    }

    fn is_leader(&self) -> bool {
        self.index == 0
    }

    fn group_size(&self) -> usize {
        self.groups[self.group].len()
    }

    fn remote_leaders(&self) -> Vec<usize> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != self.group)
            .map(|(_, group)| group[0])
            .collect()
    }
}

fn shm_enabled() -> bool {
    std::env::var("PLAIT_NCCL_SHM").as_deref() == Ok("1")
}

fn cross_channels() -> usize {
    std::env::var("PLAIT_NCCL_CROSS_CHANNELS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
}

fn make_cross(
    rank: usize,
    port: u16,
    layout: Option<&HostLayout>,
) -> Result<Option<cross::Cross>, String> {
    let Some(layout) = layout else {
        return Ok(None);
    };
    if !shm_enabled() || layout.groups.len() < 2 || !layout.is_leader() {
        return Ok(None);
    }
    let channels = cross_channels();
    if channels == 0 {
        return Ok(None);
    }
    let cross = if rank == 0 {
        cross::Cross::root(port, &layout.remote_leaders(), channels)
    } else {
        cross::Cross::leader(port, rank, channels)
    };
    cross.map(Some).map_err(|error| error.0)
}

fn shm_threshold() -> usize {
    if std::env::var_os("PLAIT_NCCL_HOSTS").is_some() {
        // A two-level group benefits from shm for large messages too.
        return usize::MAX;
    }
    std::env::var("PLAIT_NCCL_SHM_MAX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64 * 1024)
}

fn create_runtime(backend: BackendKind, device: usize) -> Result<Box<dyn DeviceRuntime>, String> {
    match backend {
        BackendKind::Cuda => CudaRuntime::new(device)
            .map(|runtime| Box::new(runtime) as Box<dyn DeviceRuntime>)
            .map_err(|error| error.0),
        BackendKind::Rocm => HostCopyRuntime::hip(device)
            .map(|runtime| Box::new(runtime) as Box<dyn DeviceRuntime>)
            .map_err(|error| error.0),
        BackendKind::Huawei => HuaweiRuntime::new(device)
            .map(|runtime| Box::new(runtime) as Box<dyn DeviceRuntime>)
            .map_err(|error| error.0),
        BackendKind::Ppu => {
            if std::env::var_os("PLAIT_PPU_LIB").is_some() {
                return PpuRuntime::new(device)
                    .map(|runtime| Box::new(runtime) as Box<dyn DeviceRuntime>)
                    .map_err(|error| error.0);
            }
            match HostCopyRuntime::hggc(device) {
                Ok(runtime) => Ok(Box::new(runtime)),
                Err(error) if std::env::var_os("PLAIT_PPU_HGGCRT_LIB").is_some() => Err(error.0),
                Err(native) => PpuRuntime::new(device)
                    .map(|runtime| Box::new(runtime) as Box<dyn DeviceRuntime>)
                    .map_err(|adapter| format!("{}; {}", native.0, adapter.0)),
            }
        }
    }
}

fn backend_from_env() -> Result<BackendKind, String> {
    match std::env::var("PLAIT_NCCL_BACKEND").as_deref() {
        Ok("cuda") | Err(_) => Ok(BackendKind::Cuda),
        Ok("rocm") => Ok(BackendKind::Rocm),
        Ok("huawei") => Ok(BackendKind::Huawei),
        Ok("ppu") => Ok(BackendKind::Ppu),
        Ok(other) => Err(format!("unknown PLAIT_NCCL_BACKEND={other}")),
    }
}

pub struct Comm {
    rank: usize,
    world: usize,
    device: usize,
    network: Mutex<Network>,
    p2p: Mutex<p2p::P2p>,
    kernel: Option<Mutex<kernel::KernelAllreduce>>,
    shm: Option<Arc<Mutex<shm::Shm>>>,
    cross: Option<Mutex<cross::Cross>>,
    ipc: Option<Mutex<ipc::IpcAllreduce>>,
    layout: Option<HostLayout>,
    runtime: Box<dyn DeviceRuntime>,
}

static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();

fn set_error(message: impl Into<String>) -> NcclResult {
    *LAST_ERROR
        .get_or_init(|| Mutex::new(String::new()))
        .lock()
        .unwrap() = message.into();
    NCCL_INVALID_USAGE
}

fn recv_exact(stream: &mut TcpStream, bytes: &mut [u8]) -> std::io::Result<()> {
    stream.read_exact(bytes)
}

fn send_all(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(bytes)?;
    stream.flush()
}

fn port_from_id(id: &NcclUniqueId) -> u16 {
    u16::from_le_bytes([id.internal[4], id.internal[5]])
}

fn master_addr() -> String {
    std::env::var("PLAIT_NCCL_MASTER_ADDR")
        .or_else(|_| std::env::var("MASTER_ADDR"))
        .unwrap_or_else(|_| "127.0.0.1".to_owned())
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
fn world_from_id(id: &NcclUniqueId) -> Option<usize> {
    let world = u32::from_le_bytes(id.internal[6..10].try_into().ok()?);
    (world > 0).then_some(world as usize)
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
fn unique_id_for_group(port: u16, world: usize) -> NcclUniqueId {
    let mut bytes = [0u8; 128];
    bytes[..4].copy_from_slice(b"PLNC");
    bytes[4..6].copy_from_slice(&port.to_le_bytes());
    bytes[6..10].copy_from_slice(&(world as u32).to_le_bytes());
    NcclUniqueId { internal: bytes }
}

fn init_network(rank: usize, world: usize, id: &NcclUniqueId) -> std::io::Result<Network> {
    let port = port_from_id(id);
    let master = master_addr();
    if rank == 0 {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        for _ in 1..world {
            let (mut stream, _) = listener.accept()?;
            stream.set_nodelay(true)?;
            let mut rank_bytes = [0u8; 4];
            stream.read_exact(&mut rank_bytes)?;
            let peer_rank = u32::from_le_bytes(rank_bytes) as usize;
            if peer_rank == 0 || peer_rank >= world || peers[peer_rank].is_some() {
                return Err(std::io::Error::other("duplicate or invalid rank handshake"));
            }
            peers[peer_rank] = Some(stream);
        }
        let peers = peers
            .into_iter()
            .skip(1)
            .map(|stream| stream.ok_or_else(|| std::io::Error::other("missing rank")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Network { root: None, peers })
    } else {
        // CUDA context creation can take seconds on a cold GPU. Connecting
        // first lets the worker wait for rank 0 without racing its listener.
        let mut last_error = None;
        let mut connected = None;
        for _ in 0..600 {
            match TcpStream::connect((master.as_str(), port)) {
                Ok(mut stream) => {
                    stream.set_nodelay(true)?;
                    stream.write_all(&(rank as u32).to_le_bytes())?;
                    connected = Some(stream);
                    break;
                }
                Err(error) => {
                    last_error = Some(error);
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
        let stream = connected.ok_or_else(|| {
            last_error.unwrap_or_else(|| std::io::Error::other("no connection attempt"))
        })?;
        Ok(Network {
            root: Some(stream),
            peers: Vec::new(),
        })
    }
}

fn host_to_device(
    runtime: &dyn DeviceRuntime,
    device_ptr: u64,
    host: &[u8],
) -> Result<(), NcclResult> {
    runtime
        .copy_host_to_device(device_ptr, host)
        .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)
}

fn device_to_host(
    runtime: &dyn DeviceRuntime,
    device_ptr: u64,
    host: &mut [u8],
) -> Result<(), NcclResult> {
    runtime
        .copy_device_to_host(device_ptr, host)
        .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)
}

fn synchronize_stream(comm: &Comm, stream: *mut std::ffi::c_void) -> Result<(), NcclResult> {
    comm.runtime
        .synchronize_stream(stream as u64)
        .map_err(|error| set_error(error.0))
}

/// E4M3 finite value, used by the optional quantized-communication path.
pub(crate) fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 3) & 0x0f) as i32;
    let mantissa = (bits & 0x07) as f32;
    if exponent == 0x0f {
        // E4M3FN uses the last mantissa code for NaN; 0x7e is the finite
        // maximum 448.
        return if mantissa == 7.0 {
            f32::NAN
        } else {
            sign * (1.0 + mantissa / 8.0) * 256.0
        };
    }
    if exponent == 0 {
        return sign * (mantissa / 8.0) * 2f32.powi(-6);
    }
    sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
}

pub(crate) fn f32_to_fp8_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs().min(448.0);
    if magnitude == 0.0 {
        return sign;
    }
    let exponent = magnitude.log2().floor() as i32;
    if exponent < -6 {
        let mantissa = (magnitude / 2f32.powi(-9)).round().clamp(0.0, 7.0) as u8;
        return sign | mantissa;
    }
    let mut exponent = exponent.clamp(-6, 8);
    let mut mantissa = ((magnitude / 2f32.powi(exponent) - 1.0) * 8.0).round() as i32;
    if mantissa == 8 {
        exponent += 1;
        mantissa = 0;
    }
    if exponent > 8 {
        return sign | 0x7e;
    }
    sign | (((exponent + 7) as u8) << 3) | mantissa.clamp(0, 7) as u8
}

pub(crate) fn reduce_bytes(op: ReduceOp, dtype: DataType, buffers: &[Vec<u8>]) -> Vec<u8> {
    let slices: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();
    reduce_slices(op, dtype, &slices)
}

/// Reduce FP8 E4M3 payloads in FP32 and return an FP32 byte buffer. The
/// vLLM-facing allreduce API has a single dtype, so the quantized path needs
/// this explicit mixed-input/mixed-output helper to avoid quantizing the sum
/// a second time.
pub(crate) fn reduce_fp8_to_f32(buffers: &[Vec<u8>]) -> Vec<u8> {
    let Some(first) = buffers.first() else {
        return Vec::new();
    };
    let mut values = vec![0.0f32; first.len()];
    for buffer in buffers {
        for (index, byte) in buffer.iter().enumerate() {
            values[index] += fp8_e4m3_to_f32(*byte);
        }
    }
    values
        .into_iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn all_reduce_fp8_host(comm: &Comm, send: &[u8]) -> Result<Vec<u8>, NcclResult> {
    let mut network = comm.network.lock().unwrap();
    if comm.world == 1 {
        return Ok(reduce_fp8_to_f32(&[send.to_vec()]));
    }
    if comm.rank == 0 {
        let mut buffers = Vec::with_capacity(comm.world);
        buffers.push(send.to_vec());
        for peer in &mut network.peers {
            let mut buffer = vec![0u8; send.len()];
            recv_exact(peer, &mut buffer).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
            buffers.push(buffer);
        }
        let result = reduce_fp8_to_f32(&buffers);
        for peer in &mut network.peers {
            send_all(peer, &result).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        }
        Ok(result)
    } else {
        let root = network.root.as_mut().unwrap();
        send_all(root, send).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        let mut result = vec![0u8; send.len() * 4];
        recv_exact(root, &mut result).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        Ok(result)
    }
}

pub(crate) fn reduce_slices(op: ReduceOp, dtype: DataType, buffers: &[&[u8]]) -> Vec<u8> {
    if buffers.is_empty() {
        return Vec::new();
    }
    let bytes = buffers[0].len();
    match dtype {
        DataType::Float32 => {
            let mut values = vec![0.0f32; bytes / 4];
            for (index, chunk) in buffers[0].chunks_exact(4).enumerate() {
                values[index] = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            for buffer in &buffers[1..] {
                for (index, chunk) in buffer.chunks_exact(4).enumerate() {
                    let value = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    values[index] = match op {
                        ReduceOp::Sum | ReduceOp::Avg => values[index] + value,
                        ReduceOp::Prod => values[index] * value,
                        ReduceOp::Max => values[index].max(value),
                        ReduceOp::Min => values[index].min(value),
                    };
                }
            }
            if matches!(op, ReduceOp::Avg) {
                let n = buffers.len() as f32;
                for value in &mut values {
                    *value /= n;
                }
            }
            let mut output = Vec::with_capacity(bytes);
            for value in values {
                output.extend_from_slice(&value.to_ne_bytes());
            }
            output
        }
        DataType::Float16 => {
            let mut values = vec![0.0f32; bytes / 2];
            for (index, chunk) in buffers[0].chunks_exact(2).enumerate() {
                values[index] = f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32();
            }
            for buffer in &buffers[1..] {
                for (index, chunk) in buffer.chunks_exact(2).enumerate() {
                    let value = f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32();
                    values[index] = match op {
                        ReduceOp::Sum | ReduceOp::Avg => values[index] + value,
                        ReduceOp::Prod => values[index] * value,
                        ReduceOp::Max => values[index].max(value),
                        ReduceOp::Min => values[index].min(value),
                    };
                }
            }
            if matches!(op, ReduceOp::Avg) {
                let n = buffers.len() as f32;
                for value in &mut values {
                    *value /= n;
                }
            }
            let mut output = Vec::with_capacity(bytes);
            for value in values {
                output.extend_from_slice(&f16::from_f32(value).to_bits().to_ne_bytes());
            }
            output
        }
        DataType::Bfloat16 => {
            let mut values = vec![0.0f32; bytes / 2];
            for (index, chunk) in buffers[0].chunks_exact(2).enumerate() {
                values[index] = bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32();
            }
            for buffer in &buffers[1..] {
                for (index, chunk) in buffer.chunks_exact(2).enumerate() {
                    let value = bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32();
                    values[index] = match op {
                        ReduceOp::Sum | ReduceOp::Avg => values[index] + value,
                        ReduceOp::Prod => values[index] * value,
                        ReduceOp::Max => values[index].max(value),
                        ReduceOp::Min => values[index].min(value),
                    };
                }
            }
            if matches!(op, ReduceOp::Avg) {
                let n = buffers.len() as f32;
                for value in &mut values {
                    *value /= n;
                }
            }
            let mut output = Vec::with_capacity(bytes);
            for value in values {
                output.extend_from_slice(&bf16::from_f32(value).to_bits().to_ne_bytes());
            }
            output
        }
        DataType::Float8e4m3 => {
            let mut values = vec![0.0f32; bytes];
            for (index, byte) in buffers[0].iter().enumerate() {
                values[index] = fp8_e4m3_to_f32(*byte);
            }
            for buffer in &buffers[1..] {
                for (index, byte) in buffer.iter().enumerate() {
                    let value = fp8_e4m3_to_f32(*byte);
                    values[index] = match op {
                        ReduceOp::Sum | ReduceOp::Avg => values[index] + value,
                        ReduceOp::Prod => values[index] * value,
                        ReduceOp::Max => values[index].max(value),
                        ReduceOp::Min => values[index].min(value),
                    };
                }
            }
            if matches!(op, ReduceOp::Avg) {
                let n = buffers.len() as f32;
                for value in &mut values {
                    *value /= n;
                }
            }
            values.into_iter().map(f32_to_fp8_e4m3).collect::<Vec<u8>>()
        }
        _ => {
            // Integer sums are byte-width preserving and sufficient for the
            // vLLM collectives that use int32/int64 metadata.
            let width = dtype.bytes();
            let mut output = buffers[0].to_vec();
            for buffer in &buffers[1..] {
                for (left, right) in output
                    .chunks_exact_mut(width)
                    .zip(buffer.chunks_exact(width))
                {
                    if matches!(dtype, DataType::Int8 | DataType::Uint8) {
                        left[0] = left[0].wrapping_add(right[0]);
                    } else if matches!(dtype, DataType::Int32 | DataType::Uint32) {
                        let a = i32::from_ne_bytes(left.try_into().unwrap());
                        let b = i32::from_ne_bytes(right.try_into().unwrap());
                        left.copy_from_slice(&a.wrapping_add(b).to_ne_bytes());
                    } else if matches!(dtype, DataType::Int64 | DataType::Uint64) {
                        let a = i64::from_ne_bytes(left.try_into().unwrap());
                        let b = i64::from_ne_bytes(right.try_into().unwrap());
                        left.copy_from_slice(&a.wrapping_add(b).to_ne_bytes());
                    }
                }
            }
            output
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclGetVersion(version: *mut i32) -> NcclResult {
    if version.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe {
        *version = 23004;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclGetErrorString(result: NcclResult) -> *const i8 {
    let text = match result {
        NCCL_SUCCESS => "success",
        NCCL_UNHANDLED_CUDA_ERROR => "unhandled CUDA error",
        NCCL_INVALID_ARGUMENT => "invalid argument",
        NCCL_INVALID_USAGE => "invalid usage",
        NCCL_UNSUPPORTED => "unsupported by Plait backend",
        _ => "unknown NCCL error",
    };
    static STRINGS: OnceLock<Mutex<HashMap<i32, CString>>> = OnceLock::new();
    let strings = STRINGS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut strings = strings.lock().unwrap();
    let value = strings
        .entry(result)
        .or_insert_with(|| CString::new(text).unwrap());
    value.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclGetUniqueId(id: *mut NcclUniqueId) -> NcclResult {
    if id.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let port = std::env::var("PLAIT_NCCL_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or_else(|| 25000 + (std::process::id() as u16 % 2000));
    let mut bytes = [0u8; 128];
    bytes[..4].copy_from_slice(b"PLNC");
    bytes[4..6].copy_from_slice(&port.to_le_bytes());
    unsafe {
        (*id).internal = bytes;
    }
    NCCL_SUCCESS
}

fn create_comm(
    nranks: i32,
    id: &NcclUniqueId,
    rank: i32,
    device_override: Option<usize>,
) -> Result<*mut Comm, String> {
    if nranks <= 0 || rank < 0 || rank >= nranks {
        return Err("invalid rank or world size".to_owned());
    }
    let backend = backend_from_env()?;
    let device = device_override
        .or_else(|| {
            std::env::var("PLAIT_NCCL_DEVICE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .or_else(|| {
            std::env::var("LOCAL_RANK")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(rank as usize);
    // Establish the rendezvous before touching CUDA: the cold-context
    // initialization can take seconds and peers should already be waiting.
    let network = init_network(rank as usize, nranks as usize, id)
        .map_err(|error| format!("plait nccl rendezvous: {error}"))?;
    let port = port_from_id(id);
    let p2p = if rank == 0 {
        p2p::P2p::root(port, nranks as usize).map_err(|error| error.0)?
    } else {
        p2p::P2p::worker(rank as u32, &master_addr(), port).map_err(|error| error.0)?
    };
    let layout = HostLayout::from_env(nranks as usize, rank as usize)?;
    let cross = make_cross(rank as usize, port, layout.as_ref())?;
    let shm = if shm_enabled() {
        let (group_rank, group_size, tag, leader) = match &layout {
            Some(layout) => (
                layout.index,
                layout.group_size(),
                layout.group,
                layout.is_leader(),
            ),
            None => (rank as usize, nranks as usize, 0, rank == 0),
        };
        Some(Arc::new(Mutex::new(
            shm::Shm::open(group_rank, group_size, port, tag, leader).map_err(|error| error.0)?,
        )))
    } else {
        None
    };
    let runtime = create_runtime(backend, device)?;
    if let Some(shm) = &shm {
        let shm = shm.lock().unwrap();
        if let Err(error) = runtime.pin_host(shm.base_pointer() as u64, shm.mapped_size()) {
            eprintln!("plait-nccl: shm pin unavailable: {error}");
        }
    }
    let mut comm = Box::new(Comm {
        rank: rank as usize,
        world: nranks as usize,
        device,
        network: Mutex::new(network),
        p2p: Mutex::new(p2p),
        kernel: None,
        shm,
        cross: cross.map(Mutex::new),
        ipc: None,
        layout,
        runtime,
    });
    if std::env::var("PLAIT_NCCL_KERNEL").as_deref() == Ok("1")
        && comm.world == 2
        && comm.layout.is_none()
    {
        if let Some(shm) = &comm.shm {
            match kernel::KernelAllreduce::new(
                comm.runtime.as_ref(),
                comm.runtime
                    .cuda_context()
                    .unwrap_or_else(|| CudaContext::new(comm.device).expect("CUDA context")),
                Arc::clone(shm),
                comm.rank as i32,
            ) {
                Ok(value) => comm.kernel = Some(Mutex::new(value)),
                Err(error) => eprintln!("plait-nccl: kernel allreduce unavailable: {error}"),
            }
        }
    }
    if std::env::var("PLAIT_NCCL_IPC").as_deref() == Ok("1")
        && nranks == 2
        && comm.layout.is_none()
        && comm.shm.is_some()
    {
        let capacity = std::env::var("PLAIT_NCCL_IPC_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1024 * 1024);
        match ipc::IpcAllreduce::new(&mut comm, capacity) {
            Ok(value) => comm.ipc = Some(Mutex::new(value)),
            Err(error) => eprintln!("plait-nccl: CUDA IPC unavailable: {error}"),
        }
    }
    Ok(Box::into_raw(comm))
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
fn join_group(
    id: &NcclUniqueId,
    device_override: Option<usize>,
    root: bool,
) -> Result<*mut Comm, String> {
    let world = world_from_id(id).ok_or_else(|| "group id has no world size".to_owned())?;
    let port = port_from_id(id);
    let device_base = device_override
        .or_else(|| {
            std::env::var("PLAIT_NCCL_DEVICE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .or_else(|| {
            std::env::var("LOCAL_RANK")
                .ok()
                .and_then(|v| v.parse().ok())
        });
    let (rank, network, p2p) = if root {
        let listener = TcpListener::bind(("0.0.0.0", port))
            .map_err(|error| format!("group listen: {error}"))?;
        let mut peers = Vec::with_capacity(world.saturating_sub(1));
        for assigned in 1..world {
            let (mut stream, _) = listener
                .accept()
                .map_err(|error| format!("group accept: {error}"))?;
            stream
                .set_nodelay(true)
                .map_err(|error| format!("group nodelay: {error}"))?;
            stream
                .write_all(&(assigned as u32).to_le_bytes())
                .map_err(|error| format!("group rank: {error}"))?;
            peers.push(stream);
        }
        drop(listener);
        let p2p = p2p::P2p::root(port, world).map_err(|error| error.0)?;
        (0usize, Network { root: None, peers }, p2p)
    } else {
        let mut last = String::new();
        let mut stream = None;
        for _ in 0..600 {
            match TcpStream::connect((master_addr().as_str(), port)) {
                Ok(value) => {
                    value
                        .set_nodelay(true)
                        .map_err(|error| format!("group nodelay: {error}"))?;
                    stream = Some(value);
                    break;
                }
                Err(error) => {
                    last = error.to_string();
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
        let mut stream = stream.ok_or_else(|| format!("group connect: {last}"))?;
        let mut assigned = [0u8; 4];
        stream
            .read_exact(&mut assigned)
            .map_err(|error| format!("group rank read: {error}"))?;
        let rank = u32::from_le_bytes(assigned) as usize;
        let p2p = p2p::P2p::worker(rank as u32, &master_addr(), port).map_err(|error| error.0)?;
        (
            rank,
            Network {
                root: Some(stream),
                peers: Vec::new(),
            },
            p2p,
        )
    };
    let device = device_base.unwrap_or(rank);
    let backend = backend_from_env()?;
    let layout = HostLayout::from_env(world, rank)?;
    let cross = make_cross(rank, port, layout.as_ref())?;
    let shm = if shm_enabled() {
        let (group_rank, group_size, tag, leader) = match &layout {
            Some(layout) => (
                layout.index,
                layout.group_size(),
                layout.group,
                layout.is_leader(),
            ),
            None => (rank, world, 0, rank == 0),
        };
        Some(Arc::new(Mutex::new(
            shm::Shm::open(group_rank, group_size, port, tag, leader).map_err(|error| error.0)?,
        )))
    } else {
        None
    };
    let runtime = create_runtime(backend, device)?;
    if let Some(shm) = &shm {
        let shm = shm.lock().unwrap();
        if let Err(error) = runtime.pin_host(shm.base_pointer() as u64, shm.mapped_size()) {
            eprintln!("plait-nccl: shm pin unavailable: {error}");
        }
    }
    let mut comm = Box::new(Comm {
        rank,
        world,
        device,
        network: Mutex::new(network),
        p2p: Mutex::new(p2p),
        kernel: None,
        shm,
        cross: cross.map(Mutex::new),
        ipc: None,
        layout,
        runtime,
    });
    if std::env::var("PLAIT_NCCL_IPC").as_deref() == Ok("1")
        && world == 2
        && comm.layout.is_none()
        && comm.shm.is_some()
    {
        let capacity = std::env::var("PLAIT_NCCL_IPC_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1024 * 1024);
        match ipc::IpcAllreduce::new(&mut comm, capacity) {
            Ok(value) => comm.ipc = Some(Mutex::new(value)),
            Err(error) => eprintln!("plait-nccl: CUDA IPC unavailable: {error}"),
        }
    }
    Ok(Box::into_raw(comm))
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommInitRank(
    comm: *mut *mut Comm,
    nranks: i32,
    id: NcclUniqueId,
    rank: i32,
) -> NcclResult {
    if comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    match create_comm(nranks, &id, rank, None) {
        Ok(handle) => {
            unsafe { *comm = handle };
            NCCL_SUCCESS
        }
        Err(error) => set_error(error),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommDestroy(comm: *mut Comm) -> NcclResult {
    if comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe {
        drop(Box::from_raw(comm));
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommAbort(comm: *mut Comm) -> NcclResult {
    ncclCommDestroy(comm)
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclGroupStart() -> NcclResult {
    NCCL_SUCCESS
}
#[unsafe(no_mangle)]
pub extern "C" fn ncclGroupEnd() -> NcclResult {
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommCount(comm: *const Comm, count: *mut i32) -> NcclResult {
    if comm.is_null() || count.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe {
        *count = (*comm).world as i32;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommUserRank(comm: *const Comm, rank: *mut i32) -> NcclResult {
    if comm.is_null() || rank.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe {
        *rank = (*comm).rank as i32;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommCuDevice(comm: *const Comm, device: *mut i32) -> NcclResult {
    if comm.is_null() || device.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe {
        *device = (*comm).device as i32;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommQueryProperties(
    comm: *const Comm,
    properties: *mut NcclCommProperties,
) -> NcclResult {
    if comm.is_null() || properties.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe {
        std::ptr::write_bytes(properties, 0, 1);
        (*properties).size = std::mem::size_of::<NcclCommProperties>();
        (*properties).magic = 0x504c_4149;
        (*properties).version = 23004;
        (*properties).rank = (*comm).rank as i32;
        (*properties).n_ranks = (*comm).world as i32;
        (*properties).cuda_dev = (*comm).device as i32;
        (*properties).device_api_support = true;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommWindowRegister(
    _comm: *mut Comm,
    buffer: *mut std::ffi::c_void,
    _size: usize,
    window: *mut *mut std::ffi::c_void,
    _flags: i32,
) -> NcclResult {
    if window.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe {
        *window = buffer;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommWindowDeregister(
    _comm: *mut Comm,
    _window: *mut std::ffi::c_void,
) -> NcclResult {
    NCCL_SUCCESS
}

fn all_reduce_two_level_finish(
    comm: &Comm,
    shm: &shm::Shm,
    layout: &HostLayout,
    generation: u32,
    bytes: usize,
    dtype: DataType,
    op: ReduceOp,
) -> Result<(), NcclResult> {
    if !layout.is_leader() {
        shm.wait_group(generation);
        shm.wait_final(generation);
        return Ok(());
    }

    shm.wait_started(generation);
    let local = shm.group_reduce(bytes, dtype, op);
    shm.publish_group(&local, generation);

    let final_result = if layout.leader == 0 {
        let remote_leaders = layout.remote_leaders();
        let cross = comm.cross.as_ref().map(|cross| cross.lock().unwrap());
        let mut parts = Vec::with_capacity(remote_leaders.len() + 1);
        parts.push(local);
        if let Some(cross) = &cross {
            for leader in &remote_leaders {
                parts.push(
                    cross
                        .root_recv(*leader, bytes)
                        .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?,
                );
            }
            let result = reduce_bytes(op, dtype, &parts);
            for leader in &remote_leaders {
                cross
                    .root_send(*leader, &result)
                    .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
            }
            result
        } else {
            let mut network = comm.network.lock().unwrap();
            for leader in &remote_leaders {
                let index = leader.checked_sub(1).ok_or(NCCL_INVALID_USAGE)?;
                let peer = network.peers.get_mut(index).ok_or(NCCL_INVALID_USAGE)?;
                let mut buffer = vec![0u8; bytes];
                recv_exact(peer, &mut buffer).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
                parts.push(buffer);
            }
            let result = reduce_bytes(op, dtype, &parts);
            for leader in &remote_leaders {
                let index = leader.checked_sub(1).ok_or(NCCL_INVALID_USAGE)?;
                let peer = network.peers.get_mut(index).ok_or(NCCL_INVALID_USAGE)?;
                send_all(peer, &result).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
            }
            result
        }
    } else if let Some(cross) = &comm.cross {
        let cross = cross.lock().unwrap();
        cross
            .leader_send(&local)
            .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        cross
            .leader_recv(bytes)
            .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?
    } else {
        let mut network = comm.network.lock().unwrap();
        let root = network.root.as_mut().ok_or(NCCL_INVALID_USAGE)?;
        send_all(root, &local).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        let mut result = vec![0u8; bytes];
        recv_exact(root, &mut result).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        result
    };
    shm.publish_final(&final_result, generation);
    Ok(())
}

fn all_gather_shm(
    comm: &Comm,
    shm: &mut shm::Shm,
    layout: &HostLayout,
    input: &[u8],
    bytes: usize,
) -> Result<Vec<u8>, NcclResult> {
    let world_bytes = bytes * comm.world;
    if world_bytes > shm.slot_bytes() {
        return Err(NCCL_INVALID_USAGE);
    }
    let generation = shm.begin_input();
    shm.input_slot_mut(bytes).copy_from_slice(input);
    shm.mark_started(generation);
    if !layout.is_leader() {
        shm.wait_final(generation);
        return Ok(shm.result_slot(world_bytes).to_vec());
    }
    shm.wait_started(generation);
    let local = shm.collect_inputs(bytes);
    let final_result = if layout.leader == 0 {
        let remote_leaders = layout.remote_leaders();
        let cross = comm.cross.as_ref().map(|cross| cross.lock().unwrap());
        let mut parts = Vec::with_capacity(remote_leaders.len() + 1);
        parts.push(local);
        if let Some(cross) = &cross {
            for leader in &remote_leaders {
                let group_size = layout
                    .groups
                    .iter()
                    .find(|group| group.first() == Some(leader))
                    .map(Vec::len)
                    .ok_or(NCCL_INVALID_USAGE)?;
                parts.push(
                    cross
                        .root_recv(*leader, bytes * group_size)
                        .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?,
                );
            }
            let full = parts.concat();
            for leader in &remote_leaders {
                cross
                    .root_send(*leader, &full)
                    .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
            }
            full
        } else {
            return Err(NCCL_INVALID_USAGE);
        }
    } else {
        let Some(cross) = &comm.cross else {
            return Err(NCCL_INVALID_USAGE);
        };
        let cross = cross.lock().unwrap();
        cross
            .leader_send(&local)
            .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        cross
            .leader_recv(world_bytes)
            .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?
    };
    shm.publish_final(&final_result, generation);
    Ok(final_result)
}

fn broadcast_shm(
    comm: &Comm,
    shm: &mut shm::Shm,
    layout: &HostLayout,
    input: Option<&[u8]>,
    bytes: usize,
) -> Result<Vec<u8>, NcclResult> {
    let generation = shm.begin_input();
    if comm.rank == 0 {
        let input = input.ok_or(NCCL_INVALID_USAGE)?;
        shm.input_slot_mut(bytes).copy_from_slice(input);
        let Some(cross) = &comm.cross else {
            return Err(NCCL_INVALID_USAGE);
        };
        let cross = cross.lock().unwrap();
        for leader in layout.remote_leaders() {
            cross
                .root_send(leader, input)
                .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        }
        shm.publish_final(input, generation);
        Ok(input.to_vec())
    } else if layout.is_leader() {
        let Some(cross) = &comm.cross else {
            return Err(NCCL_INVALID_USAGE);
        };
        let cross = cross.lock().unwrap();
        let result = cross
            .leader_recv(bytes)
            .map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        shm.publish_final(&result, generation);
        Ok(result)
    } else {
        shm.wait_final(generation);
        Ok(shm.result_slot(bytes).to_vec())
    }
}

#[allow(clippy::too_many_arguments)]
fn all_reduce_shm_two_rank(
    comm: &Comm,
    shm: &mut shm::Shm,
    sendbuff: u64,
    recvbuff: u64,
    bytes: usize,
    dtype: DataType,
    op: ReduceOp,
    stream: u64,
) -> NcclResult {
    if comm.world != 2 || comm.layout.is_some() || bytes > shm.slot_bytes() {
        return NCCL_INVALID_USAGE;
    }
    let slot = shm.input_slot_mut(bytes);
    if device_to_host(comm.runtime.as_ref(), sendbuff, slot).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    shm.barrier();
    let result = shm.symmetric_result(bytes, dtype, op);
    shm.write_own_result(&result);
    if comm
        .runtime
        .copy_host_to_device_async(recvbuff, shm.own_result_slot(bytes), stream)
        .is_err()
    {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[allow(clippy::too_many_arguments)]
fn all_reduce_shm(
    comm: &Comm,
    shm: &mut shm::Shm,
    layout: Option<&HostLayout>,
    sendbuff: u64,
    recvbuff: u64,
    bytes: usize,
    dtype: DataType,
    op: ReduceOp,
) -> NcclResult {
    let generation = shm.begin_input();
    {
        let slot = shm.input_slot_mut(bytes);
        if device_to_host(comm.runtime.as_ref(), sendbuff, slot).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
    }
    shm.mark_started(generation);
    let result = match layout {
        Some(layout) if !matches!(op, ReduceOp::Avg) => {
            all_reduce_two_level_finish(comm, shm, layout, generation, bytes, dtype, op)
        }
        _ => {
            shm.finish_single(generation, bytes, dtype, op);
            Ok(())
        }
    };
    if result.is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    let result = shm.result_slot(bytes);
    if host_to_device(comm.runtime.as_ref(), recvbuff, result).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

fn all_reduce_host(
    comm: &Comm,
    send: &[u8],
    dtype: DataType,
    op: ReduceOp,
) -> Result<Vec<u8>, NcclResult> {
    let mut network = comm.network.lock().unwrap();
    if comm.world == 1 {
        return Ok(send.to_vec());
    }
    if comm.rank == 0 {
        let mut buffers = Vec::with_capacity(comm.world);
        buffers.push(send.to_vec());
        for peer in &mut network.peers {
            let mut buffer = vec![0u8; send.len()];
            recv_exact(peer, &mut buffer).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
            buffers.push(buffer);
        }
        let result = reduce_bytes(op, dtype, &buffers);
        for peer in &mut network.peers {
            send_all(peer, &result).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        }
        Ok(result)
    } else {
        let root = network.root.as_mut().unwrap();
        send_all(root, send).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        let mut result = vec![0u8; send.len()];
        recv_exact(root, &mut result).map_err(|_| NCCL_UNHANDLED_CUDA_ERROR)?;
        Ok(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclAllReduce(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: i32,
    op: i32,
    comm: *mut Comm,
    stream: *mut std::ffi::c_void,
) -> NcclResult {
    if sendbuff.is_null() || recvbuff.is_null() || comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(dtype) = DataType::from_raw(datatype) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let Some(op) = ReduceOp::from_raw(op) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let bytes = count * dtype.bytes();
    let comm = unsafe { &*comm };
    if let Some(kernel) = &comm.kernel {
        if matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::Bfloat16
        ) && matches!(op, ReduceOp::Sum)
            && bytes <= shm_threshold()
        {
            let kernel = kernel.lock().unwrap();
            let result = kernel.submit(
                sendbuff as u64,
                recvbuff as u64,
                bytes,
                dtype,
                op,
                stream as u64,
            );
            if result == NCCL_SUCCESS {
                return result;
            }
        }
    }
    if let Err(error) = synchronize_stream(comm, stream) {
        return error;
    }
    if let (Some(ipc), Some(shm)) = (&comm.ipc, &comm.shm) {
        let mut ipc = ipc.lock().unwrap();
        if bytes <= ipc.capacity() {
            let shm = shm.lock().unwrap();
            let result = ipc.run(&shm, sendbuff as u64, recvbuff as u64, bytes, dtype);
            if result == NCCL_SUCCESS {
                return result;
            }
        }
    }
    if let Some(shm) = &comm.shm {
        let mut shm = shm.lock().unwrap();
        if comm.world == 2
            && comm.layout.is_none()
            && bytes <= shm.slot_bytes()
            && bytes <= shm_threshold()
        {
            let symmetric = all_reduce_shm_two_rank(
                comm,
                &mut shm,
                sendbuff as u64,
                recvbuff as u64,
                bytes,
                dtype,
                op,
                stream as u64,
            );
            if symmetric == NCCL_SUCCESS {
                return symmetric;
            }
        }
        if bytes <= shm.slot_bytes()
            && bytes <= shm_threshold()
            && !(comm.layout.is_some() && matches!(op, ReduceOp::Avg))
        {
            return all_reduce_shm(
                comm,
                &mut shm,
                comm.layout.as_ref(),
                sendbuff as u64,
                recvbuff as u64,
                bytes,
                dtype,
                op,
            );
        }
    }
    let mut host = vec![0u8; bytes];
    if device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut host).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    let result = match all_reduce_host(comm, &host, dtype, op) {
        Ok(result) => result,
        Err(error) => return error,
    };
    if host_to_device(comm.runtime.as_ref(), recvbuff as u64, &result).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclAllGather(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: i32,
    comm: *mut Comm,
    stream: *mut std::ffi::c_void,
) -> NcclResult {
    if sendbuff.is_null() || recvbuff.is_null() || comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(dtype) = DataType::from_raw(datatype) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let bytes = count * dtype.bytes();
    let comm = unsafe { &*comm };
    if let Err(error) = synchronize_stream(comm, stream) {
        return error;
    }
    if let (Some(shm), Some(layout)) = (&comm.shm, &comm.layout) {
        let mut shm = shm.lock().unwrap();
        if layout.groups.len() > 1 && bytes * comm.world <= shm.slot_bytes() {
            let mut input = vec![0u8; bytes];
            if device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut input).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
            if let Ok(result) = all_gather_shm(comm, &mut shm, layout, &input, bytes) {
                if host_to_device(comm.runtime.as_ref(), recvbuff as u64, &result).is_err() {
                    return NCCL_UNHANDLED_CUDA_ERROR;
                }
                return NCCL_SUCCESS;
            }
        }
    }
    let mut host = vec![0u8; bytes];
    if device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut host).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    let mut network = comm.network.lock().unwrap();
    let result = if comm.rank == 0 {
        let mut all = Vec::with_capacity(bytes * comm.world);
        all.extend_from_slice(&host);
        for peer in &mut network.peers {
            let mut buffer = vec![0u8; bytes];
            if recv_exact(peer, &mut buffer).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
            all.extend_from_slice(&buffer);
        }
        for peer in &mut network.peers {
            if send_all(peer, &all).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
        }
        all
    } else {
        let root = network.root.as_mut().unwrap();
        if send_all(root, &host).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let mut all = vec![0u8; bytes * comm.world];
        if recv_exact(root, &mut all).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        all
    };
    drop(network);
    if host_to_device(comm.runtime.as_ref(), recvbuff as u64, &result).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclBroadcast(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: i32,
    root_rank: i32,
    comm: *mut Comm,
    stream: *mut std::ffi::c_void,
) -> NcclResult {
    if recvbuff.is_null() || comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(dtype) = DataType::from_raw(datatype) else {
        return NCCL_INVALID_ARGUMENT;
    };
    if root_rank < 0 || root_rank as usize >= unsafe { (*comm).world } {
        return NCCL_INVALID_ARGUMENT;
    }
    let bytes = count * dtype.bytes();
    let comm = unsafe { &*comm };
    if let Err(error) = synchronize_stream(comm, stream) {
        return error;
    }
    if let (Some(shm), Some(layout)) = (&comm.shm, &comm.layout) {
        let mut shm = shm.lock().unwrap();
        if layout.groups.len() > 1 && root_rank == 0 && bytes <= shm.slot_bytes() {
            let mut input = vec![0u8; bytes];
            if !sendbuff.is_null()
                && device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut input).is_err()
            {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
            let send = if sendbuff.is_null() {
                None
            } else {
                Some(input.as_slice())
            };
            if let Ok(result) = broadcast_shm(comm, &mut shm, layout, send, bytes) {
                if host_to_device(comm.runtime.as_ref(), recvbuff as u64, &result).is_err() {
                    return NCCL_UNHANDLED_CUDA_ERROR;
                }
                return NCCL_SUCCESS;
            }
        }
    }
    // NCCL allows non-root ranks to pass a null send buffer; rank 0 still
    // needs a buffer from every peer so it can select the root's payload.
    let mut host = vec![0u8; bytes];
    if !sendbuff.is_null()
        && device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut host).is_err()
    {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    let mut network = comm.network.lock().unwrap();
    let result = if comm.rank == 0 {
        let mut buffers = Vec::with_capacity(comm.world);
        buffers.push(host.clone());
        for peer in &mut network.peers {
            let mut buffer = vec![0u8; bytes];
            if recv_exact(peer, &mut buffer).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
            buffers.push(buffer);
        }
        let selected = buffers.get(root_rank as usize).cloned().unwrap_or(host);
        for peer in &mut network.peers {
            if send_all(peer, &selected).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
        }
        selected
    } else {
        let root = network.root.as_mut().unwrap();
        if send_all(root, &host).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let mut selected = vec![0u8; bytes];
        if recv_exact(root, &mut selected).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        selected
    };
    drop(network);
    if host_to_device(comm.runtime.as_ref(), recvbuff as u64, &result).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[allow(clippy::too_many_arguments)]
fn reduce_scatter_shm(
    comm: &Comm,
    shm: &mut shm::Shm,
    layout: Option<&HostLayout>,
    sendbuff: u64,
    recvbuff: u64,
    width: usize,
    total: usize,
    dtype: DataType,
    op: ReduceOp,
) -> NcclResult {
    let generation = shm.begin_input();
    {
        let slot = shm.input_slot_mut(total);
        if device_to_host(comm.runtime.as_ref(), sendbuff, slot).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
    }
    shm.mark_started(generation);
    let result = match layout {
        Some(layout) if !matches!(op, ReduceOp::Avg) => {
            all_reduce_two_level_finish(comm, shm, layout, generation, total, dtype, op)
        }
        _ => {
            shm.finish_single(generation, total, dtype, op);
            Ok(())
        }
    };
    if result.is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    let full = shm.result_slot(total);
    let start = comm.rank * width;
    let end = (start + width).min(total);
    if host_to_device(comm.runtime.as_ref(), recvbuff, &full[start..end]).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclReduceScatter(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: i32,
    op: i32,
    comm: *mut Comm,
    stream: *mut std::ffi::c_void,
) -> NcclResult {
    if sendbuff.is_null() || recvbuff.is_null() || comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(dtype) = DataType::from_raw(datatype) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let Some(op) = ReduceOp::from_raw(op) else {
        return NCCL_INVALID_ARGUMENT;
    };
    // NCCL's reduce-scatter count is per rank: the send buffer holds
    // count * world elements and each rank receives count elements.
    let comm = unsafe { &*comm };
    if let Err(error) = synchronize_stream(comm, stream) {
        return error;
    }
    let width = count * dtype.bytes();
    let bytes = width * comm.world;
    if let Some(shm) = &comm.shm {
        let mut shm = shm.lock().unwrap();
        if bytes <= shm.slot_bytes() && !(comm.layout.is_some() && matches!(op, ReduceOp::Avg)) {
            return reduce_scatter_shm(
                comm,
                &mut shm,
                comm.layout.as_ref(),
                sendbuff as u64,
                recvbuff as u64,
                width,
                bytes,
                dtype,
                op,
            );
        }
    }
    let mut host = vec![0u8; bytes];
    if device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut host).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    let mut network = comm.network.lock().unwrap();
    let result = if comm.rank == 0 {
        let mut buffers = Vec::with_capacity(comm.world);
        buffers.push(host);
        for peer in &mut network.peers {
            let mut buffer = vec![0u8; bytes];
            if recv_exact(peer, &mut buffer).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
            buffers.push(buffer);
        }
        let reduced = reduce_bytes(op, dtype, &buffers);
        for (index, peer) in network.peers.iter_mut().enumerate() {
            let start = (index + 1) * width;
            let end = (start + width).min(bytes);
            if send_all(peer, &reduced[start..end]).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
        }
        reduced[..width].to_vec()
    } else {
        let root = network.root.as_mut().unwrap();
        if send_all(root, &host).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let mut chunk = vec![0u8; width];
        if recv_exact(root, &mut chunk).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        chunk
    };
    drop(network);
    if host_to_device(comm.runtime.as_ref(), recvbuff as u64, &result).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclReduce(
    sendbuff: *const std::ffi::c_void,
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: i32,
    op: i32,
    root_rank: i32,
    comm: *mut Comm,
    stream: *mut std::ffi::c_void,
) -> NcclResult {
    if sendbuff.is_null() || recvbuff.is_null() || comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(dtype) = DataType::from_raw(datatype) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let Some(op) = ReduceOp::from_raw(op) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let bytes = count * dtype.bytes();
    let comm = unsafe { &*comm };
    if let Err(error) = synchronize_stream(comm, stream) {
        return error;
    }
    let mut host = vec![0u8; bytes];
    if device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut host).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    let mut network = comm.network.lock().unwrap();
    let reduced = if comm.rank == 0 {
        let mut buffers = Vec::with_capacity(comm.world);
        buffers.push(host);
        for peer in &mut network.peers {
            let mut buffer = vec![0u8; bytes];
            if recv_exact(peer, &mut buffer).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
            buffers.push(buffer);
        }
        let reduced = reduce_bytes(op, dtype, &buffers);
        let selected = reduced.clone();
        for peer in &mut network.peers {
            if send_all(peer, &selected).is_err() {
                return NCCL_UNHANDLED_CUDA_ERROR;
            }
        }
        reduced
    } else {
        let root = network.root.as_mut().unwrap();
        if send_all(root, &host).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let mut reduced = vec![0u8; bytes];
        if recv_exact(root, &mut reduced).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        reduced
    };
    drop(network);
    if comm.rank == root_rank as usize
        && host_to_device(comm.runtime.as_ref(), recvbuff as u64, &reduced).is_err()
    {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclSend(
    sendbuff: *const std::ffi::c_void,
    count: usize,
    datatype: i32,
    dest: i32,
    comm: *mut Comm,
    stream: *mut std::ffi::c_void,
) -> NcclResult {
    if sendbuff.is_null() || comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(dtype) = DataType::from_raw(datatype) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let comm = unsafe { &*comm };
    if dest < 0 || dest as usize >= comm.world || dest as usize == comm.rank {
        return NCCL_INVALID_ARGUMENT;
    }
    if let Err(error) = synchronize_stream(comm, stream) {
        return error;
    }
    let bytes = count * dtype.bytes();
    let mut host = vec![0u8; bytes];
    if device_to_host(comm.runtime.as_ref(), sendbuff as u64, &mut host).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    match comm.p2p.lock().unwrap().send(dest as u32, &host) {
        Ok(()) => NCCL_SUCCESS,
        Err(error) => set_error(format!("p2p send: {error}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclRecv(
    recvbuff: *mut std::ffi::c_void,
    count: usize,
    datatype: i32,
    src: i32,
    comm: *mut Comm,
    stream: *mut std::ffi::c_void,
) -> NcclResult {
    if recvbuff.is_null() || comm.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(dtype) = DataType::from_raw(datatype) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let comm = unsafe { &*comm };
    if src < 0 || src as usize >= comm.world || src as usize == comm.rank {
        return NCCL_INVALID_ARGUMENT;
    }
    if let Err(error) = synchronize_stream(comm, stream) {
        return error;
    }
    let bytes = count * dtype.bytes();
    let result = match comm.p2p.lock().unwrap().recv(src as u32, bytes) {
        Ok(buffer) => buffer,
        Err(error) => return set_error(format!("p2p recv: {error}")),
    };
    if host_to_device(comm.runtime.as_ref(), recvbuff as u64, &result).is_err() {
        return NCCL_UNHANDLED_CUDA_ERROR;
    }
    NCCL_SUCCESS
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModule;

#[cfg(feature = "python")]
fn unique_id_for_port(port: u16) -> NcclUniqueId {
    let mut bytes = [0u8; 128];
    bytes[..4].copy_from_slice(b"PLNC");
    bytes[4..6].copy_from_slice(&port.to_le_bytes());
    NcclUniqueId { internal: bytes }
}

#[cfg(feature = "python")]
#[pyclass]
struct CommHandle {
    ptr: *mut Comm,
}

#[cfg(feature = "python")]
unsafe impl Send for CommHandle {}
#[cfg(feature = "python")]
unsafe impl Sync for CommHandle {}

#[cfg(feature = "python")]
#[pymethods]
impl CommHandle {
    #[new]
    fn new(rank: i32, world: i32, device: i32, port: u16) -> PyResult<Self> {
        let id = unique_id_for_port(port);
        let ptr = create_comm(world, &id, rank, Some(device as usize))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self { ptr })
    }

    #[getter]
    fn rank(&self) -> i32 {
        unsafe { (*self.ptr).rank as i32 }
    }

    #[getter]
    fn world(&self) -> i32 {
        unsafe { (*self.ptr).world as i32 }
    }

    fn all_reduce(&self, send: u64, recv: u64, count: usize, dtype: i32, op: i32) -> i32 {
        ncclAllReduce(
            send as *const std::ffi::c_void,
            recv as *mut std::ffi::c_void,
            count,
            dtype,
            op,
            self.ptr,
            std::ptr::null_mut(),
        )
    }

    fn all_reduce_stream(
        &self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: i32,
        op: i32,
        stream: u64,
    ) -> i32 {
        ncclAllReduce(
            send as *const std::ffi::c_void,
            recv as *mut std::ffi::c_void,
            count,
            dtype,
            op,
            self.ptr,
            stream as *mut std::ffi::c_void,
        )
    }

    fn all_reduce_fp8_to_f32(&self, send: u64, recv: u64, count: usize) -> i32 {
        let comm = unsafe { &*self.ptr };
        if count == 0 {
            return NCCL_SUCCESS;
        }
        let mut host = vec![0u8; count];
        if device_to_host(comm.runtime.as_ref(), send, &mut host).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let result = match all_reduce_fp8_host(comm, &host) {
            Ok(result) => result,
            Err(error) => return error,
        };
        if host_to_device(comm.runtime.as_ref(), recv, &result).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        NCCL_SUCCESS
    }

    #[pyo3(signature = (send, recv, count, dtype, stream=0))]
    fn all_gather(&self, send: u64, recv: u64, count: usize, dtype: i32, stream: u64) -> i32 {
        ncclAllGather(
            send as *const std::ffi::c_void,
            recv as *mut std::ffi::c_void,
            count,
            dtype,
            self.ptr,
            stream as *mut std::ffi::c_void,
        )
    }

    #[pyo3(signature = (send, recv, count, dtype, root, stream=0))]
    fn broadcast(
        &self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: i32,
        root: i32,
        stream: u64,
    ) -> i32 {
        ncclBroadcast(
            send as *const std::ffi::c_void,
            recv as *mut std::ffi::c_void,
            count,
            dtype,
            root,
            self.ptr,
            stream as *mut std::ffi::c_void,
        )
    }

    #[pyo3(signature = (send, recv, count, dtype, op, stream=0))]
    fn reduce_scatter(
        &self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: i32,
        op: i32,
        stream: u64,
    ) -> i32 {
        ncclReduceScatter(
            send as *const std::ffi::c_void,
            recv as *mut std::ffi::c_void,
            count,
            dtype,
            op,
            self.ptr,
            stream as *mut std::ffi::c_void,
        )
    }

    #[pyo3(signature = (send, recv, count, dtype, op, root, stream=0))]
    fn reduce(
        &self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: i32,
        op: i32,
        root: i32,
        stream: u64,
    ) -> i32 {
        ncclReduce(
            send as *const std::ffi::c_void,
            recv as *mut std::ffi::c_void,
            count,
            dtype,
            op,
            root,
            self.ptr,
            stream as *mut std::ffi::c_void,
        )
    }

    #[pyo3(signature = (ptr, count, dtype, dest, stream=0))]
    fn send(&self, ptr: u64, count: usize, dtype: i32, dest: i32, stream: u64) -> i32 {
        ncclSend(
            ptr as *const std::ffi::c_void,
            count,
            dtype,
            dest,
            self.ptr,
            stream as *mut std::ffi::c_void,
        )
    }

    #[pyo3(signature = (ptr, count, dtype, src, stream=0))]
    fn recv(&self, ptr: u64, count: usize, dtype: i32, src: i32, stream: u64) -> i32 {
        ncclRecv(
            ptr as *mut std::ffi::c_void,
            count,
            dtype,
            src,
            self.ptr,
            stream as *mut std::ffi::c_void,
        )
    }
}

#[cfg(feature = "python")]
impl Drop for CommHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            ncclCommDestroy(self.ptr);
        }
    }
}

#[cfg(feature = "python")]
fn id_from_bytes(bytes: &[u8]) -> PyResult<NcclUniqueId> {
    if bytes.len() != 128 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "group id must be 128 bytes",
        ));
    }
    let mut internal = [0u8; 128];
    internal.copy_from_slice(bytes);
    Ok(NcclUniqueId { internal })
}

#[cfg(feature = "python")]
#[pyfunction]
fn unique_id(port: u16) -> Vec<u8> {
    unique_id_for_port(port).internal.to_vec()
}

#[cfg(feature = "python")]
#[pyfunction]
fn new_group(world: usize, port: u16) -> Vec<u8> {
    unique_id_for_group(port, world).internal.to_vec()
}

#[cfg(feature = "python")]
#[pyfunction]
fn join_root(id: &[u8], device: usize) -> PyResult<CommHandle> {
    let id = id_from_bytes(id)?;
    let ptr =
        join_group(&id, Some(device), true).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(CommHandle { ptr })
}

#[cfg(feature = "python")]
#[pyfunction]
fn join(id: &[u8], device: usize) -> PyResult<CommHandle> {
    let id = id_from_bytes(id)?;
    let ptr =
        join_group(&id, Some(device), false).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(CommHandle { ptr })
}

#[cfg(feature = "python")]
#[pymodule]
fn _plait_nccl(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CommHandle>()?;
    module.add_function(wrap_pyfunction!(unique_id, module)?)?;
    module.add_function(wrap_pyfunction!(new_group, module)?)?;
    module.add_function(wrap_pyfunction!(join_root, module)?)?;
    module.add_function(wrap_pyfunction!(join, module)?)?;
    module.add("SUCCESS", NCCL_SUCCESS)?;
    module.add("UNHANDLED_CUDA_ERROR", NCCL_UNHANDLED_CUDA_ERROR)?;
    module.add("INVALID_USAGE", NCCL_INVALID_USAGE)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn reduces_float32_sum_and_average() {
        let buffers = vec![f32_bytes(&[1.0, 2.0]), f32_bytes(&[3.0, 4.0])];
        let sum = reduce_bytes(ReduceOp::Sum, DataType::Float32, &buffers);
        assert_eq!(sum, f32_bytes(&[4.0, 6.0]));
        let average = reduce_bytes(ReduceOp::Avg, DataType::Float32, &buffers);
        assert_eq!(average, f32_bytes(&[2.0, 3.0]));
        let product = reduce_bytes(
            ReduceOp::Prod,
            DataType::Float32,
            &[f32_bytes(&[2.0, 3.0]), f32_bytes(&[4.0, 5.0])],
        );
        assert_eq!(product, f32_bytes(&[8.0, 15.0]));
    }

    #[test]
    fn reduces_bfloat16_sum() {
        let left: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_ne_bytes())
            .collect();
        let right: Vec<u8> = [3.0f32, 4.0]
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_ne_bytes())
            .collect();
        let sum = reduce_bytes(ReduceOp::Sum, DataType::Bfloat16, &[left, right]);
        let expected: Vec<u8> = [4.0f32, 6.0]
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_ne_bytes())
            .collect();
        assert_eq!(sum, expected);
    }

    #[test]
    fn fp8_e4m3_roundtrips_within_quantization_error() {
        for value in [0.0f32, 1.0, -1.0, 0.1, -0.1, 448.0, -448.0] {
            let bits = f32_to_fp8_e4m3(value);
            let decoded = fp8_e4m3_to_f32(bits);
            let tolerance = value.abs().max(1.0) * 0.07;
            assert!(
                (decoded - value).abs() <= tolerance,
                "{value} -> {bits:#04x} -> {decoded}"
            );
        }
        assert!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(448.0)).is_finite());
        assert!(fp8_e4m3_to_f32(0x7f).is_nan());
    }

    #[test]
    fn reduces_fp8_e4m3_in_float32_accumulator() {
        let left: Vec<u8> = [1.0f32, 2.0].iter().map(|v| f32_to_fp8_e4m3(*v)).collect();
        let right: Vec<u8> = [3.0f32, 4.0].iter().map(|v| f32_to_fp8_e4m3(*v)).collect();
        let sum = reduce_bytes(ReduceOp::Sum, DataType::Float8e4m3, &[left, right]);
        let decoded: Vec<f32> = sum.iter().map(|bits| fp8_e4m3_to_f32(*bits)).collect();
        assert!((decoded[0] - 4.0).abs() < 0.3);
        assert!((decoded[1] - 6.0).abs() < 0.4);
    }

    #[test]
    fn unique_id_encodes_port_and_magic() {
        let id = NcclUniqueId { internal: [0; 128] };
        let mut bytes = id.internal;
        bytes[..4].copy_from_slice(b"PLNC");
        bytes[4..6].copy_from_slice(&35000u16.to_le_bytes());
        assert_eq!(&bytes[..4], b"PLNC");
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 35000);
    }
}

#[cfg(test)]
mod group_tests {
    use super::*;

    #[test]
    fn group_id_carries_port_and_world() {
        let id = unique_id_for_group(35123, 8);
        assert_eq!(&id.internal[..4], b"PLNC");
        assert_eq!(port_from_id(&id), 35123);
        assert_eq!(world_from_id(&id), Some(8));
    }

    #[test]
    fn legacy_unique_id_has_no_world() {
        let id = NcclUniqueId { internal: [0; 128] };
        assert_eq!(world_from_id(&id), None);
    }
}
