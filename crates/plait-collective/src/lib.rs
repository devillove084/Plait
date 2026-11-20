//! Peer-accessible GPU collectives.
//!
//! This is deliberately smaller than NCCL's full source tree, but it exercises
//! the two primitives NCCL relies on on a single host: CUDA peer access and
//! peer-to-peer copies between primary contexts. The ring and recursive
//! doubling all-reduce implementations communicate only through peer memory;
//! no host staging is used after the initial H2D and final D2H.
//!
//! Algorithms are selected explicitly. A future topology-aware scheduler can
//! choose between them using the same `PeerGroup` interface.

#![allow(clippy::needless_range_loop)]

pub mod ipc;
pub mod topology;

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(not(plait_collective_cubin))]
use cudarc::nvrtc::compile_ptx;

#[cfg(not(plait_collective_cubin))]
const ADD_KERNEL: &str = r#"
extern "C" __global__ void plait_add_inplace(
    float* dst,
    const float* src,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}
"#;

/// A logical rank. Rank and GPU ordinal are distinct concepts; this implementation
/// assigns one rank per visible GPU on the local host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankId(pub u32);

impl RankId {
    pub const fn new(index: usize) -> Self {
        Self(index as u32)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for RankId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rank{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllReduceAlgorithm {
    /// Bandwidth-optimal ring reduce-scatter + all-gather.
    Ring,
    /// Log-step exchange; requires a power-of-two rank count.
    RecursiveDoubling,
    /// Reduce to the root, then broadcast; a chain/star reference.
    ReduceBroadcast,
    /// Binary tree reduce followed by tree broadcast.
    Tree,
}

#[derive(Debug)]
pub struct CollectiveError(String);

impl CollectiveError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for CollectiveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CollectiveError {}

type Result<T, E = CollectiveError> = std::result::Result<T, E>;

struct ChannelBuffers {
    channel: usize,
    start: usize,
    len: usize,
    width: usize,
    data: Vec<CudaSlice<f32>>,
    receive: Vec<CudaSlice<f32>>,
}

struct GpuRank {
    /// Channel 0 alias, kept for the single-stream path.
    stream: Arc<CudaStream>,
    streams: Box<[Arc<CudaStream>]>,
    function: CudaFunction,
    _module: Arc<CudaModule>,
}

#[derive(Clone, Debug)]
pub struct TraceEvent {
    pub name: &'static str,
    pub category: &'static str,
    pub rank: RankId,
    pub start_us: f64,
    pub duration_us: f64,
    pub bytes: usize,
    pub algorithm: Option<&'static str>,
    pub channel: usize,
    pub channels: usize,
}

struct TraceBuffer {
    epoch: Instant,
    events: Vec<TraceEvent>,
}

/// One process owning one peer-enabled context per GPU.
pub struct PeerGroup {
    ranks: Box<[GpuRank]>,
    channels: usize,
    trace: Option<Mutex<TraceBuffer>>,
}

impl PeerGroup {
    pub fn new(gpus: usize) -> Result<Self> {
        Self::new_with_channels(gpus, 1)
    }

    /// Create one primary context per rank and `channels` nonblocking streams
    /// per context. Channel 0 is the default stream.
    pub fn new_with_channels(gpus: usize, channels: usize) -> Result<Self> {
        let channels = channels.max(1);
        let available = CudaContext::device_count()
            .map_err(|error| CollectiveError::new(format!("CUDA device count: {error}")))?;
        if gpus < 2 || gpus > available as usize {
            return Err(CollectiveError::new(format!(
                "collectives need 2..={available} GPUs, requested {gpus}"
            )));
        }
        let mut ranks = Vec::with_capacity(gpus);
        for ordinal in 0..gpus {
            let ctx = CudaContext::new(ordinal).map_err(|error| {
                CollectiveError::new(format!("CUDA context {ordinal}: {error}"))
            })?;
            let mut streams = Vec::with_capacity(channels);
            let stream = ctx.default_stream();
            streams.push(stream.clone());
            for _ in 1..channels {
                streams.push(
                    ctx.new_stream()
                        .map_err(|error| CollectiveError::new(format!("new stream: {error}")))?,
                );
            }
            #[cfg(plait_collective_cubin)]
            let ptx = {
                use cudarc::nvrtc::Ptx;
                Ptx::from_binary(std::fs::read(env!("PLAIT_COLLECTIVE_CUBIN")).map_err(
                    |error| CollectiveError::new(format!("read collective cubin: {error}")),
                )?)
            };
            #[cfg(not(plait_collective_cubin))]
            let ptx = compile_ptx(ADD_KERNEL)
                .map_err(|error| CollectiveError::new(format!("NVRTC: {error:?}")))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|error| CollectiveError::new(format!("load module: {error}")))?;
            let function = module
                .load_function("plait_add_inplace")
                .map_err(|error| CollectiveError::new(format!("load kernel: {error}")))?;
            ranks.push(GpuRank {
                stream,
                streams: streams.into_boxed_slice(),
                function,
                _module: module,
            });
        }
        let group = Self {
            ranks: ranks.into_boxed_slice(),
            channels,
            trace: None,
        };
        group.peer_copy_check()?;
        Ok(group)
    }

    pub fn world_size(&self) -> usize {
        self.ranks.len()
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Select a conservative algorithm from the detected interconnect. A
    /// missing topology falls back to ring rather than failing a collective.
    pub fn recommended_allreduce(&self, bytes: usize) -> AllReduceAlgorithm {
        topology::GpuTopology::detect()
            .map(|topology| topology.recommended_allreduce(bytes))
            .unwrap_or(AllReduceAlgorithm::Ring)
    }

    /// Compatibility alias for callers that think in devices rather than ranks.
    pub fn gpus(&self) -> usize {
        self.world_size()
    }

    pub fn rank_count(&self) -> usize {
        self.world_size()
    }

    pub fn rank(&self, ordinal: usize) -> Option<RankId> {
        (ordinal < self.world_size()).then(|| RankId::new(ordinal))
    }

    pub fn ranks(&self) -> impl Iterator<Item = RankId> + '_ {
        (0..self.world_size()).map(RankId::new)
    }

    /// Enable the Chrome trace collector with a preallocated event capacity.
    /// Tracing is opt-in; the collective hot path has no mutex when disabled.
    pub fn enable_trace(&mut self, capacity: usize) {
        self.trace = Some(Mutex::new(TraceBuffer {
            epoch: Instant::now(),
            events: Vec::with_capacity(capacity),
        }));
    }

    pub fn trace_events(&self) -> Vec<TraceEvent> {
        self.trace
            .as_ref()
            .map(|trace| trace.lock().expect("trace mutex").events.clone())
            .unwrap_or_default()
    }

    /// Write a Chrome Trace Event JSON file (`chrome://tracing` or Perfetto).
    pub fn write_trace(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let Some(trace) = &self.trace else {
            return Err(CollectiveError::new("tracing is not enabled"));
        };
        let trace = trace.lock().expect("trace mutex");
        let mut output = String::with_capacity(128 + trace.events.len() * 192);
        output.push_str("{\"traceEvents\":[");
        for (index, event) in trace.events.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            write!(
                output,
                "{{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"X\",\"ts\":{:.3},\"dur\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"rank\":{},\"world_size\":{},\"bytes\":{},\"algorithm\":\"{}\",\"channel\":{},\"channels\":{}}}}}",
                event.name,
                event.category,
                event.start_us,
                event.duration_us,
                event.rank.index() * event.channels + event.channel,
                event.rank.index(),
                self.world_size(),
                event.bytes,
                event.algorithm.unwrap_or(""),
                event.channel,
                event.channels,
            )
            .map_err(|error| CollectiveError::new(format!("trace encoding: {error}")))?;
        }
        output.push_str("],\"displayTimeUnit\":\"ms\"}");
        std::fs::write(path, output)
            .map_err(|error| CollectiveError::new(format!("write trace: {error}")))
    }

    fn trace_start(&self) -> Option<Instant> {
        self.trace.as_ref().map(|_| Instant::now())
    }

    fn trace_record(
        &self,
        name: &'static str,
        category: &'static str,
        bytes: usize,
        algorithm: Option<&'static str>,
        channels: usize,
        started: Option<Instant>,
    ) {
        let (Some(started), Some(trace)) = (started, &self.trace) else {
            return;
        };
        let now = Instant::now();
        let mut trace = trace.lock().expect("trace mutex");
        let start_us = started.duration_since(trace.epoch).as_secs_f64() * 1e6;
        let duration_us = now.duration_since(started).as_secs_f64() * 1e6;
        let channels = channels.max(1);
        for rank in 0..self.world_size() {
            for channel in 0..channels {
                trace.events.push(TraceEvent {
                    name,
                    category,
                    rank: RankId::new(rank),
                    start_us,
                    duration_us,
                    bytes,
                    algorithm,
                    channel,
                    channels,
                });
            }
        }
    }

    /// Verify that a peer copy works in both directions. This is the first
    /// failure point when IOMMU/PCIe peer access is unavailable.
    pub fn peer_copy_check(&self) -> Result<()> {
        if self.ranks.len() < 2 {
            return Ok(());
        }
        let source = self.ranks[0]
            .stream
            .clone_htod(&[1.0f32, 2.0, 3.0, 4.0])
            .map_err(driver("source allocation"))?;
        let mut destination = self.ranks[1]
            .stream
            .alloc_zeros::<f32>(4)
            .map_err(driver("destination allocation"))?;
        self.copy(0, &source, 0..4, 1, &mut destination, 0..4)?;
        self.ranks[1]
            .stream
            .synchronize()
            .map_err(driver("peer check sync"))?;
        let values = self.ranks[1]
            .stream
            .clone_dtoh(&destination)
            .map_err(driver("peer check copy"))?;
        if values != [1.0, 2.0, 3.0, 4.0] {
            return Err(CollectiveError::new(format!(
                "peer copy returned {values:?}"
            )));
        }
        Ok(())
    }

    pub fn all_reduce(
        &self,
        inputs: &[Vec<f32>],
        algorithm: AllReduceAlgorithm,
    ) -> Result<Vec<Vec<f32>>> {
        let started = self.trace_start();
        let bytes = inputs.first().map_or(0, Vec::len) * std::mem::size_of::<f32>();
        let result = (|| {
            self.validate(inputs)?;
            let count = inputs[0].len();
            if count == 0 {
                return Ok(vec![Vec::new(); inputs.len()]);
            }
            if algorithm == AllReduceAlgorithm::RecursiveDoubling
                && self.ranks.len().is_power_of_two()
            {
                return self.all_reduce_recursive(inputs);
            }
            if algorithm == AllReduceAlgorithm::ReduceBroadcast {
                return self.all_reduce_reduce_broadcast(inputs);
            }
            if algorithm == AllReduceAlgorithm::Tree {
                return self.all_reduce_tree(inputs);
            }
            if self.channels > 1 {
                return self.all_reduce_ring_channels(inputs);
            }
            self.all_reduce_ring(inputs)
        })();
        if result.is_ok() {
            let name = match algorithm {
                AllReduceAlgorithm::Ring => "Ring",
                AllReduceAlgorithm::RecursiveDoubling => "RecursiveDoubling",
                AllReduceAlgorithm::ReduceBroadcast => "ReduceBroadcast",
                AllReduceAlgorithm::Tree => "Tree",
            };
            let channels = if algorithm == AllReduceAlgorithm::Ring {
                self.channels
            } else {
                1
            };
            self.trace_record(
                "all_reduce",
                "collective",
                bytes,
                Some(name),
                channels,
                started,
            );
        }
        result
    }

    pub fn broadcast(&self, value: Vec<f32>, root: usize) -> Result<Vec<Vec<f32>>> {
        let started = self.trace_start();
        let bytes = value.len() * std::mem::size_of::<f32>();
        let result = self.broadcast_inner(value, root);
        if result.is_ok() {
            self.trace_record("broadcast", "collective", bytes, None, 1, started);
        }
        result
    }

    fn broadcast_inner(&self, value: Vec<f32>, root: usize) -> Result<Vec<Vec<f32>>> {
        if root >= self.ranks.len() {
            return Err(CollectiveError::new("broadcast root out of range"));
        }
        let count = value.len();
        let mut data = Vec::with_capacity(self.ranks.len());
        let mut receive = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            let host = if rank == root {
                value.clone()
            } else {
                vec![0.0; count]
            };
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(&host)
                    .map_err(driver("broadcast allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(count)
                    .map_err(driver("broadcast receive"))?,
            );
        }
        for rank in 0..self.ranks.len() {
            if rank == root {
                continue;
            }
            self.copy(
                root,
                &data[root],
                0..count,
                rank,
                &mut receive[rank],
                0..count,
            )?;
            self.copy(
                rank,
                &receive[rank],
                0..count,
                rank,
                &mut data[rank],
                0..count,
            )?;
        }
        self.sync()?;
        self.collect(&data, count)
    }

    pub fn reduce(&self, inputs: &[Vec<f32>], root: usize) -> Result<Vec<f32>> {
        let started = self.trace_start();
        let bytes = inputs.first().map_or(0, Vec::len) * std::mem::size_of::<f32>();
        let result = self.reduce_inner(inputs, root);
        if result.is_ok() {
            self.trace_record("reduce", "collective", bytes, None, 1, started);
        }
        result
    }

    fn reduce_inner(&self, inputs: &[Vec<f32>], root: usize) -> Result<Vec<f32>> {
        self.validate(inputs)?;
        if root >= self.ranks.len() {
            return Err(CollectiveError::new("reduce root out of range"));
        }
        let count = inputs[0].len();
        let mut data = Vec::with_capacity(self.ranks.len());
        let mut receive = self.ranks[root]
            .stream
            .alloc_zeros::<f32>(count)
            .map_err(driver("reduce receive"))?;
        for (rank, input) in inputs.iter().enumerate() {
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(input)
                    .map_err(driver("reduce allocation"))?,
            );
        }
        for rank in 0..self.ranks.len() {
            if rank == root {
                continue;
            }
            self.copy(rank, &data[rank], 0..count, root, &mut receive, 0..count)?;
            self.add(root, &mut data[root], 0..count, &receive, 0..count)?;
            self.ranks[root]
                .stream
                .synchronize()
                .map_err(driver("reduce sync"))?;
        }
        let values = self.ranks[root]
            .stream
            .clone_dtoh(&data[root])
            .map_err(driver("reduce copy"))?;
        Ok(values[..count].to_vec())
    }

    /// Binary tree broadcast rooted at `root`; root 0 uses parent `r & (r-1)`.
    pub fn broadcast_tree(&self, value: Vec<f32>, root: usize) -> Result<Vec<Vec<f32>>> {
        if root >= self.world_size() {
            return Err(CollectiveError::new("broadcast root out of range"));
        }
        let started = self.trace_start();
        let bytes = value.len() * std::mem::size_of::<f32>();
        let result = self.broadcast_tree_inner(value, root);
        if result.is_ok() {
            self.trace_record("broadcast_tree", "collective", bytes, None, 1, started);
        }
        result
    }

    fn broadcast_tree_inner(&self, value: Vec<f32>, root: usize) -> Result<Vec<Vec<f32>>> {
        let n = self.world_size();
        let count = value.len();
        let mut data = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for rank in 0..n {
            let host = if rank == root {
                value.clone()
            } else {
                vec![0.0; count]
            };
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(&host)
                    .map_err(driver("tree broadcast allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(count)
                    .map_err(driver("tree broadcast receive"))?,
            );
        }
        for logical in 1..n {
            let rank = (root + logical) % n;
            let parent_logical = logical & (logical - 1);
            let parent = (root + parent_logical) % n;
            self.copy(
                parent,
                &data[parent],
                0..count,
                rank,
                &mut receive[rank],
                0..count,
            )?;
            self.copy(
                rank,
                &receive[rank],
                0..count,
                rank,
                &mut data[rank],
                0..count,
            )?;
            self.ranks[rank]
                .stream
                .synchronize()
                .map_err(driver("tree broadcast sync"))?;
        }
        self.collect(&data, count)
    }

    /// Binary tree reduce rooted at `root`; returns the reduced value on root.
    pub fn reduce_tree(&self, inputs: &[Vec<f32>], root: usize) -> Result<Vec<f32>> {
        self.validate(inputs)?;
        if root >= self.world_size() {
            return Err(CollectiveError::new("reduce root out of range"));
        }
        let started = self.trace_start();
        let bytes = inputs.first().map_or(0, Vec::len) * std::mem::size_of::<f32>();
        let result = self.reduce_tree_inner(inputs, root);
        if result.is_ok() {
            self.trace_record("reduce_tree", "collective", bytes, None, 1, started);
        }
        result
    }

    fn reduce_tree_inner(&self, inputs: &[Vec<f32>], root: usize) -> Result<Vec<f32>> {
        let n = self.world_size();
        let count = inputs[0].len();
        let mut data = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for rank in 0..n {
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(&inputs[rank])
                    .map_err(driver("tree reduce allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(count)
                    .map_err(driver("tree reduce receive"))?,
            );
        }
        for logical in (1..n).rev() {
            let rank = (root + logical) % n;
            let parent_logical = logical & (logical - 1);
            let parent = (root + parent_logical) % n;
            self.copy(
                rank,
                &data[rank],
                0..count,
                parent,
                &mut receive[parent],
                0..count,
            )?;
            self.add(
                parent,
                &mut data[parent],
                0..count,
                &receive[parent],
                0..count,
            )?;
            self.ranks[parent]
                .stream
                .synchronize()
                .map_err(driver("tree reduce sync"))?;
        }
        let values = self.ranks[root]
            .stream
            .clone_dtoh(&data[root])
            .map_err(driver("tree reduce copy"))?;
        Ok(values[..count].to_vec())
    }

    pub fn all_gather(&self, chunks: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let started = self.trace_start();
        let bytes =
            chunks.first().map_or(0, Vec::len) * self.world_size() * std::mem::size_of::<f32>();
        let result = self.all_gather_inner(chunks);
        if result.is_ok() {
            self.trace_record("all_gather", "collective", bytes, None, 1, started);
        }
        result
    }

    fn all_gather_inner(&self, chunks: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if chunks.len() != self.ranks.len() {
            return Err(CollectiveError::new(format!(
                "all_gather expects {} chunks, got {}",
                self.ranks.len(),
                chunks.len()
            )));
        }
        let width = chunks[0].len();
        if chunks.iter().any(|chunk| chunk.len() != width) {
            return Err(CollectiveError::new("all_gather chunks differ in size"));
        }
        let n = self.ranks.len();
        let mut data = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for rank in 0..n {
            let mut host = vec![0.0; width * n];
            host[rank * width..(rank + 1) * width].copy_from_slice(&chunks[rank]);
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(&host)
                    .map_err(driver("all_gather allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(width)
                    .map_err(driver("all_gather receive"))?,
            );
        }
        for step in 0..n - 1 {
            for rank in 0..n {
                let destination = (rank + 1) % n;
                let chunk = (rank + n - step) % n;
                let range = chunk * width..(chunk + 1) * width;
                self.copy(
                    rank,
                    &data[rank],
                    range.clone(),
                    destination,
                    &mut receive[destination],
                    0..width,
                )?;
                self.copy(
                    destination,
                    &receive[destination],
                    0..width,
                    destination,
                    &mut data[destination],
                    range,
                )?;
            }
        }
        self.sync()?;
        self.collect(&data, width * n)
    }

    pub fn reduce_scatter(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let started = self.trace_start();
        let bytes = inputs.first().map_or(0, Vec::len) * std::mem::size_of::<f32>();
        let result = self.reduce_scatter_inner(inputs);
        if result.is_ok() {
            self.trace_record("reduce_scatter", "collective", bytes, None, 1, started);
        }
        result
    }

    fn reduce_scatter_inner(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.validate(inputs)?;
        let n = self.ranks.len();
        let count = inputs[0].len();
        let width = count.div_ceil(n);
        let padded = width * n;
        let mut data = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for (rank, input) in inputs.iter().enumerate() {
            let mut host = input.clone();
            host.resize(padded, 0.0);
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(&host)
                    .map_err(driver("reduce_scatter allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(width)
                    .map_err(driver("reduce_scatter receive"))?,
            );
        }
        self.reduce_scatter_core(&mut data, &mut receive, width)?;
        self.sync()?;
        let mut output = Vec::with_capacity(n);
        for rank in 0..n {
            let host = self.ranks[rank]
                .stream
                .clone_dtoh(&data[rank])
                .map_err(driver("reduce_scatter copy"))?;
            output.push(host[..count].to_vec());
        }
        Ok(output)
    }

    pub fn all_to_all(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let started = self.trace_start();
        let bytes = inputs.first().map_or(0, Vec::len) * std::mem::size_of::<f32>();
        let result = self.all_to_all_inner(inputs);
        if result.is_ok() {
            self.trace_record("all_to_all", "collective", bytes, None, 1, started);
        }
        result
    }

    fn all_to_all_inner(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.validate(inputs)?;
        let n = self.ranks.len();
        let count = inputs[0].len();
        if count % n != 0 {
            return Err(CollectiveError::new(
                "all_to_all length must divide by ranks",
            ));
        }
        let width = count / n;
        let mut input = Vec::with_capacity(n);
        let mut output = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for rank in 0..n {
            input.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(&inputs[rank])
                    .map_err(driver("all_to_all input"))?,
            );
            output.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(count)
                    .map_err(driver("all_to_all output"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(width)
                    .map_err(driver("all_to_all receive"))?,
            );
        }
        for source in 0..n {
            for destination in 0..n {
                let send = destination * width..(destination + 1) * width;
                let recv = source * width..(source + 1) * width;
                self.copy(
                    source,
                    &input[source],
                    send,
                    destination,
                    &mut receive[destination],
                    0..width,
                )?;
                self.copy(
                    destination,
                    &receive[destination],
                    0..width,
                    destination,
                    &mut output[destination],
                    recv,
                )?;
            }
        }
        self.sync()?;
        self.collect(&output, count)
    }

    fn all_reduce_ring_channels(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = self.ranks.len();
        let count = inputs[0].len();
        let segment = count.div_ceil(self.channels);
        let mut channels = Vec::with_capacity(self.channels);
        for channel in 0..self.channels {
            let start = channel * segment;
            if start >= count {
                break;
            }
            let len = (count - start).min(segment);
            let width = len.div_ceil(n);
            let padded = width * n;
            let mut data = Vec::with_capacity(n);
            let mut receive = Vec::with_capacity(n);
            for (rank, input) in inputs.iter().enumerate() {
                let mut host = vec![0.0f32; padded];
                host[..len].copy_from_slice(&input[start..start + len]);
                data.push(
                    self.ranks[rank].streams[channel]
                        .clone_htod(&host)
                        .map_err(driver("channel allocation"))?,
                );
                receive.push(
                    self.ranks[rank].streams[channel]
                        .alloc_zeros::<f32>(width)
                        .map_err(driver("channel receive"))?,
                );
            }
            channels.push(ChannelBuffers {
                channel,
                start,
                len,
                width,
                data,
                receive,
            });
        }

        for buffers in &mut channels {
            self.reduce_scatter_core_on(
                buffers.channel,
                &mut buffers.data,
                &mut buffers.receive,
                buffers.width,
            )?;
            self.all_gather_after_reduce_on(
                buffers.channel,
                &mut buffers.data,
                &mut buffers.receive,
                buffers.width,
            )?;
        }
        self.sync()?;

        let mut output = vec![vec![0.0f32; count]; n];
        for buffers in &channels {
            for rank in 0..n {
                let host = self.ranks[rank].streams[buffers.channel]
                    .clone_dtoh(&buffers.data[rank])
                    .map_err(driver("channel copy"))?;
                output[rank][buffers.start..buffers.start + buffers.len]
                    .copy_from_slice(&host[..buffers.len]);
            }
        }
        Ok(output)
    }

    fn reduce_scatter_core_on(
        &self,
        channel: usize,
        data: &mut [CudaSlice<f32>],
        receive: &mut [CudaSlice<f32>],
        width: usize,
    ) -> Result<()> {
        let n = self.ranks.len();
        for step in 0..n - 1 {
            for rank in 0..n {
                let destination = (rank + 1) % n;
                let chunk = (rank + n - step) % n;
                let range = chunk * width..(chunk + 1) * width;
                self.copy_on(
                    channel,
                    rank,
                    &data[rank],
                    range.clone(),
                    destination,
                    &mut receive[destination],
                    0..width,
                )?;
                self.add_on(
                    channel,
                    destination,
                    &mut data[destination],
                    range,
                    &receive[destination],
                    0..width,
                )?;
            }
        }
        Ok(())
    }

    fn all_gather_after_reduce_on(
        &self,
        channel: usize,
        data: &mut [CudaSlice<f32>],
        receive: &mut [CudaSlice<f32>],
        width: usize,
    ) -> Result<()> {
        let n = self.ranks.len();
        for step in 0..n - 1 {
            for rank in 0..n {
                let destination = (rank + 1) % n;
                let chunk = (rank + n - step + 1) % n;
                let range = chunk * width..(chunk + 1) * width;
                self.copy_on(
                    channel,
                    rank,
                    &data[rank],
                    range.clone(),
                    destination,
                    &mut receive[destination],
                    0..width,
                )?;
                self.copy_on(
                    channel,
                    destination,
                    &receive[destination],
                    0..width,
                    destination,
                    &mut data[destination],
                    range,
                )?;
            }
        }
        Ok(())
    }

    fn all_reduce_ring(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = self.ranks.len();
        let count = inputs[0].len();
        let width = count.div_ceil(n);
        let padded = width * n;
        let mut data = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for (rank, input) in inputs.iter().enumerate() {
            let mut host = input.clone();
            host.resize(padded, 0.0);
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(&host)
                    .map_err(driver("all_reduce allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(width)
                    .map_err(driver("all_reduce receive"))?,
            );
        }
        self.reduce_scatter_core(&mut data, &mut receive, width)?;
        self.all_gather_after_reduce(&mut data, &mut receive, width)?;
        self.sync()?;
        let mut output = Vec::with_capacity(n);
        for rank in 0..n {
            let host = self.ranks[rank]
                .stream
                .clone_dtoh(&data[rank])
                .map_err(driver("all_reduce copy"))?;
            output.push(host[..count].to_vec());
        }
        Ok(output)
    }

    fn reduce_scatter_core(
        &self,
        data: &mut [CudaSlice<f32>],
        receive: &mut [CudaSlice<f32>],
        width: usize,
    ) -> Result<()> {
        let n = self.ranks.len();
        for step in 0..n - 1 {
            for rank in 0..n {
                let destination = (rank + 1) % n;
                let chunk = (rank + n - step) % n;
                let range = chunk * width..(chunk + 1) * width;
                self.copy(
                    rank,
                    &data[rank],
                    range.clone(),
                    destination,
                    &mut receive[destination],
                    0..width,
                )?;
                self.add(
                    destination,
                    &mut data[destination],
                    range,
                    &receive[destination],
                    0..width,
                )?;
            }
        }
        Ok(())
    }

    fn all_gather_after_reduce(
        &self,
        data: &mut [CudaSlice<f32>],
        receive: &mut [CudaSlice<f32>],
        width: usize,
    ) -> Result<()> {
        let n = self.ranks.len();
        for step in 0..n - 1 {
            for rank in 0..n {
                let destination = (rank + 1) % n;
                let chunk = (rank + n - step + 1) % n;
                let range = chunk * width..(chunk + 1) * width;
                self.copy(
                    rank,
                    &data[rank],
                    range.clone(),
                    destination,
                    &mut receive[destination],
                    0..width,
                )?;
                self.copy(
                    destination,
                    &receive[destination],
                    0..width,
                    destination,
                    &mut data[destination],
                    range,
                )?;
            }
        }
        Ok(())
    }

    fn all_reduce_recursive(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = self.ranks.len();
        let count = inputs[0].len();
        let mut data = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for (rank, input) in inputs.iter().enumerate() {
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(input)
                    .map_err(driver("recursive allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(count)
                    .map_err(driver("recursive receive"))?,
            );
        }
        let mut step = 1;
        while step < n {
            for rank in 0..n {
                let partner = rank ^ step;
                self.copy(
                    rank,
                    &data[rank],
                    0..count,
                    partner,
                    &mut receive[partner],
                    0..count,
                )?;
            }
            for rank in 0..n {
                self.add(rank, &mut data[rank], 0..count, &receive[rank], 0..count)?;
            }
            step <<= 1;
        }
        self.sync()?;
        self.collect(&data, count)
    }

    fn all_reduce_tree(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = self.ranks.len();
        let count = inputs[0].len();
        let mut data = Vec::with_capacity(n);
        let mut receive = Vec::with_capacity(n);
        for (rank, input) in inputs.iter().enumerate() {
            data.push(
                self.ranks[rank]
                    .stream
                    .clone_htod(input)
                    .map_err(driver("tree allocation"))?,
            );
            receive.push(
                self.ranks[rank]
                    .stream
                    .alloc_zeros::<f32>(count)
                    .map_err(driver("tree receive"))?,
            );
        }

        // Bottom-up reduction: rank r contributes to parent r & (r - 1).
        for rank in (1..n).rev() {
            let parent = rank & (rank - 1);
            self.copy(
                rank,
                &data[rank],
                0..count,
                parent,
                &mut receive[parent],
                0..count,
            )?;
            self.add(
                parent,
                &mut data[parent],
                0..count,
                &receive[parent],
                0..count,
            )?;
            self.ranks[parent]
                .stream
                .synchronize()
                .map_err(driver("tree reduce sync"))?;
        }
        // Top-down broadcast.
        for rank in 1..n {
            let parent = rank & (rank - 1);
            self.copy(
                parent,
                &data[parent],
                0..count,
                rank,
                &mut receive[rank],
                0..count,
            )?;
            self.copy(
                rank,
                &receive[rank],
                0..count,
                rank,
                &mut data[rank],
                0..count,
            )?;
            self.ranks[rank]
                .stream
                .synchronize()
                .map_err(driver("tree broadcast sync"))?;
        }
        let mut output = Vec::with_capacity(n);
        for rank in 0..n {
            let host = self.ranks[rank]
                .stream
                .clone_dtoh(&data[rank])
                .map_err(driver("tree copy"))?;
            output.push(host[..count].to_vec());
        }
        Ok(output)
    }

    fn all_reduce_reduce_broadcast(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let reduced = self.reduce_inner(inputs, 0)?;
        self.broadcast_inner(reduced, 0)
    }

    fn validate(&self, inputs: &[Vec<f32>]) -> Result<()> {
        if inputs.len() != self.ranks.len() {
            return Err(CollectiveError::new(format!(
                "expected {} inputs, got {}",
                self.ranks.len(),
                inputs.len()
            )));
        }
        let count = inputs[0].len();
        if inputs.iter().any(|input| input.len() != count) {
            return Err(CollectiveError::new("collective inputs differ in size"));
        }
        Ok(())
    }

    fn copy(
        &self,
        source: usize,
        source_buffer: &CudaSlice<f32>,
        source_range: std::ops::Range<usize>,
        destination: usize,
        destination_buffer: &mut CudaSlice<f32>,
        destination_range: std::ops::Range<usize>,
    ) -> Result<()> {
        self.copy_on(
            0,
            source,
            source_buffer,
            source_range,
            destination,
            destination_buffer,
            destination_range,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_on(
        &self,
        channel: usize,
        source: usize,
        source_buffer: &CudaSlice<f32>,
        source_range: std::ops::Range<usize>,
        destination: usize,
        destination_buffer: &mut CudaSlice<f32>,
        destination_range: std::ops::Range<usize>,
    ) -> Result<()> {
        let source_view = source_buffer.slice(source_range);
        let mut destination_view = destination_buffer.slice_mut(destination_range);
        let stream = &self.ranks[destination].streams[channel];
        stream
            .memcpy_dtod(&source_view, &mut destination_view)
            .map_err(driver("peer copy"))?;
        let _ = source;
        Ok(())
    }

    fn add(
        &self,
        rank: usize,
        destination: &mut CudaSlice<f32>,
        destination_range: std::ops::Range<usize>,
        source: &CudaSlice<f32>,
        source_range: std::ops::Range<usize>,
    ) -> Result<()> {
        self.add_on(
            0,
            rank,
            destination,
            destination_range,
            source,
            source_range,
        )
    }

    fn add_on(
        &self,
        channel: usize,
        rank: usize,
        destination: &mut CudaSlice<f32>,
        destination_range: std::ops::Range<usize>,
        source: &CudaSlice<f32>,
        source_range: std::ops::Range<usize>,
    ) -> Result<()> {
        let count = destination_range.len();
        if count == 0 {
            return Ok(());
        }
        let stream = self.ranks[rank].streams[channel].clone();
        let function = &self.ranks[rank].function;
        let mut destination_view = destination.slice_mut(destination_range);
        let source_view = source.slice(source_range);
        let count = count as i32;
        let mut launch = stream.launch_builder(function);
        launch.arg(&mut destination_view);
        launch.arg(&source_view);
        launch.arg(&count);
        // SAFETY: both views are live, same context, and the kernel bounds
        // checks `i < n`.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }
            .map_err(driver("add kernel"))?;
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        for rank in &self.ranks {
            for stream in &rank.streams {
                stream.synchronize().map_err(driver("collective sync"))?;
            }
        }
        Ok(())
    }

    fn collect(&self, data: &[CudaSlice<f32>], count: usize) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(data.len());
        for (rank, buffer) in data.iter().enumerate() {
            let host = self.ranks[rank]
                .stream
                .clone_dtoh(buffer)
                .map_err(driver("collective copy"))?;
            output.push(host[..count].to_vec());
        }
        Ok(output)
    }
}

fn driver(context: &'static str) -> impl Fn(cudarc::driver::DriverError) -> CollectiveError {
    move |error| CollectiveError::new(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    fn reference_sum(inputs: &[Vec<f32>]) -> Vec<f32> {
        let mut output = vec![0.0; inputs[0].len()];
        for input in inputs {
            for (slot, value) in output.iter_mut().zip(input) {
                *slot += value;
            }
        }
        output
    }

    #[test]
    fn ipc_handle_roundtrips() {
        let handle = crate::ipc::IpcHandle([0x5a; 64]);
        assert_eq!(
            crate::ipc::IpcHandle::from_hex(&handle.to_hex()).unwrap(),
            handle
        );
    }

    #[test]
    fn peer_collectives_match_cpu_reference_or_skip() {
        let group = match PeerGroup::new(2) {
            Ok(group) => group,
            Err(error) => {
                eprintln!("peer collectives skipped: {error}");
                return;
            }
        };
        let inputs = vec![
            (0..64).map(|index| index as f32).collect::<Vec<_>>(),
            (0..64)
                .map(|index| (index * 3 + 1) as f32)
                .collect::<Vec<_>>(),
        ];
        let expected = reference_sum(&inputs);
        for algorithm in [
            AllReduceAlgorithm::Ring,
            AllReduceAlgorithm::ReduceBroadcast,
            AllReduceAlgorithm::Tree,
        ] {
            let output = group.all_reduce(&inputs, algorithm).expect("all_reduce");
            for values in output {
                assert_eq!(values, expected);
            }
        }

        let reduced = group.reduce(&inputs, 0).expect("reduce");
        assert_eq!(reduced, expected);
        let tree_reduced = group.reduce_tree(&inputs, 0).expect("tree reduce");
        assert_eq!(tree_reduced, expected);
        let tree_broadcast = group
            .broadcast_tree(expected.clone(), 0)
            .expect("tree broadcast");
        for values in tree_broadcast {
            assert_eq!(values, expected);
        }
        let rooted = group
            .broadcast_tree(expected.clone(), 1)
            .expect("rooted tree broadcast");
        for values in rooted {
            assert_eq!(values, expected);
        }

        let broadcast = group.broadcast(expected.clone(), 0).expect("broadcast");
        for values in broadcast {
            assert_eq!(values, expected);
        }

        let chunks = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let gathered = group.all_gather(&chunks).expect("all_gather");
        for values in gathered {
            assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
        }

        let matrix = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let all_to_all = group.all_to_all(&matrix).expect("all_to_all");
        assert_eq!(all_to_all[0], vec![1.0, 2.0, 10.0, 20.0]);
        assert_eq!(all_to_all[1], vec![3.0, 4.0, 30.0, 40.0]);

        let reduced_scatter = group.reduce_scatter(&matrix).expect("reduce_scatter");
        // Ring reduce-scatter leaves rank r with the fully reduced chunk
        // (r + 1) % n; every other chunk is a partial.
        assert_eq!(reduced_scatter[0][2..4], [33.0, 44.0]);
        assert_eq!(reduced_scatter[1][0..2], [11.0, 22.0]);
    }

    #[test]
    fn all_collectives_match_on_all_visible_ranks_or_skip() {
        let available = CudaContext::device_count().unwrap_or(0).max(0) as usize;
        let ranks = available.min(8);
        let group = match PeerGroup::new(ranks) {
            Ok(group) => group,
            Err(error) => {
                eprintln!("multi-rank collectives skipped: {error}");
                return;
            }
        };
        let n = group.gpus();
        let width = 4usize;
        let matrix: Vec<Vec<f32>> = (0..n)
            .map(|rank| {
                (0..n * width)
                    .map(|index| (rank * 100 + index) as f32)
                    .collect()
            })
            .collect();

        let all_to_all = group.all_to_all(&matrix).expect("all_to_all");
        for destination in 0..n {
            for source in 0..n {
                let expected = &matrix[source][destination * width..(destination + 1) * width];
                assert_eq!(
                    &all_to_all[destination][source * width..(source + 1) * width],
                    expected
                );
            }
        }

        let gathered = group
            .all_gather(
                &(0..n)
                    .map(|rank| vec![rank as f32, (rank + 10) as f32])
                    .collect::<Vec<_>>(),
            )
            .expect("all_gather");
        for values in gathered {
            let expected: Vec<f32> = (0..n)
                .flat_map(|rank| [rank as f32, (rank + 10) as f32])
                .collect();
            assert_eq!(values, expected);
        }

        let scattered = group.reduce_scatter(&matrix).expect("reduce_scatter");
        for rank in 0..n {
            let chunk = (rank + 1) % n;
            let mut expected = vec![0.0; width];
            for source in 0..n {
                for (slot, value) in expected.iter_mut().enumerate() {
                    *value += matrix[source][chunk * width + slot];
                }
            }
            assert_eq!(
                &scattered[rank][chunk * width..(chunk + 1) * width],
                &expected[..]
            );
        }

        let reduced = group.reduce(&matrix, 0).expect("reduce");
        let mut expected = vec![0.0; n * width];
        for row in &matrix {
            for (slot, value) in expected.iter_mut().zip(row) {
                *slot += value;
            }
        }
        assert_eq!(reduced, expected);
    }

    #[test]
    fn chrome_trace_has_one_lane_per_rank_or_skips() {
        let mut group = match PeerGroup::new(2) {
            Ok(group) => group,
            Err(error) => {
                eprintln!("trace test skipped: {error}");
                return;
            }
        };
        group.enable_trace(8);
        group
            .all_reduce(&[vec![1.0, 2.0], vec![3.0, 4.0]], AllReduceAlgorithm::Ring)
            .expect("all_reduce");
        let events = group.trace_events();
        assert_eq!(events.len(), 2, "one trace record per rank");
        let path = std::env::temp_dir().join("plait-collective-trace.json");
        group.write_trace(&path).expect("trace write");
        let text = std::fs::read_to_string(&path).expect("trace read");
        assert!(text.starts_with("{\"traceEvents\":["));
        assert!(text.contains("\"name\":\"all_reduce\""));
        assert!(text.contains("\"tid\":0"));
        assert!(text.contains("\"tid\":1"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn channelized_allreduce_matches_cpu_reference_or_skips() {
        let mut group = match PeerGroup::new_with_channels(2, 2) {
            Ok(group) => group,
            Err(error) => {
                eprintln!("channelized allreduce skipped: {error}");
                return;
            }
        };
        let inputs = vec![
            (0..65).map(|index| index as f32).collect::<Vec<_>>(),
            (0..65)
                .map(|index| (index * 2 + 1) as f32)
                .collect::<Vec<_>>(),
        ];
        let expected = reference_sum(&inputs);
        group.enable_trace(16);
        let output = group
            .all_reduce(&inputs, AllReduceAlgorithm::Ring)
            .expect("channelized ring");
        for values in output {
            assert_eq!(values, expected);
        }
        let events = group.trace_events();
        assert_eq!(events.len(), 4, "two ranks x two channels");
        let mut tids: Vec<usize> = events
            .iter()
            .map(|event| event.rank.index() * event.channels + event.channel)
            .collect();
        tids.sort();
        assert_eq!(tids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn recursive_doubling_matches_ring_or_skips() {
        let group = match PeerGroup::new(2) {
            Ok(group) => group,
            Err(error) => {
                eprintln!("recursive allreduce skipped: {error}");
                return;
            }
        };
        let inputs = vec![vec![1.0; 32], vec![2.0; 32]];
        let expected = reference_sum(&inputs);
        let output = group
            .all_reduce(&inputs, AllReduceAlgorithm::RecursiveDoubling)
            .expect("recursive doubling");
        for values in output {
            assert_eq!(values, expected);
        }
    }
}
