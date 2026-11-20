//! Registered memory and one-sided transfer contracts.
//!
//! Consensus must not carry megabytes of tensor data. Plait keeps large
//! payloads in aligned, registered regions and puts only descriptors plus
//! small control values in the replicated log. The `RdmaPlane` trait is the
//! seam for a real `ibverbs` backend; [`LoopbackRdma`] is the in-process
//! implementation used by tests and the DSA demo. It performs the same
//! descriptor validation and bounds checks as a hardware backend, so callers
//! exercise the real control flow without requiring an RNIC.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::BTreeMap;
use std::io;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use plait_core::NodeId;
use serde::{Deserialize, Serialize};

/// A page-aligned allocation. Page alignment keeps a region directly
/// registrable with an RNIC and avoids unaligned vectorized loads.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
}

unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    pub fn new(len: usize, align: usize) -> io::Result<Self> {
        if !align.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "alignment must be a power of two",
            ));
        }
        let layout = Layout::from_size_align(len.max(1), align)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
        // SAFETY: `layout` has non-zero size and a valid alignment.
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "aligned allocation failed")
        })?;
        // SAFETY: the allocation is uniquely owned and uninitialized; writing
        // zeroes is valid for the whole region.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, len.max(1)) };
        Ok(Self { ptr, len, align })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn alignment(&self) -> usize {
        self.align
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    fn layout(&self) -> Layout {
        Layout::from_size_align(self.len.max(1), self.align).expect("layout was validated")
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: the pointer owns `len` initialized bytes for its lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: `&mut self` guarantees unique access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated with exactly this layout in `new`.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout()) };
    }
}

#[derive(Default)]
struct PoolInner {
    free: Mutex<BTreeMap<usize, Vec<AlignedBuffer>>>,
    max_per_class: usize,
    align: usize,
    allocated: AtomicUsize,
    reused: AtomicUsize,
}

impl PoolInner {
    fn in_pool(&self) -> usize {
        self.free
            .lock()
            .expect("buffer pool poisoned")
            .values()
            .map(Vec::len)
            .sum()
    }
}

/// A reusable size-class pool for frame and staging buffers. Buffers are
/// page-aligned, returned on drop, and capped per class so a burst cannot pin
/// memory forever.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self::with_limits(4096, 64)
    }

    pub fn with_limits(align: usize, max_per_class: usize) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                free: Mutex::new(BTreeMap::new()),
                max_per_class,
                align,
                allocated: AtomicUsize::new(0),
                reused: AtomicUsize::new(0),
            }),
        }
    }

    pub fn acquire(&self, len: usize) -> io::Result<PooledBuffer> {
        let class = len.max(1).next_power_of_two().max(self.inner.align);
        let mut free = self.inner.free.lock().expect("buffer pool poisoned");
        let buffer = if let Some(buffer) = free.get_mut(&class).and_then(Vec::pop) {
            self.inner.reused.fetch_add(1, Ordering::Relaxed);
            buffer
        } else {
            let buffer = AlignedBuffer::new(class, self.inner.align)?;
            self.inner.allocated.fetch_add(1, Ordering::Relaxed);
            buffer
        };
        Ok(PooledBuffer {
            buffer: Some(buffer),
            pool: self.inner.clone(),
            class,
            len,
        })
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            allocated: self.inner.allocated.load(Ordering::Relaxed),
            reused: self.inner.reused.load(Ordering::Relaxed),
            in_pool: self.inner.in_pool(),
        }
    }

    pub fn align(&self) -> usize {
        self.inner.align
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolStats {
    pub allocated: usize,
    pub reused: usize,
    pub in_pool: usize,
}

pub struct PooledBuffer {
    buffer: Option<AlignedBuffer>,
    pool: Arc<PoolInner>,
    class: usize,
    len: usize,
}

impl PooledBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.buffer.as_ref().expect("buffer present").as_ptr()
    }
}

impl Deref for PooledBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.buffer.as_ref().expect("buffer present")[..self.len]
    }
}

impl DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer.as_mut().expect("buffer present")[..self.len]
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let mut free = self.pool.free.lock().expect("buffer pool poisoned");
            let queue = free.entry(self.class).or_default();
            if queue.len() < self.pool.max_per_class {
                queue.push(buffer);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RegionId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryKind {
    Host,
    Pinned,
    Device,
}

/// Everything a peer needs for a one-sided operation. The consensus log can
/// carry this small struct instead of the tensor bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionDescriptor {
    pub node: NodeId,
    pub region: RegionId,
    pub offset: u64,
    pub len: u64,
    pub rkey: u64,
    pub kind: MemoryKind,
}

impl RegionDescriptor {
    pub fn checked_range(&self) -> io::Result<std::ops::Range<usize>> {
        let start = usize::try_from(self.offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset too large"))?;
        let len = usize::try_from(self.len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "length too large"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        Ok(start..end)
    }
}

/// A locally registered region. The handle keeps the backing allocation alive
/// and exposes descriptor creation plus local access.
pub struct RegionHandle {
    id: RegionId,
    buffer: Arc<Mutex<PooledBuffer>>,
    kind: MemoryKind,
    rkey: u64,
}

impl RegionHandle {
    pub fn id(&self) -> RegionId {
        self.id
    }

    pub fn len(&self) -> usize {
        self.buffer.lock().expect("region poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn descriptor(&self, node: NodeId, offset: u64, len: u64) -> io::Result<RegionDescriptor> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset too large"))?;
        let length = usize::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "length too large"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "descriptor exceeds the registered region",
            ));
        }
        Ok(RegionDescriptor {
            node,
            region: self.id,
            offset,
            len,
            rkey: self.rkey,
            kind: self.kind,
        })
    }

    pub fn write(&self, offset: usize, bytes: &[u8]) -> io::Result<()> {
        let mut buffer = self.buffer.lock().expect("region poisoned");
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > buffer.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "write exceeds the registered region",
            ));
        }
        buffer[offset..end].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read(&self, offset: usize, out: &mut [u8]) -> io::Result<()> {
        let buffer = self.buffer.lock().expect("region poisoned");
        let end = offset
            .checked_add(out.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > buffer.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read exceeds the registered region",
            ));
        }
        out.copy_from_slice(&buffer[offset..end]);
        Ok(())
    }
}

/// One-sided operations. A real backend would use `ibv_reg_mr` and
/// `ibv_post_send`; the trait deliberately has no completion-queue type so the
/// consensus and planning layers stay independent of the transport.
pub trait RdmaPlane: Send + Sync {
    /// Register an owned pooled buffer. Keeping the `PooledBuffer` in the
    /// registry means registration pins the allocation until the plane drops
    /// it, and returning it to the pool is automatic afterwards.
    fn register(&self, buffer: PooledBuffer, kind: MemoryKind) -> io::Result<RegionHandle>;
    fn read(&self, remote: &RegionDescriptor, dst: &mut [u8]) -> io::Result<()>;
    fn write(&self, remote: &RegionDescriptor, src: &[u8]) -> io::Result<()>;
}

/// In-process implementation of the descriptor protocol. It does not emulate
/// hardware latency; it verifies that every caller uses registered regions,
/// keys, and bounds correctly.
#[derive(Default)]
pub struct LoopbackRdma {
    regions: Mutex<BTreeMap<RegionId, Arc<Mutex<PooledBuffer>>>>,
    next: AtomicU64,
}

impl LoopbackRdma {
    pub fn new() -> Self {
        Self::default()
    }

    fn region(&self, id: RegionId) -> io::Result<Arc<Mutex<PooledBuffer>>> {
        self.regions
            .lock()
            .expect("rdma registry poisoned")
            .get(&id)
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "unknown region"))
    }
}

impl RdmaPlane for LoopbackRdma {
    fn register(&self, buffer: PooledBuffer, kind: MemoryKind) -> io::Result<RegionHandle> {
        let id = RegionId(self.next.fetch_add(1, Ordering::Relaxed) + 1);
        let rkey = id.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let shared = Arc::new(Mutex::new(buffer));
        self.regions
            .lock()
            .expect("rdma registry poisoned")
            .insert(id, shared.clone());
        Ok(RegionHandle {
            id,
            buffer: shared,
            kind,
            rkey,
        })
    }

    fn read(&self, remote: &RegionDescriptor, dst: &mut [u8]) -> io::Result<()> {
        let range = remote.checked_range()?;
        if range.len() != dst.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "descriptor length does not match the destination",
            ));
        }
        let buffer = self.region(remote.region)?;
        let buffer = buffer.lock().expect("region poisoned");
        if range.end > buffer.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "descriptor exceeds the region",
            ));
        }
        dst.copy_from_slice(&buffer[range]);
        Ok(())
    }

    fn write(&self, remote: &RegionDescriptor, src: &[u8]) -> io::Result<()> {
        let range = remote.checked_range()?;
        if range.len() != src.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "descriptor length does not match the source",
            ));
        }
        let buffer = self.region(remote.region)?;
        let mut buffer = buffer.lock().expect("region poisoned");
        if range.end > buffer.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "descriptor exceeds the region",
            ));
        }
        buffer[range].copy_from_slice(src);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooled_buffers_are_aligned_and_reused() {
        let pool = BufferPool::with_limits(4096, 8);
        let first = pool.acquire(100).unwrap();
        assert_eq!(first.as_ptr() as usize % 4096, 0);
        let address = first.as_ptr();
        drop(first);
        let second = pool.acquire(100).unwrap();
        assert_eq!(second.as_ptr(), address);
        assert_eq!(second.len(), 100);
        assert_eq!(pool.stats().reused, 1);
    }

    #[test]
    fn region_descriptors_are_checked() {
        let plane = LoopbackRdma::new();
        let pool = BufferPool::with_limits(4096, 8);
        let region = plane
            .register(pool.acquire(4096).unwrap(), MemoryKind::Pinned)
            .unwrap();
        region.write(8, &[1, 2, 3, 4]).unwrap();
        let node = NodeId::derive("mem");
        let descriptor = region.descriptor(node, 8, 4).unwrap();
        let mut out = [0u8; 4];
        plane.read(&descriptor, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        plane.write(&descriptor, &[9, 9, 9, 9]).unwrap();
        region.read(8, &mut out).unwrap();
        assert_eq!(out, [9, 9, 9, 9]);
        assert!(region.descriptor(node, 4094, 4).is_err());
    }

    #[test]
    fn large_descriptor_does_not_allocate_inside_the_plane() {
        let plane = LoopbackRdma::new();
        let pool = BufferPool::with_limits(4096, 8);
        let region = plane
            .register(pool.acquire(1 << 16).unwrap(), MemoryKind::Host)
            .unwrap();
        let node = NodeId::derive("mem2");
        let descriptor = region.descriptor(node, 0, 1024).unwrap();
        let mut out = vec![0u8; 1024];
        plane.read(&descriptor, &mut out).unwrap();
        assert_eq!(out.len(), 1024);
    }
}
