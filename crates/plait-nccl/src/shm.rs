//! Single-host shared-memory all-reduce mailbox.
//!
//! Ranks on the same host exchange host-side buffers through one `/dev/shm`
//! file instead of the TCP star. The mailbox is deliberately size-capped;
//! larger messages fall back to the network path. A two-phase ready protocol
//! lets a host leader publish its local aggregate, participate in the
//! cross-host exchange, then publish the final aggregate.

use std::ffi::CString;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::{DataType, ReduceOp, reduce_slices};

const HEADER_BYTES: usize = 512;
const MAGIC: u32 = 0x504c_5348;
const GROUP_READY: usize = 256;
const FINAL_READY: usize = 260;

#[derive(Debug)]
pub struct ShmError(pub String);

impl std::fmt::Display for ShmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShmError {}

fn err(message: impl Into<String>) -> ShmError {
    ShmError(message.into())
}

pub struct Shm {
    fd: i32,
    map: *mut u8,
    size: usize,
    rank: usize,
    world: usize,
    slot_bytes: usize,
    generation: AtomicU32,
    barrier_generation: AtomicU32,
    path: CString,
    owner: bool,
}

unsafe impl Send for Shm {}
unsafe impl Sync for Shm {}

impl Shm {
    pub fn open(
        rank: usize,
        world: usize,
        port: u16,
        tag: usize,
        root: bool,
    ) -> Result<Self, ShmError> {
        let slot_bytes = std::env::var("PLAIT_NCCL_SHM_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if std::env::var_os("PLAIT_NCCL_HOSTS").is_some() {
                64 * 1024 * 1024
            } else {
                16 * 1024 * 1024
            });
        let size = HEADER_BYTES + world * slot_bytes;
        let path = format!("/dev/shm/plait-nccl-{port}-{tag}");
        let c_path = CString::new(path.clone()).map_err(|error| err(error.to_string()))?;
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(err(format!(
                "open {path}: {}",
                std::io::Error::last_os_error()
            )));
        }
        if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
            unsafe { libc::close(fd) };
            return Err(err(format!(
                "ftruncate {path}: {}",
                std::io::Error::last_os_error()
            )));
        }
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if map == libc::MAP_FAILED {
            unsafe { libc::close(fd) };
            return Err(err(format!(
                "mmap {path}: {}",
                std::io::Error::last_os_error()
            )));
        }
        if root {
            unsafe {
                std::ptr::write_bytes(map, 0, HEADER_BYTES);
                *(map.cast::<u32>()) = MAGIC;
            }
        } else {
            let mut waited = 0;
            while unsafe { *(map.cast::<u32>()) } != MAGIC {
                std::thread::sleep(std::time::Duration::from_millis(10));
                waited += 1;
                if waited > 3000 {
                    unsafe { libc::close(fd) };
                    return Err(err("timed out waiting for shared-memory header"));
                }
            }
        }
        Ok(Self {
            fd,
            map: map.cast(),
            size,
            rank,
            world,
            slot_bytes,
            generation: AtomicU32::new(0),
            barrier_generation: AtomicU32::new(0),
            path: c_path,
            owner: root,
        })
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub fn base_pointer(&self) -> usize {
        self.map as usize
    }

    pub fn mapped_size(&self) -> usize {
        self.size
    }

    #[allow(dead_code)]
    pub fn data_pointer(&self) -> u64 {
        unsafe { self.map.add(HEADER_BYTES) as u64 }
    }

    #[allow(dead_code)]
    pub fn data_bytes(&self) -> usize {
        self.world * self.slot_bytes
    }

    pub fn begin_input(&self) -> u32 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn input_slot_mut(&mut self, bytes: usize) -> &mut [u8] {
        assert!(bytes <= self.slot_bytes);
        unsafe { std::slice::from_raw_parts_mut(self.slot(self.rank), bytes) }
    }

    pub fn mark_started(&self, generation: u32) {
        self.started(self.rank).store(generation, Ordering::Release);
    }

    pub fn result_slot(&self, bytes: usize) -> &[u8] {
        assert!(bytes <= self.slot_bytes);
        unsafe { std::slice::from_raw_parts(self.slot(0), bytes) }
    }

    pub fn write_own_result(&self, result: &[u8]) {
        let own = self.slot(self.rank);
        unsafe {
            std::ptr::copy_nonoverlapping(result.as_ptr(), own, result.len());
        }
    }

    pub fn own_result_slot(&self, bytes: usize) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.slot(self.rank), bytes) }
    }

    pub fn symmetric_result(&self, bytes: usize, dtype: DataType, op: ReduceOp) -> Vec<u8> {
        let own = self.slot(self.rank);
        let peer = self.slot(1 - self.rank);
        let own_slice = unsafe { std::slice::from_raw_parts(own, bytes) };
        let peer_slice = unsafe { std::slice::from_raw_parts(peer, bytes) };
        reduce_slices(op, dtype, &[own_slice, peer_slice])
    }

    pub fn finish_single(&self, generation: u32, bytes: usize, dtype: DataType, op: ReduceOp) {
        if self.rank == 0 {
            self.wait_started(generation);
            let result = self.group_reduce(bytes, dtype, op);
            self.publish_final(&result, generation);
        } else {
            self.wait_final(generation);
        }
    }

    fn started(&self, rank: usize) -> &AtomicU32 {
        unsafe { &*(self.map.add(4 + rank * 4).cast::<AtomicU32>()) }
    }

    fn group_ready(&self) -> &AtomicU32 {
        unsafe { &*(self.map.add(GROUP_READY).cast::<AtomicU32>()) }
    }

    fn final_ready(&self) -> &AtomicU32 {
        unsafe { &*(self.map.add(FINAL_READY).cast::<AtomicU32>()) }
    }

    fn barrier_count(&self) -> &AtomicU32 {
        unsafe { &*(self.map.add(272).cast::<AtomicU32>()) }
    }

    fn barrier_sense(&self) -> &AtomicU32 {
        unsafe { &*(self.map.add(276).cast::<AtomicU32>()) }
    }

    pub fn barrier(&self) {
        let sense = self.barrier_sense().load(Ordering::Acquire);
        let arrived = self.barrier_count().fetch_add(1, Ordering::AcqRel) + 1;
        if arrived == self.world as u32 {
            self.barrier_count().store(0, Ordering::Release);
            self.barrier_sense().store(sense ^ 1, Ordering::Release);
        } else {
            while self.barrier_sense().load(Ordering::Acquire) == sense {
                std::hint::spin_loop();
            }
        }
        let _ = self.barrier_generation.fetch_add(1, Ordering::Relaxed);
    }

    fn slot(&self, rank: usize) -> *mut u8 {
        unsafe { self.map.add(HEADER_BYTES + rank * self.slot_bytes) }
    }

    pub fn wait_started(&self, generation: u32) {
        for peer in 0..self.world {
            while self.started(peer).load(Ordering::Acquire) != generation {
                std::hint::spin_loop();
            }
        }
    }

    pub fn collect_inputs(&self, bytes: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.world * bytes);
        for rank in 0..self.world {
            let pointer = self.slot(rank);
            let slice = unsafe { std::slice::from_raw_parts(pointer, bytes) };
            output.extend_from_slice(slice);
        }
        output
    }

    pub fn group_reduce(&self, bytes: usize, dtype: DataType, op: ReduceOp) -> Vec<u8> {
        let mut buffers = Vec::with_capacity(self.world);
        for rank in 0..self.world {
            let pointer = self.slot(rank);
            buffers.push(unsafe { std::slice::from_raw_parts(pointer, bytes) });
        }
        reduce_slices(op, dtype, &buffers)
    }

    pub fn publish_group(&self, result: &[u8], generation: u32) {
        unsafe {
            std::ptr::copy_nonoverlapping(result.as_ptr(), self.slot(0), result.len());
        }
        self.group_ready().store(generation, Ordering::Release);
    }

    pub fn wait_group(&self, generation: u32) {
        while self.group_ready().load(Ordering::Acquire) != generation {
            std::hint::spin_loop();
        }
    }

    pub fn publish_final(&self, result: &[u8], generation: u32) {
        unsafe {
            std::ptr::copy_nonoverlapping(result.as_ptr(), self.slot(0), result.len());
        }
        self.final_ready().store(generation, Ordering::Release);
    }

    pub fn wait_final(&self, generation: u32) {
        while self.final_ready().load(Ordering::Acquire) != generation {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Shm {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.map.cast(), self.size);
            libc::close(self.fd);
            if self.owner {
                libc::unlink(self.path.as_ptr());
            }
        }
    }
}
