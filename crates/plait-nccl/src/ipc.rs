//! Two-process CUDA IPC all-reduce for small messages.
//!
//! Both ranks allocate a local and a receive region, exchange the receive
//! handles over the existing control stream, copy their local buffer into the
//! peer's receive region, barrier, add the peer data on the GPU, then copy the
//! result back to the caller's tensor. No host staging on the data path.

use std::io::{Read, Write};

use plait_collective::ipc::{AddKernel, IpcHandle, PeerRegion};

use crate::{
    Comm, DataType, NCCL_INVALID_USAGE, NCCL_SUCCESS, NCCL_UNHANDLED_CUDA_ERROR, NcclResult,
};

pub struct IpcAllreduce {
    local: PeerRegion,
    receive: PeerRegion,
    peer_receive: PeerRegion,
    kernel: AddKernel,
    capacity: usize,
}

impl IpcAllreduce {
    pub fn new(comm: &mut Comm, capacity: usize) -> Result<Self, String> {
        let local = PeerRegion::alloc(comm.device, capacity).map_err(|error| error.to_string())?;
        let receive =
            PeerRegion::alloc(comm.device, capacity).map_err(|error| error.to_string())?;
        let handle = receive.handle().map_err(|error| error.to_string())?;
        let mut network = comm.network.lock().unwrap();
        let peer_handle = if comm.rank == 0 {
            let peer = network.peers.first_mut().ok_or("ipc needs one peer")?;
            let mut bytes = [0u8; 64];
            peer.read_exact(&mut bytes)
                .map_err(|error| error.to_string())?;
            let peer_handle = IpcHandle(bytes);
            let mut all = Vec::with_capacity(128);
            all.extend_from_slice(&handle.0);
            all.extend_from_slice(&peer_handle.0);
            peer.write_all(&all).map_err(|error| error.to_string())?;
            peer_handle
        } else {
            let root = network.root.as_mut().ok_or("ipc worker root")?;
            root.write_all(&handle.0)
                .map_err(|error| error.to_string())?;
            let mut all = [0u8; 128];
            root.read_exact(&mut all)
                .map_err(|error| error.to_string())?;
            IpcHandle(all[64..].try_into().map_err(|_| "ipc handle")?)
        };
        drop(network);
        let peer_receive = PeerRegion::open(peer_handle, comm.device, capacity)
            .map_err(|error| error.to_string())?;
        let kernel = AddKernel::current(comm.device).map_err(|error| error.to_string())?;
        Ok(Self {
            local,
            receive,
            peer_receive,
            kernel,
            capacity,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn run(
        &mut self,
        shm: &crate::shm::Shm,
        input: u64,
        output: u64,
        bytes: usize,
        dtype: DataType,
    ) -> NcclResult {
        if bytes > self.capacity {
            return NCCL_INVALID_USAGE;
        }
        if self.local.copy_from_device(input, bytes).is_err()
            || self.local.copy_to(&self.peer_receive, bytes).is_err()
        {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        shm.barrier();
        let count = bytes / dtype.bytes();
        let result = match dtype {
            DataType::Float32 => self.kernel.add(&self.local, &self.receive, count),
            DataType::Bfloat16 => self.kernel.add_bf16(&self.local, &self.receive, count),
            DataType::Float16 => self.kernel.add_f16(&self.local, &self.receive, count),
            _ => return NCCL_INVALID_USAGE,
        };
        if result.is_err() || self.local.copy_to_device(output, bytes).is_err() {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        NCCL_SUCCESS
    }
}
