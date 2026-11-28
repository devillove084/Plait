//! Single-stream GPU kernel allreduce over host-mapped shm.

use std::ffi::c_void;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use libloading::Library;

use crate::{DataType, NCCL_SUCCESS, NCCL_UNHANDLED_CUDA_ERROR, ReduceOp};

const READY_OFFSET: usize = 320;
const SLOT_OFFSET: usize = 512;

struct StreamApi {
    _library: Library,
    wait: unsafe extern "C" fn(sys::CUstream, sys::CUdeviceptr, u32, u32) -> sys::CUresult,
}
impl StreamApi {
    fn load() -> Result<Self, String> {
        let library = unsafe { Library::new("libcuda.so.1") }
            .map_err(|error| format!("load libcuda: {error}"))?;
        let wait = unsafe {
            *library
                .get::<unsafe extern "C" fn(sys::CUstream, sys::CUdeviceptr, u32, u32) -> sys::CUresult>(
                    b"cuStreamWaitValue32_v2\0",
                )
                .map_err(|error| format!("cuStreamWaitValue32: {error}"))?
        };
        Ok(Self {
            _library: library,
            wait,
        })
    }
}

pub struct KernelAllreduce {
    module: sys::CUmodule,
    function: sys::CUfunction,
    done: sys::CUdeviceptr,
    ready: u64,
    slots: u64,
    rank: i32,
    capacity: usize,
    next: AtomicU32,
    api: StreamApi,
    context: Arc<CudaContext>,
    #[allow(dead_code)]
    shm: Arc<Mutex<crate::shm::Shm>>,
}
unsafe impl Send for KernelAllreduce {}
unsafe impl Sync for KernelAllreduce {}

impl KernelAllreduce {
    pub fn new(
        runtime: &dyn crate::backend::DeviceRuntime,
        context: Arc<CudaContext>,
        #[allow(dead_code)] shm: Arc<Mutex<crate::shm::Shm>>,
        rank: i32,
    ) -> Result<Self, String> {
        let (base, size, slot_bytes) = {
            let shm = shm.lock().unwrap();
            (shm.base_pointer(), shm.mapped_size(), shm.slot_bytes())
        };
        let device_base = runtime
            .map_host(base, size)
            .map_err(|error| error.to_string())?;
        let cubin = include_bytes!(env!("PLAIT_PERSISTENT_CUBIN"));
        let mut module = std::ptr::null_mut();
        let status = unsafe { sys::cuModuleLoadData(&mut module, cubin.as_ptr().cast()) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("kernel module: {status:?}"));
        }
        let name = b"plait_kernel\0";
        let mut function = std::ptr::null_mut();
        let status =
            unsafe { sys::cuModuleGetFunction(&mut function, module, name.as_ptr().cast()) };
        if status != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = sys::cuModuleUnload(module);
            }
            return Err(format!("kernel function: {status:?}"));
        }
        let mut done = 0;
        let status = unsafe { sys::cuMemAlloc_v2(&mut done, 4) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("kernel done: {status:?}"));
        }
        let _ = unsafe { sys::cuMemsetD32_v2(done, 0, 1) };
        Ok(Self {
            module,
            function,
            done,
            ready: device_base + READY_OFFSET as u64,
            slots: device_base + SLOT_OFFSET as u64,
            rank,
            capacity: slot_bytes / 4,
            next: AtomicU32::new(0),
            api: StreamApi::load()?,
            context,
            shm,
        })
    }

    pub fn submit(
        &self,
        input: u64,
        output: u64,
        bytes: usize,
        dtype: DataType,
        op: ReduceOp,
        caller_stream: u64,
    ) -> i32 {
        if op != ReduceOp::Sum {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let count = bytes / dtype.bytes();
        if count > self.capacity {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let generation = self.next.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
        let count_i = count as i32;
        let dtype_i = dtype as i32;
        let capacity = self.capacity as i32;
        let mut args: [*mut c_void; 10] = [
            (&input as *const u64 as *mut u64).cast(),
            (&output as *const u64 as *mut u64).cast(),
            (&count_i as *const i32 as *mut i32).cast(),
            (&dtype_i as *const i32 as *mut i32).cast(),
            (&generation as *const u32 as *mut u32).cast(),
            (&self.ready as *const u64 as *mut u64).cast(),
            (&self.slots as *const u64 as *mut u64).cast(),
            (&self.done as *const u64 as *mut u64).cast(),
            (&self.rank as *const i32 as *mut i32).cast(),
            (&capacity as *const i32 as *mut i32).cast(),
        ];
        let stream = caller_stream as sys::CUstream;
        let status = unsafe {
            sys::cuLaunchKernel(
                self.function,
                1,
                1,
                1,
                256,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != sys::CUresult::CUDA_SUCCESS {
            return NCCL_UNHANDLED_CUDA_ERROR;
        }
        let status = unsafe { (self.api.wait)(stream, self.done, generation, 0) };
        if status == sys::CUresult::CUDA_SUCCESS {
            NCCL_SUCCESS
        } else {
            NCCL_UNHANDLED_CUDA_ERROR
        }
    }
}

impl Drop for KernelAllreduce {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        unsafe {
            let _ = sys::cuMemFree_v2(self.done);
            let _ = sys::cuModuleUnload(self.module);
        }
    }
}
