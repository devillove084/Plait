//! CUDA IPC peer regions for cross-process GPU collectives.
//!
//! The region owns a real `cuMemAlloc` allocation. `handle()` exports the CUDA
//! IPC handle; `open()` imports it into another process. Peer copies use
//! `cuMemcpyPeerAsync`, and the add kernel is loaded from the same cubin as the
//! single-process collective path. No host staging is used after setup.

use std::ffi::{CString, c_void};
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, DriverError};

use crate::{CollectiveError, Result};

/// 64-byte CUDA IPC memory handle, serializable as hex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IpcHandle(pub [u8; 64]);

impl IpcHandle {
    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(128);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(output, "{byte:02x}").expect("hex write");
        }
        output
    }

    pub fn from_hex(text: &str) -> Result<Self> {
        if text.len() != 128 {
            return Err(CollectiveError::new("IPC handle must be 128 hex digits"));
        }
        let mut bytes = [0u8; 64];
        for (index, slot) in bytes.iter_mut().enumerate() {
            let offset = index * 2;
            *slot = u8::from_str_radix(&text[offset..offset + 2], 16)
                .map_err(|error| CollectiveError::new(format!("IPC handle hex: {error}")))?;
        }
        Ok(Self(bytes))
    }
}

/// A CUDA allocation that can be exported to another process.
pub struct PeerRegion {
    context: Arc<CudaContext>,
    raw_context: sys::CUcontext,
    ptr: sys::CUdeviceptr,
    bytes: usize,
    imported: bool,
}

// The CUDA context and allocation are owned for the lifetime of the region.
// Access is serialized by the caller (one collective owner).
unsafe impl Send for PeerRegion {}

impl PeerRegion {
    pub fn alloc(device: usize, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(CollectiveError::new("peer region cannot be empty"));
        }
        let context = CudaContext::new(device)
            .map_err(|error| CollectiveError::new(format!("CUDA context {device}: {error}")))?;
        context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        let mut raw_context = std::ptr::null_mut();
        check(
            unsafe { sys::cuCtxGetCurrent(&mut raw_context) },
            "cuCtxGetCurrent",
        )?;
        let mut ptr = 0;
        check(unsafe { sys::cuMemAlloc_v2(&mut ptr, bytes) }, "cuMemAlloc")?;
        Ok(Self {
            context,
            raw_context,
            ptr,
            bytes,
            imported: false,
        })
    }

    pub fn open(handle: IpcHandle, device: usize, bytes: usize) -> Result<Self> {
        let context = CudaContext::new(device)
            .map_err(|error| CollectiveError::new(format!("CUDA context {device}: {error}")))?;
        context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        let mut raw_context = std::ptr::null_mut();
        check(
            unsafe { sys::cuCtxGetCurrent(&mut raw_context) },
            "cuCtxGetCurrent",
        )?;
        let raw_handle = unsafe { std::mem::transmute::<[u8; 64], sys::CUipcMemHandle>(handle.0) };
        let mut ptr = 0;
        check(
            unsafe {
                sys::cuIpcOpenMemHandle_v2(
                    &mut ptr,
                    raw_handle,
                    sys::CUipcMem_flags_enum::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
                )
            },
            "cuIpcOpenMemHandle",
        )?;
        Ok(Self {
            context,
            raw_context,
            ptr,
            bytes,
            imported: true,
        })
    }

    pub fn handle(&self) -> Result<IpcHandle> {
        let mut handle = sys::CUipcMemHandle { reserved: [0; 64] };
        check(
            unsafe { sys::cuIpcGetMemHandle(&mut handle, self.ptr) },
            "cuIpcGetMemHandle",
        )?;
        Ok(IpcHandle(unsafe {
            std::mem::transmute::<sys::CUipcMemHandle, [u8; 64]>(handle)
        }))
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn write_bytes(&self, values: &[u8]) -> Result<()> {
        if values.len() > self.bytes {
            return Err(CollectiveError::new("host write exceeds region"));
        }
        self.context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        check(
            unsafe {
                sys::cuMemcpyHtoD_v2(self.ptr, values.as_ptr().cast::<c_void>(), values.len())
            },
            "cuMemcpyHtoD",
        )
    }

    pub fn write_host(&self, values: &[f32]) -> Result<()> {
        let bytes = std::mem::size_of_val(values);
        if bytes > self.bytes {
            return Err(CollectiveError::new("host write exceeds region"));
        }
        self.bind()?;
        check(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, values.as_ptr().cast::<c_void>(), bytes) },
            "cuMemcpyHtoD",
        )
    }

    pub fn read_host(&self, count: usize) -> Result<Vec<f32>> {
        let bytes = count * std::mem::size_of::<f32>();
        if bytes > self.bytes {
            return Err(CollectiveError::new("host read exceeds region"));
        }
        let mut values = vec![0.0f32; count];
        self.bind()?;
        check(
            unsafe { sys::cuMemcpyDtoH_v2(values.as_mut_ptr().cast::<c_void>(), self.ptr, bytes) },
            "cuMemcpyDtoH",
        )?;
        Ok(values)
    }

    /// Copy bytes from this region into a peer region without host staging.
    pub fn copy_to(&self, destination: &PeerRegion, bytes: usize) -> Result<()> {
        if bytes > self.bytes || bytes > destination.bytes {
            return Err(CollectiveError::new("peer copy exceeds region"));
        }
        check(
            unsafe {
                sys::cuMemcpyPeerAsync(
                    destination.ptr,
                    destination.raw_context,
                    self.ptr,
                    self.raw_context,
                    bytes,
                    std::ptr::null_mut(),
                )
            },
            "cuMemcpyPeerAsync",
        )?;
        destination.sync()
    }

    pub fn sync(&self) -> Result<()> {
        self.bind()?;
        check(unsafe { sys::cuCtxSynchronize() }, "cuCtxSynchronize")
    }

    fn bind(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))
    }
}

impl Drop for PeerRegion {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        let _ = self.bind();
        unsafe {
            if self.imported {
                let _ = sys::cuIpcCloseMemHandle(self.ptr);
            } else {
                let _ = sys::cuMemFree_v2(self.ptr);
            }
        }
    }
}

/// Add kernel loaded from the collective cubin.
pub struct AddKernel {
    context: Arc<CudaContext>,
    module: sys::CUmodule,
    function: sys::CUfunction,
}

impl AddKernel {
    #[cfg(plait_collective_cubin)]
    pub fn current(device: usize) -> Result<Self> {
        const CUBIN: &[u8] = include_bytes!(env!("PLAIT_COLLECTIVE_CUBIN"));
        let context = CudaContext::new(device)
            .map_err(|error| CollectiveError::new(format!("CUDA context {device}: {error}")))?;
        context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        let mut module = std::ptr::null_mut();
        check(
            unsafe { sys::cuModuleLoadData(&mut module, CUBIN.as_ptr().cast::<c_void>()) },
            "cuModuleLoadData",
        )?;
        let name = CString::new("plait_add_inplace").expect("kernel name");
        let mut function = std::ptr::null_mut();
        check(
            unsafe { sys::cuModuleGetFunction(&mut function, module, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Self {
            context,
            module,
            function,
        })
    }

    #[cfg(not(plait_collective_cubin))]
    pub fn current(_device: usize) -> Result<Self> {
        Err(CollectiveError::new(
            "cross-process add kernel requires a compiled collectives cubin",
        ))
    }

    /// `destination += source` over `count` f32 values.
    pub fn add(&self, destination: &PeerRegion, source: &PeerRegion, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        self.context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        let mut dst = destination.ptr;
        let mut src = source.ptr;
        let mut n = count as i32;
        let mut args: [*mut c_void; 3] = [
            (&mut dst as *mut sys::CUdeviceptr).cast::<c_void>(),
            (&mut src as *mut sys::CUdeviceptr).cast::<c_void>(),
            (&mut n as *mut i32).cast::<c_void>(),
        ];
        let blocks = count.div_ceil(256) as u32;
        check(
            unsafe {
                sys::cuLaunchKernel(
                    self.function,
                    blocks,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        check(unsafe { sys::cuCtxSynchronize() }, "cuCtxSynchronize")
    }
}

impl Drop for AddKernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = sys::cuModuleUnload(self.module);
            }
        }
    }
}

fn check(result: sys::CUresult, operation: &'static str) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CollectiveError::new(format!(
            "{operation} failed: {result:?}"
        )))
    }
}

fn _driver_error(error: DriverError) -> CollectiveError {
    CollectiveError::new(error.to_string())
}

/// VMM allocation exported as a POSIX file descriptor.
///
/// This is the path for allocations that cannot use legacy CUDA IPC, and the
/// path that can later be registered with GPUDirect RDMA.
pub struct VmmRegion {
    context: Arc<CudaContext>,
    ptr: sys::CUdeviceptr,
    bytes: usize,
    handle: sys::CUmemGenericAllocationHandle,
}

impl VmmRegion {
    pub fn alloc(device: usize, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(CollectiveError::new("VMM region cannot be empty"));
        }
        let context = CudaContext::new(device)
            .map_err(|error| CollectiveError::new(format!("CUDA context {device}: {error}")))?;
        context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;

        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes =
            sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        prop.location = sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 { id: device as i32 },
        };
        let mut granularity = 0usize;
        check(
            unsafe {
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "cuMemGetAllocationGranularity",
        )?;
        let bytes = bytes.div_ceil(granularity) * granularity;
        let mut handle = 0;
        check(
            unsafe { sys::cuMemCreate(&mut handle, bytes, &prop, 0) },
            "cuMemCreate",
        )?;
        let mut ptr = 0;
        check(
            unsafe { sys::cuMemAddressReserve(&mut ptr, bytes, granularity, 0, 0) },
            "cuMemAddressReserve",
        )?;
        if let Err(error) = check(
            unsafe { sys::cuMemMap(ptr, bytes, 0, handle, 0) },
            "cuMemMap",
        ) {
            unsafe {
                let _ = sys::cuMemAddressFree(ptr, bytes);
                let _ = sys::cuMemRelease(handle);
            }
            return Err(error);
        }
        let access = sys::CUmemAccessDesc {
            location: prop.location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        if let Err(error) = check(
            unsafe { sys::cuMemSetAccess(ptr, bytes, &access, 1) },
            "cuMemSetAccess",
        ) {
            unsafe {
                let _ = sys::cuMemUnmap(ptr, bytes);
                let _ = sys::cuMemAddressFree(ptr, bytes);
                let _ = sys::cuMemRelease(handle);
            }
            return Err(error);
        }
        Ok(Self {
            context,
            ptr,
            bytes,
            handle,
        })
    }

    pub fn export_fd(&self) -> Result<i32> {
        self.context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        let mut fd = 0i32;
        check(
            unsafe {
                sys::cuMemExportToShareableHandle(
                    (&mut fd as *mut i32).cast::<c_void>(),
                    self.handle,
                    sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                )
            },
            "cuMemExportToShareableHandle",
        )?;
        Ok(fd)
    }

    pub fn import_fd(fd: i32, device: usize, bytes: usize) -> Result<Self> {
        let context = CudaContext::new(device)
            .map_err(|error| CollectiveError::new(format!("CUDA context {device}: {error}")))?;
        context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        let mut handle = 0;
        check(
            unsafe {
                sys::cuMemImportFromShareableHandle(
                    &mut handle,
                    fd as *mut c_void,
                    sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                )
            },
            "cuMemImportFromShareableHandle",
        )?;
        let mut granularity = 0usize;
        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location = sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 { id: device as i32 },
        };
        check(
            unsafe {
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "cuMemGetAllocationGranularity",
        )?;
        let bytes = bytes.div_ceil(granularity) * granularity;
        let mut ptr = 0;
        check(
            unsafe { sys::cuMemAddressReserve(&mut ptr, bytes, granularity, 0, 0) },
            "cuMemAddressReserve",
        )?;
        if let Err(error) = check(
            unsafe { sys::cuMemMap(ptr, bytes, 0, handle, 0) },
            "cuMemMap",
        ) {
            unsafe {
                let _ = sys::cuMemAddressFree(ptr, bytes);
                let _ = sys::cuMemRelease(handle);
            }
            return Err(error);
        }
        let access = sys::CUmemAccessDesc {
            location: prop.location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        check(
            unsafe { sys::cuMemSetAccess(ptr, bytes, &access, 1) },
            "cuMemSetAccess",
        )?;
        Ok(Self {
            context,
            ptr,
            bytes,
            handle,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }

    pub fn write_bytes(&self, values: &[u8]) -> Result<()> {
        if values.len() > self.bytes {
            return Err(CollectiveError::new("host write exceeds region"));
        }
        self.context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        check(
            unsafe {
                sys::cuMemcpyHtoD_v2(self.ptr, values.as_ptr().cast::<c_void>(), values.len())
            },
            "cuMemcpyHtoD",
        )
    }

    pub fn write_host(&self, values: &[f32]) -> Result<()> {
        let bytes = std::mem::size_of_val(values);
        if bytes > self.bytes {
            return Err(CollectiveError::new("VMM write exceeds region"));
        }
        self.context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        check(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, values.as_ptr().cast::<c_void>(), bytes) },
            "cuMemcpyHtoD",
        )
    }

    pub fn read_host(&self, count: usize) -> Result<Vec<f32>> {
        let bytes = count * std::mem::size_of::<f32>();
        if bytes > self.bytes {
            return Err(CollectiveError::new("VMM read exceeds region"));
        }
        let mut values = vec![0.0f32; count];
        self.context
            .bind_to_thread()
            .map_err(|error| CollectiveError::new(format!("bind context: {error}")))?;
        check(
            unsafe { sys::cuMemcpyDtoH_v2(values.as_mut_ptr().cast::<c_void>(), self.ptr, bytes) },
            "cuMemcpyDtoH",
        )?;
        Ok(values)
    }
}

impl Drop for VmmRegion {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        let _ = self.context.bind_to_thread();
        unsafe {
            let _ = sys::cuMemUnmap(self.ptr, self.bytes);
            let _ = sys::cuMemAddressFree(self.ptr, self.bytes);
            let _ = sys::cuMemRelease(self.handle);
        }
    }
}
