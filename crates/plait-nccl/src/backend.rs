//! Device backends used by the NCCL compatibility layer.
//!
//! Vendor libraries supply device copies and stream ordering. Collective
//! transport and tensor representation are shared across backends.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Arc;

mod host_copy;
pub use host_copy::HostCopyRuntime;

thread_local! {
    static BOUND_CUDA_DEVICE: Cell<Option<usize>> = const { Cell::new(None) };
}

use cudarc::driver::{CudaContext, sys};

#[derive(Debug)]
pub enum BackendError {
    Operation(String),
    WorkerUnavailable(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Operation(message) | Self::WorkerUnavailable(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for BackendError {}

pub trait DeviceRuntime: Send + Sync {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError>;
    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError>;

    fn synchronize_stream(&self, _stream: u64) -> Result<(), BackendError> {
        Ok(())
    }
    fn pin_host(&self, _pointer: u64, _bytes: usize) -> Result<(), BackendError> {
        Ok(())
    }

    fn map_host(&self, _pointer: usize, _bytes: usize) -> Result<u64, BackendError> {
        Err(BackendError::Operation(
            "host mapping requires CUDA".to_owned(),
        ))
    }

    fn cuda_context(&self) -> Option<Arc<CudaContext>> {
        None
    }

    fn copy_host_to_device_async(
        &self,
        target: u64,
        source: &[u8],
        _stream: u64,
    ) -> Result<(), BackendError> {
        self.copy_host_to_device(target, source)
    }
}

pub struct CudaRuntime {
    context: Arc<CudaContext>,
    device: usize,
}

impl CudaRuntime {
    pub fn new(device: usize) -> Result<Self, BackendError> {
        let context = CudaContext::new(device)
            .map_err(|error| BackendError::Operation(format!("CUDA context {device}: {error}")))?;
        Ok(Self { context, device })
    }
}

impl CudaRuntime {
    fn bind(&self) -> Result<(), BackendError> {
        if BOUND_CUDA_DEVICE.with(Cell::get) == Some(self.device) {
            return Ok(());
        }
        self.context
            .bind_to_thread()
            .map_err(|error| BackendError::Operation(format!("bind CUDA context: {error}")))?;
        BOUND_CUDA_DEVICE.with(|slot| slot.set(Some(self.device)));
        Ok(())
    }

    fn execution_status(
        &self,
        operation: &str,
        status: sys::CUresult,
        stream: sys::CUstream,
    ) -> Result<(), BackendError> {
        if status == sys::CUresult::CUDA_SUCCESS {
            return Ok(());
        }
        let message = format!("{operation}: {status:?}");
        if status == sys::CUresult::CUDA_ERROR_ECC_UNCORRECTABLE {
            let mut current = std::ptr::null_mut();
            let context_status = unsafe { sys::cuCtxGetCurrent(&mut current) };
            if context_status == sys::CUresult::CUDA_SUCCESS
                && current == self.context.cu_ctx()
                && (stream.is_null() || self.stream_on_context(stream))
            {
                return Err(BackendError::WorkerUnavailable(message));
            }
        }
        Err(BackendError::Operation(message))
    }

    fn stream_on_context(&self, stream: sys::CUstream) -> bool {
        let mut context = std::ptr::null_mut();
        let status = unsafe { sys::cuStreamGetCtx(stream, &mut context) };
        status == sys::CUresult::CUDA_SUCCESS && context == self.context.cu_ctx()
    }
}

impl DeviceRuntime for CudaRuntime {
    fn synchronize_stream(&self, stream: u64) -> Result<(), BackendError> {
        self.bind()?;
        let status = unsafe { sys::cuStreamSynchronize(stream as sys::CUstream) };
        self.execution_status("cuStreamSynchronize", status, stream as sys::CUstream)
    }

    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        self.bind()?;
        let status = unsafe { sys::cuMemcpyHtoD_v2(target, source.as_ptr().cast(), source.len()) };
        self.execution_status("cuMemcpyHtoD", status, std::ptr::null_mut())
    }

    fn copy_host_to_device_async(
        &self,
        target: u64,
        source: &[u8],
        stream: u64,
    ) -> Result<(), BackendError> {
        self.bind()?;
        let status = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                target,
                source.as_ptr().cast(),
                source.len(),
                stream as sys::CUstream,
            )
        };
        self.execution_status("cuMemcpyHtoDAsync", status, stream as sys::CUstream)
    }

    fn cuda_context(&self) -> Option<Arc<CudaContext>> {
        Some(Arc::clone(&self.context))
    }

    fn map_host(&self, pointer: usize, _bytes: usize) -> Result<u64, BackendError> {
        self.bind()?;
        let mut device = 0u64;
        let status =
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, pointer as *mut c_void, 0) };
        if status == sys::CUresult::CUDA_SUCCESS {
            Ok(device)
        } else {
            Err(BackendError::Operation(format!(
                "cuMemHostGetDevicePointer: {status:?}"
            )))
        }
    }

    fn pin_host(&self, pointer: u64, bytes: usize) -> Result<(), BackendError> {
        self.bind()?;
        let status = unsafe {
            sys::cuMemHostRegister_v2(
                pointer as *mut c_void,
                bytes,
                sys::CU_MEMHOSTREGISTER_DEVICEMAP,
            )
        };
        if status == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(BackendError::Operation(format!(
                "cuMemHostRegister: {status:?}"
            )))
        }
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        self.bind()?;
        let status =
            unsafe { sys::cuMemcpyDtoH_v2(target.as_mut_ptr().cast(), source, target.len()) };
        self.execution_status("cuMemcpyDtoH", status, std::ptr::null_mut())
    }
}

/// CANN ACL subset. Symbols are resolved at runtime; no CANN headers or link
/// flags are required to build the NVIDIA wheel.
struct AclApi {
    _library: libloading::Library,
    set_device: unsafe extern "C" fn(i32) -> i32,
    memcpy: unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, u32) -> i32,
    synchronize_device: unsafe extern "C" fn() -> i32,
    synchronize_stream: unsafe extern "C" fn(*mut c_void) -> i32,
}

unsafe impl Send for AclApi {}
unsafe impl Sync for AclApi {}

pub struct HuaweiRuntime {
    api: Arc<AclApi>,
    device: i32,
}

fn acl_execution_status(operation: &str, status: i32) -> Result<(), BackendError> {
    const ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR: i32 = 507054;
    match status {
        0 => Ok(()),
        ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR => Err(BackendError::WorkerUnavailable(format!(
            "{operation}: {status}"
        ))),
        _ => Err(BackendError::Operation(format!("{operation}: {status}"))),
    }
}

impl HuaweiRuntime {
    pub fn new(device: usize) -> Result<Self, BackendError> {
        let library_path =
            std::env::var("PLAIT_HUAWEI_ACL_LIB").unwrap_or_else(|_| "libascendcl.so".to_owned());
        let library = unsafe { libloading::Library::new(&library_path) }
            .map_err(|error| BackendError::Operation(format!("load {library_path}: {error}")))?;
        let init = unsafe {
            *library
                .get::<unsafe extern "C" fn(*const std::ffi::c_char) -> i32>(b"aclInit\0")
                .map_err(|error| BackendError::Operation(format!("aclInit: {error}")))?
        };
        let set_device = unsafe {
            *library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"aclrtSetDevice\0")
                .map_err(|error| BackendError::Operation(format!("aclrtSetDevice: {error}")))?
        };
        let memcpy = unsafe {
            *library
                .get::<unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, u32) -> i32>(
                    b"aclrtMemcpy\0",
                )
                .map_err(|error| BackendError::Operation(format!("aclrtMemcpy: {error}")))?
        };
        let synchronize_device = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> i32>(b"aclrtSynchronizeDevice\0")
                .map_err(|error| {
                    BackendError::Operation(format!("aclrtSynchronizeDevice: {error}"))
                })?
        };
        let synchronize_stream = unsafe {
            *library
                .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"aclrtSynchronizeStream\0")
                .map_err(|error| {
                    BackendError::Operation(format!("aclrtSynchronizeStream: {error}"))
                })?
        };
        const ACL_ERROR_REPEAT_INITIALIZE: i32 = 100002;
        let status = unsafe { init(std::ptr::null()) };
        if status != 0 && status != ACL_ERROR_REPEAT_INITIALIZE {
            return Err(BackendError::Operation(format!("aclInit: {status}")));
        }
        let runtime = Self {
            api: Arc::new(AclApi {
                _library: library,
                set_device,
                memcpy,
                synchronize_device,
                synchronize_stream,
            }),
            device: i32::try_from(device).map_err(|_| {
                BackendError::Operation(format!("invalid ACL device index {device}"))
            })?,
        };
        runtime.bind()?;
        Ok(runtime)
    }

    fn bind(&self) -> Result<(), BackendError> {
        let status = unsafe { (self.api.set_device)(self.device) };
        if status == 0 {
            Ok(())
        } else {
            Err(BackendError::Operation(format!(
                "aclrtSetDevice({}): {status}",
                self.device
            )))
        }
    }
}

impl DeviceRuntime for HuaweiRuntime {
    fn synchronize_stream(&self, stream: u64) -> Result<(), BackendError> {
        self.bind()?;
        let status = unsafe {
            if stream == 0 {
                (self.api.synchronize_device)()
            } else {
                (self.api.synchronize_stream)(stream as *mut c_void)
            }
        };
        acl_execution_status("ACL synchronization", status)
    }

    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        self.bind()?;
        // ACL_MEMCPY_HOST_TO_DEVICE = 1.
        let status = unsafe {
            (self.api.memcpy)(
                target as *mut c_void,
                source.len(),
                source.as_ptr().cast(),
                source.len(),
                1,
            )
        };
        acl_execution_status("aclrtMemcpy H2D", status)
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        self.bind()?;
        // ACL_MEMCPY_DEVICE_TO_HOST = 2.
        let status = unsafe {
            (self.api.memcpy)(
                target.as_mut_ptr().cast(),
                target.len(),
                source as *const c_void,
                target.len(),
                2,
            )
        };
        acl_execution_status("aclrtMemcpy D2H", status)
    }
}

/// A small vendor-neutral ABI for PPU runtimes:
/// `plait_ppu_set_device(int)`, `plait_ppu_memcpy_h2d(uint64_t, const void*, size_t)`,
/// `plait_ppu_memcpy_d2h(uint64_t, void*, size_t)`, `plait_ppu_sync()`.
/// A vendor adapter can be shipped as `libplait_ppu.so` without changing this
/// crate.
struct PpuApi {
    _library: libloading::Library,
    h2d: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    d2h: unsafe extern "C" fn(u64, *mut c_void, usize) -> i32,
    sync: unsafe extern "C" fn() -> i32,
}

unsafe impl Send for PpuApi {}
unsafe impl Sync for PpuApi {}

pub struct PpuRuntime {
    api: Arc<PpuApi>,
}

impl PpuRuntime {
    pub fn new(device: usize) -> Result<Self, BackendError> {
        let library_path =
            std::env::var("PLAIT_PPU_LIB").unwrap_or_else(|_| "libplait_ppu.so".to_owned());
        let library = unsafe { libloading::Library::new(&library_path) }
            .map_err(|error| BackendError::Operation(format!("load {library_path}: {error}")))?;
        let set_device = unsafe {
            *library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"plait_ppu_set_device\0")
                .map_err(|error| {
                    BackendError::Operation(format!("plait_ppu_set_device: {error}"))
                })?
        };
        let h2d = unsafe {
            *library
                .get::<unsafe extern "C" fn(u64, *const c_void, usize) -> i32>(
                    b"plait_ppu_memcpy_h2d\0",
                )
                .map_err(|error| {
                    BackendError::Operation(format!("plait_ppu_memcpy_h2d: {error}"))
                })?
        };
        let d2h = unsafe {
            *library
                .get::<unsafe extern "C" fn(u64, *mut c_void, usize) -> i32>(
                    b"plait_ppu_memcpy_d2h\0",
                )
                .map_err(|error| {
                    BackendError::Operation(format!("plait_ppu_memcpy_d2h: {error}"))
                })?
        };
        let sync = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> i32>(b"plait_ppu_sync\0")
                .map_err(|error| BackendError::Operation(format!("plait_ppu_sync: {error}")))?
        };
        if unsafe { set_device(device as i32) } != 0 {
            return Err(BackendError::Operation(format!(
                "plait_ppu_set_device({device}) failed"
            )));
        }
        Ok(Self {
            api: Arc::new(PpuApi {
                _library: library,
                h2d,
                d2h,
                sync,
            }),
        })
    }
}

impl DeviceRuntime for PpuRuntime {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        let status = unsafe { (self.api.h2d)(target, source.as_ptr().cast(), source.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(BackendError::Operation(format!(
                "plait_ppu_memcpy_h2d: {status}"
            )))
        }
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        let status = unsafe { (self.api.d2h)(source, target.as_mut_ptr().cast(), target.len()) };
        if status == 0 {
            let _ = unsafe { (self.api.sync)() };
            Ok(())
        } else {
            Err(BackendError::Operation(format!(
                "plait_ppu_memcpy_d2h: {status}"
            )))
        }
    }
}
