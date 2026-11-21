//! Device backends used by the NCCL compatibility layer.
//!
//! CUDA is the only backend exercised on this host. Huawei and PPU are loaded
//! dynamically so the same Rust/Python wheel runs on NVIDIA machines without
//! requiring CANN or a PPU runtime at build time. The vendor methods are the
//! only place where platform-specific copies happen; collectives stay in the
//! shared transport.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Arc;

thread_local! {
    static BOUND_CUDA_DEVICE: Cell<Option<usize>> = const { Cell::new(None) };
}

use cudarc::driver::{CudaContext, sys};

#[derive(Debug)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}

pub trait DeviceRuntime: Send + Sync {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError>;
    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError>;
    fn pin_host(&self, _pointer: u64, _bytes: usize) -> Result<(), BackendError> {
        Ok(())
    }

    fn map_host(&self, _pointer: usize, _bytes: usize) -> Result<u64, BackendError> {
        Err(BackendError("host mapping requires CUDA".to_owned()))
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
            .map_err(|error| BackendError(format!("CUDA context {device}: {error}")))?;
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
            .map_err(|error| BackendError(format!("bind CUDA context: {error}")))?;
        BOUND_CUDA_DEVICE.with(|slot| slot.set(Some(self.device)));
        Ok(())
    }
}

impl DeviceRuntime for CudaRuntime {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        self.bind()?;
        let status = unsafe { sys::cuMemcpyHtoD_v2(target, source.as_ptr().cast(), source.len()) };
        if status == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(BackendError(format!("cuMemcpyHtoD: {status:?}")))
        }
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
        if status == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(BackendError(format!("cuMemcpyHtoDAsync: {status:?}")))
        }
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
            Err(BackendError(format!(
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
            Err(BackendError(format!("cuMemHostRegister: {status:?}")))
        }
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        self.bind()?;
        let status =
            unsafe { sys::cuMemcpyDtoH_v2(target.as_mut_ptr().cast(), source, target.len()) };
        if status == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(BackendError(format!("cuMemcpyDtoH: {status:?}")))
        }
    }
}

/// CANN ACL subset. Symbols are resolved at runtime; no CANN headers or link
/// flags are required to build the NVIDIA wheel.
struct AclApi {
    _library: libloading::Library,
    memcpy: unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, u32) -> i32,
    synchronize: unsafe extern "C" fn() -> i32,
}

unsafe impl Send for AclApi {}
unsafe impl Sync for AclApi {}

pub struct HuaweiRuntime {
    api: Arc<AclApi>,
}

impl HuaweiRuntime {
    pub fn new(device: usize) -> Result<Self, BackendError> {
        let library_path =
            std::env::var("PLAIT_HUAWEI_ACL_LIB").unwrap_or_else(|_| "libascendcl.so".to_owned());
        let library = unsafe { libloading::Library::new(&library_path) }
            .map_err(|error| BackendError(format!("load {library_path}: {error}")))?;
        let init = unsafe {
            *library
                .get::<unsafe extern "C" fn(*const std::ffi::c_char) -> i32>(b"aclInit\0")
                .map_err(|error| BackendError(format!("aclInit: {error}")))?
        };
        let set_device = unsafe {
            *library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"aclrtSetDevice\0")
                .map_err(|error| BackendError(format!("aclrtSetDevice: {error}")))?
        };
        let memcpy = unsafe {
            *library
                .get::<unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, u32) -> i32>(
                    b"aclrtMemcpy\0",
                )
                .map_err(|error| BackendError(format!("aclrtMemcpy: {error}")))?
        };
        let synchronize = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> i32>(b"aclrtSynchronizeDevice\0")
                .map_err(|error| BackendError(format!("aclrtSynchronizeDevice: {error}")))?
        };
        if unsafe { init(std::ptr::null()) } != 0 {
            return Err(BackendError("aclInit failed".to_owned()));
        }
        if unsafe { set_device(device as i32) } != 0 {
            return Err(BackendError(format!("aclrtSetDevice({device}) failed")));
        }
        Ok(Self {
            api: Arc::new(AclApi {
                _library: library,
                memcpy,
                synchronize,
            }),
        })
    }
}

impl DeviceRuntime for HuaweiRuntime {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
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
        if status == 0 {
            Ok(())
        } else {
            Err(BackendError(format!("aclrtMemcpy H2D: {status}")))
        }
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
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
        if status == 0 {
            let _ = unsafe { (self.api.synchronize)() };
            Ok(())
        } else {
            Err(BackendError(format!("aclrtMemcpy D2H: {status}")))
        }
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
            .map_err(|error| BackendError(format!("load {library_path}: {error}")))?;
        let set_device = unsafe {
            *library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"plait_ppu_set_device\0")
                .map_err(|error| BackendError(format!("plait_ppu_set_device: {error}")))?
        };
        let h2d = unsafe {
            *library
                .get::<unsafe extern "C" fn(u64, *const c_void, usize) -> i32>(
                    b"plait_ppu_memcpy_h2d\0",
                )
                .map_err(|error| BackendError(format!("plait_ppu_memcpy_h2d: {error}")))?
        };
        let d2h = unsafe {
            *library
                .get::<unsafe extern "C" fn(u64, *mut c_void, usize) -> i32>(
                    b"plait_ppu_memcpy_d2h\0",
                )
                .map_err(|error| BackendError(format!("plait_ppu_memcpy_d2h: {error}")))?
        };
        let sync = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> i32>(b"plait_ppu_sync\0")
                .map_err(|error| BackendError(format!("plait_ppu_sync: {error}")))?
        };
        if unsafe { set_device(device as i32) } != 0 {
            return Err(BackendError(format!(
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
            Err(BackendError(format!("plait_ppu_memcpy_h2d: {status}")))
        }
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        let status = unsafe { (self.api.d2h)(source, target.as_mut_ptr().cast(), target.len()) };
        if status == 0 {
            let _ = unsafe { (self.api.sync)() };
            Ok(())
        } else {
            Err(BackendError(format!("plait_ppu_memcpy_d2h: {status}")))
        }
    }
}
