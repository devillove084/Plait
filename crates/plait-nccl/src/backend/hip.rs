use std::ffi::{c_int, c_void};

use libloading::Library;

use super::{BackendError, DeviceRuntime};

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

type SetDevice = unsafe extern "C" fn(c_int) -> c_int;
type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type StreamSynchronize = unsafe extern "C" fn(*mut c_void) -> c_int;

pub struct HipRuntime {
    _library: Library,
    device: c_int,
    set_device: SetDevice,
    memcpy: Memcpy,
    stream_synchronize: StreamSynchronize,
}

impl HipRuntime {
    pub fn new(device: usize) -> Result<Self, BackendError> {
        let device = c_int::try_from(device)
            .map_err(|_| BackendError("ROCm device ordinal is out of range".into()))?;
        let path =
            std::env::var_os("PLAIT_ROCM_HIP_LIB").unwrap_or_else(|| "libamdhip64.so".into());
        let library = unsafe { Library::new(&path) }
            .map_err(|error| BackendError(format!("load {}: {error}", path.to_string_lossy())))?;
        // The dynamically loaded HIP runtime owns these entry points.
        let set_device = *unsafe { library.get::<SetDevice>(b"hipSetDevice") }
            .map_err(|error| BackendError(format!("hipSetDevice: {error}")))?;
        let memcpy = *unsafe { library.get::<Memcpy>(b"hipMemcpy") }
            .map_err(|error| BackendError(format!("hipMemcpy: {error}")))?;
        let stream_synchronize =
            *unsafe { library.get::<StreamSynchronize>(b"hipStreamSynchronize") }
                .map_err(|error| BackendError(format!("hipStreamSynchronize: {error}")))?;
        let runtime = Self {
            _library: library,
            device,
            set_device,
            memcpy,
            stream_synchronize,
        };
        runtime.bind()?;
        Ok(runtime)
    }

    fn bind(&self) -> Result<(), BackendError> {
        // HIP's current device is thread-local. The caller can change it
        // between communicator operations.
        status("hipSetDevice", unsafe { (self.set_device)(self.device) })
    }
}

impl DeviceRuntime for HipRuntime {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        if source.is_empty() {
            return Ok(());
        }
        self.bind()?;
        status("hipMemcpy H2D", unsafe {
            (self.memcpy)(
                target as *mut c_void,
                source.as_ptr().cast(),
                source.len(),
                HOST_TO_DEVICE,
            )
        })
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        if target.is_empty() {
            return Ok(());
        }
        self.bind()?;
        status("hipMemcpy D2H", unsafe {
            (self.memcpy)(
                target.as_mut_ptr().cast(),
                source as *const c_void,
                target.len(),
                DEVICE_TO_HOST,
            )
        })
    }

    fn synchronize_stream(&self, stream: u64) -> Result<(), BackendError> {
        self.bind()?;
        status("hipStreamSynchronize", unsafe {
            (self.stream_synchronize)(stream as *mut c_void)
        })
    }
}

fn status(operation: &str, code: c_int) -> Result<(), BackendError> {
    if code == 0 {
        Ok(())
    } else {
        Err(BackendError(format!("{operation}: HIP error {code}")))
    }
}
