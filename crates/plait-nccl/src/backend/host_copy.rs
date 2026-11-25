use std::ffi::{OsString, c_int, c_void};

use libloading::Library;

use super::{BackendError, DeviceRuntime};

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

type SetDevice = unsafe extern "C" fn(c_int) -> c_int;
type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type StreamSynchronize = unsafe extern "C" fn(*mut c_void) -> c_int;

pub struct HostCopyRuntime {
    _library: Library,
    name: &'static str,
    device: c_int,
    set_device: SetDevice,
    memcpy: Memcpy,
    stream_synchronize: StreamSynchronize,
}

impl HostCopyRuntime {
    pub fn hip(device: usize) -> Result<Self, BackendError> {
        let path =
            std::env::var_os("PLAIT_ROCM_HIP_LIB").unwrap_or_else(|| "libamdhip64.so".into());
        Self::load(
            device,
            "HIP",
            path,
            "hipSetDevice",
            "hipMemcpy",
            "hipStreamSynchronize",
        )
    }

    pub fn hggc(device: usize) -> Result<Self, BackendError> {
        let path =
            std::env::var_os("PLAIT_PPU_HGGCRT_LIB").unwrap_or_else(|| "libhggcrt.13.0.so".into());
        Self::load(
            device,
            "HGGC",
            path,
            "hggcSetDevice",
            "hggcMemcpy",
            "hggcStreamSynchronize",
        )
    }

    fn load(
        device: usize,
        name: &'static str,
        path: OsString,
        set_device: &str,
        memcpy: &str,
        stream_synchronize: &str,
    ) -> Result<Self, BackendError> {
        let device = c_int::try_from(device)
            .map_err(|_| BackendError(format!("{name} device ordinal is out of range")))?;
        let library = unsafe { Library::new(&path) }
            .map_err(|error| BackendError(format!("load {}: {error}", path.to_string_lossy())))?;
        let set_device = *unsafe { library.get::<SetDevice>(set_device.as_bytes()) }
            .map_err(|error| BackendError(format!("{set_device}: {error}")))?;
        let memcpy = *unsafe { library.get::<Memcpy>(memcpy.as_bytes()) }
            .map_err(|error| BackendError(format!("{memcpy}: {error}")))?;
        let stream_synchronize =
            *unsafe { library.get::<StreamSynchronize>(stream_synchronize.as_bytes()) }
                .map_err(|error| BackendError(format!("{stream_synchronize}: {error}")))?;
        let runtime = Self {
            _library: library,
            name,
            device,
            set_device,
            memcpy,
            stream_synchronize,
        };
        runtime.bind()?;
        Ok(runtime)
    }

    fn bind(&self) -> Result<(), BackendError> {
        // The current device is thread-local. The caller can change it
        // between communicator operations.
        self.status("set device", unsafe { (self.set_device)(self.device) })
    }

    fn status(&self, operation: &str, code: c_int) -> Result<(), BackendError> {
        if code == 0 {
            Ok(())
        } else {
            Err(BackendError(format!(
                "{} {operation}: error {code}",
                self.name
            )))
        }
    }
}

impl DeviceRuntime for HostCopyRuntime {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        if source.is_empty() {
            return Ok(());
        }
        self.bind()?;
        self.status("copy host to device", unsafe {
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
        self.status("copy device to host", unsafe {
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
        self.status("synchronize stream", unsafe {
            (self.stream_synchronize)(stream as *mut c_void)
        })
    }
}
