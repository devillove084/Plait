use std::ffi::{OsString, c_int, c_void};

use libloading::Library;

use super::{BackendError, DeviceRuntime};

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

type SetDevice = unsafe extern "C" fn(c_int) -> c_int;
type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type StreamSynchronize = unsafe extern "C" fn(*mut c_void) -> c_int;
type StreamGetDevice = unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int;

pub struct HostCopyRuntime {
    _library: Library,
    name: &'static str,
    device: c_int,
    set_device: SetDevice,
    memcpy: Memcpy,
    stream_synchronize: StreamSynchronize,
    stream_get_device: Option<StreamGetDevice>,
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
            "hipStreamGetDevice",
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
            "hggcStreamGetDevice",
        )
    }

    fn load(
        device: usize,
        name: &'static str,
        path: OsString,
        set_device: &str,
        memcpy: &str,
        stream_synchronize: &str,
        stream_get_device: &str,
    ) -> Result<Self, BackendError> {
        let device = c_int::try_from(device).map_err(|_| {
            BackendError::Operation(format!("{name} device ordinal is out of range"))
        })?;
        let library = unsafe { Library::new(&path) }.map_err(|error| {
            BackendError::Operation(format!("load {}: {error}", path.to_string_lossy()))
        })?;
        let set_device = *unsafe { library.get::<SetDevice>(set_device.as_bytes()) }
            .map_err(|error| BackendError::Operation(format!("{set_device}: {error}")))?;
        let memcpy = *unsafe { library.get::<Memcpy>(memcpy.as_bytes()) }
            .map_err(|error| BackendError::Operation(format!("{memcpy}: {error}")))?;
        let stream_synchronize =
            *unsafe { library.get::<StreamSynchronize>(stream_synchronize.as_bytes()) }.map_err(
                |error| BackendError::Operation(format!("{stream_synchronize}: {error}")),
            )?;
        let stream_get_device =
            unsafe { library.get::<StreamGetDevice>(stream_get_device.as_bytes()) }
                .ok()
                .map(|symbol| *symbol);
        let runtime = Self {
            _library: library,
            name,
            device,
            set_device,
            memcpy,
            stream_synchronize,
            stream_get_device,
        };
        runtime.bind()?;
        Ok(runtime)
    }

    fn bind(&self) -> Result<(), BackendError> {
        // The current device is thread-local. The caller can change it
        // between communicator operations.
        let code = unsafe { (self.set_device)(self.device) };
        if code == 0 {
            Ok(())
        } else {
            Err(BackendError::Operation(format!(
                "{} set device: error {code}",
                self.name
            )))
        }
    }

    fn execution_status(
        &self,
        operation: &str,
        code: c_int,
        stream: u64,
    ) -> Result<(), BackendError> {
        // Native HIP and HGGC both define their uncorrectable ECC result as 214.
        const UNCORRECTABLE_ECC: c_int = 214;
        match code {
            0 => Ok(()),
            UNCORRECTABLE_ECC if self.owns_stream(stream) => Err(BackendError::WorkerUnavailable(
                format!("{} {operation}: error {code}", self.name),
            )),
            _ => Err(BackendError::Operation(format!(
                "{} {operation}: error {code}",
                self.name
            ))),
        }
    }

    fn owns_stream(&self, stream: u64) -> bool {
        if stream == 0 {
            return true;
        }
        let Some(stream_get_device) = self.stream_get_device else {
            return false;
        };
        let mut device = -1;
        let status = unsafe { stream_get_device(stream as *mut c_void, &mut device) };
        status == 0 && device == self.device
    }
}

impl DeviceRuntime for HostCopyRuntime {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        if source.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let status = unsafe {
            (self.memcpy)(
                target as *mut c_void,
                source.as_ptr().cast(),
                source.len(),
                HOST_TO_DEVICE,
            )
        };
        self.execution_status("copy host to device", status, 0)
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        if target.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let status = unsafe {
            (self.memcpy)(
                target.as_mut_ptr().cast(),
                source as *const c_void,
                target.len(),
                DEVICE_TO_HOST,
            )
        };
        self.execution_status("copy device to host", status, 0)
    }

    fn synchronize_stream(&self, stream: u64) -> Result<(), BackendError> {
        self.bind()?;
        let status = unsafe { (self.stream_synchronize)(stream as *mut c_void) };
        self.execution_status("synchronize stream", status, stream)
    }
}
