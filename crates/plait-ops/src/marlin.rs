//! The upstream Marlin Tensor Core kernel with Rust-managed packing and buffers.
//!
//! Weight and scale permutations are adapted from Marlin (Elias Frantar, 2024).
//! See kernels/marlin/LICENSE and NOTICE for the Apache-2.0 attribution.

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use libloading::Library;

use crate::marlin_layout::pack;
use crate::w4a16::checked_rows;
use crate::{LinearOutput, OperatorError, OutputType, W4Weights};

type Launch = unsafe extern "C" fn(
    *const u16,
    *const u32,
    *mut u16,
    *mut u16,
    *mut i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;

pub struct MarlinDevice {
    ordinal: i32,
    context: Arc<CudaContext>,
    launch: Launch,
    _library: Library,
}

impl MarlinDevice {
    /// Open a device and a local library built from the Marlin sources in Plait.
    ///
    /// # Safety
    /// The library must implement `plait_marlin_launch` with Plait's ABI and
    /// honor the memory bounds and stream provided by the caller.
    pub unsafe fn load(ordinal: i32, path: &Path) -> Result<Self, OperatorError> {
        if ordinal < 0 {
            return Err(OperatorError::Shape("device ordinal must be nonnegative"));
        }
        let context = CudaContext::new(ordinal as usize).map_err(backend)?;
        if context.compute_capability().map_err(backend)? != (8, 6) {
            return Err(OperatorError::Unsupported(
                "the imported Marlin implementation is enabled for NVIDIA SM86",
            ));
        }
        let library = unsafe { Library::new(path) }.map_err(backend)?;
        let launch = *unsafe { library.get::<Launch>(b"plait_marlin_launch") }.map_err(backend)?;
        Ok(Self {
            ordinal,
            context,
            launch,
            _library: library,
        })
    }

    pub fn prepare(&self, weights: W4Weights<'_>) -> Result<MarlinLinear<'_>, OperatorError> {
        let (packed, scales, group_size) = pack(weights)?;
        let stream = self.context.new_stream().map_err(backend)?;
        let packed = stream.clone_htod(&packed).map_err(backend)?;
        let scales = stream.clone_htod(&scales).map_err(backend)?;
        Ok(MarlinLinear {
            device: self,
            stream,
            packed,
            scales,
            group_size,
            k: weights.k,
            n: weights.n,
        })
    }
}

/// An already packed, device-resident W4A16 matrix. Each run allocates its own
/// Marlin locks so a failed or independent invocation cannot reuse old state.
pub struct MarlinLinear<'a> {
    device: &'a MarlinDevice,
    stream: Arc<CudaStream>,
    packed: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    group_size: i32,
    k: u32,
    n: u32,
}

impl MarlinLinear<'_> {
    pub fn run(
        &self,
        activations: &[u16],
        bias: Option<&[u16]>,
        output: OutputType,
    ) -> Result<LinearOutput, OperatorError> {
        if bias.is_some() || output != OutputType::F16 {
            return Err(OperatorError::Unsupported(
                "Marlin requires an FP16 output without fused bias",
            ));
        }
        let rows = checked_rows(self.k, self.n, activations.len())?;
        let elements = (rows as usize) * (self.n as usize);
        if activations.len().max(elements) > i32::MAX as usize {
            return Err(OperatorError::Shape(
                "Marlin activation or output exceeds the indexing range",
            ));
        }
        if rows == 0 {
            return Ok(LinearOutput::F16(Vec::new()));
        }
        let stream = &self.stream;
        self.device.context.bind_to_thread().map_err(backend)?;
        let a = stream.clone_htod(activations).map_err(backend)?;
        // The upstream split-K reduction uses C as temporary storage before
        // writing the final result. Zero here also keeps initial storage defined.
        let mut c = stream.alloc_zeros::<u16>(elements).map_err(backend)?;
        let mut locks = stream
            .alloc_zeros::<i32>(self.n as usize / 128 * 16)
            .map_err(backend)?;
        let status = {
            let (a_ptr, _a) = a.device_ptr(stream);
            let (b_ptr, _b) = self.packed.device_ptr(stream);
            let (s_ptr, _s) = self.scales.device_ptr(stream);
            let (c_ptr, _c) = c.device_ptr_mut(stream);
            let (locks_ptr, _locks) = locks.device_ptr_mut(stream);
            // Packed weights and scales live on this stream's device, the call
            // dimensions are bounded, and all pointers live until synchronization.
            unsafe {
                (self.device.launch)(
                    a_ptr as *const u16,
                    b_ptr as *const u32,
                    s_ptr as *mut u16,
                    c_ptr as *mut u16,
                    locks_ptr as *mut i32,
                    rows as i32,
                    self.n as i32,
                    self.k as i32,
                    self.group_size,
                    self.device.ordinal,
                    stream.cu_stream().cast(),
                )
            }
        };
        if status != 0 {
            return Err(backend(format!("Marlin launch failed ({status})")));
        }
        stream
            .clone_dtoh(&c)
            .map(LinearOutput::F16)
            .map_err(backend)
    }
}

fn backend(error: impl std::fmt::Display) -> OperatorError {
    OperatorError::Backend(error.to_string())
}
