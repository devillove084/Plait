//! Adapter for Plait's existing C reference CUDA-compatible and Metal kernels.

use std::ffi::c_void;
use std::path::Path;
use std::ptr;

use libloading::Library;

pub use crate::resident::{
    DeviceQuantizedActivations, DeviceSession, DeviceTensor, PreparedW4A16, PreparedW4Projection,
    W4Activation, W4ProjectionKernel,
};

use crate::{LinearOutput, OperatorError, OutputType, W4A16};

#[repr(C)]
pub(crate) struct Shape {
    pub(crate) m: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
    pub(crate) group_size: u32,
    pub(crate) output_bits: u32,
}

type RunHost = unsafe extern "C" fn(
    *const Shape,
    i32,
    *const u16,
    *const u8,
    *const u16,
    *const u8,
    *const u16,
    *mut c_void,
) -> i32;

pub struct NativeW4A16 {
    run_host: RunHost,
    library: Library,
}

impl NativeW4A16 {
    /// Load a library built from Plait's compatible C kernel sources.
    ///
    /// # Safety
    /// The library must implement the contract in `kernels/w4a16.h`, including
    /// completing all accesses before the host function returns. If the
    /// library exports a resident API or its optional kernel extensions, it
    /// must honor the corresponding queue, ownership and completion contracts.
    pub unsafe fn load(path: &Path) -> Result<Self, OperatorError> {
        let library = unsafe { Library::new(path) }
            .map_err(|error| OperatorError::Backend(error.to_string()))?;
        let run_host = *unsafe { library.get::<RunHost>(b"plait_w4a16_run_host") }
            .map_err(|error| OperatorError::Backend(error.to_string()))?;
        Ok(Self { run_host, library })
    }

    /// Create an independent queue and allocation scope for this device.
    pub fn open_device(&self, ordinal: i32) -> Result<DeviceSession<'_>, OperatorError> {
        DeviceSession::open(&self.library, ordinal)
    }

    pub fn run(&self, ordinal: i32, call: W4A16<'_>) -> Result<LinearOutput, OperatorError> {
        let mut output = LinearOutput::zeros(call.output, call.output_len());
        if call.rows == 0 {
            return Ok(output);
        }
        let weights = call.weights;
        let shape = Shape {
            m: call.rows,
            n: weights.n,
            k: weights.k,
            group_size: weights.group_size,
            output_bits: match call.output {
                OutputType::F16 => 16,
                OutputType::F32 => 32,
            },
        };
        let implicit_zero;
        let zero = match weights.zero_points {
            Some(points) => points,
            None => {
                implicit_zero = vec![8; weights.scales.len()];
                &implicit_zero
            }
        };
        let destination = match &mut output {
            LinearOutput::F16(values) => values.as_mut_ptr().cast(),
            LinearOutput::F32(values) => values.as_mut_ptr().cast(),
        };
        // All buffers were validated by W4A16 and live through this synchronous call.
        let status = unsafe {
            (self.run_host)(
                &shape,
                ordinal,
                call.activations.as_ptr(),
                weights.packed.as_ptr(),
                weights.scales.as_ptr(),
                zero.as_ptr(),
                call.bias.map_or(ptr::null(), |values| values.as_ptr()),
                destination,
            )
        };
        if status != 0 {
            return Err(OperatorError::Backend(format!(
                "native W4A16 execution failed ({status})"
            )));
        }
        Ok(output)
    }
}
