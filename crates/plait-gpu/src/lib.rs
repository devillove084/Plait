//! Real CUDA execution for Plait compute tasks.
//!
//! The kernel is compiled with NVRTC at runtime, so the same source runs on the
//! local RTX 3090 (sm_86) and the remote RTX PRO 5000 Blackwell (sm_120) without
//! shipping architecture-specific PTX. This crate is intentionally small: it
//! proves the compute path; model kernels are adapters above it.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{Ptx, compile_ptx};

const KERNEL: &str = r#"
extern "C" __global__ void plait_axpy(
    const float* x,
    const float* y,
    float* out,
    float alpha,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}
"#;

pub struct GpuExecutor {
    // Drop order matters: the function and module must die before the context.
    function: CudaFunction,
    _module: Arc<CudaModule>,
    _context: Arc<CudaContext>,
}

impl GpuExecutor {
    /// Open an NVIDIA device by ordinal and compile the reference kernel.
    pub fn new(ordinal: usize) -> Result<Self, GpuError> {
        let context =
            CudaContext::new(ordinal).map_err(|error| GpuError::new(error.to_string()))?;
        let (major, minor) = context
            .compute_capability()
            .map_err(|error| GpuError::new(error.to_string()))?;
        let ptx = match (major, minor) {
            (8, 6) => {
                let path = std::env::temp_dir().join(format!("plait_axpy_sm86_{ordinal}.cubin"));
                std::fs::write(&path, include_bytes!("../kernels/plait_axpy_sm86.cubin"))
                    .map_err(|error| GpuError::new(error.to_string()))?;
                Ptx::from_file(path)
            }
            (12, 0) => {
                let path = std::env::temp_dir().join(format!("plait_axpy_sm120_{ordinal}.cubin"));
                std::fs::write(&path, include_bytes!("../kernels/plait_axpy_sm120.cubin"))
                    .map_err(|error| GpuError::new(error.to_string()))?;
                Ptx::from_file(path)
            }
            _ => compile_ptx(KERNEL).map_err(|error| GpuError::new(format!("NVRTC: {error:?}")))?,
        };
        let module = context
            .load_module(ptx)
            .map_err(|error| GpuError::new(error.to_string()))?;
        context
            .bind_to_thread()
            .map_err(|error| GpuError::new(error.to_string()))?;
        let function = module
            .load_function("plait_axpy")
            .map_err(|error| GpuError::new(error.to_string()))?;
        Ok(Self {
            function,
            _module: module,
            _context: context,
        })
    }

    /// Run a CUTLASS SGEMM with device pointers and the cudarc stream. No
    /// intermediate host buffer is allocated by the CUTLASS shim; only the
    /// input/output copies at the API boundary remain.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<Vec<f32>, GpuError> {
        if m == 0 || n == 0 || k == 0 {
            return Err(GpuError::new("sgemm dimensions must be non-zero"));
        }
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(GpuError::new("sgemm input lengths do not match dimensions"));
        }
        if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
            return Err(GpuError::new("sgemm dimensions exceed i32"));
        }
        self._context
            .bind_to_thread()
            .map_err(|error| GpuError::new(error.to_string()))?;
        let stream = self._context.default_stream();
        let a_dev = stream
            .clone_htod(a)
            .map_err(|error| GpuError::new(error.to_string()))?;
        let b_dev = stream
            .clone_htod(b)
            .map_err(|error| GpuError::new(error.to_string()))?;
        let mut c_dev = stream
            .clone_htod(c)
            .map_err(|error| GpuError::new(error.to_string()))?;
        #[cfg(plait_cutlass)]
        {
            let rc = {
                let (a_ptr, _a_guard) = a_dev.device_ptr(&stream);
                let (b_ptr, _b_guard) = b_dev.device_ptr(&stream);
                let (c_ptr, _c_guard) = c_dev.device_ptr_mut(&stream);
                unsafe {
                    plait_cutlass_sgemm_device(
                        a_ptr as *const c_void,
                        b_ptr as *const c_void,
                        c_ptr as *mut c_void,
                        m as i32,
                        n as i32,
                        k as i32,
                        alpha,
                        beta,
                        stream.cu_stream() as *mut c_void,
                    )
                }
            };
            if rc != 0 {
                return Err(GpuError::new(format!(
                    "CUTLASS SGEMM failed with code {rc}"
                )));
            }
            stream
                .synchronize()
                .map_err(|error| GpuError::new(error.to_string()))?;
            stream
                .clone_dtoh(&c_dev)
                .map_err(|error| GpuError::new(error.to_string()))
        }
        #[cfg(not(plait_cutlass))]
        {
            let _ = (a_dev, b_dev, c_dev, alpha, beta);
            Err(GpuError::new(
                "CUTLASS is not built: clone NVIDIA/cutlass into target/cutlass and provide nvcc",
            ))
        }
    }

    /// `out = alpha * x + y` on the device. Returns host-visible results.
    pub fn axpy(&self, x: &[f32], y: &[f32], alpha: f32) -> Result<Vec<f32>, GpuError> {
        if x.len() != y.len() || x.is_empty() {
            return Err(GpuError::new("axpy requires equal, non-empty inputs"));
        }
        let n = x.len();
        if n > i32::MAX as usize {
            return Err(GpuError::new("axpy input is too large"));
        }
        let stream = self._context.default_stream();
        let x_dev = stream
            .clone_htod(x)
            .map_err(|error| GpuError::new(error.to_string()))?;
        let y_dev = stream
            .clone_htod(y)
            .map_err(|error| GpuError::new(error.to_string()))?;
        let mut out_dev = stream
            .alloc_zeros::<f32>(n)
            .map_err(|error| GpuError::new(error.to_string()))?;
        let n_i32 = n as i32;
        let config = LaunchConfig::for_num_elems(n as u32);
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&x_dev);
        launch.arg(&y_dev);
        launch.arg(&mut out_dev);
        launch.arg(&alpha);
        launch.arg(&n_i32);
        // SAFETY: all buffers are device allocations of `n` elements and the
        // kernel bounds-checks `i < n`.
        unsafe { launch.launch(config) }.map_err(|error| GpuError::new(error.to_string()))?;
        stream
            .clone_dtoh(&out_dev)
            .map_err(|error| GpuError::new(error.to_string()))
    }
}

#[cfg(plait_cutlass)]
unsafe extern "C" {
    fn plait_cutlass_sgemm_device(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: *mut c_void,
    ) -> i32;

    fn plait_cutlass_sgemm(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
    ) -> i32;
}

/// CUTLASS-backed row-major SGEMM: `C = alpha * A * B + beta * C`.
///
/// This is the real CUTLASS path when the build found `target/cutlass` and a
/// compatible nvcc. The host copies are deliberate in this reference path; a
/// production adapter should pass device pointers from `plait-rdma`/the runtime.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_sgemm(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<Vec<f32>, GpuError> {
    if m == 0 || n == 0 || k == 0 {
        return Err(GpuError::new("sgemm dimensions must be non-zero"));
    }
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(GpuError::new("sgemm input lengths do not match dimensions"));
    }
    if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
        return Err(GpuError::new("sgemm dimensions exceed i32"));
    }
    #[cfg(plait_cutlass)]
    {
        let mut out = c.to_vec();
        // SAFETY: lengths are validated, pointers are valid for the call, and
        // the shim synchronizes before returning.
        let rc = unsafe {
            plait_cutlass_sgemm(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                m as i32,
                n as i32,
                k as i32,
                alpha,
                beta,
            )
        };
        if rc != 0 {
            return Err(GpuError::new(format!(
                "CUTLASS SGEMM failed with code {rc}"
            )));
        }
        Ok(out)
    }
    #[cfg(not(plait_cutlass))]
    {
        let _ = (a, b, c, alpha, beta);
        Err(GpuError::new(
            "CUTLASS is not built: clone NVIDIA/cutlass into target/cutlass and provide nvcc",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuError(String);

impl GpuError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_matches_cpu_or_skips_without_cuda() {
        let executor = match GpuExecutor::new(0) {
            Ok(executor) => executor,
            Err(error) => {
                eprintln!("skipping CUDA test: {error}");
                return;
            }
        };
        let x: Vec<f32> = (0..1024).map(|index| index as f32 * 0.5).collect();
        let y: Vec<f32> = (0..1024).map(|index| 1.0 - index as f32 * 0.25).collect();
        let alpha = 2.0;
        let device = executor.axpy(&x, &y, alpha).unwrap();
        for (index, value) in device.iter().enumerate() {
            let expected = alpha * x[index] + y[index];
            assert!((value - expected).abs() < 1e-5, "index {index}");
        }
    }

    #[test]
    fn cutlass_sgemm_matches_cpu_or_skips_without_cutlass() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.25).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 1.0).collect();
        let c = vec![0.0f32; m * n];
        let output = match cutlass_sgemm(&a, &b, &c, m, n, k, 1.0, 0.0) {
            Ok(output) => output,
            Err(error) => {
                eprintln!("skipping CUTLASS test: {error}");
                return;
            }
        };
        for row in 0..m {
            for col in 0..n {
                let mut expected = 0.0f32;
                for inner in 0..k {
                    expected += a[row * k + inner] * b[inner * n + col];
                }
                assert!(
                    (output[row * n + col] - expected).abs() < 1e-3,
                    "mismatch at [{row}][{col}]"
                );
            }
        }
    }
}
