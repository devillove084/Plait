//! Device-resident W4A16 execution shared by the native backends.

use std::cell::OnceCell;
use std::ffi::c_void;
use std::mem::size_of_val;
use std::ptr::{self, NonNull};
use std::rc::Rc;

use libloading::Library;

use crate::native::Shape;
use crate::quantized::valid_group;
use crate::{
    ActivationBits, LinearOutput, OperatorError, OutputType, QuantizedActivations, W4Weights,
    humming_layout, marlin_layout,
};

type QuantizedLaunch = unsafe extern "C" fn(
    *mut c_void,
    *const Shape,
    u32,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut *mut c_void,
) -> i32;

type QuantizeF16 = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    *const c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut *mut c_void,
) -> i32;

#[repr(C)]
struct DeviceApi {
    open: unsafe extern "C" fn(i32, *mut *mut c_void) -> i32,
    close: unsafe extern "C" fn(*mut c_void),
    buffer_new: unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut *mut c_void) -> i32,
    buffer_free: unsafe extern "C" fn(*mut c_void, *mut c_void),
    buffer_read: unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void, usize) -> i32,
    launch: unsafe extern "C" fn(
        *mut c_void,
        *const Shape,
        *const c_void,
        *const c_void,
        *const c_void,
        *const c_void,
        *const c_void,
        *mut c_void,
        *mut *mut c_void,
    ) -> i32,
    wait: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    completion_free: unsafe extern "C" fn(*mut c_void, *mut c_void),
}

struct Session<'lib> {
    api: &'lib DeviceApi,
    library: &'lib Library,
    handle: NonNull<c_void>,
    quantized_launch: Option<QuantizedLaunch>,
    quantize_f16: Option<QuantizeF16>,
}

#[repr(C)]
struct MarlinApi {
    probe: unsafe extern "C" fn(*mut c_void) -> i32,
    launch: unsafe extern "C" fn(
        *mut c_void,
        *const Shape,
        *const c_void,
        *const c_void,
        *const c_void,
        *mut c_void,
        *mut c_void,
        *mut *mut c_void,
    ) -> i32,
}

#[repr(C)]
struct HummingApi {
    probe: unsafe extern "C" fn(*mut c_void, u32, u32, u32) -> i32,
    repack: unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void) -> i32,
    launch: unsafe extern "C" fn(
        *mut c_void,
        *const Shape,
        *const c_void,
        *const c_void,
        *const c_void,
        *const c_void,
        *mut c_void,
        *mut *mut c_void,
    ) -> i32,
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.handle.as_ptr()) };
    }
}

/// Owns an in-order device queue. Prepared weights and tensors keep the queue
/// alive and can only be used by the session that allocated them.
pub struct DeviceSession<'lib>(Rc<Session<'lib>>);

#[derive(Clone, Copy)]
pub enum W4Activation {
    F16,
    Host(ActivationBits),
    Device(ActivationBits),
}

#[derive(Clone, Copy, Default)]
pub enum W4ProjectionKernel {
    #[default]
    Native,
    Humming,
    Marlin,
}

/// The prepared kernel, precision and bias for one decoder projection.
pub struct PreparedW4Projection<'lib> {
    weights: PreparedW4A16<'lib>,
    bias: Option<DeviceTensor<'lib>>,
    activation: W4Activation,
}

impl<'lib> DeviceSession<'lib> {
    pub fn prepare_w4_projection(
        &self,
        weights: W4Weights<'_>,
        bias_f16: Option<&[u16]>,
        activation: W4Activation,
        kernel: W4ProjectionKernel,
    ) -> Result<PreparedW4Projection<'lib>, OperatorError> {
        let bias = match kernel {
            W4ProjectionKernel::Native => {
                if !matches!(activation, W4Activation::F16) {
                    self.0.quantized_launcher()?;
                }
                if matches!(activation, W4Activation::Device(_))
                    && !self.supports_f16_quantization()
                {
                    return Err(OperatorError::Unsupported(
                        "device library has no FP16 activation quantizer",
                    ));
                }
                bias_f16
                    .map(|values| self.upload_f16(1, weights.n, values))
                    .transpose()?
            }
            W4ProjectionKernel::Humming | W4ProjectionKernel::Marlin => {
                if !matches!(activation, W4Activation::F16) || bias_f16.is_some() {
                    return Err(OperatorError::Unsupported(
                        "Humming and Marlin projections require A16 without bias",
                    ));
                }
                None
            }
        };
        let weights = match kernel {
            W4ProjectionKernel::Native => self.prepare(weights)?,
            W4ProjectionKernel::Humming => self.prepare_humming(weights)?,
            W4ProjectionKernel::Marlin => self.prepare_marlin(weights)?,
        };
        Ok(PreparedW4Projection {
            weights,
            bias,
            activation,
        })
    }

    /// Return host FP32 values. Humming and Marlin round the device result to
    /// FP16 before widening it for the decoder's host operations.
    pub fn project_w4(
        &self,
        projection: &PreparedW4Projection<'lib>,
        values: &[u16],
    ) -> Result<Vec<f32>, OperatorError> {
        let weights = &projection.weights;
        if matches!(weights.kernel, Kernel::Reference { .. }) {
            return self.project_w4_f32(
                weights,
                values,
                projection.bias.as_ref(),
                projection.activation,
            );
        }
        let rows = crate::w4a16::checked_rows(weights.k, weights.n, values.len())?;
        let input = self.upload_f16(rows, weights.k, values)?;
        match weights.submit(&input, None, OutputType::F16)?.read()? {
            LinearOutput::F16(values) => Ok(values
                .into_iter()
                .map(|bits| half::f16::from_bits(bits).to_f32())
                .collect()),
            LinearOutput::F32(_) => Err(OperatorError::Backend(
                "optimized W4 projection requires FP16 output".into(),
            )),
        }
    }

    pub fn project_w4_f32(
        &self,
        weights: &PreparedW4A16<'lib>,
        activations_f16: &[u16],
        bias: Option<&DeviceTensor<'lib>>,
        activation: W4Activation,
    ) -> Result<Vec<f32>, OperatorError> {
        let rows = crate::w4a16::checked_rows(weights.k, weights.n, activations_f16.len())?;
        let upload = || self.upload_f16(rows, weights.k, activations_f16);
        let output = match activation {
            W4Activation::F16 => weights.submit(&upload()?, bias, OutputType::F32)?.read()?,
            W4Activation::Host(bits) => {
                let host = QuantizedActivations::from_f16(
                    bits,
                    weights.k,
                    weights.group_size,
                    activations_f16,
                )?;
                weights
                    .submit_quantized(&self.upload_quantized(&host)?, bias, OutputType::F32)?
                    .read()?
            }
            W4Activation::Device(bits) => {
                let input = upload()?;
                let low = self.quantize_f16(&input, bits, weights.group_size)?;
                weights
                    .submit_quantized(&low, bias, OutputType::F32)?
                    .read()?
            }
        };
        match output {
            LinearOutput::F32(values) => Ok(values),
            LinearOutput::F16(_) => Err(OperatorError::Backend(
                "W4 projection requires FP32 output".into(),
            )),
        }
    }

    pub(crate) fn open(library: &'lib Library, ordinal: i32) -> Result<Self, OperatorError> {
        if ordinal < 0 {
            return Err(OperatorError::Shape("device ordinal must be nonnegative"));
        }
        type Entry = unsafe extern "C" fn() -> *const DeviceApi;
        let entry = unsafe { library.get::<Entry>(b"plait_w4a16_device_api_v1") }
            .map_err(|_| OperatorError::Unsupported("library has no resident W4A16 device API"))?;
        // The caller of NativeW4A16::load guarantees the library's ABI.
        let api = unsafe { entry().as_ref() }.ok_or_else(|| {
            OperatorError::Backend("resident W4A16 device API is null".to_owned())
        })?;
        let mut handle = ptr::null_mut();
        status("open device session", unsafe {
            (api.open)(ordinal, &mut handle)
        })?;
        Ok(Self(Rc::new(Session {
            api,
            library,
            handle: handle_or_error(handle)?,
            quantized_launch: unsafe { library.get(b"plait_w4_quantized_launch_v1") }
                .ok()
                .map(|symbol| *symbol),
            quantize_f16: unsafe { library.get(b"plait_w4_quantize_f16_v1") }
                .ok()
                .map(|symbol| *symbol),
        })))
    }

    pub fn prepare(&self, weights: W4Weights<'_>) -> Result<PreparedW4A16<'lib>, OperatorError> {
        let implicit_zero;
        let zero = match weights.zero_points {
            Some(points) => points,
            None => {
                implicit_zero = vec![8; weights.scales.len()];
                &implicit_zero
            }
        };
        Ok(PreparedW4A16 {
            k: weights.k,
            n: weights.n,
            group_size: weights.group_size,
            packed: Buffer::upload(&self.0, weights.packed)?,
            scale: Buffer::upload(&self.0, weights.scales)?,
            kernel: Kernel::Reference {
                zero: Buffer::upload(&self.0, zero)?,
            },
        })
    }

    /// Prepare a Marlin kernel compiled for this device on the same queue. The result
    /// accepts FP16 output without bias and shares tensors with `prepare`.
    pub fn prepare_marlin(
        &self,
        weights: W4Weights<'_>,
    ) -> Result<PreparedW4A16<'lib>, OperatorError> {
        let session = &self.0;
        type Entry = unsafe extern "C" fn() -> *const MarlinApi;
        let entry = unsafe { session.library.get::<Entry>(b"plait_w4a16_marlin_api_v1") }
            .map_err(|_| OperatorError::Unsupported("library has no resident Marlin kernel"))?;
        let api = unsafe { entry().as_ref() }
            .ok_or_else(|| OperatorError::Backend("resident Marlin API is null".to_owned()))?;
        let probe = unsafe { (api.probe)(session.handle.as_ptr()) };
        if probe == -1 {
            return Err(OperatorError::Unsupported(
                "the Marlin library was built for a different device architecture",
            ));
        }
        status("probe Marlin device", probe)?;
        let (packed, scales, _) = marlin_layout::pack(weights)?;
        let workspace_bytes = weights.n as usize / 128 * 16 * size_of::<i32>();
        Ok(PreparedW4A16 {
            k: weights.k,
            n: weights.n,
            group_size: weights.group_size,
            packed: Buffer::upload(session, &packed)?,
            scale: Buffer::upload(session, &scales)?,
            kernel: Kernel::Marlin {
                api,
                workspace: Buffer::new(session, ptr::null(), workspace_bytes)?,
            },
        })
    }

    /// Prepare the Humming GEMM compiled for this device, shape and group size.
    /// The prepared kernel produces FP16 and accepts tensors from this queue.
    pub fn prepare_humming(
        &self,
        weights: W4Weights<'_>,
    ) -> Result<PreparedW4A16<'lib>, OperatorError> {
        let session = &self.0;
        type Entry = unsafe extern "C" fn() -> *const HummingApi;
        let entry = unsafe { session.library.get::<Entry>(b"plait_w4a16_humming_api_v1") }
            .map_err(|_| OperatorError::Unsupported("library has no resident Humming kernel"))?;
        let api = unsafe { entry().as_ref() }
            .ok_or_else(|| OperatorError::Backend("resident Humming API is null".to_owned()))?;
        let probe = unsafe {
            (api.probe)(
                session.handle.as_ptr(),
                weights.n,
                weights.k,
                weights.group_size,
            )
        };
        if probe == -1 {
            return Err(OperatorError::Unsupported(
                "Humming library has no kernel for this device and weight shape",
            ));
        }
        status("probe Humming device", probe)?;
        let (scales, zeros) = humming_layout::group_parameters(weights);
        let canonical = Buffer::upload(session, weights.packed)?;
        let packed = Buffer::new(session, ptr::null(), weights.packed.len())?;
        status("repack Humming weights", unsafe {
            (api.repack)(session.handle.as_ptr(), canonical.ptr(), packed.ptr())
        })?;
        Ok(PreparedW4A16 {
            k: weights.k,
            n: weights.n,
            group_size: weights.group_size,
            packed,
            scale: Buffer::upload(session, &scales)?,
            kernel: Kernel::Humming {
                api,
                zero: Buffer::upload(session, &zeros)?,
            },
        })
    }

    /// Upload a row-major FP16 tensor. The host slice may be reused on return.
    pub fn upload_f16(
        &self,
        rows: u32,
        columns: u32,
        values: &[u16],
    ) -> Result<DeviceTensor<'lib>, OperatorError> {
        let elements = matrix_elements(rows, columns)?;
        if values.len() != elements {
            return Err(OperatorError::Shape(
                "data does not match the FP16 tensor shape",
            ));
        }
        Ok(DeviceTensor {
            rows,
            columns,
            output: OutputType::F16,
            buffer: Buffer::upload(&self.0, values)?,
            completion: None,
        })
    }

    pub fn upload_quantized(
        &self,
        input: &QuantizedActivations,
    ) -> Result<DeviceQuantizedActivations<'lib>, OperatorError> {
        self.0.quantized_launcher()?;
        Ok(DeviceQuantizedActivations {
            rows: input.rows,
            columns: input.columns,
            group_size: input.group_size,
            bits: input.bits,
            codes: Buffer::upload(&self.0, &input.packed)?,
            scale: Buffer::upload(&self.0, &input.scales)?,
            completion: None,
        })
    }

    pub fn supports_f16_quantization(&self) -> bool {
        self.0.quantized_launch.is_some() && self.0.quantize_f16.is_some()
    }

    /// Quantize an existing FP16 allocation without moving its data to the host.
    /// A non-finite input is reported on a downstream wait or read.
    pub fn quantize_f16(
        &self,
        input: &DeviceTensor<'lib>,
        bits: ActivationBits,
        group_size: u32,
    ) -> Result<DeviceQuantizedActivations<'lib>, OperatorError> {
        let session = &self.0;
        session.quantized_launcher()?;
        let quantize = session.quantize_f16.ok_or(OperatorError::Unsupported(
            "device library has no FP16 activation quantizer",
        ))?;
        if !Rc::ptr_eq(session, &input.buffer.session) {
            return Err(OperatorError::Shape(
                "tensor belongs to another device session",
            ));
        }
        if input.output != OutputType::F16 || !valid_group(input.columns, group_size) {
            return Err(OperatorError::Shape(
                "invalid FP16 activation or quantization group",
            ));
        }
        let rows = input.rows;
        let columns = input.columns;
        let packed_bytes = rows as usize * bits.row_bytes(columns as usize);
        let scale_bytes = matrix_elements(rows, columns.div_ceil(group_size))?
            .checked_mul(size_of::<f32>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(OperatorError::Shape(
                "activation scales exceed the addressable size",
            ))?;
        let codes = Buffer::new(session, ptr::null(), packed_bytes)?;
        let scale = Buffer::new(session, ptr::null(), scale_bytes)?;
        let completion = if rows == 0 {
            None
        } else {
            let invalid = Buffer::new(session, ptr::null(), size_of::<u32>())?;
            let mut event = ptr::null_mut();
            status("quantize FP16 activation", unsafe {
                quantize(
                    session.handle.as_ptr(),
                    rows,
                    columns,
                    group_size,
                    bits as u32,
                    input.buffer.ptr(),
                    codes.ptr(),
                    scale.ptr(),
                    invalid.ptr(),
                    &mut event,
                )
            })?;
            Some(Rc::new(Completion {
                session: Rc::clone(session),
                handle: handle_or_error(event)?,
                dependencies: [input.completion.clone(), None],
                invalid: Some(invalid),
                result: OnceCell::new(),
            }))
        };
        Ok(DeviceQuantizedActivations {
            rows,
            columns,
            group_size,
            bits,
            codes,
            scale,
            completion,
        })
    }
}

impl Session<'_> {
    fn quantized_launcher(&self) -> Result<QuantizedLaunch, OperatorError> {
        self.quantized_launch.ok_or(OperatorError::Unsupported(
            "device library has no INT4 or INT8 activation kernel",
        ))
    }
}

/// Resident packed activation codes and one FP32 scale per row and K group.
pub struct DeviceQuantizedActivations<'lib> {
    rows: u32,
    columns: u32,
    group_size: u32,
    bits: ActivationBits,
    codes: Buffer<'lib>,
    scale: Buffer<'lib>,
    completion: Option<Rc<Completion<'lib>>>,
}

/// One resident packed weight matrix. Reuse it across submissions.
pub struct PreparedW4A16<'lib> {
    k: u32,
    n: u32,
    group_size: u32,
    packed: Buffer<'lib>,
    scale: Buffer<'lib>,
    kernel: Kernel<'lib>,
}

enum Kernel<'lib> {
    Reference {
        zero: Buffer<'lib>,
    },
    Marlin {
        api: &'lib MarlinApi,
        workspace: Buffer<'lib>,
    },
    Humming {
        api: &'lib HummingApi,
        zero: Buffer<'lib>,
    },
}

impl<'lib> PreparedW4A16<'lib> {
    pub fn input_columns(&self) -> u32 {
        self.k
    }

    pub fn group_size(&self) -> u32 {
        self.group_size
    }

    /// Run a signed INT4 or INT8 activation against canonical INT4 weights.
    pub fn submit_quantized(
        &self,
        input: &DeviceQuantizedActivations<'lib>,
        bias: Option<&DeviceTensor<'lib>>,
        output: OutputType,
    ) -> Result<DeviceTensor<'lib>, OperatorError> {
        let Kernel::Reference { zero } = &self.kernel else {
            return Err(OperatorError::Unsupported(
                "the integer activation kernel requires canonical weights",
            ));
        };
        let session = &self.packed.session;
        if !Rc::ptr_eq(session, &input.codes.session) {
            return Err(OperatorError::Shape(
                "tensor belongs to another device session",
            ));
        }
        self.check_bias(bias)?;
        if input.columns != self.k || input.group_size != self.group_size {
            return Err(OperatorError::Shape(
                "activation and weight groups must match",
            ));
        }
        let launch = session.quantized_launcher()?;
        self.submit_output(
            input.rows,
            output,
            [
                input.completion.clone(),
                bias.and_then(|tensor| tensor.completion.clone()),
            ],
            |shape, buffer, event| unsafe {
                launch(
                    session.handle.as_ptr(),
                    shape,
                    input.bits as u32,
                    input.codes.ptr(),
                    input.scale.ptr(),
                    self.packed.ptr(),
                    self.scale.ptr(),
                    zero.ptr(),
                    bias.map_or(ptr::null_mut(), |tensor| tensor.buffer.ptr()),
                    buffer,
                    event,
                )
            },
        )
    }

    fn check_bias(&self, bias: Option<&DeviceTensor<'lib>>) -> Result<(), OperatorError> {
        if let Some(bias) = bias {
            if !Rc::ptr_eq(&self.packed.session, &bias.buffer.session) {
                return Err(OperatorError::Shape(
                    "tensor belongs to another device session",
                ));
            }
            if bias.rows != 1 || bias.columns != self.n || bias.output != OutputType::F16 {
                return Err(OperatorError::Shape(
                    "bias must be a single FP16 row with N columns",
                ));
            }
        }
        Ok(())
    }

    /// Enqueue a linear operation, using tensors on this session's device.
    /// The output may immediately feed another submission on the same queue.
    pub fn submit(
        &self,
        input: &DeviceTensor<'lib>,
        bias: Option<&DeviceTensor<'lib>>,
        output: OutputType,
    ) -> Result<DeviceTensor<'lib>, OperatorError> {
        let session = &self.packed.session;
        if !Rc::ptr_eq(session, &input.buffer.session) {
            return Err(OperatorError::Shape(
                "tensor belongs to another device session",
            ));
        }
        if input.columns != self.k || input.output != OutputType::F16 {
            return Err(OperatorError::Shape(
                "activation must be FP16 with K columns",
            ));
        }
        self.check_bias(bias)?;
        if matches!(self.kernel, Kernel::Marlin { .. }) {
            if bias.is_some() || output != OutputType::F16 {
                return Err(OperatorError::Unsupported(
                    "Marlin requires an FP16 output without fused bias",
                ));
            }
            if (input.rows as u64) * (self.k.max(self.n) as u64) > i32::MAX as u64 {
                return Err(OperatorError::Shape(
                    "Marlin activation or output exceeds the indexing range",
                ));
            }
        }
        if matches!(self.kernel, Kernel::Humming { .. })
            && (bias.is_some() || output != OutputType::F16)
        {
            return Err(OperatorError::Unsupported(
                "Humming requires an FP16 output without fused bias",
            ));
        }
        self.submit_output(
            input.rows,
            output,
            [
                input.completion.clone(),
                bias.and_then(|tensor| tensor.completion.clone()),
            ],
            |shape, buffer, event| unsafe {
                match &self.kernel {
                    Kernel::Reference { zero } => (session.api.launch)(
                        session.handle.as_ptr(),
                        shape,
                        input.buffer.ptr(),
                        self.packed.ptr(),
                        self.scale.ptr(),
                        zero.ptr(),
                        bias.map_or(ptr::null_mut(), |tensor| tensor.buffer.ptr()),
                        buffer,
                        event,
                    ),
                    Kernel::Marlin { api, workspace } => (api.launch)(
                        session.handle.as_ptr(),
                        shape,
                        input.buffer.ptr(),
                        self.packed.ptr(),
                        self.scale.ptr(),
                        workspace.ptr(),
                        buffer,
                        event,
                    ),
                    Kernel::Humming { api, zero } => (api.launch)(
                        session.handle.as_ptr(),
                        shape,
                        input.buffer.ptr(),
                        self.packed.ptr(),
                        self.scale.ptr(),
                        zero.ptr(),
                        buffer,
                        event,
                    ),
                }
            },
        )
    }

    fn submit_output(
        &self,
        rows: u32,
        output: OutputType,
        dependencies: [Option<Rc<Completion<'lib>>>; 2],
        submit: impl FnOnce(&Shape, *mut c_void, *mut *mut c_void) -> i32,
    ) -> Result<DeviceTensor<'lib>, OperatorError> {
        let session = &self.packed.session;
        let elements = matrix_elements(rows, self.n)?;
        let output_bits = match output {
            OutputType::F16 => 16,
            OutputType::F32 => 32,
        };
        let bytes = elements
            .checked_mul(output_bits as usize / 8)
            .filter(|size| *size <= isize::MAX as usize)
            .ok_or(OperatorError::Shape("output exceeds the addressable size"))?;
        let buffer = Buffer::new(session, ptr::null(), bytes)?;
        let completion = if elements == 0 {
            None
        } else {
            let shape = Shape {
                m: rows,
                n: self.n,
                k: self.k,
                group_size: self.group_size,
                output_bits,
            };
            let mut event = ptr::null_mut();
            status("submit W4 linear", submit(&shape, buffer.ptr(), &mut event))?;
            Some(Rc::new(Completion {
                session: Rc::clone(session),
                handle: handle_or_error(event)?,
                dependencies,
                invalid: None,
                result: OnceCell::new(),
            }))
        };
        Ok(DeviceTensor {
            rows,
            columns: self.n,
            output,
            buffer,
            completion,
        })
    }
}

/// A row-major device allocation and the completion of its producer.
pub struct DeviceTensor<'lib> {
    rows: u32,
    columns: u32,
    output: OutputType,
    buffer: Buffer<'lib>,
    completion: Option<Rc<Completion<'lib>>>,
}

impl DeviceTensor<'_> {
    /// Wait for this tensor and propagate any failed upstream GPU operation.
    pub fn wait(&self) -> Result<(), OperatorError> {
        self.completion
            .as_ref()
            .map_or(Ok(()), |event| event.wait())
    }

    /// Materialize the result on the host. Normal same-device chaining can
    /// pass the tensor directly to the next operation.
    pub fn read(&self) -> Result<LinearOutput, OperatorError> {
        self.wait()?;
        let elements = self.rows as usize * self.columns as usize;
        let mut output = LinearOutput::zeros(self.output, elements);
        match &mut output {
            LinearOutput::F16(values) => self.buffer.read(values)?,
            LinearOutput::F32(values) => self.buffer.read(values)?,
        }
        Ok(output)
    }
}

struct Buffer<'lib> {
    session: Rc<Session<'lib>>,
    handle: Option<NonNull<c_void>>,
}

impl<'lib> Buffer<'lib> {
    fn upload<T>(session: &Rc<Session<'lib>>, source: &[T]) -> Result<Self, OperatorError> {
        Self::new(session, source.as_ptr().cast(), size_of_val(source))
    }

    fn new(
        session: &Rc<Session<'lib>>,
        source: *const c_void,
        bytes: usize,
    ) -> Result<Self, OperatorError> {
        let handle = if bytes == 0 {
            None
        } else {
            let mut buffer = ptr::null_mut();
            status("allocate device buffer", unsafe {
                (session.api.buffer_new)(session.handle.as_ptr(), source, bytes, &mut buffer)
            })?;
            Some(handle_or_error(buffer)?)
        };
        Ok(Self {
            session: Rc::clone(session),
            handle,
        })
    }

    fn ptr(&self) -> *mut c_void {
        self.handle.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    fn read<T>(&self, destination: &mut [T]) -> Result<(), OperatorError> {
        if let Some(handle) = self.handle {
            status("read device buffer", unsafe {
                (self.session.api.buffer_read)(
                    self.session.handle.as_ptr(),
                    handle.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    size_of_val(destination),
                )
            })?;
        }
        Ok(())
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle {
            unsafe {
                (self.session.api.buffer_free)(self.session.handle.as_ptr(), handle.as_ptr())
            };
        }
    }
}

struct Completion<'lib> {
    session: Rc<Session<'lib>>,
    handle: NonNull<c_void>,
    dependencies: [Option<Rc<Completion<'lib>>>; 2],
    invalid: Option<Buffer<'lib>>,
    result: OnceCell<Result<(), OperatorError>>,
}

impl Completion<'_> {
    fn wait(&self) -> Result<(), OperatorError> {
        self.result
            .get_or_init(|| {
                let own = status("wait for W4 linear", unsafe {
                    (self.session.api.wait)(self.session.handle.as_ptr(), self.handle.as_ptr())
                });
                for dependency in self.dependencies.iter().flatten() {
                    dependency.wait()?;
                }
                own?;
                if let Some(invalid) = &self.invalid {
                    let mut flag = [0u32];
                    invalid.read(&mut flag)?;
                    if flag[0] != 0 {
                        return Err(OperatorError::Shape("quantized activations must be finite"));
                    }
                }
                Ok(())
            })
            .clone()
    }
}

impl Drop for Completion<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.session.api.completion_free)(self.session.handle.as_ptr(), self.handle.as_ptr())
        };
    }
}

fn matrix_elements(rows: u32, columns: u32) -> Result<usize, OperatorError> {
    rows.checked_mul(columns)
        .filter(|_| columns != 0)
        .map(|elements| elements as usize)
        .ok_or(OperatorError::Shape(
            "matrix dimensions exceed the device indexing range",
        ))
}

fn status(operation: &str, code: i32) -> Result<(), OperatorError> {
    if code == 0 {
        Ok(())
    } else {
        Err(OperatorError::Backend(format!(
            "{operation} failed ({code})"
        )))
    }
}

fn handle_or_error(handle: *mut c_void) -> Result<NonNull<c_void>, OperatorError> {
    NonNull::new(handle)
        .ok_or_else(|| OperatorError::Backend("backend returned a null handle".to_owned()))
}
