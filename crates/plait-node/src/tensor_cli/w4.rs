use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use plait_core::serving::{GroupId, GroupPhase};
use plait_core::tensor::KPartial;
use plait_ops::native::{DeviceSession, NativeW4A16, PreparedW4A16, W4Activation};
use plait_ops::{ActivationBits, OperatorError, W4Weights};
use plait_parallel::precision::{
    W4Decision, W4Kernel, W4PlacementRequest, W4Profile, W4Quantization, W4Scope,
    W4SelectionRequest, plan_w4, select_w4,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{ControlPeers, GroupControl, Publish, partial_size, serve_partial};

pub(super) mod worker;

pub(super) use worker::run as run_worker;

#[derive(Deserialize)]
struct W4Device {
    library: PathBuf,
    ordinal: i32,
    group_size: u32,
}

#[derive(Deserialize)]
pub(super) struct PublishW4 {
    #[serde(flatten)]
    source: Publish,
    #[serde(flatten)]
    device: W4Device,
    activation_bits: Option<u32>,
    quantization: Option<W4Quantization>,
    decision: Option<W4Decision>,
    control: Option<ControlPeers>,
}

#[derive(Deserialize)]
pub(super) struct PlanW4 {
    #[serde(flatten)]
    request: W4PlacementRequest,
    control: Option<AdmissionControl>,
}

#[derive(Deserialize)]
struct AdmissionControl {
    group: GroupId,
    #[serde(flatten)]
    peers: ControlPeers,
}

#[derive(Deserialize)]
pub(super) struct ProfileW4 {
    partial: KPartial,
    #[serde(flatten)]
    device: W4Device,
    calibration_id: [u8; 32],
    layer_reference_l2: f64,
    delivery_ns: u64,
    samples: usize,
    valid_until_ms: u64,
    #[serde(default)]
    kernel_profiles: bool,
}

#[derive(Deserialize)]
struct W4Input {
    #[serde(flatten)]
    weights: W4WeightInput,
    activations_f16: Vec<u16>,
}

#[derive(Deserialize)]
struct W4WeightInput {
    packed: Vec<u8>,
    scales_f16: Vec<u16>,
    zero_points: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct W4ProfileInput {
    #[serde(flatten)]
    primary: W4Input,
    #[serde(default)]
    calibration: Vec<W4Calibration>,
}

#[derive(Deserialize)]
struct W4Calibration {
    activations_f16: Vec<u16>,
    layer_reference_l2: f64,
}

impl W4Device {
    fn check_partial(&self, partial: KPartial) -> Result<(), Box<dyn std::error::Error>> {
        partial_size(partial)?;
        if self.group_size == 0 || partial.k_start % self.group_size != 0 {
            return Err("W4 shard must start at a weight group boundary".into());
        }
        Ok(())
    }

    fn scope(&self, partial: KPartial, input: &W4WeightInput) -> io::Result<W4Scope> {
        let library_sha256 = Sha256::digest(std::fs::read(&self.library)?).into();
        Ok(W4Scope {
            device: partial.grant.device,
            incarnation: partial.grant.incarnation,
            ordinal: self.ordinal,
            library_sha256,
            weights_sha256: input.digest(
                partial.k_end - partial.k_start,
                partial.output_shape[1],
                self.group_size,
            ),
            k_range: [partial.k_start, partial.k_end],
            output_shape: partial.output_shape,
            group_size: self.group_size,
        })
    }

    fn weights<'a>(
        &self,
        partial: KPartial,
        input: &'a W4Input,
    ) -> Result<W4Weights<'a>, OperatorError> {
        let [rows, n] = partial.output_shape;
        let k = partial.k_end - partial.k_start;
        if u64::from(rows) * u64::from(k) != input.activations_f16.len() as u64 {
            return Err(OperatorError::Shape(
                "activation shape does not match the K partial",
            ));
        }
        input.weights.layout(k, n, self.group_size)
    }
}

impl W4WeightInput {
    fn layout(&self, k: u32, n: u32, group_size: u32) -> Result<W4Weights<'_>, OperatorError> {
        W4Weights::new(
            k,
            n,
            group_size,
            &self.packed,
            &self.scales_f16,
            self.zero_points.as_deref(),
        )
    }

    fn digest(&self, k: u32, n: u32, group_size: u32) -> [u8; 32] {
        let mut weights = Sha256::new();
        for dimension in [k, n, group_size] {
            weights.update(dimension.to_le_bytes());
        }
        weights.update(&self.packed);
        for scale in &self.scales_f16 {
            weights.update(scale.to_le_bytes());
        }
        weights.update([u8::from(self.zero_points.is_some())]);
        if let Some(zero) = &self.zero_points {
            weights.update(zero);
        }
        weights.finalize().into()
    }
}

struct W4Partition<'lib> {
    canonical: PreparedW4A16<'lib>,
    humming: Option<PreparedW4A16<'lib>>,
    marlin: Option<PreparedW4A16<'lib>>,
}

impl<'lib> W4Partition<'lib> {
    fn prepare(
        device: &DeviceSession<'lib>,
        weights: W4Weights<'_>,
        kernel_profiles: bool,
    ) -> Result<Self, OperatorError> {
        let canonical = device.prepare(weights)?;
        let humming = optional_kernel(device.prepare_humming_f32(weights))?;
        let marlin = if kernel_profiles || humming.is_none() {
            optional_kernel(device.prepare_marlin_f32(weights))?
        } else {
            None
        };
        Ok(Self {
            canonical,
            humming,
            marlin,
        })
    }

    fn resident_bytes(&self) -> u64 {
        [&self.humming, &self.marlin]
            .into_iter()
            .flatten()
            .fold(self.canonical.resident_bytes(), |bytes, kernel| {
                bytes.saturating_add(kernel.resident_bytes())
            })
    }

    fn project(
        &self,
        device: &DeviceSession<'lib>,
        activations: &[u16],
        activation: W4Activation,
        kernel: Option<W4Kernel>,
    ) -> Result<Vec<f32>, OperatorError> {
        let weights = match (kernel, activation) {
            (Some(W4Kernel::Humming), _) => self.humming.as_ref().ok_or(
                OperatorError::Unsupported("profiled Humming kernel is unavailable on this worker"),
            )?,
            (Some(W4Kernel::Marlin), _) => self.marlin.as_ref().ok_or(
                OperatorError::Unsupported("profiled Marlin kernel is unavailable on this worker"),
            )?,
            (None, W4Activation::F16) => self
                .humming
                .as_ref()
                .or(self.marlin.as_ref())
                .unwrap_or(&self.canonical),
            _ => &self.canonical,
        };
        device.project_w4_f32(weights, activations, None, activation)
    }
}

fn optional_kernel<'lib>(
    weights: Result<PreparedW4A16<'lib>, OperatorError>,
) -> Result<Option<PreparedW4A16<'lib>>, OperatorError> {
    match weights {
        Ok(weights) => Ok(Some(weights)),
        Err(OperatorError::Unsupported(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) async fn publish(command: PublishW4) -> Result<(), Box<dyn std::error::Error>> {
    let partial = command.source.partial;
    command.device.check_partial(partial)?;
    if command.decision.is_some()
        && (command.activation_bits.is_some() || command.quantization.is_some())
    {
        return Err("provide either a W4 decision or explicit activation settings".into());
    }
    let input: W4Input = serde_json::from_reader(io::stdin().lock())?;
    let weights = command.device.weights(partial, &input)?;
    let (bits, quantization, kernel) = if let Some(decision) = command.decision {
        let scope = command.device.scope(partial, &input.weights)?;
        validate_decision(decision, partial, scope)?;
        (
            decision.profile.activation_bits,
            decision.profile.quantization,
            decision.profile.kernel,
        )
    } else {
        (
            command.activation_bits.unwrap_or(16),
            command.quantization,
            None,
        )
    };
    let mut control = if let Some(peers) = command.control {
        let mut control = GroupControl::connect(peers).await?;
        control.check_worker(partial.grant, None).await?;
        Some(control)
    } else {
        None
    };
    if let Some(decision) = command.decision {
        check_expiry(decision)?;
    }
    let result = {
        // The local deployment configuration selects a trusted Plait device library.
        let backend = unsafe { NativeW4A16::load(&command.device.library) }?;
        let device = backend.open_device(command.device.ordinal)?;
        let activation = activation_format(bits, quantization, device.supports_f16_quantization())?;
        if let Some(kernel) = kernel {
            let prepared = match kernel {
                W4Kernel::Humming => device.prepare_humming_f32(weights)?,
                W4Kernel::Marlin => device.prepare_marlin_f32(weights)?,
                W4Kernel::Reference | W4Kernel::Integer => device.prepare(weights)?,
            };
            device.project_w4_f32(&prepared, &input.activations_f16, None, activation)?
        } else {
            let linear = W4Partition::prepare(&device, weights, false)?;
            linear.project(&device, &input.activations_f16, activation, None)?
        }
    };
    if let Some(control) = &mut control {
        control.check_worker(partial.grant, None).await?;
    }
    serve_partial(
        command.source,
        result.into_iter().flat_map(f32::to_le_bytes).collect(),
    )
    .await
}

fn validate_decision(
    decision: W4Decision,
    partial: KPartial,
    scope: W4Scope,
) -> Result<(), Box<dyn std::error::Error>> {
    if !decision.profile.is_valid() {
        return Err("invalid W4 profile".into());
    }
    if decision.shard.partial != partial
        || decision.shard.scope != scope
        || decision.profile.scope != scope
    {
        return Err(
            "W4 decision does not match the worker, invocation, or loaded artifacts".into(),
        );
    }
    Ok(())
}

fn check_expiry(decision: W4Decision) -> Result<(), Box<dyn std::error::Error>> {
    if decision.profile.valid_until_ms <= now_ms()? {
        return Err("the selected W4 profile has expired".into());
    }
    Ok(())
}

pub(super) fn profile(command: ProfileW4) -> Result<(), Box<dyn std::error::Error>> {
    let partial = command.partial;
    command.device.check_partial(partial)?;
    if command.samples == 0 || command.valid_until_ms <= now_ms()? {
        return Err("profile-w4 requires at least one sample and a future expiry".into());
    }
    if !command.layer_reference_l2.is_finite() || command.layer_reference_l2 <= 0.0 {
        return Err("profile-w4 requires a positive finite layer reference norm".into());
    }
    let input: W4ProfileInput = serde_json::from_reader(io::stdin().lock())?;
    let weights = command.device.weights(partial, &input.primary)?;
    let scope = command.device.scope(partial, &input.primary.weights)?;
    let mut cases = Vec::with_capacity(1 + input.calibration.len());
    cases.push((
        input.primary.activations_f16.as_slice(),
        command.layer_reference_l2,
    ));
    for case in &input.calibration {
        if case.activations_f16.len() != input.primary.activations_f16.len() {
            return Err("calibration activation shape does not match the K partial".into());
        }
        if !case.layer_reference_l2.is_finite() || case.layer_reference_l2 <= 0.0 {
            return Err("calibration requires a positive finite layer reference norm".into());
        }
        cases.push((case.activations_f16.as_slice(), case.layer_reference_l2));
    }
    // The local deployment configuration selects a trusted Plait device library.
    let backend = unsafe { NativeW4A16::load(&command.device.library) }?;
    let device = backend.open_device(command.device.ordinal)?;
    let linear = W4Partition::prepare(&device, weights, command.kernel_profiles)?;
    let mut profiles = Vec::new();
    let mut references: Vec<Vec<f32>> = Vec::with_capacity(cases.len());
    let device_quantization = device.supports_f16_quantization();
    let reference = command.kernel_profiles.then_some(W4Kernel::Reference);
    let integer = command.kernel_profiles.then_some(W4Kernel::Integer);
    let mut modes = vec![(16, None, reference)];
    if command.kernel_profiles {
        if linear.humming.is_some() {
            modes.push((16, None, Some(W4Kernel::Humming)));
        }
        if linear.marlin.is_some() {
            modes.push((16, None, Some(W4Kernel::Marlin)));
        }
    }
    modes.extend([
        (8, Some(W4Quantization::Host), integer),
        (8, Some(W4Quantization::Device), integer),
        (4, Some(W4Quantization::Host), integer),
        (4, Some(W4Quantization::Device), integer),
    ]);
    'modes: for (bits, quantization, kernel) in modes {
        if quantization == Some(W4Quantization::Device) && !device_quantization {
            continue;
        }
        let baseline = bits == 16 && kernel == reference;
        let activation = activation_format(bits, quantization, device_quantization)?;
        let mut latency_ns = 0;
        let mut error_ppm = 0;
        for (index, &(activations, layer_norm)) in cases.iter().enumerate() {
            let run = || linear.project(&device, activations, activation, kernel);
            match run() {
                Ok(_) => {}
                Err(OperatorError::Unsupported(_)) if !baseline => continue 'modes,
                Err(error) => return Err(error.into()),
            }
            let mut durations = Vec::with_capacity(command.samples);
            let mut reference = Vec::new();
            for _ in 0..command.samples {
                let started = Instant::now();
                let values = run()?;
                durations.push(u64::try_from(started.elapsed().as_nanos())?.max(1));
                if values.iter().any(|value| !value.is_finite()) {
                    if command.kernel_profiles && !baseline {
                        continue 'modes;
                    }
                    return Err("calibration produced a non-finite output".into());
                }
                if baseline {
                    reference = values;
                } else {
                    error_ppm =
                        error_ppm.max(calibration_error(&values, &references[index], layer_norm)?);
                }
            }
            if baseline {
                references.push(reference);
            }
            durations.sort_unstable();
            latency_ns = latency_ns.max(durations[durations.len() / 2]);
        }
        let latency_ns = latency_ns
            .checked_add(command.delivery_ns)
            .ok_or("W4 profile latency exceeds the range")?;
        profiles.push(W4Profile {
            scope,
            calibration_id: command.calibration_id,
            activation_bits: bits,
            quantization,
            kernel,
            latency_ns,
            error_ppm,
            valid_until_ms: command.valid_until_ms,
        });
    }
    write_json(&profiles)
}

fn calibration_error(
    values: &[f32],
    reference: &[f32],
    layer_norm: f64,
) -> Result<u64, &'static str> {
    let l2 = values
        .iter()
        .zip(reference)
        .map(|(actual, expected)| (f64::from(*actual) - f64::from(*expected)).powi(2))
        .sum::<f64>()
        .sqrt();
    let ppm = (l2 / layer_norm * 1_000_000.0).ceil();
    if !ppm.is_finite() || ppm >= u64::MAX as f64 {
        return Err("normalized W4 calibration error exceeds the range");
    }
    Ok(ppm as u64)
}

pub(super) fn select(request: W4SelectionRequest) -> Result<(), Box<dyn std::error::Error>> {
    write_json(&select_w4(&request)?)
}

pub(super) async fn plan(command: PlanW4) -> Result<(), Box<dyn std::error::Error>> {
    let mut request = command.request;
    if let Some(AdmissionControl { group, peers }) = command.control {
        if !request.workers.is_empty() {
            return Err("control obtains worker grants from the active group".into());
        }
        let mut control = GroupControl::connect(peers).await?;
        let view = control.observe(group, None).await?;
        if view.phase != GroupPhase::Active {
            return Err("the observed group no longer admits new work".into());
        }
        request.workers = view
            .grants
            .values()
            .copied()
            .filter(|grant| view.allows_grant(*grant))
            .collect();
        request.now_ms = now_ms()?;
    }
    write_json(&plan_w4(&request)?)
}

fn activation_format(
    bits: u32,
    quantization: Option<W4Quantization>,
    device_quantization: bool,
) -> Result<W4Activation, OperatorError> {
    let bits = match bits {
        16 if quantization.is_none() => return Ok(W4Activation::F16),
        16 => {
            return Err(OperatorError::Shape(
                "A16 takes no activation quantization path",
            ));
        }
        8 => ActivationBits::I8,
        4 => ActivationBits::I4,
        _ => {
            return Err(OperatorError::Unsupported(
                "activation_bits must be 4, 8 or 16",
            ));
        }
    };
    match quantization {
        Some(W4Quantization::Host) => Ok(W4Activation::Host(bits)),
        Some(W4Quantization::Device) if !device_quantization => Err(OperatorError::Unsupported(
            "device library has no FP16 activation quantizer",
        )),
        Some(W4Quantization::Device) => Ok(W4Activation::Device(bits)),
        None if device_quantization => Ok(W4Activation::Device(bits)),
        None => Ok(W4Activation::Host(bits)),
    }
}

pub(super) fn now_ms() -> io::Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?;
    u64::try_from(duration.as_millis()).map_err(io::Error::other)
}

fn write_json(value: &impl serde::Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)?;
    Ok(())
}
