use std::time::Instant;

use plait_ops::OperatorError;
use plait_ops::decoder::{DecoderBlock, LinearProjection, TokenHead};
use plait_ops::native::DeviceSession;
use plait_parallel::precision::W4Quantization;
use serde::{Deserialize, Serialize};

use super::{DeviceProjection, ProjectionExecution, ProjectionInput, ProjectionKernel};
use crate::tensor_cli::w4::calibration_error;

#[derive(Deserialize)]
pub(super) struct Config {
    cases: Vec<Case>,
    max_error_ppm: u64,
    #[serde(default = "default_samples")]
    samples: usize,
}

#[derive(Clone, Deserialize)]
struct Case {
    #[serde(default)]
    prefix: Vec<f32>,
    hidden: Vec<f32>,
}

#[derive(Deserialize)]
pub(super) struct ModelConfig {
    cases: Vec<TokenCase>,
    max_error_ppm: u64,
    #[serde(default = "default_samples")]
    samples: usize,
}

#[derive(Deserialize)]
struct TokenCase {
    #[serde(default)]
    prefix_tokens: Vec<u32>,
    tokens: Vec<u32>,
}

pub(super) struct ModelCalibration {
    baseline: Vec<Case>,
    tuning: Config,
}

#[derive(Serialize)]
pub(super) struct TokenComparison {
    pub(super) baseline: Vec<u32>,
    pub(super) tuned: Vec<u32>,
}

impl ModelCalibration {
    pub(super) fn new(
        config: ModelConfig,
        head: &TokenHead<impl LinearProjection>,
        max_tokens: u32,
    ) -> Result<Self, OperatorError> {
        if config.cases.is_empty() || config.samples == 0 {
            return Err(OperatorError::Shape(
                "model tuning requires calibration cases and samples",
            ));
        }
        let cases = config
            .cases
            .into_iter()
            .map(|case| {
                if case.tokens.is_empty()
                    || case.prefix_tokens.len().saturating_add(case.tokens.len())
                        > max_tokens as usize
                {
                    return Err(OperatorError::Shape("invalid model tuning token count"));
                }
                Ok(Case {
                    prefix: head.embed(&case.prefix_tokens)?,
                    hidden: head.embed(&case.tokens)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            baseline: cases.clone(),
            tuning: Config {
                cases,
                max_error_ppm: config.max_error_ppm,
                samples: config.samples,
            },
        })
    }

    pub(super) fn layer<'a, 'lib>(
        &mut self,
        block: &mut DecoderBlock<DeviceProjection<'a, 'lib>>,
        device: &'a DeviceSession<'lib>,
        sources: &[ProjectionInput; 7],
        shapes: [(u32, u32); 7],
        group_size: u32,
        max_tokens: u32,
    ) -> Result<ResultProfile, OperatorError> {
        for case in &mut self.baseline {
            case.advance(block, max_tokens)?;
        }
        let result = tune(
            block,
            device,
            sources,
            shapes,
            group_size,
            max_tokens,
            &self.tuning,
        )?;
        for case in &mut self.tuning.cases {
            case.advance(block, max_tokens)?;
        }
        Ok(result)
    }

    pub(super) fn compare(
        &self,
        head: &TokenHead<impl LinearProjection>,
    ) -> Result<TokenComparison, OperatorError> {
        let choices = |cases: &[Case]| {
            cases
                .iter()
                .map(|case| head.next_token(&case.hidden))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(TokenComparison {
            baseline: choices(&self.baseline)?,
            tuned: choices(&self.tuning.cases)?,
        })
    }
}

impl Case {
    fn advance(
        &mut self,
        block: &DecoderBlock<impl LinearProjection>,
        max_tokens: u32,
    ) -> Result<(), OperatorError> {
        let mut cache = block.empty_cache(max_tokens)?;
        if !self.prefix.is_empty() {
            self.prefix = block.forward(&self.prefix, &mut cache)?;
        }
        self.hidden = block.forward(&self.hidden, &mut cache)?;
        Ok(())
    }
}

fn default_samples() -> usize {
    5
}

#[derive(Serialize)]
pub(super) struct ResultProfile {
    pub(super) projections: [ProjectionExecution; 7],
    baseline_latency_ns: u64,
    latency_ns: u64,
    error_ppm: u64,
}

struct Profile {
    latency_ns: u64,
    error_ppm: u64,
    reference: Vec<(Vec<f32>, f64)>,
}

const CANDIDATES: [ProjectionExecution; 6] = [
    integer(4, W4Quantization::Device),
    integer(8, W4Quantization::Device),
    integer(4, W4Quantization::Host),
    integer(8, W4Quantization::Host),
    accelerated(ProjectionKernel::Humming),
    accelerated(ProjectionKernel::Marlin),
];

const fn integer(bits: u32, quantization: W4Quantization) -> ProjectionExecution {
    ProjectionExecution {
        activation_bits: Some(bits),
        quantization: Some(quantization),
        kernel: ProjectionKernel::Native,
    }
}

const fn accelerated(kernel: ProjectionKernel) -> ProjectionExecution {
    ProjectionExecution {
        activation_bits: Some(16),
        quantization: None,
        kernel,
    }
}

pub(super) fn tune<'a, 'lib>(
    block: &mut DecoderBlock<DeviceProjection<'a, 'lib>>,
    device: &'a DeviceSession<'lib>,
    sources: &[ProjectionInput; 7],
    shapes: [(u32, u32); 7],
    group_size: u32,
    max_tokens: u32,
    config: &Config,
) -> Result<ResultProfile, OperatorError> {
    if config.samples == 0 || config.cases.is_empty() {
        return Err(OperatorError::Shape(
            "decoder tuning requires calibration cases and samples",
        ));
    }
    let baseline_execution = ProjectionExecution::default();
    if sources.iter().any(|source| {
        source.execution != baseline_execution
            && source.execution != accelerated(ProjectionKernel::Native)
    }) {
        return Err(OperatorError::Shape(
            "decoder tuning requires native A16 settings on the input projections",
        ));
    }
    let baseline = profile(block, max_tokens, config, None)?;
    let mut result = ResultProfile {
        projections: [accelerated(ProjectionKernel::Native); 7],
        baseline_latency_ns: baseline.latency_ns,
        latency_ns: baseline.latency_ns,
        error_ppm: 0,
    };
    for (index, (source, shape)) in sources.iter().zip(shapes).enumerate() {
        for execution in CANDIDATES {
            let candidate = match execution.prepare(
                device,
                &source.weights,
                source.bias_f16.as_deref(),
                shape,
                group_size,
            ) {
                Ok(candidate) => candidate,
                Err(OperatorError::Unsupported(_) | OperatorError::Shape(_)) => continue,
                Err(error) => return Err(error),
            };
            let previous = block.replace_projection(index, candidate)?;
            let measured = profile(block, max_tokens, config, Some(&baseline.reference));
            match measured {
                Ok(measured)
                    if measured.error_ppm <= config.max_error_ppm
                        && measured.latency_ns < result.latency_ns =>
                {
                    result.projections[index] = execution;
                    result.latency_ns = measured.latency_ns;
                    result.error_ppm = measured.error_ppm;
                    continue;
                }
                Ok(_) | Err(OperatorError::Unsupported(_) | OperatorError::Shape(_)) => {}
                Err(error) => return Err(error),
            }
            block.replace_projection(index, previous)?;
        }
    }
    Ok(result)
}

fn profile(
    block: &DecoderBlock<DeviceProjection<'_, '_>>,
    max_tokens: u32,
    config: &Config,
    baseline: Option<&[(Vec<f32>, f64)]>,
) -> Result<Profile, OperatorError> {
    let mut profile = Profile {
        latency_ns: 0,
        error_ppm: 0,
        reference: Vec::new(),
    };
    for (index, case) in config.cases.iter().enumerate() {
        let mut prefix = block.empty_cache(max_tokens)?;
        if !case.prefix.is_empty() {
            block.forward(&case.prefix, &mut prefix)?;
        }
        let start = prefix.snapshot();
        let mut warm = block.restore_cache(start.clone(), max_tokens)?;
        let output = block.forward(&case.hidden, &mut warm)?;
        if baseline.is_none() {
            let norm = output
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>()
                .sqrt();
            profile.reference.push((output, norm));
        }
        let mut samples = Vec::with_capacity(config.samples);
        for _ in 0..config.samples {
            let mut cache = block.restore_cache(start.clone(), max_tokens)?;
            let now = Instant::now();
            let output = block.forward(&case.hidden, &mut cache)?;
            samples.push(now.elapsed().as_nanos().clamp(1, u128::from(u64::MAX)) as u64);
            if let Some(baseline) = baseline {
                let (reference, norm) = &baseline[index];
                let error = if *norm == 0.0 {
                    if output != *reference {
                        return Err(OperatorError::Shape(
                            "decoder tuning changed a zero reference",
                        ));
                    }
                    0
                } else {
                    calibration_error(&output, reference, *norm).map_err(OperatorError::Shape)?
                };
                profile.error_ppm = profile.error_ppm.max(error);
            }
        }
        samples.sort_unstable();
        profile.latency_ns = profile.latency_ns.max(samples[samples.len() / 2]);
    }
    Ok(profile)
}
