use half::f16;

use crate::OperatorError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputType {
    F16,
    F32,
}

#[derive(Debug, PartialEq)]
pub enum LinearOutput {
    F16(Vec<u16>),
    F32(Vec<f32>),
}

impl LinearOutput {
    pub(crate) fn zeros(output: OutputType, elements: usize) -> Self {
        match output {
            OutputType::F16 => Self::F16(vec![0; elements]),
            OutputType::F32 => Self::F32(vec![0.0; elements]),
        }
    }
}

/// Row-major INT4 weights indexed by output channel and input column. Scales
/// are binary16 bits. Zero points are unsigned nibbles, with 8 implied when omitted.
#[derive(Clone, Copy, Debug)]
pub struct W4Weights<'a> {
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) group_size: u32,
    pub(crate) packed: &'a [u8],
    pub(crate) scales: &'a [u16],
    pub(crate) zero_points: Option<&'a [u8]>,
}

impl<'a> W4Weights<'a> {
    pub fn new(
        k: u32,
        n: u32,
        group_size: u32,
        packed: &'a [u8],
        scales: &'a [u16],
        zero_points: Option<&'a [u8]>,
    ) -> Result<Self, OperatorError> {
        if k == 0 || n == 0 || group_size == 0 {
            return Err(OperatorError::Shape(
                "W4 weights require positive K, N and group size",
            ));
        }
        let groups = k.div_ceil(group_size) as usize;
        let matrix = (n as usize).checked_mul(k.div_ceil(2) as usize);
        let parameters = (n as usize).checked_mul(groups);
        if matrix != Some(packed.len()) || parameters != Some(scales.len()) {
            return Err(OperatorError::Shape(
                "packed INT4 weights and FP16 scales must match the layout",
            ));
        }
        if zero_points
            .is_some_and(|points| points.len() != scales.len() || points.iter().any(|&z| z > 15))
        {
            return Err(OperatorError::Shape(
                "W4 zero points require one unsigned nibble per channel and group",
            ));
        }
        Ok(Self {
            k,
            n,
            group_size,
            packed,
            scales,
            zero_points,
        })
    }

    pub fn input_columns(self) -> u32 {
        self.k
    }

    pub fn output_columns(self) -> u32 {
        self.n
    }

    pub fn group_size(self) -> u32 {
        self.group_size
    }

    pub(crate) fn groups(self) -> usize {
        self.k.div_ceil(self.group_size) as usize
    }

    pub(crate) fn code(self, channel: usize, column: usize) -> u8 {
        let byte = self.packed[channel * self.k.div_ceil(2) as usize + column / 2];
        (byte >> (4 * (column % 2))) & 15
    }
}

/// A W4A16 invocation. Tensors are row-major, and activations and bias are
/// binary16 bits. F32 outputs can be reduced across logical K shards.
#[derive(Clone, Copy, Debug)]
pub struct W4A16<'a> {
    pub(crate) weights: W4Weights<'a>,
    pub(crate) rows: u32,
    pub(crate) activations: &'a [u16],
    pub(crate) bias: Option<&'a [u16]>,
    pub(crate) output: OutputType,
}

impl<'a> W4A16<'a> {
    pub fn new(
        weights: W4Weights<'a>,
        activations: &'a [u16],
        bias: Option<&'a [u16]>,
        output: OutputType,
    ) -> Result<Self, OperatorError> {
        if bias.is_some_and(|values| values.len() != weights.n as usize) {
            return Err(OperatorError::Shape(
                "bias does not match W4A16 output width",
            ));
        }
        let rows = checked_rows(weights.k, weights.n, activations.len())?;
        Ok(Self {
            weights,
            rows,
            activations,
            bias,
            output,
        })
    }

    pub fn rows(self) -> u32 {
        self.rows
    }

    pub(crate) fn output_len(self) -> usize {
        self.rows as usize * self.weights.n as usize
    }

    /// Portable FP32 accumulation with the same dequantization equation as
    /// the C reference kernels. This path also serves as a conformance oracle.
    pub fn run_cpu(self) -> LinearOutput {
        if self.rows == 0 {
            return LinearOutput::zeros(self.output, 0);
        }
        let weights = self.weights;
        let n = weights.n as usize;
        let k = weights.k as usize;
        let groups = weights.groups();
        let activations: Vec<f32> = self.activations.iter().map(|&x| decode(x)).collect();
        let mut output = LinearOutput::zeros(self.output, self.output_len());
        let mut dequantized = vec![0.0; k];
        for channel in 0..n {
            for (column, weight) in dequantized.iter_mut().enumerate() {
                let group = channel * groups + column / weights.group_size as usize;
                let zero = weights.zero_points.map_or(8, |points| points[group]);
                *weight = (f32::from(weights.code(channel, column)) - f32::from(zero))
                    * decode(weights.scales[group]);
            }
            for (row, activations) in activations.chunks_exact(k).enumerate() {
                let mut sum = 0.0;
                for (&activation, &weight) in activations.iter().zip(&dequantized) {
                    sum = activation.mul_add(weight, sum);
                }
                if let Some(bias) = self.bias {
                    sum += decode(bias[channel]);
                }
                let index = row * n + channel;
                match &mut output {
                    LinearOutput::F16(values) => values[index] = f16::from_f32(sum).to_bits(),
                    LinearOutput::F32(values) => values[index] = sum,
                }
            }
        }
        output
    }
}

pub(crate) fn checked_rows(k: u32, n: u32, activations: usize) -> Result<u32, OperatorError> {
    let k = k as usize;
    if activations % k != 0 {
        return Err(OperatorError::Shape(
            "activations must contain complete W4A16 rows",
        ));
    }
    u32::try_from(activations / k)
        .ok()
        .filter(|rows| rows.checked_mul(n).is_some())
        .ok_or(OperatorError::Shape(
            "W4A16 output exceeds the device indexing range",
        ))
}

fn decode(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}
