//! Symmetric activation quantization for groupwise INT4-weight linear layers.

use half::f16;
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};

use crate::{LinearOutput, OperatorError, OutputType, W4Weights};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ActivationBits {
    I4 = 4,
    I8 = 8,
}

impl ActivationBits {
    fn max_code(self) -> f32 {
        match self {
            Self::I4 => 7.0,
            Self::I8 => 127.0,
        }
    }

    pub(crate) fn row_bytes(self, columns: usize) -> usize {
        match self {
            Self::I4 => columns.div_ceil(2),
            Self::I8 => columns,
        }
    }
}

pub(crate) fn valid_group(columns: u32, group_size: u32) -> bool {
    columns != 0 && group_size != 0 && columns.min(group_size) <= i32::MAX as u32 / (128 * 15)
}

/// Row-major signed INT4 or INT8 activations with an FP32 scale per row and
/// K group. INT4 stores the even column in the low nibble, in two's complement.
pub struct QuantizedActivations {
    pub(crate) bits: ActivationBits,
    pub(crate) rows: u32,
    pub(crate) columns: u32,
    pub(crate) group_size: u32,
    pub(crate) packed: Vec<u8>,
    pub(crate) scales: Vec<f32>,
}

impl QuantizedActivations {
    /// Quantize finite IEEE binary16 input with one absmax scale per K group.
    pub fn from_f16(
        bits: ActivationBits,
        columns: u32,
        group_size: u32,
        activations: &[u16],
    ) -> Result<Self, OperatorError> {
        if !valid_group(columns, group_size)
            || activations.len() > u32::MAX as usize
            || activations.len() % columns as usize != 0
        {
            return Err(OperatorError::Shape(
                "invalid quantized activation dimensions",
            ));
        }
        let rows = activations.len() / columns as usize;
        let packed_stride = bits.row_bytes(columns as usize);
        let groups = columns.div_ceil(group_size) as usize;
        let mut packed = vec![0; rows * packed_stride];
        let mut scales = Vec::with_capacity(rows * groups);
        let limit = bits.max_code();
        let mut converted = vec![0.0; (columns.min(group_size) as usize).min(activations.len())];
        for (values, packed) in activations
            .chunks_exact(columns as usize)
            .zip(packed.chunks_exact_mut(packed_stride))
        {
            for (group, values) in values.chunks(group_size as usize).enumerate() {
                // IEEE binary16 encodings are ordered by magnitude after removing the sign.
                let max_bits = values.iter().map(|bits| bits & 0x7fff).max().unwrap();
                if max_bits >= f16::INFINITY.to_bits() {
                    return Err(OperatorError::Shape("quantized activations must be finite"));
                }
                let maximum = f16::from_bits(max_bits).to_f32();
                let scale = if maximum == 0.0 { 1.0 } else { maximum / limit };
                scales.push(scale);
                let group_values = &mut converted[..values.len()];
                values
                    .reinterpret_cast::<f16>()
                    .convert_to_f32_slice(group_values);
                for (offset, &value) in group_values.iter().enumerate() {
                    let column = group * group_size as usize + offset;
                    let code = (value / scale).round_ties_even().clamp(-limit, limit) as i8 as u8;
                    match bits {
                        ActivationBits::I4 => {
                            packed[column / 2] |= (code & 15) << (4 * (column % 2))
                        }
                        ActivationBits::I8 => packed[column] = code,
                    };
                }
            }
        }
        Ok(Self {
            bits,
            rows: rows as u32,
            columns,
            group_size,
            packed,
            scales,
        })
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    /// Oracle for the integer dot, with FP32 dequantization per group.
    pub fn run_cpu(
        &self,
        weights: W4Weights<'_>,
        bias: Option<&[u16]>,
        output: OutputType,
    ) -> Result<LinearOutput, OperatorError> {
        if self.columns != weights.k || self.group_size != weights.group_size {
            return Err(OperatorError::Shape(
                "activation and weight groups must match",
            ));
        }
        if bias.is_some_and(|values| values.len() != weights.n as usize) {
            return Err(OperatorError::Shape("bias does not match output width"));
        }
        let count = self
            .rows
            .checked_mul(weights.n)
            .ok_or(OperatorError::Shape(
                "output exceeds the device indexing range",
            ))? as usize;
        let mut result = LinearOutput::zeros(output, count);
        let groups = weights.groups();
        let k = self.columns as usize;
        let group_size = self.group_size as usize;
        let n = weights.n as usize;
        for row in 0..self.rows as usize {
            for channel in 0..n {
                let mut sum = 0.0f32;
                for group in 0..groups {
                    let parameter = channel * groups + group;
                    let zero = weights.zero_points.map_or(8, |zeros| zeros[parameter]) as i32;
                    let start = group * group_size;
                    let mut dot = 0i32;
                    for column in start..k.min(start + group_size) {
                        dot += self.code(row, column)
                            * (i32::from(weights.code(channel, column)) - zero);
                    }
                    let scale = self.scales[row * groups + group]
                        * f16::from_bits(weights.scales[parameter]).to_f32();
                    sum = (dot as f32).mul_add(scale, sum);
                }
                if let Some(bias) = bias {
                    sum += f16::from_bits(bias[channel]).to_f32();
                }
                match &mut result {
                    LinearOutput::F16(values) => {
                        values[row * n + channel] = f16::from_f32(sum).to_bits()
                    }
                    LinearOutput::F32(values) => values[row * n + channel] = sum,
                }
            }
        }
        Ok(result)
    }

    fn code(&self, row: usize, column: usize) -> i32 {
        let offset = row * self.bits.row_bytes(self.columns as usize);
        match self.bits {
            ActivationBits::I4 => {
                let value = (self.packed[offset + column / 2] >> (4 * (column % 2))) & 15;
                i32::from(value ^ 8) - 8
            }
            ActivationBits::I8 => i32::from(self.packed[offset + column] as i8),
        }
    }
}
