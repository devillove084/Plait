//! Bulk conversion for the host-staged sum of two FP16 buffers.

use half::f16;
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};

const MIN_PAIR_BYTES: usize = 8 * 1024;
const BLOCK_ELEMENTS: usize = 512;
const BLOCK_BYTES: usize = BLOCK_ELEMENTS * 2;

pub(super) fn sum_fp16_pair(left: &[u8], right: &[u8]) -> Option<Vec<u8>> {
    if left.len() < MIN_PAIR_BYTES || left.len() != right.len() || left.len() % 2 != 0 {
        return None;
    }
    const HIGH_BYTE: usize = cfg!(target_endian = "little") as usize;
    // Keep the scalar implementation's handling of infinities and NaN payloads.
    if left
        .chunks_exact(2)
        .chain(right.chunks_exact(2))
        .any(|word| word[HIGH_BYTE] & 0x7c == 0x7c)
    {
        return None;
    }

    let mut output = vec![0; left.len()];
    let mut bits = [0u16; BLOCK_ELEMENTS];
    let mut left_values = [0.0f32; BLOCK_ELEMENTS];
    let mut right_values = [0.0f32; BLOCK_ELEMENTS];
    for ((left, right), result) in left
        .chunks(BLOCK_BYTES)
        .zip(right.chunks(BLOCK_BYTES))
        .zip(output.chunks_mut(BLOCK_BYTES))
    {
        let elements = left.len() / 2;
        let words = &mut bits[..elements];
        let sums = &mut left_values[..elements];
        let other = &mut right_values[..elements];
        widen(left, words, sums);
        widen(right, words, other);
        for (sum, addend) in sums.iter_mut().zip(other) {
            *sum += *addend;
        }
        words
            .reinterpret_cast_mut::<f16>()
            .convert_from_f32_slice(sums);
        for (word, bytes) in words.iter().zip(result.chunks_exact_mut(2)) {
            bytes.copy_from_slice(&word.to_ne_bytes());
        }
    }
    Some(output)
}

fn widen(bytes: &[u8], words: &mut [u16], values: &mut [f32]) {
    for (word, bytes) in words.iter_mut().zip(bytes.chunks_exact(2)) {
        *word = u16::from_ne_bytes([bytes[0], bytes[1]]);
    }
    words.reinterpret_cast::<f16>().convert_to_f32_slice(values);
}
