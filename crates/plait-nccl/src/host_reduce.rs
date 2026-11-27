//! Blocked host reduction with one FP32 accumulation per element.

use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use half::{bf16, f16};

const MIN_SUM_BYTES: usize = 8 * 1024;
const BLOCK_ELEMENTS: usize = 512;
const BLOCK_BYTES: usize = BLOCK_ELEMENTS * 2;

pub(super) fn sum_fp16(buffers: &[&[u8]]) -> Option<Vec<u8>> {
    if let [left, right] = buffers {
        return sum_fp16_pair(left, right);
    }
    sum_blocks(
        buffers,
        0x7c00,
        |words, values| words.reinterpret_cast::<f16>().convert_to_f32_slice(values),
        |words, values| {
            words
                .reinterpret_cast_mut::<f16>()
                .convert_from_f32_slice(values)
        },
    )
}

pub(super) fn sum_bf16(buffers: &[&[u8]]) -> Option<Vec<u8>> {
    sum_blocks(
        buffers,
        0x7f80,
        |words, values| {
            words
                .reinterpret_cast::<bf16>()
                .convert_to_f32_slice(values)
        },
        |words, values| {
            words
                .reinterpret_cast_mut::<bf16>()
                .convert_from_f32_slice(values)
        },
    )
}

fn sum_blocks(
    buffers: &[&[u8]],
    exponent_mask: u16,
    widen: impl Fn(&[u16], &mut [f32]),
    narrow: impl Fn(&mut [u16], &[f32]),
) -> Option<Vec<u8>> {
    let (&first, peers) = buffers.split_first()?;
    let bytes = first.len();
    if peers.is_empty()
        || bytes < MIN_SUM_BYTES
        || bytes % 2 != 0
        || peers.iter().any(|peer| peer.len() != bytes)
    {
        return None;
    }
    // Keep the scalar implementation's handling of nonfinite inputs.
    if buffers.iter().any(|buffer| {
        buffer
            .chunks_exact(2)
            .any(|word| u16::from_ne_bytes([word[0], word[1]]) & exponent_mask == exponent_mask)
    }) {
        return None;
    }

    let mut output = vec![0; bytes];
    let mut bits = [0u16; BLOCK_ELEMENTS];
    let mut values = [0.0f32; BLOCK_ELEMENTS];
    let mut addends = [0.0f32; BLOCK_ELEMENTS];
    for (block, result) in output.chunks_mut(BLOCK_BYTES).enumerate() {
        let start = block * BLOCK_BYTES;
        let range = start..start + result.len();
        let elements = result.len() / 2;
        let words = &mut bits[..elements];
        let sums = &mut values[..elements];
        let addends = &mut addends[..elements];
        load_words(&first[range.clone()], words);
        widen(words, sums);
        // Each element follows the original slot order and is rounded once.
        for peer in peers {
            load_words(&peer[range.clone()], words);
            widen(words, addends);
            for (sum, addend) in sums.iter_mut().zip(addends.iter()) {
                *sum += *addend;
            }
        }
        narrow(words, sums);
        for (word, bytes) in words.iter().zip(result.chunks_exact_mut(2)) {
            bytes.copy_from_slice(&word.to_ne_bytes());
        }
    }
    Some(output)
}

fn load_words(bytes: &[u8], words: &mut [u16]) {
    for (word, bytes) in words.iter_mut().zip(bytes.chunks_exact(2)) {
        *word = u16::from_ne_bytes([bytes[0], bytes[1]]);
    }
}

fn sum_fp16_pair(left: &[u8], right: &[u8]) -> Option<Vec<u8>> {
    if left.len() < MIN_SUM_BYTES || left.len() != right.len() || left.len() % 2 != 0 {
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
        widen_fp16(left, words, sums);
        widen_fp16(right, words, other);
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

fn widen_fp16(bytes: &[u8], words: &mut [u16], values: &mut [f32]) {
    load_words(bytes, words);
    words.reinterpret_cast::<f16>().convert_to_f32_slice(values);
}
