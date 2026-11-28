//! Blocked host reduction with one FP32 accumulation per element.

use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use half::{bf16, f16};

const MIN_SUM_BYTES: usize = 8 * 1024;
#[cfg(target_arch = "x86_64")]
const MIN_F16C_BYTES: usize = 512;
const BLOCK_ELEMENTS: usize = 512;
const BLOCK_BYTES: usize = BLOCK_ELEMENTS * 2;

pub(super) fn sum_fp16(buffers: &[&[u8]]) -> Option<Vec<u8>> {
    if let [left, right] = buffers {
        return sum_fp16_pair(left, right);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some((&first, peers)) = buffers.split_first()
        && !peers.is_empty()
        && first.len() >= MIN_F16C_BYTES
        && first.len() % 2 == 0
        && peers.iter().all(|peer| peer.len() == first.len())
        && std::is_x86_feature_detected!("f16c")
    {
        let mut output = vec![0; first.len()];
        // CPU support and matching, even-length source slices were checked above.
        return unsafe { sum_fp16_many_f16c(first, peers, &mut output) }.then_some(output);
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
    let kernel = Fp16PairSum::select(left, right)?;
    let mut output = vec![0; left.len()];
    kernel.write(left, right, &mut output).then_some(output)
}

/// The caller replaces the output with the scalar result if this returns false.
pub(super) fn try_sum_fp16_pair_into(left: &[u8], right: &[u8], output: &mut [u8]) -> bool {
    output.len() == left.len()
        && Fp16PairSum::select(left, right).is_some_and(|kernel| kernel.write(left, right, output))
}

enum Fp16PairSum {
    Blocked,
    #[cfg(target_arch = "x86_64")]
    F16c,
}

impl Fp16PairSum {
    fn select(left: &[u8], right: &[u8]) -> Option<Self> {
        let bytes = left.len();
        if bytes != right.len() || bytes % 2 != 0 {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        if bytes >= MIN_F16C_BYTES && std::is_x86_feature_detected!("f16c") {
            return Some(Self::F16c);
        }
        (bytes >= MIN_SUM_BYTES).then_some(Self::Blocked)
    }

    fn write(self, left: &[u8], right: &[u8], output: &mut [u8]) -> bool {
        match self {
            Self::Blocked => sum_fp16_pair_blocked(left, right, output),
            #[cfg(target_arch = "x86_64")]
            // `select` verifies CPU support and the caller supplies equal-sized buffers.
            Self::F16c => unsafe { sum_fp16_pair_f16c(left, right, output) },
        }
    }
}

fn sum_fp16_pair_blocked(left: &[u8], right: &[u8], output: &mut [u8]) -> bool {
    const HIGH_BYTE: usize = cfg!(target_endian = "little") as usize;
    // Keep the scalar implementation's handling of infinities and NaN payloads.
    if left
        .chunks_exact(2)
        .chain(right.chunks_exact(2))
        .any(|word| word[HIGH_BYTE] & 0x7c == 0x7c)
    {
        return false;
    }

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
    true
}

fn widen_fp16(bytes: &[u8], words: &mut [u16], values: &mut [f32]) {
    load_words(bytes, words);
    words.reinterpret_cast::<f16>().convert_to_f32_slice(values);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn sum_fp16_pair_f16c(left: &[u8], right: &[u8], output: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        _MM_FROUND_TO_NEAREST_INT, _mm_and_si128, _mm_cmpeq_epi16, _mm_loadu_si128,
        _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi16, _mm_storeu_si128, _mm256_add_ps,
        _mm256_cvtph_ps, _mm256_cvtps_ph,
    };

    const LANES_BYTES: usize = 16;
    const EXPONENT: u16 = 0x7c00;
    let exponent = _mm_set1_epi16(EXPONENT as i16);
    for ((left, right), output) in left
        .chunks_exact(LANES_BYTES)
        .zip(right.chunks_exact(LANES_BYTES))
        .zip(output.chunks_exact_mut(LANES_BYTES))
    {
        // Each slice holds exactly 16 bytes. The loads and store allow unaligned addresses.
        let left = unsafe { _mm_loadu_si128(left.as_ptr().cast()) };
        let right = unsafe { _mm_loadu_si128(right.as_ptr().cast()) };
        let left_special = _mm_cmpeq_epi16(_mm_and_si128(left, exponent), exponent);
        let right_special = _mm_cmpeq_epi16(_mm_and_si128(right, exponent), exponent);
        if _mm_movemask_epi8(_mm_or_si128(left_special, right_special)) != 0 {
            return false;
        }
        let sums = _mm256_add_ps(_mm256_cvtph_ps(left), _mm256_cvtph_ps(right));
        let rounded = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(sums);
        unsafe { _mm_storeu_si128(output.as_mut_ptr().cast(), rounded) };
    }
    let tail = left.len() / LANES_BYTES * LANES_BYTES;
    for ((left, right), output) in left[tail..]
        .chunks_exact(2)
        .zip(right[tail..].chunks_exact(2))
        .zip(output[tail..].chunks_exact_mut(2))
    {
        let left = u16::from_ne_bytes([left[0], left[1]]);
        let right = u16::from_ne_bytes([right[0], right[1]]);
        if left & EXPONENT == EXPONENT || right & EXPONENT == EXPONENT {
            return false;
        }
        let sum = f16::from_bits(left).to_f32() + f16::from_bits(right).to_f32();
        output.copy_from_slice(&f16::from_f32(sum).to_bits().to_ne_bytes());
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn sum_fp16_many_f16c(first: &[u8], peers: &[&[u8]], output: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        _MM_FROUND_TO_NEAREST_INT, _mm_and_si128, _mm_cmpeq_epi16, _mm_loadu_si128,
        _mm_movemask_epi8, _mm_set1_epi16, _mm_storeu_si128, _mm256_add_ps, _mm256_cvtph_ps,
        _mm256_cvtps_ph,
    };

    const VECTOR_BYTES: usize = 16;
    const EXPONENT: u16 = 0x7c00;
    let exponent = _mm_set1_epi16(EXPONENT as i16);
    let finite =
        |bits| _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_and_si128(bits, exponent), exponent)) == 0;
    let vector_end = first.len() / VECTOR_BYTES * VECTOR_BYTES;
    for (block, output) in output.chunks_exact_mut(VECTOR_BYTES).enumerate() {
        let start = block * VECTOR_BYTES;
        // The checked lengths provide a full vector at this offset for every slot.
        let head = unsafe { _mm_loadu_si128(first.as_ptr().add(start).cast()) };
        if !finite(head) {
            return false;
        }
        let mut sums = _mm256_cvtph_ps(head);
        for peer in peers {
            let lanes = unsafe { _mm_loadu_si128(peer.as_ptr().add(start).cast()) };
            if !finite(lanes) {
                return false;
            }
            sums = _mm256_add_ps(sums, _mm256_cvtph_ps(lanes));
        }
        let rounded = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(sums);
        unsafe { _mm_storeu_si128(output.as_mut_ptr().cast(), rounded) };
    }
    for (index, result) in output[vector_end..].chunks_exact_mut(2).enumerate() {
        let offset = vector_end + 2 * index;
        let first = u16::from_ne_bytes([first[offset], first[offset + 1]]);
        if first & EXPONENT == EXPONENT {
            return false;
        }
        let mut sum = f16::from_bits(first).to_f32();
        for peer in peers {
            let word = u16::from_ne_bytes([peer[offset], peer[offset + 1]]);
            if word & EXPONENT == EXPONENT {
                return false;
            }
            sum += f16::from_bits(word).to_f32();
        }
        result.copy_from_slice(&f16::from_f32(sum).to_bits().to_ne_bytes());
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fp16_many_handles_misaligned_vectors_and_nonfinite_tails() {
        if !std::is_x86_feature_detected!("f16c") {
            return;
        }
        let bytes = MIN_F16C_BYTES + 6;
        let cases: [[u16; 4]; 4] = [
            [0x7800, 0x0001, 0xf800, 0x0001],
            [0x8000, 0x8000, 0x8000, 0x8000],
            [0x3c01, 0x1000, 0x0000, 0x0000],
            [0x7bff, 0x7bff, 0x7bff, 0x7bff],
        ];
        let mut inputs = [(); 4].map(|()| vec![0u8; bytes + 1]);
        let mut expected = Vec::new();
        for (index, words) in cases.iter().cycle().take(bytes / 2).enumerate() {
            let offset = 1 + 2 * index;
            for (input, word) in inputs.iter_mut().zip(words) {
                input[offset..offset + 2].copy_from_slice(&word.to_ne_bytes());
            }
            let sum = words[1..]
                .iter()
                .fold(f16::from_bits(words[0]).to_f32(), |sum, &word| {
                    sum + f16::from_bits(word).to_f32()
                });
            expected.extend(f16::from_f32(sum).to_bits().to_ne_bytes());
        }
        let reduce = |inputs: &[Vec<u8>; 4]| sum_fp16(&inputs.each_ref().map(|input| &input[1..]));
        assert_eq!(reduce(&inputs), Some(expected));
        for index in [0, bytes / 2 - 1] {
            let offset = 1 + 2 * index;
            let word = &mut inputs[2][offset..offset + 2];
            let original = [word[0], word[1]];
            word.copy_from_slice(&0x7c01u16.to_ne_bytes());
            assert!(reduce(&inputs).is_none());
            inputs[2][offset..offset + 2].copy_from_slice(&original);
        }
    }

    #[test]
    fn fp16_pair_writes_unaligned_blocks_and_checks_the_tail() {
        // Exercise both the vector body and an odd number of FP16 elements.
        let bytes = MIN_SUM_BYTES + 6;
        let cases: [(u16, u16); 8] = [
            (0x8000, 0x8000),
            (0x0000, 0x8000),
            (0x0001, 0x0001),
            (0x03ff, 0x0001),
            (0x3c00, 0x1000),
            (0x3c01, 0x1000),
            (0x7bff, 0x7bff),
            (0xfbff, 0x7bff),
        ];
        let mut left = vec![0; bytes + 2];
        let mut right = vec![0; bytes + 2];
        let mut output = vec![0x55; bytes + 2];
        let mut expected = Vec::with_capacity(bytes);
        for (index, (a, b)) in cases.iter().copied().cycle().take(bytes / 2).enumerate() {
            let start = 2 * index + 1;
            left[start..start + 2].copy_from_slice(&a.to_ne_bytes());
            right[start..start + 2].copy_from_slice(&b.to_ne_bytes());
            let sum = f16::from_bits(a).to_f32() + f16::from_bits(b).to_f32();
            expected.extend(f16::from_f32(sum).to_bits().to_ne_bytes());
        }
        assert!(try_sum_fp16_pair_into(
            &left[1..=bytes],
            &right[1..=bytes],
            &mut output[1..=bytes],
        ));
        assert_eq!(output[1..=bytes], expected);
        assert_eq!([output[0], output[bytes + 1]], [0x55; 2]);
        output[1..=bytes].fill(0x55);
        assert!(Fp16PairSum::Blocked.write(
            &left[1..=bytes],
            &right[1..=bytes],
            &mut output[1..=bytes],
        ));
        assert_eq!(output[1..=bytes], expected);
        assert_eq!([output[0], output[bytes + 1]], [0x55; 2]);
        right[bytes - 1..=bytes].copy_from_slice(&0x7e21u16.to_ne_bytes());
        assert!(!try_sum_fp16_pair_into(
            &left[1..=bytes],
            &right[1..=bytes],
            &mut output[1..=bytes],
        ));
    }
}
