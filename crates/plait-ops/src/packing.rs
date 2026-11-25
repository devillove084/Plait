//! Humming-compatible integer word packing. Adapted under Apache-2.0.
//! See kernels/humming/LICENSE and NOTICE for the upstream revision.

/// Pack one upstream tile of 32 codes into BITS words. Code 0 starts
/// at bit 0. Code widths that do not divide 32 can span two words.
pub(crate) fn pack_codes<const BITS: usize>(codes: &[u8; 32]) -> [u32; BITS] {
    assert!((1..=8).contains(&BITS));
    let mask = (1u32 << BITS) - 1;
    let mut words = [0; BITS];
    for (index, &code) in codes.iter().enumerate() {
        let bit = index * BITS;
        let (word, shift) = (bit / 32, bit % 32);
        let code = u32::from(code) & mask;
        words[word] |= code << shift;
        if shift + BITS > 32 {
            words[word + 1] |= code >> (32 - shift);
        }
    }
    words
}

#[cfg(test)]
mod tests {
    use super::pack_codes;

    #[test]
    fn matches_bit_order_across_word_boundaries() {
        fn check<const BITS: usize>() {
            let codes = std::array::from_fn(|i| (187 * i + 39) as u8);
            let words = pack_codes::<BITS>(&codes);
            for (index, &code) in codes.iter().enumerate() {
                for bit in 0..BITS {
                    let position = index * BITS + bit;
                    let observed = (words[position / 32] >> (position % 32)) & 1;
                    assert_eq!(observed, u32::from((code >> bit) & 1));
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
        check::<6>();
        check::<7>();
        check::<8>();
    }
}
