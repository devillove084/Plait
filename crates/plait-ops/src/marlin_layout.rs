//! Marlin's weight and scale permutations for canonical INT4 weights.
//! Adapted from Marlin (Elias Frantar, 2024), under Apache-2.0.

use crate::packing::pack_codes;
use crate::{OperatorError, W4Weights};

pub(crate) fn pack(weights: W4Weights<'_>) -> Result<(Vec<u32>, Vec<u16>, i32), OperatorError> {
    let (k, n) = (weights.k as usize, weights.n as usize);
    if k % 128 != 0 || n % 256 != 0 || (k as u64) * (n as u64) > i32::MAX as u64 {
        return Err(OperatorError::Unsupported(
            "Marlin requires K aligned to 128 and N aligned to 256",
        ));
    }
    let grouped = weights.group_size != weights.k;
    if grouped && weights.group_size != 128 {
        return Err(OperatorError::Unsupported(
            "Marlin supports channelwise scales or groups of 128",
        ));
    }
    if weights
        .zero_points
        .is_some_and(|zeros| zeros.iter().any(|&value| value != 8))
    {
        return Err(OperatorError::Unsupported(
            "this Marlin kernel requires symmetric INT4 weights",
        ));
    }
    let mut packed = Vec::with_capacity(k * n / 8);
    const NIBBLE_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    for k_tile in (0..k).step_by(16) {
        for n_tile in (0..n).step_by(64) {
            for lane in 0..32 {
                let first_row = 2 * (lane % 4);
                let first_column = lane / 4;
                let mut codes = [0; 32];
                for (tile, group) in codes.chunks_exact_mut(8).enumerate() {
                    for (code, index) in group.iter_mut().zip(NIBBLE_ORDER) {
                        let row = first_row + (index % 4 / 2) * 8 + index % 2;
                        let column = first_column + (index / 4) * 8;
                        *code = weights.code(n_tile + tile * 16 + column, k_tile + row);
                    }
                }
                packed.extend(pack_codes::<4>(&codes));
            }
        }
    }
    let groups = weights.groups();
    let mut scales = Vec::with_capacity(weights.scales.len());
    for group in 0..groups {
        if grouped {
            for tile in (0..n).step_by(64) {
                for lane in 0..8 {
                    for row in 0..8 {
                        scales.push(weights.scales[(tile + lane + 8 * row) * groups + group]);
                    }
                }
            }
        } else {
            for tile in (0..n).step_by(32) {
                for lane in 0..4 {
                    for index in [0, 1, 8, 9, 16, 17, 24, 25] {
                        scales.push(weights.scales[tile + 2 * lane + index]);
                    }
                }
            }
        }
    }
    Ok((packed, scales, if grouped { 128 } else { -1 }))
}
