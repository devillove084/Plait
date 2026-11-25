//! Group parameter layout adapted from Humming's transform.py (Apache-2.0).
//! The GEMM reads parameter groups first, with each 64-channel tile transposed.

use crate::W4Weights;

pub(crate) fn group_parameters(weights: W4Weights<'_>) -> (Vec<u16>, Vec<u8>) {
    let groups = weights.groups();
    let mut scales = Vec::with_capacity(weights.scales.len());
    let mut zeros = Vec::with_capacity(weights.scales.len() / 2);
    for group in 0..groups {
        for tile in (0..weights.n as usize).step_by(64) {
            for lane in 0..8 {
                for pair in 0..4 {
                    let low = (tile + lane + 16 * pair) * groups + group;
                    let high = low + 8 * groups;
                    scales.extend([weights.scales[low], weights.scales[high]]);
                    let zero = weights
                        .zero_points
                        .map_or(0x88, |points| points[low] | (points[high] << 4));
                    zeros.push(zero);
                }
            }
        }
    }
    (scales, zeros)
}
