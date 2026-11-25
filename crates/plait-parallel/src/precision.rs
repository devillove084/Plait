//! Select measured W4 kernels and layouts for a K partition.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use plait_core::DeviceId;
use plait_core::tensor::KPartial;
use plait_fabric::reduction::KSum;
use serde::{Deserialize, Serialize};

mod placement;

pub use placement::{W4PlacementRequest, plan_w4};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct W4Scope {
    pub device: DeviceId,
    pub incarnation: u64,
    pub ordinal: i32,
    pub library_sha256: [u8; 32],
    pub weights_sha256: [u8; 32],
    pub k_range: [u32; 2],
    pub output_shape: [u32; 2],
    pub group_size: u32,
}

impl W4Scope {
    pub fn matches(self, partial: KPartial, group_size: u32) -> bool {
        self.device == partial.grant.device
            && self.incarnation == partial.grant.incarnation
            && self.k_range == [partial.k_start, partial.k_end]
            && self.output_shape == partial.output_shape
            && self.group_size == group_size
            && self.ordinal >= 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct W4Shard {
    pub partial: KPartial,
    pub scope: W4Scope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum W4Quantization {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct W4Profile {
    pub scope: W4Scope,
    pub calibration_id: [u8; 32],
    pub activation_bits: u32,
    /// Older profiles defer to the loaded library's default low-bit path.
    #[serde(default)]
    pub quantization: Option<W4Quantization>,
    /// Includes preparation for this invocation, execution, and delivery to the reducer.
    pub latency_ns: u64,
    /// Upper bound for this shard's L2 error, normalized by a common layer norm.
    pub error_ppm: u64,
    pub valid_until_ms: u64,
}

impl W4Profile {
    pub fn is_valid(self) -> bool {
        self.latency_ns != 0
            && match self.activation_bits {
                4 | 8 => true,
                16 => self.quantization.is_none() && self.error_ppm == 0,
                _ => false,
            }
    }

    fn mode_key(&self) -> (u32, Option<W4Quantization>) {
        (self.activation_bits, self.quantization)
    }

    fn preference(&self) -> (u64, u64, Reverse<u32>, bool) {
        (
            self.error_ppm,
            self.latency_ns,
            Reverse(self.activation_bits),
            self.quantization.is_none(),
        )
    }
}

#[derive(Deserialize)]
pub struct W4SelectionRequest {
    pub k: u32,
    pub group_size: u32,
    pub now_ms: u64,
    pub calibration_id: [u8; 32],
    pub max_error_ppm: u64,
    pub shards: Vec<W4Shard>,
    pub profiles: Vec<W4Profile>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct W4Decision {
    pub shard: W4Shard,
    pub profile: W4Profile,
}

#[derive(Debug, Serialize)]
pub struct W4Selection {
    pub critical_path_ns: u64,
    pub error_bound_ppm: u64,
    pub decisions: Vec<W4Decision>,
}

/// Minimize the slowest shard while the sum of calibrated error bounds fits.
/// At a fixed deadline, choosing the least error independently per shard
/// gives the exact feasibility test. Search the observed deadlines.
pub fn select_w4(request: &W4SelectionRequest) -> Result<W4Selection, &'static str> {
    KSum::new(
        request.k,
        request.group_size,
        request.shards.iter().map(|shard| shard.partial).collect(),
    )
    .map_err(|_| "precision selection requires a complete aligned K partition")?;
    let mut devices = BTreeSet::new();
    let mut candidates = BTreeMap::new();
    for shard in &request.shards {
        if !shard.scope.matches(shard.partial, request.group_size) {
            return Err("W4 profile scope does not match its shard");
        }
        if !devices.insert(shard.scope.device) {
            return Err("parallel W4 selection requires distinct devices");
        }
        candidates.insert(shard.scope, BTreeMap::new());
    }
    let mut deadlines = BTreeSet::new();
    for profile in &request.profiles {
        let Some(modes) = candidates.get_mut(&profile.scope) else {
            continue;
        };
        if profile.calibration_id != request.calibration_id
            || profile.valid_until_ms <= request.now_ms
        {
            continue;
        }
        if !profile.is_valid() {
            return Err("invalid W4 profile");
        }
        if modes.insert(profile.mode_key(), *profile).is_some() {
            return Err(
                "duplicate current W4 profile for the same precision and quantization path",
            );
        }
        deadlines.insert(profile.latency_ns);
    }
    if candidates.values().any(BTreeMap::is_empty) {
        return Err("a W4 shard has no current profile for its device, shape and artifacts");
    }
    let best_by_deadline = |deadline: u64| {
        candidates.iter().map(move |(scope, modes)| {
            modes
                .values()
                .filter(|profile| profile.latency_ns <= deadline)
                .min_by_key(|p| p.preference())
                .map(|profile| (*scope, *profile))
        })
    };
    let deadlines: Vec<_> = deadlines.into_iter().collect();
    let position = deadlines.partition_point(|&deadline| {
        let mut error = 0u128;
        for choice in best_by_deadline(deadline) {
            let Some((_, profile)) = choice else {
                return true;
            };
            error += u128::from(profile.error_ppm);
        }
        error > u128::from(request.max_error_ppm)
    });
    let &critical_path_ns = deadlines
        .get(position)
        .ok_or("no W4 selection fits the calibration error budget")?;
    let selected: BTreeMap<_, _> = best_by_deadline(critical_path_ns).flatten().collect();
    let mut decisions = Vec::with_capacity(request.shards.len());
    for &shard in &request.shards {
        decisions.push(W4Decision {
            shard,
            profile: selected[&shard.scope],
        });
    }
    decisions.sort_unstable_by_key(|decision| decision.shard.partial.k_start);
    let error_bound_ppm = decisions
        .iter()
        .map(|decision| decision.profile.error_ppm)
        .sum();
    Ok(W4Selection {
        critical_path_ns,
        error_bound_ppm,
        decisions,
    })
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{GroupId, GroupVersion, WorkerGrant};
    use plait_core::{NodeId, OpId, ShardId};

    use super::*;

    fn request() -> W4SelectionRequest {
        let group = NodeId::derive("linear-group");
        let shards: Vec<_> = ["amd", "metal"]
            .iter()
            .enumerate()
            .map(|(index, host)| {
                let device = DeviceId::new(NodeId::derive(host), 0);
                let start = index as u32 * 16;
                let end = start + 16;
                let partial = KPartial {
                    grant: WorkerGrant {
                        version: GroupVersion {
                            group: GroupId(group),
                            epoch: 2,
                        },
                        device,
                        incarnation: 5,
                        fence: 19,
                    },
                    op: OpId::new(group, 3),
                    shard: ShardId(index as u32),
                    k_start: start,
                    k_end: end,
                    output_shape: [1, 8],
                };
                let scope = W4Scope {
                    device,
                    incarnation: 5,
                    ordinal: 0,
                    library_sha256: [index as u8; 32],
                    weights_sha256: [7; 32],
                    k_range: [start, end],
                    output_shape: [1, 8],
                    group_size: 8,
                };
                W4Shard { partial, scope }
            })
            .collect();
        let profiles = [
            [(4, 40, 5), (8, 60, 3), (16, 120, 0)],
            [(4, 70, 8), (8, 45, 2), (16, 100, 0)],
        ]
        .into_iter()
        .zip(&shards)
        .flat_map(|(profiles, shard)| {
            profiles.into_iter().map(|(bits, time, error)| W4Profile {
                scope: shard.scope,
                calibration_id: [9; 32],
                activation_bits: bits,
                quantization: None,
                latency_ns: time,
                error_ppm: error,
                valid_until_ms: 100,
            })
        })
        .collect();
        W4SelectionRequest {
            k: 32,
            group_size: 8,
            now_ms: 10,
            calibration_id: [9; 32],
            max_error_ppm: 7,
            shards,
            profiles,
        }
    }

    fn bits(selection: &W4Selection) -> Vec<u32> {
        selection
            .decisions
            .iter()
            .map(|d| d.profile.activation_bits)
            .collect()
    }

    #[test]
    fn finds_fastest_parallel_deadline_under_the_layer_budget() {
        let mut request = request();
        request.shards.reverse();
        let mixed = select_w4(&request).unwrap();
        assert_eq!((mixed.critical_path_ns, mixed.error_bound_ppm), (45, 7));
        assert_eq!(bits(&mixed), [4, 8]);

        request.max_error_ppm = 6;
        let tighter = select_w4(&request).unwrap();
        assert_eq!((tighter.critical_path_ns, tighter.error_bound_ppm), (60, 5));
        assert_eq!(bits(&tighter), [8, 8]);

        request.max_error_ppm = 0;
        assert_eq!(bits(&select_w4(&request).unwrap()), [16, 16]);
    }

    #[test]
    fn rejects_duplicate_and_mismatched_or_expired_evidence() {
        let mut request = request();
        request.profiles[0].valid_until_ms = request.now_ms;
        request.profiles[1].scope.library_sha256 = [99; 32];
        request.profiles[2].calibration_id = [0; 32];
        assert_eq!(
            select_w4(&request).unwrap_err(),
            "a W4 shard has no current profile for its device, shape and artifacts"
        );
        let mut request = self::request();
        request.profiles.push(request.profiles[0]);
        assert_eq!(
            select_w4(&request).unwrap_err(),
            "duplicate current W4 profile for the same precision and quantization path"
        );
    }

    #[test]
    fn chooses_measured_quantization_paths_under_the_layer_budget() {
        let mut request = request();
        request.profiles[0].quantization = Some(W4Quantization::Host);
        request.profiles[0].error_ppm = 4;
        request.profiles[4].quantization = Some(W4Quantization::Device);
        request.profiles.push(W4Profile {
            quantization: Some(W4Quantization::Device),
            latency_ns: 30,
            error_ppm: 6,
            ..request.profiles[0]
        });
        request.profiles.push(W4Profile {
            quantization: Some(W4Quantization::Host),
            latency_ns: 125,
            ..request.profiles[4]
        });
        request.max_error_ppm = 8;
        let faster = select_w4(&request).unwrap();
        assert_eq!(faster.critical_path_ns, 45);
        // The faster device path has slack in this plan. Spend that slack on accuracy.
        assert_eq!(
            faster.decisions[0].profile.quantization,
            Some(W4Quantization::Host)
        );
        request.profiles[4].latency_ns = 35;
        let faster = select_w4(&request).unwrap();
        assert_eq!((faster.critical_path_ns, faster.error_bound_ppm), (35, 8));
        assert_eq!(
            faster.decisions[0].profile.quantization,
            Some(W4Quantization::Device)
        );

        request.max_error_ppm = 6;
        let tighter = select_w4(&request).unwrap();
        assert_eq!((tighter.critical_path_ns, tighter.error_bound_ppm), (40, 6));
        assert_eq!(
            tighter.decisions[0].profile.quantization,
            Some(W4Quantization::Host)
        );
    }
}
