use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use plait_core::serving::WorkerGrant;
use plait_core::tensor::KPartial;
use plait_core::{OpId, ShardId};
use plait_fabric::reduction::KSum;
use serde::Deserialize;

use super::{W4Decision, W4Profile, W4Scope, W4Selection, W4Shard};

#[derive(Deserialize)]
pub struct W4PlacementRequest {
    pub k: u32,
    pub group_size: u32,
    pub output_shape: [u32; 2],
    pub op: OpId,
    pub now_ms: u64,
    pub calibration_id: [u8; 32],
    pub max_error_ppm: u64,
    #[serde(default)]
    pub workers: Vec<WorkerGrant>,
    /// Exact scopes of the available weight slices and libraries.
    pub scopes: Vec<W4Scope>,
    pub profiles: Vec<W4Profile>,
}

/// Find a complete aligned K partition over the currently available artifacts.
/// A worker can contribute at most one interval. Worker order is independent
/// of both the caller's order and the K order.
pub fn plan_w4(request: &W4PlacementRequest) -> Result<W4Selection, &'static str> {
    let Some(first) = request.workers.first() else {
        return Err("W4 placement requires worker grants");
    };
    let template = KPartial {
        grant: *first,
        op: request.op,
        shard: ShardId(0),
        k_start: 0,
        k_end: request.k,
        output_shape: request.output_shape,
    };
    if request.group_size == 0 || template.payload_bytes().is_none() {
        return Err("invalid W4 placement dimensions");
    }
    let mut grants = request.workers.clone();
    grants.sort_unstable_by_key(|grant| grant.device);
    if grants.iter().any(|grant| grant.version != first.version)
        || grants
            .windows(2)
            .any(|pair| pair[0].device == pair[1].device)
    {
        return Err("W4 placement requires distinct worker grants from one group version");
    }
    let available: BTreeSet<_> = request.scopes.iter().copied().collect();
    let mut catalog = Catalog::new();
    let mut modes = BTreeSet::new();
    let mut deadlines = BTreeSet::new();
    let mut explicit = None;
    for &profile in &request.profiles {
        let scope = profile.scope;
        if profile.calibration_id != request.calibration_id
            || profile.valid_until_ms <= request.now_ms
            || !available.contains(&scope)
        {
            continue;
        }
        let Ok(worker) = grants.binary_search_by_key(&scope.device, |grant| grant.device) else {
            continue;
        };
        let [start, end] = scope.k_range;
        let partial = KPartial {
            grant: grants[worker],
            k_start: start,
            k_end: end,
            ..template
        };
        if !scope.matches(partial, request.group_size)
            || start >= end
            || end > request.k
            || start % request.group_size != 0
            || (end != request.k && end % request.group_size != 0)
        {
            continue;
        }
        profile.check_calibration(&mut explicit)?;
        if !modes.insert((scope, profile.mode_key())) {
            return Err("duplicate current W4 profile for the same execution mode");
        }
        catalog
            .entry(start)
            .or_default()
            .entry((end, worker))
            .or_default()
            .push(profile);
        deadlines.insert(profile.latency_ns);
    }
    let at_deadline = |deadline| {
        partition_at_deadline(
            &catalog,
            grants.len(),
            request.k,
            request.max_error_ppm,
            deadline,
        )
    };
    let deadlines: Vec<_> = deadlines.into_iter().collect();
    let position = deadlines.partition_point(|&deadline| at_deadline(deadline).is_none());
    let &critical_path_ns = deadlines
        .get(position)
        .ok_or("no measured W4 partition fits the calibration error budget")?;
    let State {
        error_ppm,
        mut last,
    } = at_deadline(critical_path_ns)
        .ok_or("no measured W4 partition fits the calibration error budget")?;
    let mut chosen = Vec::new();
    while let Some(step) = last {
        chosen.push((step.worker, step.profile));
        last = step.prev.clone();
    }
    let decisions: Vec<_> = chosen
        .into_iter()
        .rev()
        .enumerate()
        .map(|(shard, (worker, profile))| {
            let [k_start, k_end] = profile.scope.k_range;
            W4Decision {
                shard: W4Shard {
                    partial: KPartial {
                        grant: grants[worker],
                        shard: ShardId(shard as u32),
                        k_start,
                        k_end,
                        ..template
                    },
                    scope: profile.scope,
                },
                profile,
            }
        })
        .collect();
    KSum::new(
        request.k,
        request.group_size,
        decisions
            .iter()
            .map(|decision| decision.shard.partial)
            .collect(),
    )
    .map_err(|_| "W4 placement produced an invalid partition")?;
    Ok(W4Selection {
        critical_path_ns,
        error_bound_ppm: error_ppm,
        decisions,
    })
}

// For each start, index the measured paths by end and worker.
type Catalog = BTreeMap<u32, BTreeMap<(u32, usize), Vec<W4Profile>>>;

struct Step {
    worker: usize,
    profile: W4Profile,
    prev: Option<Rc<Step>>,
}

struct State {
    error_ppm: u64,
    last: Option<Rc<Step>>,
}

struct Edge {
    end: u32,
    worker: usize,
    profile: W4Profile,
    lower_bound_ppm: u64,
}

struct Suffix {
    lower_bound_ppm: u64,
    edges: Vec<Edge>,
}

fn relaxed_suffixes(
    catalog: &Catalog,
    k: u32,
    max_error_ppm: u64,
    deadline: u64,
) -> BTreeMap<u32, Suffix> {
    // Ignoring worker reuse gives a lower bound on the remaining error.
    let mut suffixes = BTreeMap::from([(
        k,
        Suffix {
            lower_bound_ppm: 0,
            edges: Vec::new(),
        },
    )]);
    for (&start, choices) in catalog.iter().rev() {
        let edges: Vec<_> = choices
            .iter()
            .filter_map(|(&(end, worker), profiles)| {
                let remaining = suffixes.get(&end)?.lower_bound_ppm;
                let &profile = profiles
                    .iter()
                    .filter(|profile| profile.latency_ns <= deadline)
                    .min_by_key(|profile| (profile.preference(), profile.scope))?;
                (profile.error_ppm <= max_error_ppm - remaining).then(|| Edge {
                    end,
                    worker,
                    profile,
                    lower_bound_ppm: remaining + profile.error_ppm,
                })
            })
            .collect();
        if let Some(lower_bound_ppm) = edges.iter().map(|edge| edge.lower_bound_ppm).min() {
            suffixes.insert(
                start,
                Suffix {
                    lower_bound_ppm,
                    edges,
                },
            );
        }
    }
    suffixes
}

fn attain_lower_bound(suffixes: &BTreeMap<u32, Suffix>, words: usize, k: u32) -> Option<State> {
    let error_ppm = suffixes.get(&0)?.lower_bound_ppm;
    let mut used = vec![0u64; words];
    let mut start = 0;
    let mut last = None;
    while start < k {
        let suffix = &suffixes[&start];
        let edge = suffix.edges.iter().find(|edge| {
            edge.lower_bound_ppm == suffix.lower_bound_ppm
                && used[edge.worker / 64] & (1u64 << (edge.worker % 64)) == 0
        })?;
        used[edge.worker / 64] |= 1u64 << (edge.worker % 64);
        last = Some(Rc::new(Step {
            worker: edge.worker,
            profile: edge.profile,
            prev: last,
        }));
        start = edge.end;
    }
    Some(State { error_ppm, last })
}

fn partition_at_deadline(
    catalog: &Catalog,
    workers: usize,
    k: u32,
    max_error_ppm: u64,
    deadline: u64,
) -> Option<State> {
    let suffixes = relaxed_suffixes(catalog, k, max_error_ppm, deadline);
    suffixes.get(&0)?;
    let words = workers.div_ceil(64);
    if let Some(optimal) = attain_lower_bound(&suffixes, words, k) {
        return Some(optimal);
    }
    let mut frontier = BTreeMap::from([(
        0,
        BTreeMap::from([(
            vec![0u64; words],
            State {
                error_ppm: 0,
                last: None,
            },
        )]),
    )]);
    for (&start, suffix) in suffixes.range(..k) {
        let Some(states) = frontier.remove(&start) else {
            continue;
        };
        for &Edge {
            end,
            worker,
            profile,
            lower_bound_ppm,
        } in &suffix.edges
        {
            let next = frontier.entry(end).or_default();
            let word = worker / 64;
            let bit = 1u64 << (worker % 64);
            for (used, state) in &states {
                if used[word] & bit != 0 || lower_bound_ppm > max_error_ppm - state.error_ppm {
                    continue;
                }
                let error_ppm = state.error_ppm + profile.error_ppm;
                let mut used = used.clone();
                used[word] |= bit;
                // At a given boundary, equal worker sets have the same possible suffixes.
                if next
                    .get(&used)
                    .is_some_and(|current| current.error_ppm <= error_ppm)
                {
                    continue;
                }
                next.insert(
                    used,
                    State {
                        error_ppm,
                        last: Some(Rc::new(Step {
                            worker,
                            profile,
                            prev: state.last.clone(),
                        })),
                    },
                );
            }
        }
    }
    frontier
        .remove(&k)?
        .into_values()
        .min_by_key(|state| state.error_ppm)
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{GroupId, GroupVersion};
    use plait_core::{DeviceId, NodeId};

    use super::*;
    use crate::precision::W4Quantization;

    fn request() -> W4PlacementRequest {
        let group = NodeId::derive("placement-group");
        let workers: Vec<_> = ["wide", "narrow"]
            .into_iter()
            .map(|device| WorkerGrant {
                version: GroupVersion {
                    group: GroupId(group),
                    epoch: 3,
                },
                device: DeviceId::new(NodeId::derive(device), 0),
                incarnation: 2,
                fence: 13,
            })
            .collect();
        let profiles: Vec<_> = [
            // Worker 1 first and worker 0 second is the fastest valid arrangement.
            (1, [0, 8], 20, 2),
            (0, [8, 35], 27, 4),
            // Two low-latency edges on the same worker cannot run as separate shards.
            (1, [8, 35], 15, 0),
            (0, [0, 16], 80, 1),
            (1, [16, 35], 60, 1),
            (0, [0, 35], 135, 0),
            // An interior boundary cannot bisect a weight group.
            (1, [0, 7], 1, 0),
            (0, [7, 35], 1, 0),
        ]
        .into_iter()
        .map(|(worker, k_range, latency_ns, error_ppm)| W4Profile {
            scope: W4Scope {
                device: workers[worker].device,
                incarnation: workers[worker].incarnation,
                ordinal: 0,
                library_sha256: [worker as u8; 32],
                weights_sha256: [k_range[0] as u8; 32],
                k_range,
                output_shape: [1, 8],
                group_size: 8,
            },
            calibration_id: [7; 32],
            activation_bits: if worker == 0 { 8 } else { 4 },
            quantization: Some(if worker == 0 {
                W4Quantization::Device
            } else {
                W4Quantization::Host
            }),
            kernel: None,
            latency_ns,
            error_ppm,
            valid_until_ms: 90,
        })
        .collect();
        W4PlacementRequest {
            k: 35,
            group_size: 8,
            output_shape: [1, 8],
            op: OpId::new(group, 7),
            now_ms: 1,
            calibration_id: [7; 32],
            max_error_ppm: 6,
            scopes: profiles.iter().map(|profile| profile.scope).collect(),
            workers,
            profiles,
        }
    }

    #[test]
    fn places_asymmetric_slices_and_optimizes_under_error_budget() {
        let mut request = request();
        let mut unavailable = request.profiles[1];
        unavailable.scope.library_sha256 = [99; 32];
        unavailable.latency_ns = 1;
        request.profiles.push(unavailable);
        request.profiles.reverse();

        let plan = plan_w4(&request).unwrap();
        assert_eq!((plan.critical_path_ns, plan.error_bound_ppm), (27, 6));
        let first = plan.decisions[0].shard.partial;
        let second = plan.decisions[1].shard.partial;
        assert_eq!(
            (first.grant, first.shard, first.k_end),
            (request.workers[1], ShardId(0), 8)
        );
        assert_eq!(
            (second.grant, second.shard, second.k_end),
            (request.workers[0], ShardId(1), 35)
        );
        assert_eq!(
            plan.decisions[1].profile.quantization,
            Some(W4Quantization::Device)
        );
        request.workers.reverse();
        assert_eq!(plan_w4(&request).unwrap().decisions, plan.decisions);

        request.max_error_ppm = 3;
        let tighter = plan_w4(&request).unwrap();
        assert_eq!((tighter.critical_path_ns, tighter.error_bound_ppm), (80, 2));
        assert_eq!(tighter.decisions[0].shard.partial.k_end, 16);
    }

    #[test]
    fn places_integer_and_named_a16_kernels_in_one_operation() {
        use super::super::W4Kernel;

        let mut request = request();
        for profile in &mut request.profiles {
            profile.kernel = Some(W4Kernel::Integer);
        }
        request.profiles.extend([
            W4Profile {
                activation_bits: 16,
                quantization: None,
                kernel: Some(W4Kernel::Humming),
                latency_ns: 18,
                error_ppm: 2,
                ..request.profiles[1]
            },
            W4Profile {
                activation_bits: 16,
                quantization: None,
                kernel: Some(W4Kernel::Marlin),
                latency_ns: 21,
                error_ppm: 0,
                ..request.profiles[1]
            },
        ]);
        request.max_error_ppm = 4;
        let fast = plan_w4(&request).unwrap();
        assert_eq!((fast.critical_path_ns, fast.error_bound_ppm), (20, 4));
        assert_eq!(fast.decisions[0].profile.kernel, Some(W4Kernel::Integer));
        assert_eq!(fast.decisions[1].profile.kernel, Some(W4Kernel::Humming));
        request.max_error_ppm = 2;
        let tight = plan_w4(&request).unwrap();
        assert_eq!((tight.critical_path_ns, tight.error_bound_ppm), (21, 2));
        assert_eq!(tight.decisions[1].profile.kernel, Some(W4Kernel::Marlin));
    }

    #[test]
    fn replans_after_worker_removal_and_requires_current_artifacts() {
        let mut request = request();
        request.workers.pop();
        let fallback = plan_w4(&request).unwrap();
        assert_eq!(fallback.critical_path_ns, 135);
        assert_eq!(fallback.decisions.len(), 1);
        assert_eq!(fallback.decisions[0].shard.partial.k_end, 35);
        request.workers[0].incarnation += 1;
        assert_eq!(
            plan_w4(&request).unwrap_err(),
            "no measured W4 partition fits the calibration error budget"
        );

        let mut request = self::request();
        request.now_ms = request.profiles[0].valid_until_ms;
        assert!(plan_w4(&request).is_err());
        request.now_ms = 1;
        request.workers[0].version.epoch += 1;
        assert_eq!(
            plan_w4(&request).unwrap_err(),
            "W4 placement requires distinct worker grants from one group version"
        );
    }
}
