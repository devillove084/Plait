use std::collections::{BTreeMap, BTreeSet};

use plait_core::serving::{
    DecoderEndpoint, DecoderRoute, GroupId, GroupPlan, GroupVersion, HostMemory, PdRoute,
    ServingResources, ServingRole, WorkerDemand,
};
use plait_core::{DeviceId, NodeId};
use serde::{Deserialize, Serialize};

use super::super::{KvPath, ServingForecast, SloProfile, valid_rate, validate_placement};
use super::{DecoderLayoutCandidate, DecoderPlacement};

#[derive(Deserialize, Serialize)]
pub struct DecoderPool {
    artifacts: Vec<Artifact>,
    links: Vec<LinkProfile>,
    transition_loss_requests: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
struct DeviceVersion {
    device: DeviceId,
    incarnation: u64,
}

#[derive(Deserialize, Serialize)]
struct Artifact {
    #[serde(flatten)]
    device: DeviceVersion,
    role: ServingRole,
    device_bytes: u64,
    host_memory: HostMemory,
    launch_digest: [u8; 32],
    capacity: SloProfile,
    ready_after_ms: u64,
}

#[derive(Deserialize, Serialize)]
struct LinkProfile {
    prefill: DeviceVersion,
    decode: DeviceVersion,
    bytes_per_request: u64,
    links: Vec<String>,
    capacity: SloProfile,
}

struct Endpoint {
    version: GroupVersion,
    prepare: Option<DecoderPlacement>,
}

pub fn generate_decoder_layouts(
    resources: &ServingResources,
    incumbent: &[DecoderRoute],
    forecast: &mut ServingForecast,
    pool: &DecoderPool,
) -> Result<Vec<DecoderLayoutCandidate>, &'static str> {
    if !forecast.paths.is_empty() || !valid_rate(pool.transition_loss_requests) {
        return Err("decoder pool requires device link profiles and a valid transition cost");
    }
    if pool
        .artifacts
        .iter()
        .any(|artifact| artifact.role == ServingRole::Aggregated)
    {
        return Err("decoder pool artifacts require a Prefill or Decode role");
    }
    let prefill = endpoints(resources, incumbent, forecast, pool, ServingRole::Prefill)?;
    let decode = endpoints(resources, incumbent, forecast, pool, ServingRole::Decode)?;
    let mut candidates = Vec::new();
    let mut profiles = Vec::new();
    let mut measured = BTreeSet::new();
    let current: Vec<_> = incumbent.iter().map(|route| route.path()).collect();
    for profile in &pool.links {
        if !measured.insert((profile.prefill, profile.decode)) {
            return Err("duplicate decoder device link profile");
        }
        if profile.prefill.device == profile.decode.device {
            continue;
        }
        let (Some(p), Some(d)) = (prefill.get(&profile.prefill), decode.get(&profile.decode))
        else {
            continue;
        };
        let path = PdRoute {
            prefill: p.version,
            decode: d.version,
        };
        profiles.push(KvPath {
            prefill: path.prefill,
            decode: path.decode,
            bytes_per_request: profile.bytes_per_request,
            links: profile.links.clone(),
            capacity: profile.capacity,
        });
        if current.contains(&path) {
            continue;
        }
        let mut add_candidate = |paths| {
            candidates.push(DecoderLayoutCandidate {
                placements: p.prepare.iter().chain(d.prepare.iter()).cloned().collect(),
                paths,
                standbys: Vec::new(),
                transition_loss_requests: pool.transition_loss_requests,
            });
        };
        for index in 0..current.len() {
            let mut paths = current.clone();
            paths[index] = path;
            add_candidate(paths);
        }
        for index in 0..=current.len() {
            let mut paths = current.clone();
            paths.insert(index, path);
            add_candidate(paths);
        }
    }
    forecast.paths = profiles;
    Ok(candidates)
}

fn endpoints(
    resources: &ServingResources,
    incumbent: &[DecoderRoute],
    forecast: &ServingForecast,
    pool: &DecoderPool,
    role: ServingRole,
) -> Result<BTreeMap<DeviceVersion, Endpoint>, &'static str> {
    let mut endpoints = BTreeMap::new();
    for route in incumbent {
        let DecoderEndpoint { grant, .. } = match role {
            ServingRole::Prefill => route.prefill,
            ServingRole::Decode => route.decode,
            ServingRole::Aggregated => unreachable!("pool has only decoder roles"),
        };
        let key = DeviceVersion {
            device: grant.device,
            incarnation: grant.incarnation,
        };
        endpoints.insert(
            key,
            Endpoint {
                version: grant.version,
                prepare: None,
            },
        );
    }
    let mut seen = BTreeSet::new();
    for artifact in pool
        .artifacts
        .iter()
        .filter(|artifact| artifact.role == role)
    {
        if !seen.insert(artifact.device) {
            return Err("duplicate decoder device artifact");
        }
        if endpoints.contains_key(&artifact.device) || !artifact.capacity.is_valid(forecast.now_ms)
        {
            continue;
        }
        let DeviceVersion {
            device,
            incarnation,
        } = artifact.device;
        let group = GroupId(NodeId::derive(&format!(
            "plait/decoder-device/{}/{role:?}/{device}",
            forecast.model,
        )));
        let expected_epoch = resources
            .groups
            .iter()
            .find(|entry| entry.latest.group == group)
            .map_or(0, |entry| entry.latest.epoch);
        let plan = GroupPlan {
            role,
            model: forecast.model.clone(),
            tensor_parallel: 1,
            pipeline_parallel: 1,
            launch_digest: artifact.launch_digest,
            workers: BTreeMap::from([(
                device,
                WorkerDemand {
                    incarnation,
                    device_bytes: artifact.device_bytes,
                },
            )]),
            hosts: BTreeMap::from([(device.node, artifact.host_memory)]),
        };
        let Ok(version) =
            validate_placement(resources, &forecast.model, group, expected_epoch, &plan)
        else {
            continue;
        };
        endpoints.insert(
            artifact.device,
            Endpoint {
                version,
                prepare: Some(DecoderPlacement {
                    group,
                    expected_epoch,
                    plan,
                    capacity: artifact.capacity,
                    ready_after_ms: artifact.ready_after_ms,
                }),
            },
        );
    }
    Ok(endpoints)
}
