//! Evaluate a bridge on today's resources and a target after the old grants retire.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::DeviceId;
use plait_core::serving::{
    ActiveGroup, GroupPhase, GroupResources, GroupVersion, HostedPdPlanning, HostedPdRelease,
    HostedPdStage, PdRoute,
};
use serde::{Deserialize, Serialize};

use super::{Bridge, HostedPdDecision, HostedPdSelection, Layout, Search, available};

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Reconfiguration {
    pub drain_ms: u64,
}

impl Search<'_> {
    pub(super) fn reconfigure(
        &self,
        snapshot: &HostedPdPlanning,
        layouts: &[Layout],
        releases: &[HostedPdRelease],
        policy: Reconfiguration,
        selection: &mut HostedPdSelection,
    ) -> Result<(), &'static str> {
        let baseline = selection.baseline_rps;
        let retiring: BTreeMap<_, _> = releases
            .iter()
            .map(|release| (release.version, release))
            .collect();
        let present = versions(&snapshot.paths);
        let mut bridges: Vec<_> = layouts
            .iter()
            .filter(|layout| layout.potential >= baseline)
            .filter_map(|layout| {
                let paths = layout.paths();
                let removed: Vec<_> = present
                    .difference(&versions(&paths))
                    .map(|version| retiring.get(version).copied())
                    .collect::<Option<_>>()?;
                if removed.is_empty() {
                    return None;
                }
                let elapsed = layout.ready_after_ms.checked_add(policy.drain_ms)?;
                (elapsed < self.forecast.horizon_ms).then_some((layout, paths, removed, elapsed))
            })
            .collect();
        // The outer beam and this second bound cap work before looking past retirement.
        bridges.sort_by(|(a, _, _, _), (b, _, _, _)| {
            b.potential
                .total_cmp(&a.potential)
                .then_with(|| a.ready_after_ms.cmp(&b.ready_after_ms))
                .then_with(|| a.placements.len().cmp(&b.placements.len()))
        });
        for (bridge, paths, release, start_target_after_ms) in bridges.into_iter().take(16) {
            let Some(projected) = project(snapshot, bridge, &release, &self.forecast.model) else {
                continue;
            };
            let mut forecast = self.forecast.clone();
            let Some(now_ms) = forecast.now_ms.checked_add(start_target_after_ms) else {
                continue;
            };
            forecast.now_ms = now_ms;
            forecast.horizon_ms -= start_target_after_ms;
            let search = Search {
                forecast: &forecast,
                ..*self
            };
            let active = bridge
                .endpoints
                .iter()
                .map(|endpoint| (endpoint.version.group, endpoint.clone()))
                .collect();
            for target in search.layouts(&projected.snapshot, &active)? {
                if !target.placements.iter().any(|placement| {
                    placement
                        .plan
                        .workers
                        .keys()
                        .any(|device| projected.devices.contains(device))
                }) {
                    continue;
                }
                let ready_after_ms = start_target_after_ms + target.ready_after_ms;
                let Some(bridge_rate) =
                    self.layout_rate(bridge, self.forecast.now_ms.saturating_add(ready_after_ms))
                else {
                    continue;
                };
                if bridge_rate < baseline {
                    continue;
                }
                selection.layouts_evaluated += 1;
                let bridge_gain =
                    (bridge_rate - baseline) * (ready_after_ms - bridge.ready_after_ms) as f64;
                let target_gain = (target.potential - baseline)
                    * (self.forecast.horizon_ms - ready_after_ms) as f64;
                let gain = (bridge_gain + target_gain) / 1000.0
                    - bridge.transition_loss_requests
                    - target.transition_loss_requests;
                let devices = target.devices();
                if gain > 0.0
                    && gain >= self.forecast.minimum_gain_requests
                    && target.potential >= baseline
                    && selection.selected.as_ref().is_none_or(|best| {
                        gain > best.gain_requests
                            || (gain == best.gain_requests && devices < best.devices)
                    })
                {
                    selection.selected = Some(HostedPdDecision {
                        paths: target.paths(),
                        placements: target.placements,
                        steady_rps: target.potential,
                        gain_requests: gain,
                        devices,
                        bridge: Some(Bridge {
                            stage: HostedPdStage {
                                paths: paths.clone(),
                                placements: bridge.placements.clone(),
                            },
                            release: release.iter().map(|&info| info.clone()).collect(),
                            rate: bridge_rate,
                            target_ready_after_ms: ready_after_ms,
                        }),
                    });
                }
            }
        }
        Ok(())
    }
}

fn versions(paths: &[PdRoute]) -> BTreeSet<GroupVersion> {
    paths
        .iter()
        .flat_map(|path| [path.prefill, path.decode])
        .collect()
}

struct Projection {
    snapshot: HostedPdPlanning,
    devices: BTreeSet<DeviceId>,
}

fn project(
    snapshot: &HostedPdPlanning,
    bridge: &Layout,
    release: &[&HostedPdRelease],
    model: &str,
) -> Option<Projection> {
    let (mut resources, mut inventory) = available(snapshot, bridge);
    let mut devices = BTreeSet::new();
    for retired in release {
        let position = inventory
            .occupied
            .iter()
            .position(|ports| *ports == retired.ports)?;
        inventory.occupied.remove(position);
        for (host, demand) in &retired.hosts {
            let budget = resources
                .hosts
                .iter_mut()
                .find(|entry| entry.host == *host)?;
            budget.reserved.bytes = budget.reserved.bytes.checked_sub(demand.bytes)?;
            budget.reserved.pinned_bytes = budget
                .reserved
                .pinned_bytes
                .checked_sub(demand.pinned_bytes)?;
        }
        let history = resources
            .groups
            .iter_mut()
            .find(|entry| entry.latest.group == retired.version.group)?;
        if history.active.as_ref()?.version != retired.version {
            return None;
        }
        history.active = None;
        if history.latest == retired.version {
            history.phase = GroupPhase::Retired;
        }
        for worker in &mut resources.workers {
            if worker
                .grant
                .is_some_and(|grant| grant.version == retired.version)
            {
                devices.insert(worker.device);
                worker.grant = None;
            }
        }
    }
    for endpoint in &bridge.endpoints {
        let version = endpoint.version;
        if !bridge
            .placements
            .iter()
            .any(|placement| placement.group == version.group)
        {
            continue;
        }
        let group = GroupResources {
            latest: version,
            phase: GroupPhase::Active,
            active: Some(ActiveGroup {
                version,
                model: model.to_owned(),
                role: endpoint.role,
            }),
        };
        if let Some(entry) = resources
            .groups
            .iter_mut()
            .find(|entry| entry.latest.group == version.group)
        {
            *entry = group;
        } else {
            resources.groups.push(group);
        }
    }
    Some(Projection {
        snapshot: HostedPdPlanning {
            revision: snapshot.revision + 1,
            managed: true,
            pending: None,
            active: Vec::new(),
            paths: bridge.paths(),
            resources,
            inventory,
        },
        devices,
    })
}
