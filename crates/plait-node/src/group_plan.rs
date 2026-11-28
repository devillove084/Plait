//! Placement snapshots and reservation evidence shared by native planners.

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use plait_core::NodeId;
use plait_core::group::{
    ActivePlacement, ComputeRate, GroupAction, GroupId, GroupLookup, GroupPreparation, GroupReply,
    PlacementInventory, RateEvidence, ResourcePlan,
};
use plait_core::link::ProbeRegistration;
use plait_parallel::group_placement::{self, Decision, DirectedLink, Workload};

pub(super) struct Snapshot {
    pub(super) inventory: PlacementInventory,
    pub(super) rates: Option<Vec<ComputeRate>>,
    pub(super) active_recipe: Option<NodeId>,
    calibration: Option<NodeId>,
}

/// Project only the grants and host charges of the caller's exact active set.
/// Eligibility and charges for other owners remain in the snapshot.
pub(super) fn project_release(
    inventory: &mut PlacementInventory,
    releasing: &[ActivePlacement],
) -> bool {
    let workers: BTreeMap<_, _> = inventory
        .workers
        .iter()
        .map(|entry| (entry.device, entry.grant))
        .collect();
    let mut reservations: BTreeMap<_, _> = inventory
        .hosts
        .iter()
        .map(|entry| (entry.host, entry.reserved))
        .collect();
    let mut released = BTreeSet::new();
    for active in releasing {
        for (device, demand) in &active.workers {
            if !workers.get(device).is_some_and(|grant| {
                grant.is_some_and(|grant| {
                    grant.version == active.version && grant.incarnation == demand.incarnation
                })
            }) || !released.insert(*device)
            {
                return false;
            }
        }
        for (host, demand) in &active.hosts {
            let Some(reserved) = reservations.get_mut(host) else {
                return false;
            };
            let Some(bytes) = reserved.bytes.checked_sub(demand.bytes) else {
                return false;
            };
            let Some(pinned_bytes) = reserved.pinned_bytes.checked_sub(demand.pinned_bytes) else {
                return false;
            };
            reserved.bytes = bytes;
            reserved.pinned_bytes = pinned_bytes;
        }
    }
    for entry in &mut inventory.workers {
        if released.contains(&entry.device) {
            entry.grant = None;
        }
    }
    for entry in &mut inventory.hosts {
        entry.reserved = reservations[&entry.host];
    }
    true
}

impl Snapshot {
    pub(super) fn lookup(group: GroupId, calibration: Option<NodeId>) -> GroupLookup {
        calibration.map_or(GroupLookup::Placement(group), |calibration| {
            GroupLookup::PlacementWithRates { group, calibration }
        })
    }

    pub(super) fn from_reply(reply: GroupReply, calibration: Option<NodeId>) -> io::Result<Self> {
        let (inventory, rates) = match reply {
            GroupReply::Placement(view) if calibration.is_none() => (view, None),
            GroupReply::PlacementWithRates { inventory, rates } if calibration.is_some() => {
                (inventory, Some(rates))
            }
            _ => return Err(io::Error::other("expected group placement inventory")),
        };
        Ok(Self {
            inventory,
            rates,
            active_recipe: None,
            calibration,
        })
    }

    pub(super) fn select(
        &self,
        workload: &Workload,
        links: &[DirectedLink],
    ) -> io::Result<Decision> {
        group_placement::select_with_recipes(
            &self.inventory,
            workload,
            links,
            crate::hosted_probe::now_ms()?,
            self.rates.as_deref(),
            self.active_recipe,
        )
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))
    }

    pub(super) fn restrict_probes(&self, probes: &mut Vec<ProbeRegistration>) {
        let hosts: BTreeSet<_> = self
            .inventory
            .hosts
            .iter()
            .map(|entry| entry.host)
            .collect();
        probes.retain(|probe| hosts.contains(&probe.endpoint.host));
    }

    pub(super) fn prepare(
        self,
        resources: ResourcePlan,
        compared_active: bool,
        probes: Option<Vec<ProbeRegistration>>,
        recipe: Option<NodeId>,
    ) -> GroupAction {
        let measured = probes.is_some();
        let GroupPreparation {
            group,
            expected_epoch,
            expected_active,
            resources,
            recipe,
            probes,
            rates,
        } = self.preparation(
            resources,
            compared_active,
            probes.unwrap_or_default(),
            recipe,
        );
        if let Some(recipe) = recipe {
            GroupAction::PrepareRecipe {
                group,
                expected_epoch,
                expected_active,
                resources,
                recipe,
                probes,
                rates,
            }
        } else if let Some(RateEvidence {
            calibration,
            compared_active,
            rates,
        }) = rates
        {
            GroupAction::PrepareWithRates {
                group,
                expected_epoch,
                expected_active,
                resources,
                probes,
                calibration,
                compared_active,
                rates,
            }
        } else if measured {
            GroupAction::PrepareMeasured {
                group,
                expected_epoch,
                expected_active,
                resources,
                probes,
            }
        } else {
            GroupAction::PreparePlanned {
                group,
                expected_epoch,
                expected_active,
                resources,
            }
        }
    }

    pub(super) fn preparation(
        self,
        resources: ResourcePlan,
        compared_active: bool,
        mut probes: Vec<ProbeRegistration>,
        recipe: Option<NodeId>,
    ) -> GroupPreparation {
        let group = self.inventory.group;
        let expected_epoch = self
            .inventory
            .latest
            .map_or(0, |(version, _)| version.epoch);
        let incumbent = self.inventory.active.as_ref();
        let expected_active = incumbent.map(|active| active.version.epoch);
        let mut used = BTreeSet::new();
        if resources.hosts.len() > 1 {
            used.extend(resources.hosts.keys().copied());
        }
        if compared_active
            && let Some(active) = incumbent
            && active.hosts.len() > 1
        {
            used.extend(active.hosts.iter().map(|&(host, _)| host));
        }
        probes.retain(|probe| used.contains(&probe.endpoint.host));
        let evidence = if let (Some(calibration), Some(mut rates)) = (self.calibration, self.rates)
        {
            let mut used: BTreeSet<_> = resources.workers.keys().copied().collect();
            if compared_active && let Some(active) = incumbent {
                used.extend(active.workers.iter().map(|&(device, _)| device));
            }
            rates.retain(|rate| used.contains(&rate.device));
            Some(RateEvidence {
                calibration,
                compared_active,
                rates,
            })
        } else {
            None
        };
        GroupPreparation {
            group,
            expected_epoch,
            expected_active,
            resources,
            recipe,
            probes,
            rates: evidence,
        }
    }
}

pub(super) async fn collect_links(probes: &[ProbeRegistration]) -> Vec<DirectedLink> {
    let endpoints = probes.iter().map(|probe| probe.endpoint).collect();
    crate::hosted_probe::collect(endpoints, None)
        .await
        .into_iter()
        .map(|((source, target), sample)| DirectedLink {
            source,
            target,
            sampled_at_ms: sample.sampled_at_ms,
            bytes_per_second: sample.bytes_per_second,
            rtt_ns: sample.rtt_ns,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::DeviceId;
    use plait_core::group::{
        GroupVersion, HostMemory, PlacementHost, PlacementWorker, WorkerDemand, WorkerGrant,
    };

    #[test]
    fn release_projection_credits_only_current_grants_and_their_host_memory() {
        let host = NodeId::derive("host");
        let group = GroupId(NodeId::derive("primary"));
        let version = GroupVersion { group, epoch: 2 };
        let separate = GroupVersion {
            group: GroupId(NodeId::derive("bridge")),
            epoch: 1,
        };
        let memory = HostMemory {
            bytes: 8,
            pinned_bytes: 2,
        };
        let target = ActivePlacement {
            version,
            workers: vec![(
                DeviceId::new(host, 0),
                WorkerDemand {
                    incarnation: 2,
                    device_bytes: 8,
                },
            )],
            hosts: vec![(host, memory)],
        };
        let mut inventory = PlacementInventory {
            group,
            latest: None,
            active: None,
            hosts: vec![PlacementHost {
                host,
                capacity: HostMemory {
                    bytes: 24,
                    pinned_bytes: 6,
                },
                reserved: HostMemory {
                    bytes: 16,
                    pinned_bytes: 4,
                },
                eligible: true,
            }],
            workers: [version, separate]
                .into_iter()
                .enumerate()
                .map(|(ordinal, version)| {
                    let device = DeviceId::new(host, ordinal as u32);
                    PlacementWorker {
                        device,
                        incarnation: 2,
                        device_bytes: 8,
                        grant: Some(WorkerGrant {
                            version,
                            device,
                            incarnation: 2,
                            fence: 1,
                        }),
                        eligible: ordinal != 0,
                        profile: None,
                    }
                })
                .collect(),
        };
        let mut stale = target.clone();
        stale.version.epoch -= 1;
        let original = inventory.clone();
        assert!(!project_release(&mut inventory, &[stale]));
        assert_eq!(inventory, original);

        assert!(project_release(&mut inventory, &[target]));
        assert_eq!(inventory.hosts[0].reserved, memory);
        assert_eq!(inventory.workers[0].grant, None);
        assert!(!inventory.workers[0].eligible);
        assert_eq!(inventory.workers[1], original.workers[1]);
    }
}
