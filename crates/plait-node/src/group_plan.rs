//! Placement snapshots and reservation evidence shared by native planners.

use std::collections::BTreeSet;
use std::io;

use plait_core::NodeId;
use plait_core::group::{
    ComputeRate, GroupAction, GroupId, GroupLookup, GroupReply, PlacementInventory, ResourcePlan,
};
use plait_core::link::ProbeRegistration;
use plait_parallel::group_placement::{self, Decision, DirectedLink, Workload};

pub(super) struct Snapshot {
    pub(super) inventory: PlacementInventory,
    pub(super) rates: Option<Vec<ComputeRate>>,
    calibration: Option<NodeId>,
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
            calibration,
        })
    }

    pub(super) fn select(
        &self,
        workload: &Workload,
        links: &[DirectedLink],
    ) -> io::Result<Decision> {
        group_placement::select_with_rates(
            &self.inventory,
            workload,
            links,
            crate::hosted_probe::now_ms()?,
            self.rates.as_deref(),
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
    ) -> GroupAction {
        let group = self.inventory.group;
        let expected_epoch = self
            .inventory
            .latest
            .map_or(0, |(version, _)| version.epoch);
        let incumbent = self.inventory.active.as_ref();
        let expected_active = incumbent.map(|active| active.version.epoch);
        let probes = probes.map(|mut probes| {
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
            probes
        });
        if let (Some(calibration), Some(mut rates)) = (self.calibration, self.rates) {
            let mut used: BTreeSet<_> = resources.workers.keys().copied().collect();
            if compared_active && let Some(active) = incumbent {
                used.extend(active.workers.iter().map(|&(device, _)| device));
            }
            rates.retain(|rate| used.contains(&rate.device));
            GroupAction::PrepareWithRates {
                group,
                expected_epoch,
                expected_active,
                resources,
                probes: probes.unwrap_or_default(),
                calibration,
                compared_active,
                rates,
            }
        } else if let Some(probes) = probes {
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
