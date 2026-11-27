//! Generic execution groups use the same device fences and budgets as hosted groups.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;

use plait_core::group::{
    ActivePlacement, ComputeRate, GrantStatus, GroupAction, GroupError, GroupId, GroupManifest,
    GroupMember, GroupOperation, GroupOutcome, GroupPhase, GroupPreparation, GroupReceipt,
    GroupRecipe, GroupResult, GroupTransition, GroupVersion, GroupView, PeerEndpoint,
    PlacementHost, PlacementInventory, PlacementWorker, PreparedSetView, RateEvidence,
    ResourcePlan, WorkerGrant,
};
use plait_core::link::{ProbeEndpoint, ProbeRegistration};
use plait_core::serving::{ExecutionDrainTarget, ServingError};
use plait_core::{DeviceId, DeviceProfile, NodeId, OpId};

use super::{Reservations, ServingState};

#[derive(Clone, Default)]
pub(super) struct PeerGroups {
    history: BTreeMap<GroupId, History>,
    receipts: BTreeMap<OpId, GroupReceipt>,
    profiles: BTreeMap<DeviceId, (u64, DeviceProfile)>,
    probes: BTreeMap<NodeId, ProbeRegistration>,
    rates: BTreeMap<DeviceId, BTreeMap<NodeId, ComputeRate>>,
}

#[derive(Clone, Default)]
struct History {
    versions: BTreeMap<u64, Version>,
    active: Option<u64>,
    standby: Option<u64>,
}

impl History {
    fn check_candidate(&self, version: GroupVersion) -> Result<(), GroupError> {
        let (&current, view) = self.versions.last_key_value().expect("recorded version");
        if version.epoch != current {
            return Err(GroupError::StaleEpoch { current });
        }
        if view.phase != GroupPhase::Ready {
            return Err(GroupError::InvalidPhase(view.phase));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Version {
    version: GroupVersion,
    recipe: Option<NodeId>,
    preparation: Option<OpId>,
    ever_activated: bool,
    phase: GroupPhase,
    resources: ResourcePlan,
    grants: BTreeMap<DeviceId, WorkerGrant>,
    endpoints: BTreeMap<DeviceId, PeerEndpoint>,
    manifest: Option<(GroupManifest, [u8; 32])>,
    ready: BTreeSet<DeviceId>,
    released: BTreeSet<DeviceId>,
}

impl Version {
    fn outcome(&self) -> GroupOutcome {
        GroupOutcome::Version {
            version: self.version,
            phase: self.phase,
        }
    }

    fn check_grant(&self, grant: WorkerGrant) -> Result<(), GroupError> {
        if self.grants.get(&grant.device) != Some(&grant) {
            return Err(GroupError::GrantMismatch);
        }
        Ok(())
    }

    fn view(&self) -> GroupView {
        GroupView {
            version: self.version,
            phase: self.phase,
            grants: self.grants.values().copied().collect(),
            manifest: self.manifest.as_ref().map(|(manifest, _)| manifest.clone()),
            ready: self.ready.clone(),
            released: self.released.clone(),
        }
    }

    fn set_phase(&mut self, phase: GroupPhase) {
        self.phase = phase;
        self.ever_activated |= phase == GroupPhase::Active;
        if matches!(
            phase,
            GroupPhase::Draining | GroupPhase::Revoked | GroupPhase::Retired
        ) {
            self.manifest = None;
            self.endpoints.clear();
        }
    }
}

impl PeerGroups {
    pub(super) fn clear_compute_rates(&mut self, device: DeviceId) {
        self.rates.remove(&device);
    }

    pub(super) fn has_group(&self, group: GroupId) -> bool {
        self.history.contains_key(&group)
    }

    fn get(&self, version: GroupVersion) -> Option<&Version> {
        self.history
            .get(&version.group)?
            .versions
            .get(&version.epoch)
    }

    fn get_mut(&mut self, version: GroupVersion) -> Result<&mut Version, GroupError> {
        self.history
            .get_mut(&version.group)
            .and_then(|history| history.versions.get_mut(&version.epoch))
            .ok_or(GroupError::UnknownVersion(version))
    }

    fn active_epoch(&self, group: GroupId) -> Option<u64> {
        self.history.get(&group)?.active
    }
}

impl ServingState {
    pub(crate) fn link_probes(&self) -> Vec<ProbeRegistration> {
        self.peer_groups.probes.values().copied().collect()
    }

    pub(crate) fn compute_rates(&self, calibration: NodeId) -> Vec<ComputeRate> {
        self.peer_groups
            .rates
            .values()
            .filter_map(|entries| entries.get(&calibration).copied())
            .collect()
    }

    pub(crate) fn compute_rate(
        &self,
        device: DeviceId,
        calibration: NodeId,
    ) -> Option<ComputeRate> {
        self.peer_groups
            .rates
            .get(&device)?
            .get(&calibration)
            .copied()
    }

    pub(crate) fn placement_inventory(&self, group: GroupId) -> PlacementInventory {
        let history = self.peer_groups.history.get(&group);
        let latest = history
            .and_then(|history| history.versions.last_key_value())
            .map(|(_, version)| (version.version, version.phase));
        let active = history
            .and_then(|history| {
                history
                    .active
                    .and_then(|epoch| history.versions.get(&epoch))
            })
            .map(|version| ActivePlacement {
                version: version.version,
                workers: version
                    .resources
                    .workers
                    .iter()
                    .map(|(&id, &demand)| (id, demand))
                    .collect(),
                hosts: version
                    .resources
                    .hosts
                    .iter()
                    .map(|(&id, &demand)| (id, demand))
                    .collect(),
            });
        let hosts = self
            .hosts
            .iter()
            .map(|(&host, record)| PlacementHost {
                host,
                capacity: record.capacity,
                reserved: super::add_memory(record.reserved, self.deferred_pd_host_memory(host))
                    .expect("host reservations are within capacity"),
                eligible: self.require_execution_host(host).is_ok()
                    && !self
                        .execution_drains
                        .contains_key(&ExecutionDrainTarget::Host(host)),
            })
            .collect();
        let deferred: BTreeSet<_> = self
            .deferred_pd_placements()
            .flat_map(|(_, placement)| placement.plan.workers.keys().copied())
            .collect();
        let workers = self
            .workers
            .iter()
            .map(|(&device, record)| PlacementWorker {
                device,
                incarnation: record.incarnation,
                device_bytes: record.device_bytes,
                grant: record.grant,
                eligible: self.execution_drain_for(device).is_none() && !deferred.contains(&device),
                profile: self
                    .peer_groups
                    .profiles
                    .get(&device)
                    .filter(|(incarnation, _)| *incarnation == record.incarnation)
                    .map(|(_, profile)| profile.clone()),
            })
            .collect();
        PlacementInventory {
            group,
            latest,
            active,
            hosts,
            workers,
        }
    }

    pub(crate) fn peer_group_view(&self, version: GroupVersion) -> Option<GroupView> {
        self.peer_groups.get(version).map(Version::view)
    }

    pub(crate) fn peer_group_recipe(&self, version: GroupVersion) -> Option<GroupRecipe> {
        self.peer_groups.get(version).map(|record| GroupRecipe {
            version,
            recipe: record.recipe,
        })
    }

    pub(crate) fn peer_grant_status(&self, grant: WorkerGrant) -> GrantStatus {
        let Some(view) = self.peer_groups.get(grant.version) else {
            return GrantStatus::Unknown;
        };
        view.phase.grant_status(
            view.grants.get(&grant.device) == Some(&grant),
            view.released.contains(&grant.device),
        )
    }

    pub(crate) fn peer_staging_bytes(&self, grant: WorkerGrant) -> Option<u64> {
        let view = self.peer_groups.get(grant.version)?;
        view.check_grant(grant).ok()?;
        if !view.phase.accepts_peer_join() || view.released.contains(&grant.device) {
            return None;
        }
        let host = grant.device.node;
        let memory = view.resources.hosts[&host];
        // Keep shares fixed for the epoch, even if a worker releases early.
        let workers = view
            .grants
            .keys()
            .filter(|device| device.node == host)
            .count() as u64;
        Some((memory.bytes - memory.pinned_bytes) / workers)
    }

    pub(crate) fn peer_standby(&self, group: GroupId) -> Option<GroupVersion> {
        let epoch = self.peer_groups.history.get(&group)?.standby?;
        Some(GroupVersion { group, epoch })
    }

    pub(crate) fn peer_group_receipt(&self, operation: OpId) -> Option<&GroupReceipt> {
        self.peer_groups.receipts.get(&operation)
    }

    pub(crate) fn latest_peer_preparation(&self, group: GroupId) -> Option<PreparedSetView> {
        let (_, latest) = self
            .peer_groups
            .history
            .get(&group)?
            .versions
            .last_key_value()?;
        let prepare = latest.preparation?;
        let versions = self
            .prepared_versions(prepare)
            .expect("recorded prepare-set receipt");
        let mut activated = false;
        let members = versions
            .iter()
            .map(|&version| {
                let view = self
                    .peer_groups
                    .get(version)
                    .expect("recorded prepared version");
                activated |= view.ever_activated;
                (version, view.phase)
            })
            .collect();
        Some(PreparedSetView {
            prepare,
            members,
            activated,
        })
    }

    fn prepared_versions(&self, prepare: OpId) -> Result<&[GroupVersion], GroupError> {
        let Some(GroupReceipt {
            result: Ok(GroupOutcome::PreparedSet(versions)),
            ..
        }) = self.peer_group_receipt(prepare)
        else {
            return Err(GroupError::UnknownPreparation(prepare));
        };
        Ok(versions)
    }

    pub(crate) fn apply_peer_group(&mut self, operation: &GroupOperation) -> GroupResult {
        if let Some(receipt) = self.peer_group_receipt(operation.id) {
            return if &receipt.operation == operation {
                receipt.result.clone()
            } else {
                Err(GroupError::OperationIdReused)
            };
        }
        let result = self.peer_group_transition(&operation.action);
        if let Ok(GroupOutcome::PreparedSet(versions)) = &result {
            for &version in versions {
                self.peer_groups
                    .get_mut(version)
                    .expect("committed preparation")
                    .preparation = Some(operation.id);
            }
        }
        self.peer_groups.receipts.insert(
            operation.id,
            GroupReceipt {
                operation: operation.clone(),
                result: result.clone(),
            },
        );
        result
    }

    fn peer_group_transition(&mut self, action: &GroupAction) -> GroupResult {
        match *action {
            GroupAction::SetHostMemory { host, capacity } => {
                self.set_host_memory(host, capacity)
                    .map_err(resource_error)?;
                Ok(GroupOutcome::HostConfigured)
            }
            GroupAction::RegisterWorker {
                device,
                incarnation,
                device_bytes,
            } => {
                self.register_worker(device, incarnation, device_bytes)
                    .map_err(resource_error)?;
                Ok(GroupOutcome::WorkerRegistered)
            }
            GroupAction::Prepare {
                group,
                expected_epoch,
                ref resources,
            } => self.prepare_peer_group(group, expected_epoch, resources),
            GroupAction::PublishEndpoint(member) => self.publish_peer_endpoint(member),
            GroupAction::Ready { grant, manifest } => self.ready_peer_group(grant, manifest),
            GroupAction::Activate(version) => self.activate_peer_group(version),
            GroupAction::Drain(version) => self.quiesce_peer_group(version, GroupPhase::Draining),
            GroupAction::Revoke(version) => self.quiesce_peer_group(version, GroupPhase::Revoked),
            GroupAction::Release(grant) => self.release_peer_worker(grant),
            GroupAction::SetStandby { active, standby } => self.set_peer_standby(active, standby),
            GroupAction::ReportFailure { grant, .. } => self.report_peer_failure(grant),
            GroupAction::Observe { group, epoch } => {
                let epoch = epoch
                    .or_else(|| self.peer_groups.active_epoch(group))
                    .ok_or(GroupError::NoActiveGroup(group))?;
                let version = GroupVersion { group, epoch };
                self.peer_groups
                    .get(version)
                    .map(Version::outcome)
                    .ok_or(GroupError::UnknownVersion(version))
            }
            GroupAction::SwitchActiveSet { ref transitions } => self.switch_active_set(transitions),
            GroupAction::ObserveActiveSet { ref groups } => {
                let group_ids: BTreeSet<_> = groups.iter().copied().collect();
                if group_ids.is_empty() || group_ids.len() != groups.len() {
                    return Err(GroupError::InvalidPlan);
                }
                Ok(GroupOutcome::ActiveSet(
                    groups
                        .iter()
                        .map(|&group| (group, self.peer_groups.active_epoch(group)))
                        .collect(),
                ))
            }
            GroupAction::PrepareSet {
                ref preparations,
                ref guards,
            } => self.prepare_peer_set(preparations, guards),
            GroupAction::AbortFailedSet {
                prepare,
                ref versions,
            } => self.abort_failed_peer_set(prepare, versions),
            GroupAction::AdvertiseWorker {
                ref profile,
                incarnation,
                device_bytes,
            } => {
                if profile.name.is_empty() || device_bytes > profile.memory_bytes {
                    return Err(GroupError::InvalidCapacity);
                }
                if self.peer_groups.profiles.get(&profile.id).is_some_and(
                    |(version, registered)| *version == incarnation && registered != profile,
                ) {
                    return Err(GroupError::ProfileMismatch(profile.id));
                }
                self.register_worker(profile.id, incarnation, device_bytes)
                    .map_err(resource_error)?;
                self.peer_groups
                    .profiles
                    .insert(profile.id, (incarnation, profile.clone()));
                Ok(GroupOutcome::WorkerRegistered)
            }
            GroupAction::PreparePlanned {
                group,
                expected_epoch,
                expected_active,
                ref resources,
            } => self.prepare_planned_peer_group(group, expected_epoch, expected_active, resources),
            GroupAction::ObserveProbe(host) => Ok(GroupOutcome::Probe(
                self.peer_groups.probes.get(&host).copied(),
            )),
            GroupAction::PublishProbe {
                endpoint,
                expected_revision,
            } => self.publish_link_probe(endpoint, expected_revision),
            GroupAction::PrepareMeasured {
                group,
                expected_epoch,
                expected_active,
                ref resources,
                ref probes,
            } => {
                self.check_link_probes(probes)?;
                self.prepare_planned_peer_group(group, expected_epoch, expected_active, resources)
            }
            GroupAction::PublishComputeRate {
                calibration,
                device,
                incarnation,
                expected_revision,
                operations_per_second,
            } => self.publish_compute_rate(
                calibration,
                device,
                incarnation,
                expected_revision,
                operations_per_second,
            ),
            GroupAction::PublishGrantedComputeRate {
                grant,
                calibration,
                expected_revision,
                operations_per_second,
            } => {
                let view = self
                    .peer_groups
                    .get(grant.version)
                    .ok_or(GroupError::UnknownVersion(grant.version))?;
                if !matches!(
                    view.phase,
                    GroupPhase::Preparing | GroupPhase::Ready | GroupPhase::Active
                ) {
                    return Err(GroupError::InvalidPhase(view.phase));
                }
                view.check_grant(grant)?;
                if self
                    .workers
                    .get(&grant.device)
                    .and_then(|worker| worker.grant)
                    != Some(grant)
                {
                    return Err(GroupError::GrantMismatch);
                }
                self.publish_compute_rate(
                    calibration,
                    grant.device,
                    grant.incarnation,
                    expected_revision,
                    Some(operations_per_second),
                )
            }
            GroupAction::PrepareWithRates {
                group,
                expected_epoch,
                expected_active,
                ref resources,
                ref probes,
                calibration,
                compared_active,
                ref rates,
            } => {
                self.check_link_probes(probes)?;
                let incumbent = if compared_active {
                    Some(GroupVersion {
                        group,
                        epoch: expected_active.ok_or(GroupError::InvalidPlan)?,
                    })
                } else {
                    None
                };
                self.check_compute_rates(calibration, resources, incumbent, rates)?;
                self.prepare_planned_peer_group(group, expected_epoch, expected_active, resources)
            }
            GroupAction::PrepareRecipe {
                group,
                expected_epoch,
                expected_active,
                ref resources,
                recipe,
                ref probes,
                ref rates,
            } => {
                self.check_preparation_evidence(
                    group,
                    expected_active,
                    resources,
                    probes,
                    rates.as_ref(),
                )?;
                let result = self.prepare_planned_peer_group(
                    group,
                    expected_epoch,
                    expected_active,
                    resources,
                )?;
                let GroupOutcome::Version { version, .. } = result else {
                    unreachable!("preparation returns its new group version");
                };
                self.peer_groups
                    .get_mut(version)
                    .expect("new group version")
                    .recipe = Some(recipe);
                Ok(result)
            }
        }
    }

    fn check_link_probes(&self, probes: &[ProbeRegistration]) -> Result<(), GroupError> {
        for probe in probes {
            let host = probe.endpoint.host;
            let current = self.peer_groups.probes.get(&host);
            if current != Some(probe) {
                return Err(GroupError::StaleProbe {
                    host,
                    revision: current.map_or(0, |entry| entry.revision),
                });
            }
        }
        Ok(())
    }

    fn check_preparation_evidence(
        &self,
        group: GroupId,
        expected_active: Option<u64>,
        resources: &ResourcePlan,
        probes: &[ProbeRegistration],
        rates: Option<&RateEvidence>,
    ) -> Result<(), GroupError> {
        self.check_link_probes(probes)?;
        if let Some(evidence) = rates {
            let incumbent = if evidence.compared_active {
                Some(GroupVersion {
                    group,
                    epoch: expected_active.ok_or(GroupError::InvalidPlan)?,
                })
            } else {
                None
            };
            self.check_compute_rates(evidence.calibration, resources, incumbent, &evidence.rates)?;
        }
        Ok(())
    }

    fn publish_compute_rate(
        &mut self,
        calibration: NodeId,
        device: DeviceId,
        incarnation: u64,
        expected_revision: u64,
        operations_per_second: Option<NonZeroU64>,
    ) -> GroupResult {
        let worker = self
            .workers
            .get(&device)
            .ok_or(GroupError::UnknownWorker(device))?;
        if worker.incarnation != incarnation {
            return Err(GroupError::StaleIncarnation(device));
        }
        let entries = self.peer_groups.rates.entry(device).or_default();
        let current = entries.get(&calibration).copied();
        let revision = current.map_or(0, |rate| rate.revision);
        if revision != expected_revision {
            return Err(GroupError::StaleComputeRate { device, revision });
        }
        if let Some(rate) =
            current.filter(|rate| rate.operations_per_second == operations_per_second)
        {
            return Ok(GroupOutcome::ComputeRate(rate));
        }
        let rate = ComputeRate {
            device,
            incarnation,
            revision: revision
                .checked_add(1)
                .ok_or(GroupError::SequenceExhausted)?,
            operations_per_second,
        };
        entries.insert(calibration, rate);
        Ok(GroupOutcome::ComputeRate(rate))
    }

    fn check_compute_rates(
        &self,
        calibration: NodeId,
        resources: &ResourcePlan,
        incumbent: Option<GroupVersion>,
        rates: &[ComputeRate],
    ) -> Result<(), GroupError> {
        let mut devices: BTreeSet<_> = resources.workers.keys().copied().collect();
        if let Some(version) = incumbent {
            let view = self
                .peer_groups
                .get(version)
                .ok_or(GroupError::UnknownVersion(version))?;
            devices.extend(view.resources.workers.keys().copied());
        }
        for rate in rates {
            let device = rate.device;
            if !devices.remove(&device) || rate.operations_per_second.is_none() {
                return Err(GroupError::InvalidPlan);
            }
            let current = self.compute_rate(device, calibration);
            if current != Some(*rate) {
                return Err(GroupError::StaleComputeRate {
                    device,
                    revision: current.map_or(0, |entry| entry.revision),
                });
            }
        }
        if !devices.is_empty() {
            return Err(GroupError::InvalidPlan);
        }
        Ok(())
    }

    fn publish_link_probe(
        &mut self,
        endpoint: ProbeEndpoint,
        expected_revision: u64,
    ) -> GroupResult {
        if !endpoint.valid() {
            return Err(GroupError::InvalidEndpoint);
        }
        let current = self.peer_groups.probes.get(&endpoint.host).copied();
        let revision = current.map_or(0, |entry| entry.revision);
        if revision != expected_revision {
            return Err(GroupError::StaleProbe {
                host: endpoint.host,
                revision,
            });
        }
        if let Some(entry) = current.filter(|entry| entry.endpoint.session == endpoint.session) {
            return if entry.endpoint == endpoint {
                Ok(GroupOutcome::Probe(current))
            } else {
                Err(GroupError::InvalidEndpoint)
            };
        }
        let entry = ProbeRegistration {
            revision: revision
                .checked_add(1)
                .ok_or(GroupError::SequenceExhausted)?,
            endpoint,
        };
        self.peer_groups.probes.insert(endpoint.host, entry);
        Ok(GroupOutcome::Probe(Some(entry)))
    }

    fn prepare_planned_peer_group(
        &mut self,
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        resources: &ResourcePlan,
    ) -> GroupResult {
        let current = self
            .peer_groups
            .history
            .get(&group)
            .and_then(|history| history.active);
        if current != expected_active {
            return Err(GroupError::ActiveEpochChanged { current });
        }
        self.prepare_peer_group(group, expected_epoch, resources)
    }

    fn prepare_peer_group(
        &mut self,
        group: GroupId,
        expected_epoch: u64,
        resources: &ResourcePlan,
    ) -> GroupResult {
        let mut reservations = Reservations::default();
        let record =
            self.reserve_peer_group(group, expected_epoch, resources, &mut reservations)?;
        let result = record.outcome();
        self.install_peer_groups([record], reservations);
        Ok(result)
    }

    fn prepare_peer_set(
        &mut self,
        preparations: &[GroupPreparation],
        guards: &[(GroupId, Option<u64>)],
    ) -> GroupResult {
        if preparations.is_empty() {
            return Err(GroupError::InvalidPlan);
        }
        let mut guarded = BTreeSet::new();
        for &(group, expected) in guards {
            if !guarded.insert(group) {
                return Err(GroupError::InvalidPlan);
            }
            self.check_active_epoch(group, expected)?;
        }
        let mut groups = BTreeSet::new();
        let mut reservations = Reservations::default();
        let mut records = Vec::with_capacity(preparations.len());
        for plan in preparations {
            if !groups.insert(plan.group) {
                return Err(GroupError::InvalidPlan);
            }
            self.check_active_epoch(plan.group, plan.expected_active)?;
            self.check_preparation_evidence(
                plan.group,
                plan.expected_active,
                &plan.resources,
                &plan.probes,
                plan.rates.as_ref(),
            )?;
            let mut record = self.reserve_peer_group(
                plan.group,
                plan.expected_epoch,
                &plan.resources,
                &mut reservations,
            )?;
            record.recipe = plan.recipe;
            records.push(record);
        }
        let versions = records.iter().map(|record| record.version).collect();
        self.install_peer_groups(records, reservations);
        Ok(GroupOutcome::PreparedSet(versions))
    }

    fn check_active_epoch(&self, group: GroupId, expected: Option<u64>) -> Result<(), GroupError> {
        let current = self.peer_groups.active_epoch(group);
        if expected != current {
            return Err(GroupError::GroupSetChanged { group, current });
        }
        Ok(())
    }

    fn abort_failed_peer_set(&mut self, prepare: OpId, versions: &[GroupVersion]) -> GroupResult {
        if self.prepared_versions(prepare)? != versions {
            return Err(GroupError::InvalidPlan);
        }
        let mut failed = false;
        let mut candidates = Vec::new();
        for &version in versions {
            let view = self
                .peer_groups
                .get(version)
                .expect("recorded prepared version");
            if view.ever_activated {
                return Err(GroupError::PreparationActivated);
            }
            match view.phase {
                GroupPhase::Revoked | GroupPhase::Retired => failed = true,
                GroupPhase::Ready | GroupPhase::Preparing => candidates.push(version),
                phase => return Err(GroupError::InvalidPhase(phase)),
            }
        }
        if !failed {
            return Err(GroupError::PreparationHealthy);
        }
        for version in candidates {
            self.quiesce_peer_group(version, GroupPhase::Revoked)
                .expect("prechecked candidate");
        }
        Ok(GroupOutcome::AbortedSet(versions.to_vec()))
    }

    fn reserve_peer_group(
        &self,
        group: GroupId,
        expected_epoch: u64,
        resources: &ResourcePlan,
        reservations: &mut Reservations,
    ) -> Result<Version, GroupError> {
        if !resources.valid()
            || self.groups.contains_key(&group)
            || self
                .deferred_pd_placements()
                .any(|(_, placement)| placement.group == group)
        {
            return Err(GroupError::InvalidPlan);
        }
        let latest = self
            .peer_groups
            .history
            .get(&group)
            .and_then(|history| history.versions.last_key_value());
        let current = latest.map_or(0, |(&epoch, _)| epoch);
        if expected_epoch != current {
            return Err(GroupError::StaleEpoch { current });
        }
        if let Some((_, view)) = latest
            .filter(|(_, view)| matches!(view.phase, GroupPhase::Preparing | GroupPhase::Ready))
        {
            return Err(GroupError::CandidateExists(view.version));
        }
        let version = GroupVersion {
            group,
            epoch: current
                .checked_add(1)
                .ok_or(GroupError::SequenceExhausted)?,
        };
        let grants = self
            .reserve_resources(version, &resources.workers, &resources.hosts, reservations)
            .map_err(resource_error)?;
        Ok(Version {
            version,
            recipe: None,
            preparation: None,
            ever_activated: false,
            phase: GroupPhase::Preparing,
            resources: resources.clone(),
            grants,
            endpoints: BTreeMap::new(),
            manifest: None,
            ready: BTreeSet::new(),
            released: BTreeSet::new(),
        })
    }

    fn install_peer_groups(
        &mut self,
        records: impl IntoIterator<Item = Version>,
        reservations: Reservations,
    ) {
        self.install_host_reservations(reservations.hosts);
        for record in records {
            self.install_grants(&record.grants);
            self.peer_groups
                .history
                .entry(record.version.group)
                .or_default()
                .versions
                .insert(record.version.epoch, record);
        }
    }

    fn publish_peer_endpoint(&mut self, member: GroupMember) -> GroupResult {
        let grant = member.grant;
        let view = self.peer_groups.get_mut(grant.version)?;
        view.check_grant(grant)?;
        if !view.phase.accepts_peer_join() {
            return Err(GroupError::InvalidPhase(view.phase));
        }
        let endpoint = member.endpoint;
        if let Some(old) = view.endpoints.get(&grant.device) {
            return if old == &endpoint {
                Ok(view.outcome())
            } else {
                Err(GroupError::InvalidEndpoint)
            };
        }
        if !endpoint.valid()
            || view.endpoints.values().any(|existing| {
                existing.address == endpoint.address || existing.ticket == endpoint.ticket
            })
        {
            return Err(GroupError::InvalidEndpoint);
        }
        view.endpoints.insert(grant.device, endpoint);
        if view.endpoints.len() == view.grants.len() {
            let members = view
                .grants
                .iter()
                .map(|(device, &grant)| GroupMember {
                    grant,
                    endpoint: view.endpoints[device],
                })
                .collect();
            let manifest = GroupManifest::new(view.version, members).expect("validated peer group");
            let digest = manifest.digest();
            view.manifest = Some((manifest, digest));
        }
        Ok(view.outcome())
    }

    fn ready_peer_group(&mut self, grant: WorkerGrant, digest: [u8; 32]) -> GroupResult {
        let view = self.peer_groups.get_mut(grant.version)?;
        view.check_grant(grant)?;
        if !view.phase.accepts_peer_join() {
            return Err(GroupError::InvalidPhase(view.phase));
        }
        if view
            .manifest
            .as_ref()
            .is_none_or(|(_, expected)| expected != &digest)
        {
            return Err(GroupError::ManifestMismatch);
        }
        view.ready.insert(grant.device);
        if view.phase == GroupPhase::Preparing && view.ready.len() == view.grants.len() {
            view.phase = GroupPhase::Ready;
        }
        Ok(view.outcome())
    }

    fn activate_peer_group(&mut self, version: GroupVersion) -> GroupResult {
        let history = self
            .peer_groups
            .history
            .get_mut(&version.group)
            .ok_or(GroupError::UnknownVersion(version))?;
        history.check_candidate(version)?;
        history.standby = None;
        if let Some(old) = history.active.replace(version.epoch) {
            history
                .versions
                .get_mut(&old)
                .expect("active version")
                .set_phase(GroupPhase::Draining);
        }
        let view = history
            .versions
            .get_mut(&version.epoch)
            .expect("latest version");
        view.set_phase(GroupPhase::Active);
        Ok(view.outcome())
    }

    fn switch_active_set(&mut self, transitions: &[GroupTransition]) -> GroupResult {
        let mut groups = BTreeSet::new();
        if transitions.is_empty() {
            return Err(GroupError::InvalidPlan);
        }
        for &GroupTransition {
            group,
            expected,
            target,
        } in transitions
        {
            if !groups.insert(group) {
                return Err(GroupError::InvalidPlan);
            }
            let current = self.peer_groups.active_epoch(group);
            if expected != current {
                return Err(GroupError::GroupSetChanged { group, current });
            }
            if let Some(epoch) = target.filter(|next| Some(*next) != expected) {
                let version = GroupVersion { group, epoch };
                self.peer_groups
                    .history
                    .get(&group)
                    .ok_or(GroupError::UnknownVersion(version))?
                    .check_candidate(version)?;
            }
        }

        // Each group was checked above and occurs at most once. Applying a
        // switch leaves its previous grants with the draining epoch.
        for &GroupTransition {
            group,
            expected,
            target,
        } in transitions
        {
            match (expected, target) {
                (old, Some(epoch)) if old != Some(epoch) => {
                    self.activate_peer_group(GroupVersion { group, epoch })
                        .expect("prechecked target epoch");
                }
                (Some(epoch), None) => {
                    self.quiesce_peer_group(GroupVersion { group, epoch }, GroupPhase::Draining)
                        .expect("prechecked active epoch");
                }
                _ => {}
            }
        }
        Ok(GroupOutcome::ActiveSet(
            transitions
                .iter()
                .map(|change| (change.group, change.target))
                .collect(),
        ))
    }

    fn set_peer_standby(&mut self, active: GroupVersion, standby: Option<u64>) -> GroupResult {
        let history = self
            .peer_groups
            .history
            .get_mut(&active.group)
            .ok_or(GroupError::UnknownVersion(active))?;
        if history.active != Some(active.epoch) {
            return Err(GroupError::ActiveEpochChanged {
                current: history.active,
            });
        }
        if let Some(epoch) = standby {
            history.check_candidate(GroupVersion {
                group: active.group,
                epoch,
            })?;
        }
        history.standby = standby;
        Ok(history.versions[&active.epoch].outcome())
    }

    fn report_peer_failure(&mut self, grant: WorkerGrant) -> GroupResult {
        let view = self
            .peer_groups
            .get(grant.version)
            .ok_or(GroupError::UnknownVersion(grant.version))?;
        view.check_grant(grant)?;
        if matches!(view.phase, GroupPhase::Revoked | GroupPhase::Retired) {
            return Ok(view.outcome());
        }
        if view.released.contains(&grant.device) {
            return Err(GroupError::GrantMismatch);
        }
        self.fail_peer_group(grant.version, None)
    }

    pub(super) fn fail_peer_group(
        &mut self,
        version: GroupVersion,
        affected: Option<&BTreeSet<GroupVersion>>,
    ) -> GroupResult {
        let standby = self
            .peer_groups
            .history
            .get(&version.group)
            .and_then(|history| {
                let epoch = history.standby?;
                let candidate = GroupVersion {
                    group: version.group,
                    epoch,
                };
                (history.active == Some(version.epoch)
                    && affected.is_none_or(|lost| !lost.contains(&candidate))
                    && history.check_candidate(candidate).is_ok())
                .then_some(candidate)
            });
        let revoked = self.quiesce_peer_group(version, GroupPhase::Revoked)?;
        if let Some(standby) = standby {
            self.activate_peer_group(standby)
                .expect("candidate checked before revocation");
        }
        Ok(revoked)
    }

    pub(super) fn quiesce_peer_group(
        &mut self,
        version: GroupVersion,
        phase: GroupPhase,
    ) -> GroupResult {
        let view = self.peer_groups.get_mut(version)?;
        let allowed = match phase {
            GroupPhase::Draining => view.phase == GroupPhase::Active,
            GroupPhase::Revoked => matches!(
                view.phase,
                GroupPhase::Preparing
                    | GroupPhase::Ready
                    | GroupPhase::Active
                    | GroupPhase::Draining
            ),
            _ => false,
        };
        if !allowed {
            return Err(GroupError::InvalidPhase(view.phase));
        }
        view.set_phase(phase);
        let result = view.outcome();
        let history = self
            .peer_groups
            .history
            .get_mut(&version.group)
            .expect("known group");
        if history.active == Some(version.epoch) {
            history.active = None;
            history.standby = None;
        } else if history.standby == Some(version.epoch) {
            history.standby = None;
        }
        Ok(result)
    }

    fn release_peer_worker(&mut self, grant: WorkerGrant) -> GroupResult {
        let view = self.peer_groups.get_mut(grant.version)?;
        view.check_grant(grant)?;
        if view.released.contains(&grant.device) {
            return Ok(view.outcome());
        }
        if !matches!(view.phase, GroupPhase::Draining | GroupPhase::Revoked) {
            return Err(GroupError::InvalidPhase(view.phase));
        }
        let worker = self
            .workers
            .get_mut(&grant.device)
            .expect("reserved worker");
        if worker.grant != Some(grant) {
            return Err(GroupError::GrantMismatch);
        }
        worker.grant = None;
        view.released.insert(grant.device);
        let host = grant.device.node;
        if view
            .grants
            .keys()
            .filter(|device| device.node == host)
            .all(|device| view.released.contains(device))
        {
            let demand = view.resources.hosts[&host];
            let reserved = &mut self.hosts.get_mut(&host).expect("reserved host").reserved;
            reserved.bytes -= demand.bytes;
            reserved.pinned_bytes -= demand.pinned_bytes;
        }
        if view.released.len() == view.grants.len() {
            view.set_phase(GroupPhase::Retired);
        }
        Ok(view.outcome())
    }

    pub(super) fn peer_version_is_live(&self, version: GroupVersion) -> bool {
        self.peer_groups.get(version).is_some_and(|view| {
            matches!(
                view.phase,
                GroupPhase::Preparing
                    | GroupPhase::Ready
                    | GroupPhase::Active
                    | GroupPhase::Draining
            )
        })
    }
}

fn resource_error(error: ServingError) -> GroupError {
    match error {
        ServingError::InvalidCapacity => GroupError::InvalidCapacity,
        ServingError::UnknownHost(host) => GroupError::UnknownHost(host),
        ServingError::HostCapacityExceeded(host) => GroupError::HostCapacityExceeded(host),
        ServingError::ExecutionHostOffline(host) => GroupError::HostOffline(host),
        ServingError::UnknownWorker(device) => GroupError::UnknownWorker(device),
        ServingError::WorkerBusy(device) => GroupError::WorkerBusy(device),
        ServingError::ExecutionTargetDraining(ExecutionDrainTarget::Device(device)) => {
            GroupError::WorkerDraining(device)
        }
        ServingError::ExecutionTargetDraining(ExecutionDrainTarget::Host(host)) => {
            GroupError::HostDraining(host)
        }
        ServingError::StaleIncarnation(device) => GroupError::StaleIncarnation(device),
        ServingError::DeviceCapacityExceeded(device) => GroupError::DeviceCapacityExceeded(device),
        ServingError::SequenceExhausted => GroupError::SequenceExhausted,
        _ => GroupError::InvalidPlan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::AcceleratorKind;
    use plait_core::NodeId;
    use plait_core::group::{GroupFault, HostMemory, WorkerDemand};
    use plait_core::serving::{GroupPlan, ServingRole};

    fn ready_single_worker(state: &mut ServingState, version: GroupVersion) -> WorkerGrant {
        let grant = state.peer_group_view(version).unwrap().grants[0];
        state
            .publish_peer_endpoint(GroupMember {
                grant,
                endpoint: PeerEndpoint {
                    address: ([127, 0, 0, 1], 9100 + version.epoch as u16).into(),
                    ticket: [version.epoch as u8; 16],
                },
            })
            .unwrap();
        let manifest = state.peer_group_view(version).unwrap().manifest.unwrap();
        state.ready_peer_group(grant, manifest.digest()).unwrap();
        grant
    }

    fn prepared_successor(
        same_host: bool,
    ) -> (ServingState, GroupVersion, GroupVersion, WorkerGrant) {
        let mut state = ServingState::default();
        let first_host = NodeId::derive("active worker");
        let second_host = if same_host {
            first_host
        } else {
            NodeId::derive("standby worker")
        };
        let group = GroupId(NodeId::derive("failover group"));
        for (host, ordinal) in [(first_host, 0), (second_host, 1)] {
            state
                .set_host_memory(
                    host,
                    HostMemory {
                        bytes: 16,
                        pinned_bytes: 0,
                    },
                )
                .unwrap();
            let device = DeviceId::new(host, ordinal);
            state.register_worker(device, 1, 16).unwrap();
            let resources = ResourcePlan {
                workers: BTreeMap::from([(
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 8,
                    },
                )]),
                hosts: BTreeMap::from([(
                    host,
                    HostMemory {
                        bytes: 4,
                        pinned_bytes: 0,
                    },
                )]),
            };
            state
                .prepare_peer_group(group, u64::from(ordinal), &resources)
                .unwrap();
            if ordinal == 0 {
                let active = GroupVersion { group, epoch: 1 };
                ready_single_worker(&mut state, active);
                state.activate_peer_group(active).unwrap();
            }
        }
        let active = GroupVersion { group, epoch: 1 };
        let standby = GroupVersion { group, epoch: 2 };
        let grant = state.peer_group_view(active).unwrap().grants[0];
        (state, active, standby, grant)
    }

    #[test]
    fn prepare_set_leaves_no_fences_or_budget_after_a_later_plan_fails() {
        let mut state = ServingState::default();
        let host = NodeId::derive("host");
        let first = DeviceId::new(host, 0);
        let second = DeviceId::new(host, 1);
        state
            .set_host_memory(
                host,
                HostMemory {
                    bytes: 24,
                    pinned_bytes: 6,
                },
            )
            .unwrap();
        for device in [first, second] {
            state.register_worker(device, 1, 16).unwrap();
        }
        let make_plan = |name, device, pinned_bytes| GroupPreparation {
            group: GroupId(NodeId::derive(name)),
            expected_epoch: 0,
            expected_active: None,
            resources: ResourcePlan {
                workers: BTreeMap::from([(
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 8,
                    },
                )]),
                hosts: BTreeMap::from([(
                    host,
                    HostMemory {
                        bytes: 8,
                        pinned_bytes,
                    },
                )]),
            },
            recipe: Some(NodeId::derive(name)),
            probes: Vec::new(),
            rates: None,
        };
        let a = make_plan("first group", first, 4);
        let mut b = make_plan("second group", second, 4);
        let version = |plan: &GroupPreparation| GroupVersion {
            group: plan.group,
            epoch: 1,
        };
        assert_eq!(
            state.prepare_peer_set(&[a.clone(), b.clone()], &[]),
            Err(GroupError::HostCapacityExceeded(host))
        );
        assert!(state.peer_group_view(version(&a)).is_none());
        assert!(state.peer_group_view(version(&b)).is_none());
        assert_eq!(state.hosts[&host].reserved, HostMemory::default());
        for device in [first, second] {
            assert_eq!(state.workers[&device].fence, 0);
            assert!(state.workers[&device].grant.is_none());
        }
        b.resources.hosts.get_mut(&host).unwrap().pinned_bytes = 2;
        assert_eq!(
            state.prepare_peer_set(&[a.clone(), b.clone()], &[(a.group, Some(9))]),
            Err(GroupError::GroupSetChanged {
                group: a.group,
                current: None
            })
        );
        let result = Ok(GroupOutcome::PreparedSet(vec![version(&a), version(&b)]));
        let operation = GroupOperation {
            id: OpId {
                origin: host,
                counter: 1,
            },
            action: GroupAction::PrepareSet {
                preparations: vec![a.clone(), b.clone()],
                guards: Vec::new(),
            },
        };
        assert_eq!(state.apply_peer_group(&operation), result);
        assert_eq!(state.apply_peer_group(&operation), result);
        assert_eq!(
            state.hosts[&host].reserved,
            HostMemory {
                bytes: 16,
                pinned_bytes: 6
            }
        );
        for plan in [&a, &b] {
            let view = state.peer_group_view(version(plan)).unwrap();
            assert_eq!(view.phase, GroupPhase::Preparing);
            assert_eq!(view.grants[0].fence, 1);
            assert_eq!(
                state.peer_group_recipe(version(plan)).unwrap().recipe,
                plan.recipe
            );
        }
    }

    #[test]
    fn abort_failed_set_preserves_published_epochs_and_holds_grants_until_release() {
        let mut state = ServingState::default();
        let host = NodeId::derive("shared host");
        let capacity = HostMemory {
            bytes: 40,
            pinned_bytes: 16,
        };
        let demand = HostMemory {
            bytes: 8,
            pinned_bytes: 4,
        };
        state.set_host_memory(host, capacity).unwrap();
        let preparations: Vec<_> = ["first", "second"]
            .into_iter()
            .enumerate()
            .map(|(ordinal, name)| {
                let device = DeviceId::new(host, ordinal as u32);
                state.register_worker(device, 1, 16).unwrap();
                GroupPreparation {
                    group: GroupId(NodeId::derive(name)),
                    expected_epoch: 0,
                    expected_active: None,
                    resources: ResourcePlan {
                        workers: BTreeMap::from([(
                            device,
                            WorkerDemand {
                                incarnation: 1,
                                device_bytes: 8,
                            },
                        )]),
                        hosts: BTreeMap::from([(host, demand)]),
                    },
                    recipe: None,
                    probes: vec![],
                    rates: None,
                }
            })
            .collect();
        let prepare = OpId {
            origin: host,
            counter: 1,
        };
        let op = GroupOperation {
            id: prepare,
            action: GroupAction::PrepareSet {
                preparations: preparations.clone(),
                guards: vec![],
            },
        };
        let Ok(GroupOutcome::PreparedSet(versions)) = state.apply_peer_group(&op) else {
            panic!("expected a prepared set");
        };
        let [first, second] = versions.as_slice() else {
            panic!("expected a pair")
        };
        let abort = GroupAction::AbortFailedSet {
            prepare,
            versions: versions.clone(),
        };
        assert_eq!(
            state.peer_group_transition(&abort),
            Err(GroupError::PreparationHealthy)
        );
        assert_eq!(
            state.latest_peer_preparation(second.group).unwrap().prepare,
            prepare
        );
        let first_grant = ready_single_worker(&mut state, *first);
        let second_grant = ready_single_worker(&mut state, *second);

        let mut previously_active = state.clone();
        previously_active.activate_peer_group(*first).unwrap();
        previously_active.report_peer_failure(first_grant).unwrap();
        previously_active.release_peer_worker(first_grant).unwrap();
        assert!(
            previously_active
                .latest_peer_preparation(second.group)
                .unwrap()
                .activated
        );
        assert_eq!(
            previously_active.peer_group_transition(&abort),
            Err(GroupError::PreparationActivated)
        );
        assert_eq!(
            previously_active.peer_group_view(*second).unwrap().phase,
            GroupPhase::Ready
        );

        state.report_peer_failure(first_grant).unwrap();
        state.release_peer_worker(first_grant).unwrap();
        state
            .prepare_peer_group(first.group, 1, &preparations[0].resources)
            .unwrap();
        let successor = GroupVersion {
            group: first.group,
            epoch: 2,
        };
        let successor_grant = state.peer_group_view(successor).unwrap().grants[0];
        assert_eq!(
            state.peer_group_transition(&GroupAction::AbortFailedSet {
                prepare,
                versions: vec![successor, *second],
            }),
            Err(GroupError::InvalidPlan)
        );
        for _ in 0..2 {
            assert_eq!(
                state.peer_group_transition(&abort),
                Ok(GroupOutcome::AbortedSet(versions.clone()))
            );
        }
        assert_eq!(
            state.peer_group_view(*second).unwrap().phase,
            GroupPhase::Revoked
        );
        assert_eq!(
            state.workers[&second_grant.device].grant,
            Some(second_grant)
        );
        assert_eq!(
            state.hosts[&host].reserved,
            HostMemory {
                bytes: 16,
                pinned_bytes: 8
            }
        );
        state.release_peer_worker(second_grant).unwrap();
        assert_eq!(state.hosts[&host].reserved, demand);
        assert_eq!(
            state.workers[&successor_grant.device].grant,
            Some(successor_grant)
        );
    }

    #[test]
    fn active_set_switch_checks_every_target_and_replays_the_same_observation() {
        let mut state = ServingState::default();
        let groups = ["aggregate", "prefill", "decode"].map(|name| GroupId(NodeId::derive(name)));
        let versions = groups.map(|group| GroupVersion { group, epoch: 1 });
        let [aggregate, prefill, decode] = versions;
        for version in versions {
            let host = version.group.0;
            let device = DeviceId::new(host, 0);
            state.set_host_memory(host, HostMemory::default()).unwrap();
            state.register_worker(device, 1, 16).unwrap();
            state
                .prepare_peer_group(
                    version.group,
                    0,
                    &ResourcePlan {
                        hosts: BTreeMap::from([(host, HostMemory::default())]),
                        workers: BTreeMap::from([(
                            device,
                            WorkerDemand {
                                incarnation: 1,
                                device_bytes: 16,
                            },
                        )]),
                    },
                )
                .unwrap();
        }
        let original = ready_single_worker(&mut state, aggregate);
        state.activate_peer_group(aggregate).unwrap();
        ready_single_worker(&mut state, prefill);
        let [a, p, d] = groups;
        let transition = GroupAction::SwitchActiveSet {
            transitions: vec![
                GroupTransition {
                    group: a,
                    expected: Some(1),
                    target: None,
                },
                GroupTransition {
                    group: p,
                    expected: None,
                    target: Some(1),
                },
                GroupTransition {
                    group: d,
                    expected: None,
                    target: Some(1),
                },
            ],
        };
        let read = GroupOperation {
            id: OpId {
                origin: a.0,
                counter: 1,
            },
            action: GroupAction::ObserveActiveSet {
                groups: groups.to_vec(),
            },
        };
        let old_set = Ok(GroupOutcome::ActiveSet(vec![
            (a, Some(1)),
            (p, None),
            (d, None),
        ]));
        assert_eq!(state.apply_peer_group(&read), old_set);
        assert_eq!(
            state.peer_group_transition(&transition),
            Err(GroupError::InvalidPhase(GroupPhase::Preparing))
        );
        assert_eq!(state.peer_group_transition(&read.action), old_set);

        let decode_grant = ready_single_worker(&mut state, decode);
        let new_set = Ok(GroupOutcome::ActiveSet(vec![
            (a, None),
            (p, Some(1)),
            (d, Some(1)),
        ]));
        assert_eq!(state.peer_group_transition(&transition), new_set);
        assert_eq!(state.peer_group_transition(&read.action), new_set);
        assert_eq!(state.apply_peer_group(&read), old_set);
        assert_eq!(
            state.peer_group_view(aggregate).unwrap().phase,
            GroupPhase::Draining
        );
        assert_eq!(state.peer_grant_status(original), GrantStatus::Held);

        state.report_peer_failure(decode_grant).unwrap();
        assert_eq!(
            state.peer_group_transition(&read.action),
            Ok(GroupOutcome::ActiveSet(vec![
                (a, None),
                (p, Some(1)),
                (d, None)
            ]))
        );
        assert_eq!(
            state.peer_group_transition(&transition),
            Err(GroupError::GroupSetChanged {
                group: a,
                current: None
            })
        );
    }

    #[test]
    fn exact_fault_promotes_only_the_armed_ready_successor() {
        let (mut state, active, standby, old) = prepared_successor(false);
        let arm = GroupAction::SetStandby {
            active,
            standby: Some(standby.epoch),
        };
        assert_eq!(
            state.peer_group_transition(&arm),
            Err(GroupError::InvalidPhase(GroupPhase::Preparing))
        );
        let successor = ready_single_worker(&mut state, standby);
        let fault = |grant| GroupAction::ReportFailure {
            grant,
            fault: GroupFault::Transport,
        };
        assert!(state.peer_group_transition(&arm).is_ok());
        assert_eq!(state.peer_standby(active.group), Some(standby));

        let mut manual = state.clone();
        manual
            .peer_group_transition(&GroupAction::Revoke(active))
            .unwrap();
        assert_eq!(
            manual.peer_group_view(standby).unwrap().phase,
            GroupPhase::Ready
        );
        assert_eq!(manual.peer_standby(active.group), None);

        let mut lost_standby = state.clone();
        lost_standby
            .peer_group_transition(&fault(successor))
            .unwrap();
        lost_standby.peer_group_transition(&fault(old)).unwrap();
        assert!(
            lost_standby
                .placement_inventory(active.group)
                .active
                .is_none()
        );

        assert_eq!(
            state.peer_group_transition(&fault(WorkerGrant {
                fence: old.fence + 1,
                ..old
            })),
            Err(GroupError::GrantMismatch)
        );
        let first = state.peer_group_transition(&fault(old));
        assert_eq!(
            first,
            Ok(GroupOutcome::Version {
                version: active,
                phase: GroupPhase::Revoked
            })
        );
        assert_eq!(
            state.peer_group_view(standby).unwrap().phase,
            GroupPhase::Active
        );
        assert_eq!(state.workers[&old.device].grant, Some(old));
        assert_eq!(state.peer_group_transition(&fault(old)), first);
        assert_eq!(
            state.peer_group_transition(&arm),
            Err(GroupError::ActiveEpochChanged {
                current: Some(standby.epoch)
            })
        );
    }

    #[test]
    fn host_loss_only_promotes_a_standby_outside_the_affected_groups() {
        for same_host in [false, true] {
            let (mut state, active, standby, old) = prepared_successor(same_host);
            ready_single_worker(&mut state, standby);
            state.set_peer_standby(active, Some(standby.epoch)).unwrap();
            let host = old.device.node;
            let pulse = OpId::new(host, 1);
            state.pulse_execution_host(host, pulse, [1, 2]).unwrap();
            let affected = state.execution_versions_on(host);
            state.expire_execution_host(host, pulse, &affected).unwrap();
            assert_eq!(
                state.peer_group_view(active).unwrap().phase,
                GroupPhase::Revoked
            );
            assert_eq!(
                state.peer_group_view(standby).unwrap().phase,
                if same_host {
                    GroupPhase::Revoked
                } else {
                    GroupPhase::Active
                }
            );
            assert_eq!(state.peer_standby(active.group), None);
        }
    }

    #[test]
    fn a_planned_reservation_checks_the_incumbent_and_filters_old_profiles() {
        let host = NodeId::derive("placement host");
        let [a, b] = [0, 1].map(|ordinal| DeviceId::new(host, ordinal));
        let group = GroupId(NodeId::derive("planned group"));
        let capacity = HostMemory {
            bytes: 32,
            pinned_bytes: 0,
        };
        let demand = WorkerDemand {
            incarnation: 1,
            device_bytes: 8,
        };
        let resources = |device| ResourcePlan {
            workers: BTreeMap::from([(device, demand)]),
            hosts: BTreeMap::from([(
                host,
                HostMemory {
                    bytes: 4,
                    pinned_bytes: 0,
                },
            )]),
        };
        let profile = DeviceProfile {
            id: b,
            name: "test".into(),
            arch: AcceleratorKind::Amd,
            memory_bytes: 32,
            flops: 800,
            compute_units: 2,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp64: false,
            rdma_capable: false,
            numa_node: None,
        };
        let mut state = ServingState::default();
        state.set_host_memory(host, capacity).unwrap();
        state.register_worker(a, 1, 16).unwrap();
        let advertise = |profile, incarnation| GroupAction::AdvertiseWorker {
            profile,
            incarnation,
            device_bytes: 16,
        };
        state
            .peer_group_transition(&advertise(profile.clone(), 1))
            .unwrap();
        let mut different = profile.clone();
        different.flops += 1;
        assert_eq!(
            state.peer_group_transition(&advertise(different, 1)),
            Err(GroupError::ProfileMismatch(b))
        );
        state.prepare_peer_group(group, 0, &resources(a)).unwrap();
        let version = GroupVersion { group, epoch: 1 };
        let grant = state.peer_group_view(version).unwrap().grants[0];
        state
            .publish_peer_endpoint(GroupMember {
                grant,
                endpoint: PeerEndpoint {
                    address: "127.0.0.1:1025".parse().unwrap(),
                    ticket: [1; 16],
                },
            })
            .unwrap();
        let digest = state
            .peer_group_view(version)
            .unwrap()
            .manifest
            .unwrap()
            .digest();
        state.ready_peer_group(grant, digest).unwrap();
        state.activate_peer_group(version).unwrap();
        assert_eq!(
            state.placement_inventory(group).active.unwrap().version,
            version
        );
        state
            .quiesce_peer_group(version, GroupPhase::Revoked)
            .unwrap();
        assert_eq!(
            state.peer_group_transition(&GroupAction::PreparePlanned {
                group,
                expected_epoch: 1,
                expected_active: Some(1),
                resources: resources(b),
            }),
            Err(GroupError::ActiveEpochChanged { current: None })
        );
        assert!(state.workers[&b].grant.is_none());
        state.register_worker(b, 2, 16).unwrap();
        let inventory = state.placement_inventory(group);
        assert!(
            inventory
                .workers
                .iter()
                .find(|worker| worker.device == b)
                .unwrap()
                .profile
                .is_none()
        );
        state.peer_group_transition(&advertise(profile, 2)).unwrap();
        assert!(
            state
                .placement_inventory(group)
                .workers
                .iter()
                .find(|worker| worker.device == b)
                .unwrap()
                .profile
                .is_some()
        );
    }

    #[test]
    fn probe_restart_fences_a_measured_reservation_and_old_publisher() {
        let hosts = [NodeId::derive("source"), NodeId::derive("target")];
        let devices = hosts.map(|host| DeviceId::new(host, 0));
        let endpoint = |host, session| ProbeEndpoint {
            host,
            session: NodeId::derive(session),
            address: ([127, 0, 0, 1], 9010).into(),
        };
        let mut state = ServingState::default();
        let capacity = HostMemory {
            bytes: 16,
            pinned_bytes: 0,
        };
        for device in devices {
            state.set_host_memory(device.node, capacity).unwrap();
            state.register_worker(device, 1, 16).unwrap();
        }
        let first = [
            endpoint(hosts[0], "first source"),
            endpoint(hosts[1], "first target"),
        ];
        let mut probes = first.map(|endpoint| {
            let operation = GroupOperation {
                id: OpId::new(endpoint.session, 1),
                action: GroupAction::PublishProbe {
                    endpoint,
                    expected_revision: 0,
                },
            };
            let Ok(GroupOutcome::Probe(Some(registration))) = state.apply_peer_group(&operation)
            else {
                panic!("expected probe registration");
            };
            assert_eq!(registration.revision, 1);
            assert_eq!(
                state.apply_peer_group(&operation),
                Ok(GroupOutcome::Probe(Some(registration)))
            );
            registration
        });
        let plan = |probes: Vec<_>| GroupAction::PrepareMeasured {
            group: GroupId(NodeId::derive("measured group")),
            expected_epoch: 0,
            expected_active: None,
            resources: ResourcePlan {
                workers: devices
                    .into_iter()
                    .map(|device| {
                        (
                            device,
                            WorkerDemand {
                                incarnation: 1,
                                device_bytes: 8,
                            },
                        )
                    })
                    .collect(),
                hosts: hosts.into_iter().map(|host| (host, capacity)).collect(),
            },
            probes,
        };
        let successor = endpoint(hosts[1], "second target");
        let Ok(GroupOutcome::Probe(Some(current))) = state.publish_link_probe(successor, 1) else {
            panic!("expected successor registration");
        };
        assert_eq!(current.revision, 2);
        let stale = Err(GroupError::StaleProbe {
            host: hosts[1],
            revision: 2,
        });
        assert_eq!(state.publish_link_probe(first[1], 1), stale);
        assert_eq!(state.peer_group_transition(&plan(probes.to_vec())), stale);
        assert!(
            devices
                .into_iter()
                .all(|device| state.workers[&device].grant.is_none())
        );
        probes[1] = current;
        assert!(state.peer_group_transition(&plan(probes.to_vec())).is_ok());
    }

    #[test]
    fn rate_changes_fence_the_candidate_and_incumbent_before_any_reservation() {
        let host = NodeId::derive("rate host");
        let calibration = NodeId::derive("test calibration");
        let [a, b] = [0, 1].map(|ordinal| DeviceId::new(host, ordinal));
        let group = GroupId(NodeId::derive("rated group"));
        let memory = HostMemory {
            bytes: 8,
            pinned_bytes: 4,
        };
        let plan = |device| ResourcePlan {
            workers: BTreeMap::from([(
                device,
                WorkerDemand {
                    incarnation: 1,
                    device_bytes: 8,
                },
            )]),
            hosts: BTreeMap::from([(host, memory)]),
        };
        let mut state = ServingState::default();
        state
            .set_host_memory(
                host,
                HostMemory {
                    bytes: 32,
                    pinned_bytes: 16,
                },
            )
            .unwrap();
        state.register_worker(a, 1, 16).unwrap();
        state.register_worker(b, 1, 16).unwrap();
        state.prepare_peer_group(group, 0, &plan(a)).unwrap();
        let active = GroupVersion { group, epoch: 1 };
        let grant = ready_single_worker(&mut state, active);
        state.activate_peer_group(active).unwrap();

        let update = |state: &mut ServingState, device, old, rate| {
            let Ok(GroupOutcome::ComputeRate(rate)) =
                state.publish_compute_rate(calibration, device, 1, old, NonZeroU64::new(rate))
            else {
                panic!("expected an applied rate");
            };
            rate
        };
        let mut rates = [update(&mut state, a, 0, 100), update(&mut state, b, 0, 200)];
        let action = |rates: Vec<_>| GroupAction::PrepareWithRates {
            group,
            expected_epoch: 1,
            expected_active: Some(1),
            resources: plan(b),
            probes: vec![],
            calibration,
            compared_active: true,
            rates,
        };
        assert_eq!(
            state.peer_group_transition(&action(vec![rates[1]])),
            Err(GroupError::InvalidPlan)
        );
        let publish = |grant, expected_revision| GroupAction::PublishGrantedComputeRate {
            grant,
            calibration,
            expected_revision,
            operations_per_second: NonZeroU64::new(300).unwrap(),
        };
        let wrong_fence = WorkerGrant {
            fence: grant.fence + 1,
            ..grant
        };
        assert_eq!(
            state.peer_group_transition(&publish(wrong_fence, 1)),
            Err(GroupError::GrantMismatch)
        );
        let Ok(GroupOutcome::ComputeRate(faster)) = state.peer_group_transition(&publish(grant, 1))
        else {
            panic!("expected a measurement from the active grant");
        };
        let stale = GroupOperation {
            id: OpId::new(host, 1),
            action: action(rates.to_vec()),
        };
        let stale_result = Err(GroupError::StaleComputeRate {
            device: a,
            revision: 2,
        });
        assert_eq!(state.apply_peer_group(&stale), stale_result);
        assert_eq!(state.apply_peer_group(&stale), stale_result);
        assert_eq!(state.peer_grant_status(grant), GrantStatus::Held);
        rates[0] = faster;

        let invalidated = update(&mut state, b, 1, 0);
        assert_eq!(
            state.peer_group_transition(&action(rates.to_vec())),
            Err(GroupError::StaleComputeRate {
                device: b,
                revision: 2
            })
        );
        assert_eq!(state.compute_rates(calibration), [faster, invalidated]);
        assert_eq!(state.placement_inventory(group).hosts[0].reserved, memory);
        assert!(state.workers[&b].grant.is_none());

        state.register_worker(b, 2, 16).unwrap();
        assert_eq!(state.compute_rates(calibration), [faster]);
        assert_eq!(
            state.publish_compute_rate(calibration, b, 1, 2, NonZeroU64::new(400)),
            Err(GroupError::StaleIncarnation(b))
        );
        let Ok(GroupOutcome::ComputeRate(current)) =
            state.publish_compute_rate(calibration, b, 2, 0, NonZeroU64::new(400))
        else {
            panic!("expected a rate for the new incarnation");
        };
        rates[1] = current;
        let mut prepare = action(rates.to_vec());
        if let GroupAction::PrepareWithRates { resources, .. } = &mut prepare {
            resources.workers.get_mut(&b).unwrap().incarnation = 2;
        }
        assert!(matches!(
            state.peer_group_transition(&prepare),
            Ok(GroupOutcome::Version {
                phase: GroupPhase::Preparing,
                ..
            })
        ));
        assert_eq!(state.peer_grant_status(grant), GrantStatus::Held);
        state
            .quiesce_peer_group(active, GroupPhase::Draining)
            .unwrap();
        assert_eq!(
            state.peer_group_transition(&publish(grant, faster.revision)),
            Err(GroupError::InvalidPhase(GroupPhase::Draining))
        );
        assert_eq!(state.compute_rate(a, calibration), Some(faster));
    }

    #[test]
    fn publication_is_immutable_and_release_cannot_clear_a_successor_grant() {
        let host = NodeId::derive("peer devices");
        let [a, b] = [0, 1].map(|ordinal| DeviceId::new(host, ordinal));
        let memory = |bytes| HostMemory {
            bytes,
            pinned_bytes: bytes / 2,
        };
        let demand = WorkerDemand {
            incarnation: 1,
            device_bytes: 16,
        };
        let plan = |devices: &[DeviceId]| ResourcePlan {
            workers: devices.iter().map(|&device| (device, demand)).collect(),
            hosts: BTreeMap::from([(host, memory(8))]),
        };
        let group = GroupId(NodeId::derive("peer group"));
        let old = GroupVersion { group, epoch: 1 };
        let new = GroupVersion { group, epoch: 2 };
        let mut state = ServingState::default();
        state.set_host_memory(host, memory(32)).unwrap();
        for device in [a, b] {
            state.register_worker(device, 1, 32).unwrap();
        }
        state.prepare_peer_group(group, 0, &plan(&[a, b])).unwrap();
        let legacy = GroupPlan {
            role: ServingRole::Aggregated,
            model: "integration".into(),
            tensor_parallel: 1,
            pipeline_parallel: 1,
            launch_digest: [0; 32],
            workers: plan(&[a]).workers,
            hosts: plan(&[a]).hosts,
        };
        assert_eq!(
            state.prepare(GroupId(host), 0, &legacy),
            Err(ServingError::WorkerBusy(a))
        );
        let [first, second]: [WorkerGrant; 2] = state
            .peer_group_view(old)
            .unwrap()
            .grants
            .try_into()
            .unwrap();
        assert_eq!(state.peer_staging_bytes(first), Some(2));
        assert_eq!(state.peer_staging_bytes(second), Some(2));
        assert_eq!(
            state.peer_staging_bytes(WorkerGrant {
                fence: first.fence + 1,
                ..first
            }),
            None
        );
        let member = |grant, port, ticket| GroupMember {
            grant,
            endpoint: PeerEndpoint {
                address: ([127, 0, 0, 1], port).into(),
                ticket: [ticket; 16],
            },
        };
        let left = member(first, 9010, 1);
        let right = member(second, 9011, 2);
        state.publish_peer_endpoint(left).unwrap();
        assert!(state.peer_group_view(old).unwrap().manifest.is_none());
        assert_eq!(
            state.publish_peer_endpoint(member(first, 9012, 1)),
            Err(GroupError::InvalidEndpoint)
        );
        assert_eq!(
            state.publish_peer_endpoint(member(second, 9010, 2)),
            Err(GroupError::InvalidEndpoint)
        );
        state.publish_peer_endpoint(right).unwrap();
        let manifest = state.peer_group_view(old).unwrap().manifest.unwrap();
        assert_eq!(manifest.members, [left, right]);
        let digest = manifest.digest();
        assert_eq!(
            state.ready_peer_group(first, [0; 32]),
            Err(GroupError::ManifestMismatch)
        );
        state.ready_peer_group(first, digest).unwrap();
        assert_eq!(
            state.activate_peer_group(old),
            Err(GroupError::InvalidPhase(GroupPhase::Preparing))
        );
        state.ready_peer_group(second, digest).unwrap();
        state.activate_peer_group(old).unwrap();
        assert_eq!(state.peer_grant_status(first), GrantStatus::Held);
        assert_eq!(
            state.peer_grant_status(WorkerGrant {
                fence: first.fence + 1,
                ..first
            }),
            GrantStatus::Ended
        );
        assert_eq!(
            state.peer_grant_status(WorkerGrant {
                version: new,
                ..first
            }),
            GrantStatus::Unknown
        );
        state.quiesce_peer_group(old, GroupPhase::Draining).unwrap();
        assert_eq!(state.peer_grant_status(first), GrantStatus::Held);
        assert_eq!(state.peer_staging_bytes(first), None);
        state.release_peer_worker(first).unwrap();
        assert_eq!(state.peer_grant_status(first), GrantStatus::Ended);
        assert_eq!(state.peer_grant_status(second), GrantStatus::Held);
        state.quiesce_peer_group(old, GroupPhase::Revoked).unwrap();
        assert_eq!(state.peer_grant_status(second), GrantStatus::Ended);
        assert!(state.peer_group_view(old).unwrap().manifest.is_none());
        assert_eq!(
            state.ready_peer_group(second, digest),
            Err(GroupError::InvalidPhase(GroupPhase::Revoked))
        );

        state.release_peer_worker(first).unwrap();
        assert_eq!(state.hosts[&host].reserved, memory(8));
        state.prepare_peer_group(group, 1, &plan(&[a])).unwrap();
        let successor = state.peer_group_view(new).unwrap().grants[0];
        assert!(successor.fence > first.fence);
        assert_eq!(state.peer_staging_bytes(successor), Some(4));
        assert_eq!(state.peer_staging_bytes(first), None);
        state.release_peer_worker(first).unwrap();
        assert_eq!(state.workers[&a].grant, Some(successor));
        assert_eq!(state.peer_grant_status(first), GrantStatus::Ended);
        assert_eq!(state.peer_grant_status(successor), GrantStatus::Held);
        assert_eq!(state.hosts[&host].reserved, memory(16));
        state.release_peer_worker(second).unwrap();
        assert_eq!(
            state.peer_group_view(old).unwrap().phase,
            GroupPhase::Retired
        );
        assert_eq!(state.hosts[&host].reserved, memory(8));
        assert_eq!(state.workers[&a].grant, Some(successor));
    }

    #[test]
    fn host_expiry_revokes_only_peer_groups_with_grants_on_that_host() {
        let hosts = [
            NodeId::derive("expiring peer"),
            NodeId::derive("independent peer"),
        ];
        let mut state = ServingState::default();
        let mut versions = Vec::new();
        for host in hosts {
            let device = DeviceId::new(host, 0);
            let capacity = HostMemory {
                bytes: 16,
                pinned_bytes: 0,
            };
            state.set_host_memory(host, capacity).unwrap();
            state.register_worker(device, 1, 32).unwrap();
            state
                .pulse_execution_host(host, OpId::new(host, 1), [1, 2])
                .unwrap();
            let resources = ResourcePlan {
                workers: BTreeMap::from([(
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 16,
                    },
                )]),
                hosts: BTreeMap::from([(host, capacity)]),
            };
            let group = GroupId(host);
            state.prepare_peer_group(group, 0, &resources).unwrap();
            versions.push(GroupVersion { group, epoch: 1 });
        }
        let affected = state.execution_versions_on(hosts[0]);
        assert_eq!(affected, BTreeSet::from([versions[0]]));
        state
            .expire_execution_host(hosts[0], OpId::new(hosts[0], 1), &affected)
            .unwrap();
        let view = state.peer_group_view(versions[0]).unwrap();
        assert_eq!(view.phase, GroupPhase::Revoked);
        assert!(view.released.is_empty());
        assert_eq!(
            state.peer_group_view(versions[1]).unwrap().phase,
            GroupPhase::Preparing
        );
    }
}
