//! Generic execution groups use the same device fences and budgets as hosted groups.

use std::collections::{BTreeMap, BTreeSet};

use plait_core::group::{
    ActivePlacement, GroupAction, GroupError, GroupId, GroupManifest, GroupMember, GroupOperation,
    GroupOutcome, GroupPhase, GroupReceipt, GroupResult, GroupVersion, GroupView, PeerEndpoint,
    PlacementHost, PlacementInventory, PlacementWorker, ResourcePlan, WorkerGrant,
};
use plait_core::serving::{ExecutionDrainTarget, ServingError};
use plait_core::{DeviceId, DeviceProfile, OpId};

use super::{Reservations, ServingState};

#[derive(Clone, Default)]
pub(super) struct PeerGroups {
    history: BTreeMap<GroupId, History>,
    receipts: BTreeMap<OpId, GroupReceipt>,
    profiles: BTreeMap<DeviceId, (u64, DeviceProfile)>,
}

#[derive(Clone, Default)]
struct History {
    versions: BTreeMap<u64, Version>,
    active: Option<u64>,
}

#[derive(Clone)]
struct Version {
    version: GroupVersion,
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
}

impl ServingState {
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

    pub(crate) fn peer_group_receipt(&self, operation: OpId) -> Option<&GroupReceipt> {
        self.peer_groups.receipts.get(&operation)
    }

    pub(crate) fn apply_peer_group(&mut self, operation: &GroupOperation) -> GroupResult {
        if let Some(receipt) = self.peer_group_receipt(operation.id) {
            return if &receipt.operation == operation {
                receipt.result
            } else {
                Err(GroupError::OperationIdReused)
            };
        }
        let result = self.peer_group_transition(&operation.action);
        self.peer_groups.receipts.insert(
            operation.id,
            GroupReceipt {
                operation: operation.clone(),
                result,
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
            GroupAction::Observe { group, epoch } => {
                let epoch = epoch
                    .or_else(|| self.peer_groups.history.get(&group)?.active)
                    .ok_or(GroupError::NoActiveGroup(group))?;
                let version = GroupVersion { group, epoch };
                self.peer_groups
                    .get(version)
                    .map(Version::outcome)
                    .ok_or(GroupError::UnknownVersion(version))
            }
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
            } => {
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
        }
    }

    fn prepare_peer_group(
        &mut self,
        group: GroupId,
        expected_epoch: u64,
        resources: &ResourcePlan,
    ) -> GroupResult {
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
        let mut reservations = Reservations::default();
        let grants = self
            .reserve_resources(
                version,
                &resources.workers,
                &resources.hosts,
                &mut reservations,
            )
            .map_err(resource_error)?;
        self.install_host_reservations(reservations.hosts);
        self.install_grants(&grants);
        let record = Version {
            version,
            phase: GroupPhase::Preparing,
            resources: resources.clone(),
            grants,
            endpoints: BTreeMap::new(),
            manifest: None,
            ready: BTreeSet::new(),
            released: BTreeSet::new(),
        };
        let result = record.outcome();
        self.peer_groups
            .history
            .entry(group)
            .or_default()
            .versions
            .insert(version.epoch, record);
        Ok(result)
    }

    fn publish_peer_endpoint(&mut self, member: GroupMember) -> GroupResult {
        let grant = member.grant;
        let view = self.peer_groups.get_mut(grant.version)?;
        view.check_grant(grant)?;
        if !matches!(
            view.phase,
            GroupPhase::Preparing | GroupPhase::Ready | GroupPhase::Active
        ) {
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
        if !matches!(
            view.phase,
            GroupPhase::Preparing | GroupPhase::Ready | GroupPhase::Active
        ) {
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
        let (&current, view) = history.versions.last_key_value().expect("recorded version");
        if version.epoch != current {
            return Err(GroupError::StaleEpoch { current });
        }
        if view.phase != GroupPhase::Ready {
            return Err(GroupError::InvalidPhase(view.phase));
        }
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
    use plait_core::group::{HostMemory, WorkerDemand};
    use plait_core::serving::{GroupPlan, ServingRole};

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
        state.quiesce_peer_group(old, GroupPhase::Revoked).unwrap();
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
        state.release_peer_worker(first).unwrap();
        assert_eq!(state.workers[&a].grant, Some(successor));
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
