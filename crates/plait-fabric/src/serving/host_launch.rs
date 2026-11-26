use std::collections::{BTreeMap, btree_map::Entry};

use plait_core::NodeId;
use plait_core::serving::{GroupPhase, GroupVersion, ServingEndpoint, ServingError, ServingResult};

use super::{
    ServingState, check_launch_endpoint, check_launch_owner, check_quiescing, group_outcome,
};

#[derive(Clone, Default)]
pub(super) struct HostLaunch {
    owners: BTreeMap<NodeId, NodeId>,
    http_host: Option<NodeId>,
}

impl ServingState {
    pub(super) fn require_unhosted(&self, version: GroupVersion) -> Result<(), ServingError> {
        if self.host_launches.contains_key(&version) {
            return Err(ServingError::LaunchOwnerMismatch);
        }
        Ok(())
    }

    pub(super) fn host_has_launch(&self, version: GroupVersion, host: NodeId) -> bool {
        self.host_launches
            .get(&version)
            .is_some_and(|launch| launch.owners.contains_key(&host))
    }

    pub(super) fn claim_host_launch(
        &mut self,
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
        launch_digest: [u8; 32],
    ) -> ServingResult {
        self.require_execution_host(host)?;
        let view = self
            .group(version)
            .ok_or(ServingError::UnknownVersion(version))?;
        check_launch_owner(view, None)?;
        if view.phase != GroupPhase::Preparing {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if !view.plan.hosts.contains_key(&host)
            || self.w4_switches.contains_key(&version)
            || self.resident.contains_key(&version)
            || self.decoder.managed.contains(&version)
            || (!view.ready.is_empty() && !self.host_launches.contains_key(&version))
        {
            return Err(ServingError::InvalidPlan);
        }
        if view.plan.launch_digest != launch_digest {
            return Err(ServingError::LaunchDigestMismatch);
        }
        match self
            .host_launches
            .entry(version)
            .or_default()
            .owners
            .entry(host)
        {
            Entry::Occupied(existing) if *existing.get() != holder => {
                return Err(ServingError::LaunchOwnerMismatch);
            }
            Entry::Vacant(entry) => {
                entry.insert(holder);
            }
            Entry::Occupied(_) => {}
        }
        Ok(group_outcome(version, GroupPhase::Preparing))
    }

    fn host_launch(
        &self,
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
    ) -> Result<&HostLaunch, ServingError> {
        self.host_launches
            .get(&version)
            .filter(|launch| launch.owners.get(&host) == Some(&holder))
            .ok_or(ServingError::LaunchOwnerMismatch)
    }

    pub(super) fn host_launch_ready(
        &mut self,
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
        endpoint: Option<&ServingEndpoint>,
    ) -> ServingResult {
        let launch = self.host_launch(version, host, holder)?;
        let view = self.group(version).expect("claimed host has a version");
        if !matches!(view.phase, GroupPhase::Preparing | GroupPhase::Ready) {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if let Some(endpoint) = endpoint {
            if launch.http_host.is_some_and(|previous| previous != host) {
                return Err(ServingError::InvalidEndpoint);
            }
            check_launch_endpoint(view, endpoint)?;
            self.host_launches
                .get_mut(&version)
                .expect("checked launch")
                .http_host = Some(host);
        }
        let view = self.group_mut(version)?;
        if let Some(endpoint) = endpoint {
            view.endpoint = Some(endpoint.clone());
        }
        view.ready.extend(
            view.grants
                .keys()
                .copied()
                .filter(|device| device.node == host),
        );
        if view.ready.len() == view.grants.len() && view.endpoint.is_some() {
            view.phase = GroupPhase::Ready;
        }
        Ok(group_outcome(version, view.phase))
    }

    pub(super) fn observe_host_launch(
        &self,
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
    ) -> ServingResult {
        self.host_launch(version, host, holder)?;
        let view = self.group(version).expect("claimed host has a version");
        Ok(group_outcome(version, view.phase))
    }

    pub(super) fn host_launch_stopped(
        &mut self,
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
    ) -> ServingResult {
        let http_host = self.host_launch(version, host, holder)?.http_host;
        let view = self.group_mut(version)?;
        check_quiescing(view)?;
        if view.phase == GroupPhase::Draining {
            let http_host = http_host.expect("active hosted version has an endpoint");
            if host != http_host
                && view
                    .grants
                    .keys()
                    .any(|device| device.node == http_host && !view.stopped.contains(device))
            {
                return Err(ServingError::WorkersStillRunning);
            }
        }
        view.stopped.extend(
            view.grants
                .keys()
                .copied()
                .filter(|device| device.node == host),
        );
        Ok(group_outcome(version, view.phase))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use plait_core::serving::{GroupId, GroupPlan, HostMemory, ServingRole, WorkerDemand};
    use plait_core::{DeviceId, OpId};

    use super::*;

    fn prepare(
        state: &mut ServingState,
        group: GroupId,
        previous: u64,
        hosts: &[NodeId],
    ) -> GroupVersion {
        let plan = GroupPlan {
            role: ServingRole::Aggregated,
            model: "test".into(),
            tensor_parallel: hosts.len() as u32,
            pipeline_parallel: 1,
            launch_digest: [previous as u8; 32],
            workers: hosts
                .iter()
                .map(|&host| {
                    let device = DeviceId::new(host, 0);
                    state.register_worker(device, 1, 32).unwrap();
                    (
                        device,
                        WorkerDemand {
                            incarnation: 1,
                            device_bytes: 16,
                        },
                    )
                })
                .collect(),
            hosts: hosts
                .iter()
                .map(|&host| {
                    state
                        .set_host_memory(
                            host,
                            HostMemory {
                                bytes: 64,
                                pinned_bytes: 16,
                            },
                        )
                        .unwrap();
                    (
                        host,
                        HostMemory {
                            bytes: 16,
                            pinned_bytes: 4,
                        },
                    )
                })
                .collect(),
        };
        state.prepare(group, previous, &plan).unwrap();
        GroupVersion {
            group,
            epoch: previous + 1,
        }
    }

    #[test]
    fn a_spare_group_replaces_the_active_version_and_old_followers_wait_for_http() {
        let mut state = ServingState::default();
        let group = GroupId(NodeId::derive("hosted"));
        let [first, follower, spare] = ["first", "follower", "spare"].map(NodeId::derive);
        let [owner, follower_owner, next_owner] =
            ["owner", "follower-owner", "next-owner"].map(NodeId::derive);
        let old = prepare(&mut state, group, 0, &[first, follower]);
        let endpoint = ServingEndpoint {
            api_model: "test".into(),
            http: "127.0.0.1:8100".parse().unwrap(),
            prefill_kv: None,
        };
        let old_devices = state.group(old).unwrap().grants.keys().copied().collect();
        let follower_grant = state.group(old).unwrap().grants[&DeviceId::new(follower, 0)];

        state.claim_host_launch(old, first, owner, [0; 32]).unwrap();
        assert_eq!(
            state.claim_host_launch(old, first, follower_owner, [0; 32]),
            Err(ServingError::LaunchOwnerMismatch)
        );
        assert_eq!(
            state.worker_ready(follower_grant, [0; 32]),
            Err(ServingError::LaunchOwnerMismatch)
        );
        assert_eq!(
            state.host_launch_ready(old, first, owner, Some(&endpoint)),
            Ok(group_outcome(old, GroupPhase::Preparing))
        );
        assert_eq!(
            state.activate(old),
            Err(ServingError::InvalidPhase(GroupPhase::Preparing))
        );
        state
            .claim_host_launch(old, follower, follower_owner, [0; 32])
            .unwrap();
        assert_eq!(
            state.host_launch_ready(old, follower, follower_owner, Some(&endpoint)),
            Err(ServingError::InvalidEndpoint)
        );
        state
            .host_launch_ready(old, follower, follower_owner, None)
            .unwrap();
        state.activate(old).unwrap();

        let next = prepare(&mut state, group, 1, &[spare]);
        assert_eq!(
            state.observe_host_launch(old, first, owner),
            Ok(group_outcome(old, GroupPhase::Active))
        );
        state
            .claim_host_launch(next, spare, next_owner, [1; 32])
            .unwrap();
        let next_endpoint = ServingEndpoint {
            http: "127.0.0.1:8200".parse().unwrap(),
            ..endpoint
        };
        state
            .host_launch_ready(next, spare, next_owner, Some(&next_endpoint))
            .unwrap();
        state.activate(next).unwrap();
        assert_eq!(state.groups[&group].active, Some(next.epoch));
        assert_eq!(state.group(old).unwrap().phase, GroupPhase::Draining);
        assert_eq!(
            state.host_launch_stopped(old, follower, follower_owner),
            Err(ServingError::WorkersStillRunning)
        );
        assert_eq!(
            state.worker_stopped(follower_grant),
            Err(ServingError::LaunchOwnerMismatch)
        );
        assert_eq!(
            state.retire(old, &old_devices),
            Err(ServingError::WorkersStillRunning)
        );
        state.host_launch_stopped(old, first, owner).unwrap();
        assert_eq!(
            state.retire(old, &old_devices),
            Err(ServingError::WorkersStillRunning)
        );
        state
            .host_launch_stopped(old, follower, follower_owner)
            .unwrap();
        state.retire(old, &old_devices).unwrap();
        assert_eq!(state.group(next).unwrap().phase, GroupPhase::Active);
    }

    #[test]
    fn a_host_expiry_revokes_the_group_and_unstarted_peers_can_be_released() {
        let mut state = ServingState::default();
        let group = GroupId(NodeId::derive("hosted-failure"));
        let host = NodeId::derive("host");
        let peer = NodeId::derive("unstarted-peer");
        let owner = NodeId::derive("holder");
        let pulse = OpId::new(host, 1);
        let version = prepare(&mut state, group, 0, &[host, peer]);
        state.pulse_execution_host(host, pulse, [2, 3]).unwrap();
        state
            .claim_host_launch(version, host, owner, [0; 32])
            .unwrap();
        let view = state.group(version).unwrap();
        let workers: BTreeSet<_> = view.grants.keys().copied().collect();
        let unstarted = view.grants[&DeviceId::new(peer, 0)];
        let affected = state.execution_versions_on(host);
        assert_eq!(affected, BTreeSet::from([version]));
        assert_eq!(
            state.expire_execution_host(host, pulse, &affected),
            Ok(plait_core::serving::ServingOutcome::ExecutionHostOffline)
        );
        assert_eq!(state.group(version).unwrap().phase, GroupPhase::Revoked);
        state.worker_stopped(unstarted).unwrap();
        assert_eq!(
            state.retire(version, &workers),
            Err(ServingError::WorkersStillRunning)
        );
        state.host_launch_stopped(version, host, owner).unwrap();
        state.retire(version, &workers).unwrap();
        assert!(state.execution_versions_on(host).is_empty());
        assert_eq!(
            state.observe_host_launch(version, host, owner),
            Err(ServingError::LaunchOwnerMismatch)
        );
    }
}
