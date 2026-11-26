use std::collections::{BTreeMap, btree_map::Entry};

use plait_core::NodeId;
use plait_core::serving::{
    GroupId, GroupPhase, GroupPlan, GroupVersion, HostedActivation, ServingEndpoint, ServingError,
    ServingOutcome, ServingResult, ServingRole,
};

use super::{
    ServingState, check_launch_endpoint, check_launch_owner, check_quiescing, group_outcome,
};

#[derive(Clone, Default)]
pub(super) struct HostLaunch {
    owners: BTreeMap<NodeId, NodeId>,
    http_host: Option<NodeId>,
}

#[derive(Clone)]
pub(super) struct HostedSwitch {
    incumbent: Option<u64>,
    activation: HostedActivation,
    revoked: bool,
    api_model: Option<String>,
}

impl ServingState {
    pub(super) fn prepare_hosted_switch(
        &mut self,
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: &GroupPlan,
        activation: HostedActivation,
    ) -> ServingResult {
        if plan.role != ServingRole::Aggregated
            || plan.pipeline_parallel != 1
            || (activation == HostedActivation::OnRevocation && expected_active.is_none())
        {
            return Err(ServingError::InvalidPlan);
        }
        let incumbent = self.groups.get(&group).and_then(|history| history.active);
        if incumbent != expected_active {
            return Err(ServingError::ActiveEpochChanged { current: incumbent });
        }
        let api_model = if let Some(epoch) = incumbent {
            let view = self
                .group(GroupVersion { group, epoch })
                .expect("active version");
            let endpoint = view.endpoint.as_ref().ok_or(ServingError::InvalidPlan)?;
            if view.plan.role != ServingRole::Aggregated || view.plan.model != plan.model {
                return Err(ServingError::InvalidPlan);
            }
            Some(endpoint.api_model.clone())
        } else {
            None
        };
        let result = self.prepare(group, expected_epoch, plan)?;
        let ServingOutcome::Group { version, .. } = result else {
            unreachable!("prepare creates a group version");
        };
        self.hosted_switches.insert(
            version,
            HostedSwitch {
                incumbent,
                activation,
                revoked: false,
                api_model,
            },
        );
        Ok(result)
    }

    pub(super) fn hosted_incumbent_revoked(&mut self, incumbent: GroupVersion) {
        let history = &self.groups[&incumbent.group];
        let (_, latest) = history.versions.last_key_value().expect("revoked version");
        let version = latest.version;
        if let Some(switch) = self.hosted_switches.get_mut(&version)
            && switch.incumbent == Some(incumbent.epoch)
        {
            switch.revoked = true;
            self.maybe_activate_hosted(version);
        }
    }

    fn maybe_activate_hosted(&mut self, version: GroupVersion) {
        let Some(switch) = self.hosted_switches.get(&version) else {
            return;
        };
        let current = self.groups[&version.group].active;
        let lost_incumbent = current.is_none() && switch.revoked;
        let eligible = match switch.activation {
            HostedActivation::OnReady => current == switch.incumbent || lost_incumbent,
            HostedActivation::OnRevocation => lost_incumbent,
        };
        if eligible
            && self.check_activation(version).is_ok()
            && self.groups[&version.group].versions[&version.epoch]
                .plan
                .hosts
                .keys()
                .all(|&host| self.require_execution_host(host).is_ok())
        {
            self.activate(version).expect("checked hosted activation");
        }
    }

    pub(super) fn require_unhosted(&self, version: GroupVersion) -> Result<(), ServingError> {
        if self.host_launches.contains_key(&version) || self.hosted_switches.contains_key(&version)
        {
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
        if !matches!(
            view.phase,
            GroupPhase::Preparing | GroupPhase::Ready | GroupPhase::Active | GroupPhase::Draining
        ) {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if let Some(endpoint) = endpoint {
            if launch.http_host.is_some_and(|previous| previous != host)
                || self
                    .hosted_switches
                    .get(&version)
                    .and_then(|switch| switch.api_model.as_ref())
                    .is_some_and(|model| *model != endpoint.api_model)
            {
                return Err(ServingError::InvalidEndpoint);
            }
            check_launch_endpoint(view, endpoint)?;
        }
        if matches!(view.phase, GroupPhase::Active | GroupPhase::Draining) {
            return Ok(group_outcome(version, view.phase));
        }
        if endpoint.is_some() {
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
            self.maybe_activate_hosted(version);
        }
        Ok(group_outcome(
            version,
            self.group(version).expect("known version").phase,
        ))
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

    use plait_core::serving::{HostMemory, WorkerDemand};
    use plait_core::{DeviceId, OpId};

    use super::*;

    fn plan(state: &mut ServingState, previous: u64, hosts: &[NodeId]) -> GroupPlan {
        GroupPlan {
            role: ServingRole::Aggregated,
            model: "test".into(),
            tensor_parallel: hosts.len() as u32,
            pipeline_parallel: 1,
            launch_digest: [previous as u8; 32],
            workers: hosts
                .iter()
                .map(|&host| {
                    let device = DeviceId::new(host, previous as u32);
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
        }
    }

    fn prepare(
        state: &mut ServingState,
        group: GroupId,
        previous: u64,
        hosts: &[NodeId],
    ) -> GroupVersion {
        let plan = plan(state, previous, hosts);
        state.prepare(group, previous, &plan).unwrap();
        GroupVersion {
            group,
            epoch: previous + 1,
        }
    }

    fn stage(
        state: &mut ServingState,
        group: GroupId,
        previous: u64,
        hosts: &[NodeId],
        activation: HostedActivation,
    ) -> GroupVersion {
        let plan = plan(state, previous, hosts);
        state
            .prepare_hosted_switch(
                group,
                previous,
                (previous != 0).then_some(previous),
                &plan,
                activation,
            )
            .unwrap();
        let version = GroupVersion {
            group,
            epoch: previous + 1,
        };
        for &host in hosts {
            state
                .claim_host_launch(version, host, host, plan.launch_digest)
                .unwrap();
        }
        version
    }

    fn endpoint() -> ServingEndpoint {
        ServingEndpoint {
            api_model: "alias".into(),
            http: "127.0.0.1:8100".parse().unwrap(),
            prefill_kv: None,
        }
    }

    #[test]
    fn auto_switch_uses_the_last_ready_report_and_keeps_the_api_alias() {
        let mut state = ServingState::default();
        let group = GroupId(NodeId::derive("rollout"));
        let [first, a, b] = ["initial", "replacement", "follower"].map(NodeId::derive);
        let old = stage(&mut state, group, 0, &[first], HostedActivation::OnReady);
        let endpoint = endpoint();
        assert_eq!(
            state.host_launch_ready(old, first, first, Some(&endpoint)),
            Ok(group_outcome(old, GroupPhase::Active))
        );

        let next = stage(&mut state, group, 1, &[a, b], HostedActivation::OnReady);
        let wrong_alias = ServingEndpoint {
            api_model: "other".into(),
            ..endpoint.clone()
        };
        assert_eq!(
            state.host_launch_ready(next, a, a, Some(&wrong_alias)),
            Err(ServingError::InvalidEndpoint)
        );
        assert_eq!(
            state.host_launch_ready(next, a, a, Some(&endpoint)),
            Ok(group_outcome(next, GroupPhase::Preparing))
        );
        assert_eq!(state.groups[&group].active, Some(old.epoch));
        assert_eq!(
            state.host_launch_ready(next, b, b, None),
            Ok(group_outcome(next, GroupPhase::Active))
        );
        assert_eq!(state.group(old).unwrap().phase, GroupPhase::Draining);
        assert_eq!(
            state.host_launch_ready(next, b, b, None),
            Ok(group_outcome(next, GroupPhase::Active))
        );
    }

    #[test]
    fn failures_and_ready_reports_can_arrive_in_either_order() {
        let group = GroupId(NodeId::derive("failover"));
        let [first, second] = ["incumbent", "standby"].map(NodeId::derive);
        let endpoint = endpoint();
        for revoked_first in [false, true] {
            let mut state = ServingState::default();
            let old = stage(&mut state, group, 0, &[first], HostedActivation::OnReady);
            state
                .host_launch_ready(old, first, first, Some(&endpoint))
                .unwrap();
            let next = stage(
                &mut state,
                group,
                1,
                &[second],
                HostedActivation::OnRevocation,
            );
            if revoked_first {
                state.quiesce(old, GroupPhase::Revoked).unwrap();
                assert_eq!(state.groups[&group].active, None);
            }
            let ready_phase = if revoked_first {
                GroupPhase::Active
            } else {
                GroupPhase::Ready
            };
            assert_eq!(
                state.host_launch_ready(next, second, second, Some(&endpoint)),
                Ok(group_outcome(next, ready_phase))
            );
            if !revoked_first {
                assert_eq!(state.groups[&group].active, Some(old.epoch));
                state.quiesce(old, GroupPhase::Revoked).unwrap();
            }
            assert_eq!(state.groups[&group].active, Some(next.epoch));
        }
    }

    #[test]
    fn a_drained_incumbent_does_not_arm_failover() {
        let mut state = ServingState::default();
        let group = GroupId(NodeId::derive("drained"));
        let [first, second] = ["incumbent", "standby"].map(NodeId::derive);
        let old = stage(&mut state, group, 0, &[first], HostedActivation::OnReady);
        let endpoint = endpoint();
        state
            .host_launch_ready(old, first, first, Some(&endpoint))
            .unwrap();
        let next = stage(
            &mut state,
            group,
            1,
            &[second],
            HostedActivation::OnRevocation,
        );
        state.quiesce(old, GroupPhase::Draining).unwrap();
        state.quiesce(old, GroupPhase::Revoked).unwrap();
        assert_eq!(
            state.host_launch_ready(next, second, second, Some(&endpoint)),
            Ok(group_outcome(next, GroupPhase::Ready))
        );
        assert_eq!(state.groups[&group].active, None);
    }

    #[test]
    fn host_loss_invalidates_a_standby_on_another_device_of_that_host() {
        let mut state = ServingState::default();
        let group = GroupId(NodeId::derive("shared-host-fault"));
        let host = NodeId::derive("shared-host");
        let pulse = OpId::new(host, 1);
        let old = stage(&mut state, group, 0, &[host], HostedActivation::OnReady);
        state.pulse_execution_host(host, pulse, [2, 3]).unwrap();
        let endpoint = endpoint();
        state
            .host_launch_ready(old, host, host, Some(&endpoint))
            .unwrap();
        let next = stage(
            &mut state,
            group,
            1,
            &[host],
            HostedActivation::OnRevocation,
        );
        state
            .host_launch_ready(next, host, host, Some(&endpoint))
            .unwrap();
        let affected = BTreeSet::from([old, next]);
        state.expire_execution_host(host, pulse, &affected).unwrap();
        assert_eq!(state.group(old).unwrap().phase, GroupPhase::Revoked);
        assert_eq!(state.group(next).unwrap().phase, GroupPhase::Revoked);
        assert_eq!(state.groups[&group].active, None);
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
