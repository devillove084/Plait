use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use plait_core::serving::{
    GroupId, GroupPhase, GroupPlan, GroupVersion, HostedActivation, HostedAddress,
    HostedAssignment, HostedManagedAssignment, HostedPlanning, HostedPortUse, HostedRecipe,
    ServingEndpoint, ServingError, ServingOutcome, ServingResult, ServingRole,
};
use plait_core::{DeviceId, NodeId};

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
    pub(crate) fn hosted_planning(&self) -> HostedPlanning {
        HostedPlanning {
            hosts: self.hosted_inventories.values().cloned().collect(),
            occupied: self
                .hosted_recipes
                .values()
                .map(HostedPortUse::from)
                .chain(self.deferred_pd_ports())
                .collect(),
        }
    }

    pub(super) fn prepare_hosted_deployment(
        &mut self,
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: &GroupPlan,
        activation: HostedActivation,
        recipe: &HostedRecipe,
        pooled: bool,
    ) -> ServingResult {
        recipe.validate(plan)?;
        let ports = HostedPortUse::from(recipe);
        if self.hosted_recipes.iter().any(|(version, other)| {
            (pooled || self.hosted_pool_versions.contains(version))
                && ports.conflicts(HostedPortUse::from(other))
        }) || self.deferred_pd_ports().any(|other| ports.conflicts(other))
        {
            return Err(ServingError::HostedPortsInUse(recipe.api_host));
        }
        if pooled {
            for host in &recipe.hosts {
                self.check_hosted_inventory(host, plan, &recipe.profile)?;
            }
        }
        if let Some(epoch) = expected_active
            && let Some(endpoint) = self
                .group(GroupVersion { group, epoch })
                .and_then(|view| view.endpoint.as_ref())
            && endpoint.api_model != recipe.api_model
        {
            return Err(ServingError::InvalidEndpoint);
        }
        let result =
            self.prepare_hosted_switch(group, expected_epoch, expected_active, plan, activation)?;
        let ServingOutcome::Group { version, .. } = result else {
            unreachable!("hosted preparation creates a group version");
        };
        self.hosted_recipes.insert(version, recipe.clone());
        if pooled {
            self.hosted_pool_versions.insert(version);
        }
        Ok(result)
    }

    pub(super) fn check_hosted_inventory(
        &self,
        address: &HostedAddress,
        plan: &GroupPlan,
        profile: &str,
    ) -> Result<(), ServingError> {
        let host = address.host;
        self.require_execution_host(host)?;
        let inventory = self
            .hosted_inventories
            .get(&host)
            .ok_or(ServingError::HostedInventoryChanged(host))?;
        if inventory.address != address.address
            || !inventory.profiles.contains(profile)
            || plan
                .workers
                .range(DeviceId::new(host, 0)..=DeviceId::new(host, u32::MAX))
                .any(|(device, demand)| {
                    inventory.devices.get(&device.ordinal) != Some(&demand.incarnation)
                })
        {
            return Err(ServingError::HostedInventoryChanged(host));
        }
        Ok(())
    }

    pub(crate) fn hosted_assignments(&self, host: NodeId) -> Vec<HostedAssignment> {
        self.hosted_managed_assignments(host)
            .into_iter()
            .filter_map(|found| found.pd.is_none().then_some(found.assignment))
            .collect()
    }

    pub(crate) fn hosted_managed_assignments(&self, host: NodeId) -> Vec<HostedManagedAssignment> {
        let versions: BTreeSet<_> = self
            .workers
            .range(DeviceId::new(host, 0)..=DeviceId::new(host, u32::MAX))
            .filter_map(|(_, worker)| worker.grant.map(|grant| grant.version))
            .collect();
        versions
            .into_iter()
            .filter_map(|version| {
                let view = self.group(version)?;
                let recipe = self.hosted_recipes.get(&version)?;
                matches!(
                    view.phase,
                    GroupPhase::Preparing
                        | GroupPhase::Ready
                        | GroupPhase::Active
                        | GroupPhase::Draining
                )
                .then(|| HostedManagedAssignment {
                    pd: self.hosted_pd_launches.get(&version).cloned(),
                    assignment: HostedAssignment {
                        version,
                        model: view.plan.model.clone(),
                        tensor_parallel: view.plan.tensor_parallel,
                        phase: view.phase,
                        holder: self
                            .host_launches
                            .get(&version)
                            .and_then(|launch| launch.owners.get(&host))
                            .copied(),
                        recipe: recipe.clone(),
                    },
                })
            })
            .collect()
    }

    pub(super) fn prepare_hosted_switch(
        &mut self,
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: &GroupPlan,
        activation: HostedActivation,
    ) -> ServingResult {
        if plan.role != ServingRole::Aggregated
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
        if self.host_launches.contains_key(&version)
            || self.hosted_switches.contains_key(&version)
            || self.hosted_pd_launches.contains_key(&version)
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
        if self.hosted_pool_versions.contains(&version) {
            let recipe = &self.hosted_recipes[&version];
            let address = recipe
                .hosts
                .iter()
                .find(|entry| entry.host == host)
                .expect("pooled recipe covers the planned hosts");
            self.check_hosted_inventory(address, &view.plan, &recipe.profile)?;
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
            if self.hosted_recipes.get(&version).is_some_and(|recipe| {
                recipe.api_host != host
                    || recipe.api_model != endpoint.api_model
                    || recipe
                        .hosts
                        .iter()
                        .find(|entry| entry.host == host)
                        .is_none_or(|entry| {
                            endpoint.http
                                != std::net::SocketAddr::new(entry.address, recipe.http_port)
                        })
            }) || launch.http_host.is_some_and(|previous| previous != host)
                || self
                    .hosted_switches
                    .get(&version)
                    .and_then(|switch| switch.api_model.as_ref())
                    .is_some_and(|model| *model != endpoint.api_model)
            {
                return Err(ServingError::InvalidEndpoint);
            }
            check_launch_endpoint(view, endpoint)?;
            if let Some(pd) = self.hosted_pd_launches.get(&version)
                && view.plan.role == ServingRole::Prefill
                && endpoint.prefill_kv != Some(pd.kv.store)
            {
                return Err(ServingError::InvalidEndpoint);
            }
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
    fn hosted_assignment_is_committed_with_the_grants_and_endpoint() {
        use plait_core::serving::{HostedAddress, HostedRecipe, ServingAction, ServingOperation};

        let mut state = ServingState::default();
        let group = GroupId(NodeId::derive("portable-plan"));
        let [api, follower, spare] = ["api", "follower", "unused"].map(NodeId::derive);
        let mut plan = plan(&mut state, 0, &[api, follower]);
        let recipe = HostedRecipe {
            profile: "local-engine".into(),
            api_host: api,
            hosts: vec![
                HostedAddress {
                    host: api,
                    address: "192.0.2.1".parse().unwrap(),
                },
                HostedAddress {
                    host: follower,
                    address: "192.0.2.2".parse().unwrap(),
                },
            ],
            master_port: 29000,
            http_port: 8080,
            backend_port: 8081,
            api_model: "alias".into(),
            max_model_len: 4096,
            gpu_memory_bps: 8000,
            prefix_caching: true,
        };
        let prepare = |id, plan| ServingOperation {
            id: OpId::new(api, id),
            action: ServingAction::PrepareHostedDeployment {
                group,
                expected_epoch: 0,
                expected_active: None,
                plan,
                activation: HostedActivation::OnReady,
                request_sha256: None,
                recipe: recipe.clone(),
            },
        };
        assert_eq!(
            state.apply(&prepare(1, plan.clone())),
            Err(ServingError::InvalidPlan)
        );
        assert!(state.hosted_assignments(api).is_empty());
        plan.launch_digest = recipe.launch_digest();
        let version = GroupVersion { group, epoch: 1 };
        assert_eq!(
            state.apply(&prepare(2, plan.clone())),
            Ok(group_outcome(version, GroupPhase::Preparing))
        );
        let assignment = &state.hosted_assignments(api)[0];
        assert_eq!(assignment.recipe, recipe);
        assert_eq!(assignment.holder, None);
        assert_eq!(assignment.tensor_parallel, 2);
        let managed = &state.hosted_managed_assignments(api)[0];
        assert_eq!(&managed.assignment, assignment);
        assert_eq!(managed.launch_digest(), recipe.launch_digest());
        assert!(state.hosted_assignments(spare).is_empty());
        for host in [api, follower] {
            state
                .claim_host_launch(version, host, host, plan.launch_digest)
                .unwrap();
        }
        assert_eq!(state.hosted_assignments(api)[0].holder, Some(api));
        let endpoint = ServingEndpoint {
            http: "192.0.2.1:8080".parse().unwrap(),
            ..endpoint()
        };
        assert_eq!(
            state.host_launch_ready(version, follower, follower, Some(&endpoint)),
            Err(ServingError::InvalidEndpoint)
        );
        state
            .host_launch_ready(version, follower, follower, None)
            .unwrap();
        assert_eq!(
            state.host_launch_ready(version, api, api, Some(&endpoint)),
            Ok(group_outcome(version, GroupPhase::Active))
        );
        state.quiesce(version, GroupPhase::Revoked).unwrap();
        assert!(state.hosted_assignments(api).is_empty());
    }

    #[test]
    fn pooled_preparation_rechecks_inventory_and_holds_ports_until_retirement() {
        use plait_core::serving::HostedInventory;

        let mut state = ServingState::default();
        let [api, peer] = ["pool-api", "pool-peer"].map(NodeId::derive);
        let [old, next] =
            ["first-placement", "second-placement"].map(|name| GroupId(NodeId::derive(name)));
        let mut first = plan(&mut state, 0, &[api, peer]);
        let mut second = plan(&mut state, 1, &[api, peer]);
        let recipe = HostedRecipe {
            profile: "engine".into(),
            api_host: api,
            hosts: [api, peer]
                .into_iter()
                .enumerate()
                .map(|(index, host)| HostedAddress {
                    host,
                    address: std::net::Ipv4Addr::new(192, 0, 2, index as u8 + 1).into(),
                })
                .collect(),
            master_port: 19100,
            http_port: 19101,
            backend_port: 19102,
            api_model: "alias".into(),
            max_model_len: 1024,
            gpu_memory_bps: 6000,
            prefix_caching: false,
        };
        let digest = recipe.launch_digest();
        first.launch_digest = digest;
        second.launch_digest = digest;
        let prepare = |state: &mut ServingState, group, plan| {
            state.prepare_hosted_deployment(
                group,
                0,
                None,
                plan,
                HostedActivation::OnReady,
                &recipe,
                true,
            )
        };
        assert_eq!(
            prepare(&mut state, old, &first),
            Err(ServingError::HostedInventoryChanged(api)),
        );
        assert!(
            state
                .resources()
                .workers
                .iter()
                .all(|worker| worker.grant.is_none())
        );
        let advertise = |state: &mut ServingState, address: &HostedAddress, counter| {
            state
                .advertise_hosted_inventory(
                    &HostedInventory {
                        host: address.host,
                        address: address.address,
                        profiles: BTreeSet::from([recipe.profile.clone()]),
                        devices: BTreeMap::from([(0, 1), (1, 1)]),
                    },
                    OpId::new(address.host, counter),
                    [7, 9],
                )
                .unwrap();
        };
        for address in &recipe.hosts {
            advertise(&mut state, address, 1);
        }
        let version = GroupVersion {
            group: old,
            epoch: 1,
        };
        assert_eq!(
            prepare(&mut state, old, &first),
            Ok(group_outcome(version, GroupPhase::Preparing)),
        );
        state
            .expire_execution_host(peer, OpId::new(peer, 1), &BTreeSet::from([version]))
            .unwrap();
        assert_eq!(state.group(version).unwrap().phase, GroupPhase::Revoked);
        assert_eq!(
            prepare(&mut state, next, &second),
            Err(ServingError::HostedPortsInUse(api)),
        );
        assert_eq!(
            state.hosted_planning().occupied,
            vec![HostedPortUse::from(&recipe)]
        );
        let workers = first.workers.keys().copied().collect();
        for grant in state.group(version).unwrap().grants.clone().into_values() {
            state.worker_stopped(grant).unwrap();
        }
        state.retire(version, &workers).unwrap();
        assert_eq!(
            prepare(&mut state, next, &second),
            Err(ServingError::ExecutionHostOffline(peer)),
        );
        advertise(&mut state, &recipe.hosts[1], 2);
        prepare(&mut state, next, &second).unwrap();
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
