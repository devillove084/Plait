use std::collections::BTreeSet;

use plait_core::serving::{
    ExecutionHostLiveness, GroupPhase, GroupVersion, ServingError, ServingOutcome, ServingResult,
    WorkerGrant,
};
use plait_core::{DeviceId, NodeId, OpId};

use super::ServingState;

impl ServingState {
    pub(crate) fn execution_hosts(
        &self,
    ) -> impl Iterator<Item = (NodeId, ExecutionHostLiveness)> + '_ {
        self.execution_hosts
            .iter()
            .map(|(&host, &liveness)| (host, liveness))
    }

    pub(crate) fn execution_host(&self, host: NodeId) -> Option<ExecutionHostLiveness> {
        self.execution_hosts.get(&host).copied()
    }

    pub(super) fn require_execution_host(&self, host: NodeId) -> Result<(), ServingError> {
        if self
            .execution_host(host)
            .is_some_and(|health| !health.online)
        {
            return Err(ServingError::ExecutionHostOffline(host));
        }
        Ok(())
    }

    pub(super) fn pulse_execution_host(
        &mut self,
        host: NodeId,
        pulse: OpId,
        supervisor_lock: [u64; 2],
    ) -> ServingResult {
        if !self.hosts.contains_key(&host) {
            return Err(ServingError::UnknownHost(host));
        }
        if self
            .execution_host(host)
            .is_some_and(|previous| previous.supervisor_lock != supervisor_lock)
        {
            return Err(ServingError::LaunchOwnerMismatch);
        }
        self.execution_hosts.insert(
            host,
            ExecutionHostLiveness {
                pulse,
                online: true,
                supervisor_lock,
            },
        );
        Ok(ServingOutcome::ExecutionHostOnline)
    }

    pub(crate) fn execution_versions_on(&self, host: NodeId) -> BTreeSet<GroupVersion> {
        self.versions_on(host, |grant| {
            self.resident_owner(grant).is_some() || self.host_has_launch(grant.version, host)
        })
    }

    pub(crate) fn resident_versions_on(&self, host: NodeId) -> BTreeSet<GroupVersion> {
        self.versions_on(host, |grant| self.resident_owner(grant).is_some())
    }

    fn versions_on(
        &self,
        host: NodeId,
        holds_launch: impl Fn(WorkerGrant) -> bool,
    ) -> BTreeSet<GroupVersion> {
        self.workers
            .range(DeviceId::new(host, 0)..=DeviceId::new(host, u32::MAX))
            .filter_map(|(_, worker)| {
                let grant = worker.grant?;
                if !holds_launch(grant) {
                    return None;
                }
                let group = self.group(grant.version)?;
                matches!(
                    group.phase,
                    GroupPhase::Preparing
                        | GroupPhase::Ready
                        | GroupPhase::Active
                        | GroupPhase::Draining
                )
                .then_some(grant.version)
            })
            .collect()
    }

    pub(super) fn expire_execution_host(
        &mut self,
        host: NodeId,
        pulse: OpId,
        groups: &BTreeSet<GroupVersion>,
    ) -> ServingResult {
        let old = self
            .execution_host(host)
            .ok_or(ServingError::StaleExecutionHost)?;
        if old.pulse != pulse || !old.online || &self.execution_versions_on(host) != groups {
            return Err(ServingError::StaleExecutionHost);
        }
        for &version in groups {
            self.fail_resident_version(version)
                .expect("affected groups checked");
        }
        self.execution_hosts.insert(
            host,
            ExecutionHostLiveness {
                online: false,
                ..old
            },
        );
        Ok(ServingOutcome::ExecutionHostOffline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use plait_core::serving::{
        GroupId, GroupPlan, HostMemory, ResidentOwner, ServingRole, WorkerDemand,
    };

    #[test]
    fn host_expiry_compares_the_pulse_and_keeps_device_reservations() {
        let mut state = ServingState::default();
        let host = NodeId::derive("pulse-host");
        let group = GroupId(NodeId::derive("pulse-group"));
        let device = DeviceId::new(host, 0);
        let lock = [1, 2];
        let memory = HostMemory {
            bytes: 16,
            pinned_bytes: 4,
        };
        let pulses = [1, 2, 3].map(|counter| OpId::new(host, counter));
        let plan = GroupPlan {
            role: ServingRole::Aggregated,
            model: "lease".into(),
            tensor_parallel: 1,
            pipeline_parallel: 1,
            launch_digest: [4; 32],
            workers: BTreeMap::from([(
                device,
                WorkerDemand {
                    incarnation: 1,
                    device_bytes: 16,
                },
            )]),
            hosts: BTreeMap::from([(host, memory)]),
        };
        state.set_host_memory(host, memory).unwrap();
        state.register_worker(device, 1, 32).unwrap();
        state.pulse_execution_host(host, pulses[0], lock).unwrap();
        state.prepare(group, 0, &plan).unwrap();
        let version = GroupVersion { group, epoch: 1 };
        let grant = state.group(version).unwrap().grants[&device];
        let owner = ResidentOwner {
            session: host,
            supervisor_lock: [1, 3],
            device_lock: [1, 4],
        };
        state
            .claim_resident(grant, owner, plan.launch_digest)
            .unwrap();
        state.worker_ready(grant, plan.launch_digest).unwrap();
        state.activate(version).unwrap();
        state.pulse_execution_host(host, pulses[1], lock).unwrap();

        let affected = state.execution_versions_on(host);
        assert_eq!(affected, BTreeSet::from([version]));
        assert_eq!(
            state.expire_execution_host(host, pulses[0], &affected),
            Err(ServingError::StaleExecutionHost)
        );
        assert_eq!(
            state.pulse_execution_host(host, pulses[2], [2, 3]),
            Err(ServingError::LaunchOwnerMismatch)
        );
        assert_eq!(
            state.expire_execution_host(host, pulses[1], &BTreeSet::new()),
            Err(ServingError::StaleExecutionHost)
        );
        state
            .expire_execution_host(host, pulses[1], &affected)
            .unwrap();
        assert!(!state.execution_host(host).unwrap().online);
        assert_eq!(state.group(version).unwrap().phase, GroupPhase::Revoked);
        assert_eq!(state.resources().workers[0].grant, Some(grant));
        assert_eq!(
            state.require_execution_host(host),
            Err(ServingError::ExecutionHostOffline(host))
        );
        assert_eq!(
            state.retire(version, &BTreeSet::from([device])),
            Err(ServingError::WorkersStillRunning)
        );
        state.pulse_execution_host(host, pulses[2], lock).unwrap();
        assert!(state.execution_host(host).unwrap().online);
        assert_eq!(state.group(version).unwrap().phase, GroupPhase::Revoked);
    }
}
