use plait_core::serving::{
    ExecutionDrainStatus, ExecutionDrainTarget, HostedEvacuationPlanning, RouteSetId, ServingError,
    ServingOutcome, ServingResult,
};
use plait_core::{DeviceId, OpId};

use super::ServingState;

impl ServingState {
    pub(super) fn execution_drain_for(&self, device: DeviceId) -> Option<ExecutionDrainTarget> {
        [
            ExecutionDrainTarget::Host(device.node),
            ExecutionDrainTarget::Device(device),
        ]
        .into_iter()
        .find(|target| self.execution_drains.contains_key(target))
    }

    pub(super) fn begin_execution_drain(
        &mut self,
        target: ExecutionDrainTarget,
        intent: OpId,
    ) -> ServingResult {
        match target {
            ExecutionDrainTarget::Host(host) if !self.hosts.contains_key(&host) => {
                return Err(ServingError::UnknownHost(host));
            }
            ExecutionDrainTarget::Device(device) if !self.workers.contains_key(&device) => {
                return Err(ServingError::UnknownWorker(device));
            }
            _ => {}
        }
        let drain = *self.execution_drains.entry(target).or_insert(intent);
        Ok(ServingOutcome::ExecutionDrainBegun { drain })
    }

    pub(crate) fn execution_drain(&self, target: ExecutionDrainTarget) -> ExecutionDrainStatus {
        ExecutionDrainStatus {
            target,
            drain: self.execution_drains.get(&target).copied(),
            holders: self
                .workers
                .values()
                .filter_map(|worker| worker.grant)
                .filter(|grant| target.covers(grant.device))
                .map(|grant| grant.version)
                .collect(),
            deferred: self
                .deferred_pd_placements()
                .filter(|(_, placement)| {
                    placement
                        .plan
                        .workers
                        .keys()
                        .any(|&device| target.covers(device))
                })
                .map(|(transition, _)| transition.staged)
                .collect(),
        }
    }

    pub(super) fn confirm_execution_drain(
        &self,
        target: ExecutionDrainTarget,
        drain: OpId,
    ) -> ServingResult {
        let status = self.execution_drain(target);
        if status.drain != Some(drain) {
            return Err(ServingError::StaleExecutionDrain);
        }
        if !status.holders.is_empty() || !status.deferred.is_empty() {
            return Err(ServingError::ExecutionTargetInUse(target));
        }
        Ok(ServingOutcome::ExecutionDrainConfirmed { drain })
    }

    pub(super) fn resume_execution_target(
        &mut self,
        target: ExecutionDrainTarget,
        drain: OpId,
    ) -> ServingResult {
        if self.execution_drains.get(&target) != Some(&drain) {
            return Err(ServingError::StaleExecutionDrain);
        }
        self.execution_drains.remove(&target);
        Ok(ServingOutcome::ExecutionTargetResumed)
    }

    pub(crate) fn hosted_evacuation_planning(&self, key: RouteSetId) -> HostedEvacuationPlanning {
        let current = self.hosted_serving_planning(key);
        let mut drains = std::collections::BTreeMap::new();
        let withdrawing = current
            .split
            .resources
            .workers
            .iter()
            .filter_map(|worker| {
                let grant = worker.grant?;
                let target = self.execution_drain_for(grant.device)?;
                if !current.assignments.contains_key(&grant.version) {
                    return None;
                }
                drains.insert(target, self.execution_drains[&target]);
                Some(grant.version)
            })
            .collect();
        let releases = self.hosted_pd_releases(key, &current.split);
        HostedEvacuationPlanning {
            current,
            withdrawing,
            drains,
            releases,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use plait_core::NodeId;
    use plait_core::serving::{
        GroupId, GroupPhase, GroupPlan, HostMemory, HostedInventory, ServingRole, WorkerDemand,
    };

    use super::*;

    #[test]
    fn a_drain_preserves_live_work_blocks_new_grants_and_fences_resume() {
        let mut state = ServingState::default();
        let host = NodeId::derive("maintenance-host");
        let [old, peer] = [0, 1].map(|ordinal| DeviceId::new(host, ordinal));
        let memory = HostMemory {
            bytes: 64,
            pinned_bytes: 8,
        };
        state.set_host_memory(host, memory).unwrap();
        for device in [old, peer] {
            state.register_worker(device, 1, 16).unwrap();
        }
        let inventory = HostedInventory {
            host,
            address: "192.0.2.1".parse().unwrap(),
            profiles: BTreeSet::from(["engine".into()]),
            devices: BTreeMap::from([(0, 1), (1, 1)]),
        };
        let op = |i| OpId::new(host, i);
        state
            .advertise_hosted_inventory(&inventory, op(1), [1, 2])
            .unwrap();
        let plan = |device| GroupPlan {
            role: ServingRole::Aggregated,
            model: "weights".into(),
            tensor_parallel: 1,
            pipeline_parallel: 1,
            launch_digest: [0; 32],
            hosts: BTreeMap::from([(
                host,
                HostMemory {
                    bytes: 8,
                    pinned_bytes: 1,
                },
            )]),
            workers: BTreeMap::from([(
                device,
                WorkerDemand {
                    incarnation: 1,
                    device_bytes: 8,
                },
            )]),
        };
        let old_group = GroupId(NodeId::derive("old"));
        let other = GroupId(NodeId::derive("other"));
        let ServingOutcome::Group { version, .. } =
            state.prepare(old_group, 0, &plan(old)).unwrap()
        else {
            unreachable!()
        };
        let grant = state.group(version).unwrap().grants[&old];
        state.worker_ready(grant, [0; 32]).unwrap();
        state.activate(version).unwrap();

        let target = ExecutionDrainTarget::Device(old);
        assert_eq!(
            state.begin_execution_drain(target, op(2)),
            Ok(ServingOutcome::ExecutionDrainBegun { drain: op(2) })
        );
        assert_eq!(
            state.begin_execution_drain(target, op(3)),
            Ok(ServingOutcome::ExecutionDrainBegun { drain: op(2) })
        );
        assert_eq!(state.group(version).unwrap().phase, GroupPhase::Active);
        assert_eq!(
            state.execution_drain(target).holders,
            BTreeSet::from([version])
        );
        assert_eq!(
            state.confirm_execution_drain(target, op(2)),
            Err(ServingError::ExecutionTargetInUse(target))
        );
        state
            .advertise_hosted_inventory(&inventory, op(4), [1, 2])
            .unwrap();
        assert_eq!(
            state.hosted_planning().hosts[0].devices,
            BTreeMap::from([(1, 1)])
        );
        assert!(state.prepare(other, 0, &plan(peer)).is_ok());
        assert_eq!(
            state.prepare(old_group, 1, &plan(old)),
            Err(ServingError::ExecutionTargetDraining(target))
        );

        state.quiesce(version, GroupPhase::Draining).unwrap();
        state.worker_stopped(grant).unwrap();
        assert_eq!(
            state.confirm_execution_drain(target, op(2)),
            Err(ServingError::ExecutionTargetInUse(target))
        );
        state.retire(version, &BTreeSet::from([old])).unwrap();
        assert_eq!(
            state.confirm_execution_drain(target, op(2)),
            Ok(ServingOutcome::ExecutionDrainConfirmed { drain: op(2) })
        );
        assert!(
            state
                .resources()
                .workers
                .iter()
                .all(|worker| worker.device != old)
        );
        assert_eq!(
            state.resume_execution_target(target, op(3)),
            Err(ServingError::StaleExecutionDrain)
        );
        assert_eq!(
            state.resume_execution_target(target, op(2)),
            Ok(ServingOutcome::ExecutionTargetResumed)
        );
        state.begin_execution_drain(target, op(5)).unwrap();
        assert_eq!(
            state.confirm_execution_drain(target, op(2)),
            Err(ServingError::StaleExecutionDrain)
        );
        let stale = plait_core::serving::ServingOperation {
            id: op(7),
            action: plait_core::serving::ServingAction::PrepareHostedEvacuation {
                model: "weights".into(),
                expected_revision: 0,
                plan: plait_core::serving::HostedServingPlan {
                    kv: None,
                    placements: Vec::new(),
                    routes: Vec::new(),
                },
                drains: BTreeMap::from([(target, op(2))]),
            },
        };
        assert_eq!(state.apply(&stale), Err(ServingError::StaleExecutionDrain));

        let all = ExecutionDrainTarget::Host(host);
        state.begin_execution_drain(all, op(6)).unwrap();
        state.resume_execution_target(target, op(5)).unwrap();
        assert!(state.hosted_planning().hosts[0].devices.is_empty());
        assert_eq!(
            state.prepare(old_group, 1, &plan(old)),
            Err(ServingError::ExecutionTargetDraining(all))
        );
    }
}
