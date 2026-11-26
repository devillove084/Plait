use std::collections::BTreeSet;

use plait_core::OpId;
use plait_core::serving::{
    GroupPhase, GroupVersion, HostMemory, HostedPdIntent, HostedPdKv, HostedPdPlacement,
    HostedPdStage, HostedPdTransition, HostedPdTransitionPhase, HostedPdTransitionStatus,
    HostedPortUse, RouteSetId, ServingAction, ServingError, ServingOutcome, ServingResult,
};

use plait_core::NodeId;

use super::{RouteSet, ServingState, hosted_pd::route_versions};

impl ServingState {
    pub(super) fn prepare_hosted_pd_transition(
        &mut self,
        staged: OpId,
        model: &str,
        expected_revision: u64,
        kv: &HostedPdKv,
        bridge: &HostedPdStage,
        release: &BTreeSet<GroupVersion>,
        target: &HostedPdStage,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let current = self.routes.get(&key);
        let revision = current.map_or(0, |directory| directory.revision);
        if expected_revision != revision {
            return Err(ServingError::StaleDirectory { current: revision });
        }
        if let Some(staged) = self.pending_hosted_pd_id(key) {
            return Err(ServingError::HostedPdInProgress { staged });
        }
        let Some(current) =
            current.filter(|directory| directory.hosted && !directory.paths.is_empty())
        else {
            return Err(ServingError::InvalidDirectory);
        };
        if bridge.paths.is_empty()
            || target.paths.is_empty()
            || !route_versions(&current.paths)
                .difference(&route_versions(&bridge.paths))
                .copied()
                .eq(release.iter().copied())
        {
            return Err(ServingError::InvalidDirectory);
        }
        revision
            .checked_add(2)
            .ok_or(ServingError::SequenceExhausted)?;

        let transition = HostedPdTransition {
            staged,
            model: model.into(),
            expected_revision,
            kv: kv.clone(),
            bridge: bridge.paths.clone(),
            release: release.clone(),
            target: target.clone(),
        };
        self.preview_pd_transition(key, &transition, bridge)?;
        let outcome = self.prepare_hosted_pd(
            staged,
            model,
            revision,
            kv,
            &bridge.placements,
            &bridge.paths,
        )?;
        self.record_pd_transition(
            key,
            &transition,
            HostedPdTransitionPhase::PreparingBridge,
            None,
        );
        self.hosted_pd_transitions.insert(key, transition);
        Ok(outcome)
    }

    fn preview_pd_transition(
        &self,
        key: RouteSetId,
        transition: &HostedPdTransition,
        bridge: &HostedPdStage,
    ) -> Result<(), ServingError> {
        // Project the layout using the same validators as the real cutovers.
        // Receipts and in-flight requests stay with the live state.
        let mut preview = Self {
            groups: self.groups.clone(),
            hosts: self.hosts.clone(),
            workers: self.workers.clone(),
            routes: self.routes.clone(),
            hosted_recipes: self.hosted_recipes.clone(),
            hosted_pd_launches: self.hosted_pd_launches.clone(),
            hosted_pd_transitions: self.hosted_pd_transitions.clone(),
            hosted_pd_transition_status: self.hosted_pd_transition_status.clone(),
            hosted_inventories: self.hosted_inventories.clone(),
            execution_hosts: self.execution_hosts.clone(),
            ..Self::default()
        };
        let HostedPdTransition {
            staged,
            model,
            expected_revision,
            kv,
            release,
            target,
            ..
        } = transition;
        preview.prepare_hosted_pd(
            *staged,
            model,
            *expected_revision,
            kv,
            &bridge.placements,
            &bridge.paths,
        )?;
        let intent = preview
            .hosted_pd_intents
            .remove(&key)
            .expect("prepared bridge");
        for version in intent.prepared {
            preview.group_mut(version)?.phase = GroupPhase::Ready;
            preview.activate(version)?;
        }
        for &version in release {
            if preview
                .group(version)
                .is_some_and(|view| view.phase == GroupPhase::Active)
            {
                preview.quiesce(version, GroupPhase::Draining)?;
            }
            let view = preview.group_mut(version)?;
            if view.phase != GroupPhase::Retired {
                let stopped: BTreeSet<_> = view.grants.keys().copied().collect();
                view.stopped.clone_from(&stopped);
                preview.retire(version, &stopped)?;
            }
        }
        let bridge_revision = expected_revision + 1;
        preview.routes.insert(
            key,
            RouteSet {
                model: model.clone(),
                revision: bridge_revision,
                paths: bridge.paths.clone(),
                hosted: true,
            },
        );
        preview.prepare_hosted_pd(
            *staged,
            model,
            bridge_revision,
            kv,
            &target.placements,
            &target.paths,
        )?;
        Ok(())
    }

    pub(crate) fn hosted_pd_transition_status(
        &self,
        key: RouteSetId,
    ) -> Option<HostedPdTransitionStatus> {
        self.hosted_pd_transition_status.get(&key).copied()
    }

    fn deferred_pd_transitions(&self) -> impl Iterator<Item = &HostedPdTransition> {
        self.hosted_pd_transitions
            .iter()
            .filter_map(|(key, transition)| {
                (self.hosted_pd_transition_status[key].phase
                    != HostedPdTransitionPhase::PreparingTarget)
                    .then_some(transition)
            })
    }

    pub(super) fn deferred_pd_placements(
        &self,
    ) -> impl Iterator<Item = (&HostedPdTransition, &HostedPdPlacement)> {
        self.deferred_pd_transitions().flat_map(|transition| {
            transition
                .target
                .placements
                .iter()
                .map(move |placement| (transition, placement))
        })
    }

    pub(super) fn deferred_pd_ports(&self) -> impl Iterator<Item = HostedPortUse> {
        self.deferred_pd_placements()
            .map(|(_, placement)| HostedPortUse::from(&placement.recipe))
    }

    pub(super) fn deferred_pd_host_memory(&self, host: NodeId) -> HostMemory {
        // The retiring grants already consume their share of the budget. When
        // they release it, the same amount becomes part of the deferred hold.
        let sum = |total: HostMemory, memory: &HostMemory| HostMemory {
            bytes: total.bytes.saturating_add(memory.bytes),
            pinned_bytes: total.pinned_bytes.saturating_add(memory.pinned_bytes),
        };
        self.deferred_pd_transitions()
            .fold(HostMemory::default(), |total, transition| {
                let target = transition
                    .target
                    .placements
                    .iter()
                    .filter_map(|placement| placement.plan.hosts.get(&host))
                    .fold(HostMemory::default(), sum);
                let retiring = transition
                    .release
                    .iter()
                    .filter_map(|&version| {
                        let view = self.group(version)?;
                        if view.phase == GroupPhase::Retired {
                            return None;
                        }
                        view.plan.hosts.get(&host)
                    })
                    .fold(HostMemory::default(), sum);
                sum(
                    total,
                    &HostMemory {
                        bytes: target.bytes.saturating_sub(retiring.bytes),
                        pinned_bytes: target.pinned_bytes.saturating_sub(retiring.pinned_bytes),
                    },
                )
            })
    }

    pub(crate) fn hosted_pd_transition_work(&self) -> Vec<ServingAction> {
        self.hosted_pd_transitions
            .iter()
            .filter_map(|(&key, transition)| {
                if let Some(intent) = self.hosted_pd_intents.get(&key) {
                    self.hosted_pd_can_resolve(intent).then(|| {
                        ServingAction::ResolveHostedPdTransition {
                            transition: transition.clone(),
                            intent: intent.clone(),
                        }
                    })
                } else {
                    match self.check_pd_release(key, transition) {
                        Ok(false) => None,
                        Ok(true) | Err(_) => {
                            Some(ServingAction::AdvanceHostedPdTransition(transition.clone()))
                        }
                    }
                }
            })
            .collect()
    }

    pub(super) fn resolve_hosted_pd_transition(
        &mut self,
        transition: &HostedPdTransition,
        intent: &HostedPdIntent,
    ) -> ServingResult {
        let key = RouteSetId::for_model(&transition.model);
        if self.hosted_pd_transitions.get(&key) != Some(transition)
            || intent.model != transition.model
            || intent.staged != transition.staged
        {
            return Err(ServingError::InvalidDirectory);
        }
        let outcome = self.resolve_hosted_pd(intent, false)?;
        let phase = match outcome {
            ServingOutcome::HostedPdCancelled => HostedPdTransitionPhase::Cancelled,
            ServingOutcome::RoutesPublished { .. }
                if intent.expected_revision == transition.expected_revision =>
            {
                HostedPdTransitionPhase::WaitingForRelease
            }
            ServingOutcome::RoutesPublished { .. } => HostedPdTransitionPhase::Published,
            _ => unreachable!("hosted P/D resolution publishes or cancels"),
        };
        if matches!(
            phase,
            HostedPdTransitionPhase::Cancelled | HostedPdTransitionPhase::Published
        ) {
            self.hosted_pd_transitions.remove(&key);
        }
        self.record_pd_transition(key, transition, phase, None);
        Ok(outcome)
    }

    pub(super) fn advance_hosted_pd_transition(
        &mut self,
        transition: &HostedPdTransition,
    ) -> ServingResult {
        let key = RouteSetId::for_model(&transition.model);
        if self.hosted_pd_transitions.get(&key) != Some(transition)
            || self.hosted_pd_intents.contains_key(&key)
        {
            return Err(ServingError::InvalidDirectory);
        }
        match self.check_pd_release(key, transition) {
            Ok(false) => return Err(ServingError::WorkersStillRunning),
            Err(error) => {
                self.hosted_pd_transitions.remove(&key);
                self.record_pd_transition(
                    key,
                    transition,
                    HostedPdTransitionPhase::Cancelled,
                    Some(error),
                );
                return Ok(ServingOutcome::HostedPdCancelled);
            }
            Ok(true) => {}
        }

        // Consume the deferred claims and reserve real grants in this same command.
        let current = self
            .hosted_pd_transitions
            .remove(&key)
            .expect("checked transition");
        let stage = &current.target;
        match self.prepare_hosted_pd(
            current.staged,
            &current.model,
            current.expected_revision + 1,
            &current.kv,
            &stage.placements,
            &stage.paths,
        ) {
            Ok(outcome) => {
                self.record_pd_transition(
                    key,
                    &current,
                    HostedPdTransitionPhase::PreparingTarget,
                    None,
                );
                self.hosted_pd_transitions.insert(key, current);
                Ok(outcome)
            }
            Err(error) => {
                self.record_pd_transition(
                    key,
                    &current,
                    HostedPdTransitionPhase::Cancelled,
                    Some(error),
                );
                Ok(ServingOutcome::HostedPdCancelled)
            }
        }
    }

    fn check_pd_release(
        &self,
        key: RouteSetId,
        transition: &HostedPdTransition,
    ) -> Result<bool, ServingError> {
        let directory = self
            .routes
            .get(&key)
            .ok_or(ServingError::InvalidDirectory)?;
        if directory.revision != transition.expected_revision + 1
            || directory.paths != transition.bridge
            || directory
                .paths
                .iter()
                .any(|&path| self.resolve_route(&transition.model, path, None).is_none())
        {
            return Err(ServingError::InvalidDirectory);
        }
        for placement in &transition.target.placements {
            for host in &placement.recipe.hosts {
                self.check_hosted_inventory(host, &placement.plan, &placement.recipe.profile)?;
            }
        }
        Ok(transition.release.iter().all(|&version| {
            self.group(version)
                .is_some_and(|view| view.phase == GroupPhase::Retired)
        }))
    }

    fn record_pd_transition(
        &mut self,
        key: RouteSetId,
        transition: &HostedPdTransition,
        phase: HostedPdTransitionPhase,
        error: Option<ServingError>,
    ) {
        self.hosted_pd_transition_status.insert(
            key,
            HostedPdTransitionStatus {
                staged: transition.staged,
                revision: self
                    .routes
                    .get(&key)
                    .map_or(0, |directory| directory.revision),
                phase,
                error,
            },
        );
    }
}
