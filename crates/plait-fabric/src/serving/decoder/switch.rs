use std::collections::{BTreeMap, BTreeSet};

use plait_core::OpId;
use plait_core::serving::{
    DecoderPreparation, DecoderRouteWeight, DecoderRouteWork, DecoderStandby, DecoderSwitch,
    GroupPhase, GroupVersion, GroupView, PdRoute, RouteSetId, ServingError, ServingOutcome,
    ServingResult,
};

use super::routing::decoder_weights;
use super::{RouteSet, ServingState};
use crate::serving::Reservations;

#[derive(Clone)]
pub(super) struct StagedSwitch {
    intent: DecoderSwitch,
    weights: BTreeMap<PdRoute, u32>,
}

struct Commit {
    directory: RouteSet,
    activate: Vec<GroupVersion>,
    drain: BTreeSet<GroupVersion>,
}

impl ServingState {
    pub(in crate::serving) fn supports_decoder_pool(
        &self,
        model: &str,
        multiple_paths: bool,
    ) -> bool {
        self.decoder
            .routes
            .get(&RouteSetId::for_model(model))
            .is_none_or(|directory| {
                (multiple_paths || directory.paths.len() <= 1) && directory.standbys.is_empty()
            })
    }

    pub(in crate::serving) fn stage_decoder_switch(
        &mut self,
        model: &str,
        revision: u64,
        paths: &[PdRoute],
        standbys: &[DecoderStandby],
        staged: OpId,
    ) -> ServingResult {
        let intent =
            self.plan_decoder_switch(model, revision, paths, standbys, staged, &BTreeMap::new())?;
        self.decoder.switches.insert(
            RouteSetId::for_model(model),
            StagedSwitch {
                intent,
                weights: BTreeMap::new(),
            },
        );
        Ok(ServingOutcome::DecoderSwitchStaged { revision })
    }

    pub(in crate::serving) fn prepare_decoder_switch(
        &mut self,
        model: &str,
        revision: u64,
        prepare: &[DecoderPreparation],
        paths: &[PdRoute],
        standbys: &[DecoderStandby],
        staged: OpId,
        weights: Option<&[DecoderRouteWeight]>,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        if weights.is_some()
            && let Some(pending) = self.decoder.switches.get(&key)
            && pending.intent.revision == revision
            && self
                .decoder
                .routes
                .get(&key)
                .map_or(0, |directory| directory.revision)
                == revision
        {
            return Err(ServingError::DecoderSwitchInProgress {
                staged: pending.intent.staged,
            });
        }
        let weights = weights
            .map(|weights| decoder_weights(paths, weights, true))
            .transpose()?
            .unwrap_or_default();
        let mut reservations = Reservations::default();
        let mut groups = BTreeSet::new();
        let mut prepared = BTreeMap::new();
        let recipients: BTreeSet<_> = paths.iter().flat_map(versions).collect();
        let needed: BTreeSet<_> = recipients
            .iter()
            .copied()
            .chain(
                standbys
                    .iter()
                    .flat_map(|rule| rule.alternatives.iter().flat_map(versions)),
            )
            .collect();
        for candidate in prepare {
            if !groups.insert(candidate.group) || candidate.plan.model != model {
                return Err(ServingError::InvalidPlan);
            }
            let active = self
                .groups
                .get(&candidate.group)
                .and_then(|history| history.active);
            if active != candidate.expected_active {
                return Err(ServingError::ActiveEpochChanged { current: active });
            }
            let view = self.reserve_group(
                candidate.group,
                candidate.expected_epoch,
                &candidate.plan,
                &mut reservations,
            )?;
            if !needed.contains(&view.version) {
                return Err(ServingError::InvalidPlan);
            }
            prepared.insert(view.version, view);
        }
        let intent =
            self.plan_decoder_switch(model, revision, paths, standbys, staged, &prepared)?;
        let prev: BTreeSet<_> = intent.previous.iter().flat_map(versions).collect();
        for version in recipients {
            if let Some(epoch) = self
                .groups
                .get(&version.group)
                .and_then(|history| history.active)
                && epoch != version.epoch
            {
                let active = GroupVersion {
                    group: version.group,
                    epoch,
                };
                if !prev.contains(&active) || needed.contains(&active) {
                    return Err(ServingError::ActiveEpochChanged {
                        current: Some(epoch),
                    });
                }
            }
        }
        self.install_groups(prepared.into_values(), reservations);
        self.decoder
            .switches
            .insert(key, StagedSwitch { intent, weights });
        Ok(ServingOutcome::DecoderSwitchStaged { revision })
    }

    fn plan_decoder_switch(
        &self,
        model: &str,
        revision: u64,
        paths: &[PdRoute],
        standbys: &[DecoderStandby],
        staged: OpId,
        prepared: &BTreeMap<GroupVersion, GroupView>,
    ) -> Result<DecoderSwitch, ServingError> {
        let key = RouteSetId::for_model(model);
        let current = self.decoder.routes.get(&key);
        let current_revision = current.map_or(0, |directory| directory.revision);
        if revision != current_revision {
            return Err(ServingError::StaleDirectory {
                current: current_revision,
            });
        }
        if model.is_empty()
            || paths.is_empty()
            || current.is_some_and(|directory| directory.model != model)
            || paths.iter().copied().collect::<BTreeSet<_>>().len() != paths.len()
        {
            return Err(ServingError::InvalidDirectory);
        }
        let mut candidate_groups = BTreeMap::new();
        let valid = |path| {
            Self::valid_decoder_groups(model, path, |version| {
                prepared.get(&version).or_else(|| self.group(version))
            })
        };
        for &path in paths {
            if !valid(path)
                || versions(&path).any(|version| {
                    candidate_groups
                        .insert(version.group, version)
                        .is_some_and(|old| old != version)
                })
            {
                return Err(ServingError::InvalidRoute(path));
            }
        }
        let candidates = paths.iter().map(|&path| (path, [0; 32])).collect();
        Self::decoder_standbys_using(&candidates, standbys, valid)?;
        Ok(DecoderSwitch {
            staged,
            revision,
            previous: current.map_or_else(Vec::new, |directory| directory.paths.clone()),
            paths: paths.to_vec(),
            standbys: standbys.to_vec(),
        })
    }

    pub(crate) fn decoder_switch(&self, id: RouteSetId) -> Option<DecoderSwitch> {
        self.decoder
            .switches
            .get(&id)
            .map(|pending| pending.intent.clone())
    }

    pub(crate) fn decoder_route_work(&self, id: RouteSetId) -> Option<DecoderRouteWork> {
        if let Some(failover) = self.decoder_failover(id) {
            return Some(DecoderRouteWork::Failover(failover));
        }
        let pending = self.decoder.switches.get(&id)?;
        let model = &self
            .group(pending.intent.paths.first()?.prefill)?
            .plan
            .model;
        self.validate_decoder_switch(model, pending).ok()?;
        Some(DecoderRouteWork::Switch(pending.intent.clone()))
    }

    pub(in crate::serving) fn cancel_decoder_switch(
        &mut self,
        model: &str,
        staged: OpId,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        if self
            .decoder
            .switches
            .get(&key)
            .is_none_or(|pending| pending.intent.staged != staged)
        {
            return Err(ServingError::InvalidDirectory);
        }
        self.decoder.switches.remove(&key);
        Ok(ServingOutcome::DecoderSwitchCancelled)
    }

    pub(in crate::serving) fn commit_decoder_switch(
        &mut self,
        model: &str,
        intent: &DecoderSwitch,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let pending = self
            .decoder
            .switches
            .get(&key)
            .filter(|pending| &pending.intent == intent)
            .ok_or(ServingError::InvalidDirectory)?;
        let Commit {
            directory,
            activate,
            drain,
        } = self.validate_decoder_switch(model, pending)?;
        let revision = directory.revision;
        for version in activate {
            self.activate(version)
                .expect("validated independent decoder activation");
        }
        self.decoder.routes.insert(key, directory);
        for version in drain {
            if self
                .group(version)
                .is_some_and(|group| group.phase == GroupPhase::Active)
            {
                self.quiesce(version, GroupPhase::Draining)
                    .expect("validated active decoder drain");
            }
        }
        self.decoder.switches.remove(&key);
        Ok(ServingOutcome::RoutesPublished { revision })
    }

    fn validate_decoder_switch(
        &self,
        model: &str,
        pending: &StagedSwitch,
    ) -> Result<Commit, ServingError> {
        let intent = &pending.intent;
        let current = self.decoder.routes.get(&RouteSetId::for_model(model));
        let current_revision = current.map_or(0, |directory| directory.revision);
        if current_revision != intent.revision {
            return Err(ServingError::StaleDirectory {
                current: current_revision,
            });
        }
        let revision = current_revision
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?;
        if current.map_or(&[][..], |directory| directory.paths.as_slice())
            != intent.previous.as_slice()
        {
            return Err(ServingError::InvalidDirectory);
        }
        let recipients: BTreeSet<_> = intent.paths.iter().flat_map(versions).collect();
        let retained: BTreeSet<_> = intent
            .standbys
            .iter()
            .flat_map(|rule| rule.alternatives.iter().flat_map(versions))
            .chain(recipients.iter().copied())
            .collect();
        let drain: BTreeSet<_> = intent
            .previous
            .iter()
            .flat_map(versions)
            .filter(|version| !retained.contains(version))
            .collect();
        let mut activate = Vec::new();
        for new in recipients {
            let phase = self
                .group(new)
                .ok_or(ServingError::UnknownVersion(new))?
                .phase;
            if phase == GroupPhase::Active {
                continue;
            }
            self.check_activation(new)?;
            if let Some(epoch) = self.groups[&new.group].active
                && !drain.contains(&GroupVersion {
                    group: new.group,
                    epoch,
                })
            {
                return Err(ServingError::ActiveEpochChanged {
                    current: Some(epoch),
                });
            }
            activate.push(new);
        }
        let mut fingerprints = BTreeMap::new();
        for &path in &intent.paths {
            let endpoint = self
                .decoder_endpoints(model, path)
                .ok_or(ServingError::InvalidRoute(path))?;
            let fingerprint = endpoint.prefill.fingerprint;
            if current.is_some_and(|directory| {
                !directory.fingerprints.is_empty() && !directory.fingerprints.contains(&fingerprint)
            }) {
                return Err(ServingError::InvalidRoute(path));
            }
            fingerprints.insert(path, fingerprint);
        }
        let standbys = self.decoder_standbys(model, &fingerprints, &intent.standbys)?;
        let directory = RouteSet {
            model: model.to_owned(),
            revision,
            paths: intent.paths.clone(),
            standbys,
            fingerprints: fingerprints.into_values().collect(),
            admission_weights: pending.weights.clone(),
        };
        Ok(Commit {
            directory,
            activate,
            drain,
        })
    }
}

fn versions(path: &PdRoute) -> impl Iterator<Item = GroupVersion> {
    [path.prefill, path.decode].into_iter()
}
