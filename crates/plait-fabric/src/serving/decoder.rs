use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;

use plait_core::serving::{
    DecoderCheckpoint, DecoderClaim, DecoderDirectory, DecoderDrainStatus, DecoderEndpoint,
    DecoderOffer, DecoderReplayState, DecoderRoute, DecoderStandby, DecoderStep, GroupPhase,
    GroupVersion, PdRoute, RouteSetId, ServingError, ServingOutcome, ServingResult, ServingRole,
    WorkerGrant,
};
use plait_core::{NodeId, OpId};

use super::{ServingState, routable};

mod failover;
mod replay;
mod routing;
mod switch;
use failover::Standby;
use replay::Replay;

const ADMISSION_MS: u64 = 120_000;

#[derive(Clone, Default)]
pub(super) struct DecoderState {
    pub(super) endpoints: BTreeMap<GroupVersion, DecoderEndpoint>,
    pub(super) managed: BTreeSet<GroupVersion>,
    pub(super) faulted: BTreeSet<GroupVersion>,
    routes: BTreeMap<RouteSetId, RouteSet>,
    work: BTreeMap<RouteSetId, BTreeMap<OpId, Work>>,
    replays: BTreeMap<RouteSetId, BTreeMap<OpId, Replay>>,
    submissions: BTreeMap<OpId, RouteSetId>,
    switches: BTreeMap<RouteSetId, switch::StagedSwitch>,
}

#[derive(Clone)]
struct RouteSet {
    model: String,
    revision: u64,
    paths: Vec<PdRoute>,
    standbys: BTreeMap<PdRoute, Standby>,
    fingerprints: BTreeSet<[u8; 32]>,
    admission_weights: BTreeMap<PdRoute, u32>,
}

#[derive(Clone)]
struct Work {
    route: DecoderRoute,
    expires_at_ms: u64,
    stage: Stage,
    recoverable: bool,
    retry_prefill: bool,
    checkpoints: Vec<DecoderCheckpoint>,
}

impl Work {
    fn depends_on(&self, version: GroupVersion) -> bool {
        self.route.decode.grant.version == version
            || (self.route.prefill.grant.version == version
                && (self.recoverable || !matches!(self.stage, Stage::Decode { .. })))
    }
}

#[derive(Clone, Copy)]
enum Stage {
    Admitted,
    Prefill(OpId),
    Offered(DecoderOffer),
    Decode { owner: OpId, offer: DecoderOffer },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RecoveryPolicy {
    None,
    Decode,
    Pipeline,
}

impl ServingState {
    pub(crate) fn decoder_routes(&self, id: RouteSetId) -> Option<DecoderDirectory> {
        let declared = self.decoder.routes.get(&id)?;
        Some(DecoderDirectory {
            model: declared.model.clone(),
            revision: declared.revision,
            paths: declared
                .paths
                .iter()
                .filter_map(|&path| self.resolve_decoder(&declared.model, path, None))
                .collect(),
        })
    }

    pub(super) fn decoder_ready(
        &mut self,
        endpoint: DecoderEndpoint,
        digest: [u8; 32],
        managed: bool,
    ) -> ServingResult {
        let version = endpoint.grant.version;
        let view = self
            .group(version)
            .ok_or(ServingError::UnknownVersion(version))?;
        if !matches!(view.plan.role, ServingRole::Prefill | ServingRole::Decode)
            || view.grants.len() != 1
            || !usable_addr(endpoint.addr)
            || self.decoder.endpoints.get(&version).is_some_and(|old| {
                *old != endpoint || self.decoder.managed.contains(&version) != managed
            })
        {
            return Err(ServingError::InvalidEndpoint);
        }
        let result = self.worker_ready(endpoint.grant, digest)?;
        self.decoder.endpoints.insert(version, endpoint);
        if managed {
            self.decoder.managed.insert(version);
        }
        Ok(result)
    }

    pub(super) fn publish_decoder_routes(
        &mut self,
        model: &str,
        expected_revision: u64,
        paths: &[PdRoute],
        activate: Option<GroupVersion>,
        standbys: &[DecoderStandby],
    ) -> ServingResult {
        let id = RouteSetId::for_model(model);
        let previous = self.decoder.routes.get(&id);
        if model.is_empty() || previous.is_some_and(|entry| entry.model != model) {
            return Err(ServingError::InvalidDirectory);
        }
        let current = previous.map_or(0, |entry| entry.revision);
        if expected_revision != current {
            return Err(ServingError::StaleDirectory { current });
        }
        let revision = current
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?;
        if let Some(version) = activate {
            self.check_activation(version)?;
            if !paths
                .iter()
                .any(|path| path.prefill == version || path.decode == version)
            {
                return Err(ServingError::InvalidDirectory);
            }
        }
        let mut fingerprints = BTreeMap::new();
        for &path in paths {
            let Some(route) = self.resolve_decoder(model, path, activate) else {
                return Err(ServingError::InvalidRoute(path));
            };
            if fingerprints
                .insert(path, route.prefill.fingerprint)
                .is_some()
            {
                return Err(ServingError::InvalidRoute(path));
            }
        }
        let standbys = self.decoder_standbys(model, &fingerprints, standbys)?;
        if let Some(version) = activate {
            self.activate(version)?;
        }
        self.decoder.routes.insert(
            id,
            RouteSet {
                model: model.to_owned(),
                revision,
                paths: paths.to_vec(),
                standbys,
                fingerprints: fingerprints.into_values().collect(),
                admission_weights: BTreeMap::new(),
            },
        );
        Ok(ServingOutcome::RoutesPublished { revision })
    }

    fn resolve_decoder(
        &self,
        model: &str,
        path: PdRoute,
        activate: Option<GroupVersion>,
    ) -> Option<DecoderRoute> {
        if !routable(self.group(path.prefill)?, activate)
            || !routable(self.group(path.decode)?, activate)
        {
            return None;
        }
        self.decoder_endpoints(model, path)
    }

    fn decoder_endpoints(&self, model: &str, path: PdRoute) -> Option<DecoderRoute> {
        let p = self.group(path.prefill)?;
        let d = self.group(path.decode)?;
        let prefill = *self.decoder.endpoints.get(&path.prefill)?;
        let decode = *self.decoder.endpoints.get(&path.decode)?;
        (p.plan.role == ServingRole::Prefill
            && d.plan.role == ServingRole::Decode
            && p.plan.model == model
            && d.plan.model == model
            && p.holds_grant(prefill.grant)
            && d.holds_grant(decode.grant)
            && prefill.fingerprint == decode.fingerprint)
            .then_some(DecoderRoute { prefill, decode })
    }

    pub(super) fn admit_decoder(
        &mut self,
        model: &str,
        revision: u64,
        path: PdRoute,
        id: OpId,
        now_ms: u64,
        recovery: RecoveryPolicy,
    ) -> ServingResult {
        if self.decoder.submissions.contains_key(&id) {
            return Err(ServingError::OperationIdReused);
        }
        let key = RouteSetId::for_model(model);
        let directory = self
            .decoder
            .routes
            .get(&key)
            .ok_or(ServingError::InvalidDirectory)?;
        if directory.revision != revision {
            return Err(ServingError::StaleDirectory {
                current: directory.revision,
            });
        }
        let route = directory
            .paths
            .contains(&path)
            .then(|| self.resolve_decoder(model, path, None))
            .flatten()
            .ok_or(ServingError::InvalidRoute(path))?;
        let standby = directory.standbys.get(&path);
        let recoverable = recovery != RecoveryPolicy::None
            && standby.is_some_and(|standby| standby.has_decode_alternative(path));
        let retry_prefill = recovery == RecoveryPolicy::Pipeline
            && standby.is_some_and(|standby| standby.has_prefill_alternative(path));
        let expires_at_ms = now_ms
            .checked_add(ADMISSION_MS)
            .ok_or(ServingError::SequenceExhausted)?;
        let work = self.decoder.work.entry(key).or_default();
        work.retain(|_, job| job.route.path() != path || job.expires_at_ms > now_ms);
        work.insert(
            id,
            Work {
                route,
                expires_at_ms,
                stage: Stage::Admitted,
                recoverable,
                retry_prefill,
                checkpoints: Vec::new(),
            },
        );
        if recoverable || retry_prefill {
            Ok(ServingOutcome::DecoderRecoveryAdmitted { revision, route })
        } else {
            Ok(ServingOutcome::DecoderAdmitted { revision, route })
        }
    }

    pub(super) fn advance_decoder(
        &mut self,
        model: &str,
        id: OpId,
        path: PdRoute,
        grant: WorkerGrant,
        now_ms: u64,
        step: DecoderStep,
        operation: OpId,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let work = self
            .decoder
            .work
            .get(&key)
            .and_then(|entries| entries.get(&id))
            .ok_or(ServingError::DecoderNotAdmitted)?;
        if work.expires_at_ms <= now_ms || work.route.path() != path {
            return Err(ServingError::DecoderNotAdmitted);
        }
        let prefilling = work.route.prefill.grant == grant;
        let decoding = work.route.decode.grant == grant;
        if !(prefilling || decoding)
            || !self
                .group(grant.version)
                .is_some_and(|view| view.allows_grant(grant))
        {
            return Err(ServingError::GrantMismatch);
        }
        let jobs = self.decoder.work.get_mut(&key).expect("checked job map");
        let work = jobs.get_mut(&id).expect("checked job");
        let replay = self
            .decoder
            .replays
            .get_mut(&key)
            .and_then(|records| records.get_mut(&id));
        let current_entry = replay.as_ref().and_then(|record| record.entry);
        let step = match step {
            DecoderStep::StartPrefillEntry { entry } if Some(entry) == current_entry => {
                DecoderStep::StartPrefill
            }
            DecoderStep::ReadOfferEntry { entry } if Some(entry) == current_entry => {
                DecoderStep::ReadOffer
            }
            DecoderStep::StartPrefillEntry { .. } | DecoderStep::ReadOfferEntry { .. } => {
                return Err(ServingError::StaleDecoderEntry);
            }
            DecoderStep::StartPrefill | DecoderStep::ReadOffer if current_entry.is_some() => {
                return Err(ServingError::StaleDecoderEntry);
            }
            step => step,
        };
        match (step, work.stage) {
            (DecoderStep::StartPrefill, Stage::Admitted) if prefilling => {
                work.stage = Stage::Prefill(operation)
            }
            (DecoderStep::Offer { owner, kv }, Stage::Prefill(expected))
                if prefilling && owner == expected =>
            {
                if kv.prefix_tokens == 0
                    || kv.source.bytes == 0
                    || !usable_addr(kv.source.addr)
                    || kv.source.addr.ip() != work.route.prefill.addr.ip()
                {
                    return Err(ServingError::InvalidEndpoint);
                }
                if let Some(replay) = replay {
                    replay.first(kv.first_token)?;
                }
                work.stage = Stage::Offered(kv);
            }
            (DecoderStep::ReadOffer, Stage::Offered(kv)) if decoding => {
                return Ok(ServingOutcome::DecoderOffered(kv));
            }
            (DecoderStep::ClaimDecode { transfer }, Stage::Offered(kv))
                if decoding && transfer == kv.source.transfer =>
            {
                work.checkpoints.clear();
                work.stage = Stage::Decode {
                    owner: operation,
                    offer: kv,
                };
            }
            (DecoderStep::ClaimDecodeCheckpoint { transfer, offset }, Stage::Offered(kv))
                if decoding && transfer == kv.source.transfer =>
            {
                if offset != 1
                    && !work
                        .checkpoints
                        .iter()
                        .any(|checkpoint| checkpoint.end_offset == offset)
                {
                    return Err(ServingError::InvalidDecoderStage);
                }
                work.checkpoints
                    .retain(|checkpoint| checkpoint.end_offset <= offset);
                work.stage = Stage::Decode {
                    owner: operation,
                    offer: kv,
                };
            }
            (
                DecoderStep::CheckOutput { owner },
                Stage::Decode {
                    owner: expected, ..
                },
            ) if decoding && owner == expected => {
                return Ok(ServingOutcome::DecoderAdvanced);
            }
            (
                DecoderStep::CheckOutputCheckpoint { owner, checkpoint },
                Stage::Decode {
                    owner: expected,
                    offer,
                },
            ) if decoding && owner == expected => {
                let next = work
                    .checkpoints
                    .last()
                    .map_or(1, |previous| previous.end_offset);
                if checkpoint.start_offset != next
                    || checkpoint.end_offset <= next
                    || checkpoint.source.addr != offer.source.addr
                    || checkpoint.source.transfer == offer.source.transfer
                    || checkpoint.source.bytes == 0
                {
                    return Err(ServingError::InvalidDecoderStage);
                }
                work.checkpoints.push(checkpoint);
                return Ok(ServingOutcome::DecoderAdvanced);
            }
            (DecoderStep::Finish { owner }, Stage::Prefill(expected))
                if prefilling && owner == expected && replay.is_none() =>
            {
                jobs.remove(&id);
            }
            (
                DecoderStep::Finish { owner },
                Stage::Decode {
                    owner: expected, ..
                },
            ) if decoding && owner == expected && replay.is_none() => {
                jobs.remove(&id);
            }
            (DecoderStep::FinishPrefillToken { owner, token }, Stage::Prefill(expected))
                if prefilling && owner == expected =>
            {
                if let Some(replay) = replay {
                    replay.first(token)?;
                    if replay.tokens.len() != 1 {
                        return Err(ServingError::InvalidDecoderOutput);
                    }
                    replay.state = DecoderReplayState::Complete;
                }
                jobs.remove(&id);
            }
            (
                DecoderStep::FinishDecodeStream { owner, end_offset },
                Stage::Decode {
                    owner: expected, ..
                },
            ) if decoding && owner == expected => {
                if let Some(replay) = replay {
                    if end_offset < 2 || replay.tokens.len() != end_offset as usize {
                        return Err(ServingError::InvalidDecoderOutput);
                    }
                    replay.state = DecoderReplayState::Complete;
                }
                jobs.remove(&id);
            }
            _ => return Err(ServingError::InvalidDecoderStage),
        }
        self.revoke_drained_decoder(path.prefill);
        self.revoke_drained_decoder(path.decode);
        Ok(ServingOutcome::DecoderAdvanced)
    }

    pub(super) fn recover_decoder_work(
        &mut self,
        model: &str,
        id: OpId,
        revision: u64,
        failed: PdRoute,
        replacement: PdRoute,
        now_ms: u64,
        restart_prefill: bool,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let directory = self
            .decoder
            .routes
            .get(&key)
            .ok_or(ServingError::InvalidDirectory)?;
        if directory.revision != revision {
            return Err(ServingError::StaleDirectory {
                current: directory.revision,
            });
        }
        let work = self
            .decoder
            .work
            .get(&key)
            .and_then(|jobs| jobs.get(&id))
            .filter(|work| work.route.path() == failed && work.expires_at_ms > now_ms)
            .ok_or(ServingError::DecoderNotAdmitted)?;
        let decode_changed = failed.decode != replacement.decode;
        let prefill_changed = failed.prefill != replacement.prefill;
        let decode_failed = self.decoder.faulted.contains(&failed.decode);
        let stage = if restart_prefill {
            if !work.retry_prefill
                || !prefill_changed
                || !self.decoder.faulted.contains(&failed.prefill)
                || (decode_changed && !decode_failed)
                || (matches!(work.stage, Stage::Decode { .. }) && !decode_failed)
            {
                return Err(ServingError::InvalidDecoderStage);
            }
            Stage::Admitted
        } else {
            if !work.recoverable || !decode_failed || prefill_changed || !decode_changed {
                return Err(ServingError::InvalidDecoderStage);
            }
            match work.stage {
                Stage::Offered(offer) | Stage::Decode { offer, .. } => Stage::Offered(offer),
                _ => return Err(ServingError::InvalidDecoderStage),
            }
        };
        let route = directory
            .paths
            .contains(&replacement)
            .then(|| self.resolve_decoder(model, replacement, None))
            .flatten()
            .filter(|route| {
                route.prefill.fingerprint == work.route.prefill.fingerprint
                    && (restart_prefill || route.prefill == work.route.prefill)
            })
            .ok_or(ServingError::InvalidRoute(replacement))?;
        let work = self
            .decoder
            .work
            .get_mut(&key)
            .and_then(|jobs| jobs.get_mut(&id))
            .expect("checked decoder work");
        work.route = route;
        work.stage = stage;
        if restart_prefill {
            work.checkpoints.clear();
        }
        if let Some(replay) = self
            .decoder
            .replays
            .get_mut(&key)
            .and_then(|records| records.get_mut(&id))
        {
            replay.revision = revision;
            replay.path = replacement;
        }
        self.revoke_drained_decoder(failed.prefill);
        self.revoke_drained_decoder(failed.decode);
        Ok(ServingOutcome::DecoderRecoveryAdmitted { revision, route })
    }

    pub(crate) fn decoder_claims(&self, version: GroupVersion) -> Vec<DecoderClaim> {
        let Some(group) = self.group(version) else {
            return Vec::new();
        };
        self.decoder
            .work
            .get(&RouteSetId::for_model(&group.plan.model))
            .into_iter()
            .flatten()
            .filter_map(|(&work, entry)| match entry.stage {
                Stage::Prefill(owner) if entry.route.path().prefill == version => {
                    Some(DecoderClaim { work, owner })
                }
                Stage::Decode { owner, .. } if entry.route.path().decode == version => {
                    Some(DecoderClaim { work, owner })
                }
                _ => None,
            })
            .collect()
    }

    pub(crate) fn decoder_checkpoints(
        &self,
        model: RouteSetId,
        id: OpId,
        transfer: NodeId,
    ) -> Vec<DecoderCheckpoint> {
        let Some(work) = self.decoder.work.get(&model).and_then(|jobs| jobs.get(&id)) else {
            return Vec::new();
        };
        match work.stage {
            Stage::Offered(offer) | Stage::Decode { offer, .. }
                if offer.source.transfer == transfer =>
            {
                work.checkpoints.clone()
            }
            _ => Vec::new(),
        }
    }

    pub(super) fn cancel_decoder(&mut self, model: &str, id: OpId, path: PdRoute) -> ServingResult {
        if let Some(jobs) = self.decoder.work.get_mut(&RouteSetId::for_model(model))
            && let Some(work) = jobs.get(&id)
        {
            if work.route.path() != path {
                return Err(ServingError::DecoderNotAdmitted);
            }
            jobs.remove(&id);
            if let Some(replay) = self
                .decoder
                .replays
                .get_mut(&RouteSetId::for_model(model))
                .and_then(|records| records.get_mut(&id))
            {
                replay.state = DecoderReplayState::Failed;
            }
        }
        self.revoke_drained_decoder(path.prefill);
        self.revoke_drained_decoder(path.decode);
        Ok(ServingOutcome::DecoderAdvanced)
    }

    pub(crate) fn decoder_drain(&self, version: GroupVersion) -> Option<DecoderDrainStatus> {
        if !self.decoder.managed.contains(&version) {
            return None;
        }
        let model = &self.group(version)?.plan.model;
        let mut status = DecoderDrainStatus {
            pending: 0,
            next_expiry: None,
        };
        if let Some(jobs) = self.decoder.work.get(&RouteSetId::for_model(model)) {
            for job in jobs.values().filter(|job| job.depends_on(version)) {
                status.pending += 1;
                if status
                    .next_expiry
                    .is_none_or(|(ms, _)| job.expires_at_ms < ms)
                {
                    status.next_expiry = Some((job.expires_at_ms, job.route.path()));
                }
            }
        }
        Some(status)
    }

    pub(super) fn expire_decoder(
        &mut self,
        model: &str,
        path: PdRoute,
        through_ms: u64,
    ) -> ServingResult {
        if ![path.prefill, path.decode].iter().any(|version| {
            self.decoder.managed.contains(version)
                && self.group(*version).is_some_and(|view| {
                    view.phase == GroupPhase::Draining && view.plan.model == model
                })
        }) {
            return Err(ServingError::InvalidDecoderStage);
        }
        if let Some(jobs) = self.decoder.work.get_mut(&RouteSetId::for_model(model)) {
            jobs.retain(|_, job| job.route.path() != path || job.expires_at_ms > through_ms);
        }
        self.revoke_drained_decoder(path.prefill);
        self.revoke_drained_decoder(path.decode);
        Ok(ServingOutcome::DecoderAdvanced)
    }

    pub(super) fn revoke_drained_decoder(&mut self, version: GroupVersion) {
        if self
            .group(version)
            .is_some_and(|view| view.phase == GroupPhase::Draining)
            && self
                .decoder_drain(version)
                .is_some_and(|status| status.pending == 0)
        {
            self.group_mut(version).expect("known draining group").phase = GroupPhase::Revoked;
        }
    }
}

fn usable_addr(addr: SocketAddr) -> bool {
    addr.port() != 0 && !addr.ip().is_unspecified() && !addr.ip().is_multicast()
}
