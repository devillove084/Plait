use plait_core::OpId;
use plait_core::serving::{
    DECODER_REPLAY_MAX_TOKENS, DecoderCheckpoint, DecoderEntryRetry, DecoderReplayPage,
    DecoderReplayState, DecoderStep, DecoderSubmission, PdRoute, RouteSetId, ServingError,
    ServingOutcome, ServingResult, WorkerGrant,
};

use super::{ADMISSION_MS, RecoveryPolicy, ServingState, Stage};

const RETENTION_MS: u64 = 300_000;
const MAX_STREAMS_PER_MODEL: usize = 1024;
const READ_PAGE: usize = 256;

#[derive(Clone)]
pub(super) struct Replay {
    pub(super) revision: u64,
    pub(super) path: PdRoute,
    pub(super) state: DecoderReplayState,
    pub(super) tokens: Vec<u32>,
    pub(super) entry: Option<OpId>,
    entry_retry: Option<(OpId, DecoderEntryRetry)>,
    input_sha256: Option<[u8; 32]>,
    max_tokens: u32,
    admission_until: u64,
    retain_until: u64,
}

impl Replay {
    fn check_takeover(
        &self,
        expected: OpId,
        retry: Option<DecoderEntryRetry>,
    ) -> Result<(), ServingError> {
        if retry.is_some_and(|retry| retry.sequence == 0) {
            return Err(ServingError::StaleDecoderEntry);
        }
        let continues = retry
            .zip(self.entry_retry)
            .is_some_and(|(next, (anchor, current))| {
                anchor == expected
                    && next.series == current.series
                    && next.sequence > current.sequence
            });
        if self.entry != Some(expected) && !continues {
            return Err(ServingError::StaleDecoderEntry);
        }
        Ok(())
    }

    pub(super) fn first(&mut self, token: u32) -> Result<(), ServingError> {
        if let Some(&first) = self.tokens.first() {
            if first != token {
                return Err(ServingError::InvalidDecoderOutput);
            }
        } else {
            self.tokens.push(token);
        }
        Ok(())
    }

    fn suffix(&self, offset: u32, tokens: &[u32]) -> Result<usize, ServingError> {
        let published = self
            .tokens
            .get(offset as usize..)
            .ok_or(ServingError::InvalidDecoderOutput)?;
        let overlap = published.len().min(tokens.len());
        if offset == 0
            || tokens.is_empty()
            || tokens.len() > self.max_tokens.saturating_sub(offset) as usize
            || published[..overlap] != tokens[..overlap]
            || self.state != DecoderReplayState::Running
        {
            return Err(ServingError::InvalidDecoderOutput);
        }
        Ok(overlap)
    }
}

impl ServingState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::serving) fn admit_decoder_replay(
        &mut self,
        model: &str,
        revision: u64,
        path: PdRoute,
        work: OpId,
        now: u64,
        max_tokens: u32,
        input_sha256: Option<[u8; 32]>,
    ) -> ServingResult {
        if !(1..=DECODER_REPLAY_MAX_TOKENS).contains(&max_tokens) {
            return Err(ServingError::InvalidDecoderOutput);
        }
        let retain_until = now
            .checked_add(RETENTION_MS)
            .ok_or(ServingError::SequenceExhausted)?;
        let key = RouteSetId::for_model(model);
        if self
            .decoder
            .replays
            .get(&key)
            .into_iter()
            .flat_map(|records| records.values())
            .filter(|record| record.retain_until > now)
            .count()
            >= MAX_STREAMS_PER_MODEL
        {
            return Err(ServingError::DecoderReplayCapacity);
        }
        let result =
            self.admit_decoder(model, revision, path, work, now, RecoveryPolicy::Pipeline)?;
        let records = self.decoder.replays.entry(key).or_default();
        records.retain(|_, record| record.retain_until > now);
        records.insert(
            work,
            Replay {
                revision,
                path,
                state: DecoderReplayState::Running,
                tokens: Vec::new(),
                entry: input_sha256.map(|_| work),
                entry_retry: None,
                input_sha256,
                max_tokens,
                admission_until: now + ADMISSION_MS,
                retain_until,
            },
        );
        Ok(result)
    }

    pub(in crate::serving) fn check_decoder_entry(
        &self,
        model: &str,
        work: OpId,
        expected: Option<OpId>,
    ) -> Result<(), ServingError> {
        let current = self
            .decoder
            .replays
            .get(&RouteSetId::for_model(model))
            .and_then(|records| records.get(&work))
            .and_then(|record| record.entry);
        if current != expected {
            return Err(ServingError::StaleDecoderEntry);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::serving) fn submit_decoder(
        &mut self,
        model: &str,
        entry: OpId,
        revision: u64,
        path: Option<PdRoute>,
        previous: Option<PdRoute>,
        now: u64,
        request: DecoderSubmission,
    ) -> ServingResult {
        let work = DecoderSubmission::work(request.id);
        if work == entry {
            return Err(ServingError::OperationIdReused);
        }
        if request.sequence == 0 {
            return Err(ServingError::StaleDecoderEntry);
        }
        let key = RouteSetId::for_model(model);
        let retry = DecoderEntryRetry {
            series: request.id,
            sequence: request.sequence,
        };
        if let Some(bound) = self.decoder.submissions.get(&work) {
            if *bound != key {
                return Err(ServingError::OperationIdReused);
            }
            let record = self
                .decoder
                .replays
                .get(&key)
                .and_then(|records| records.get(&work))
                .filter(|record| now < record.retain_until)
                .ok_or(ServingError::DecoderNotAdmitted)?;
            if record.input_sha256 != Some(request.input_sha256)
                || record.max_tokens != request.max_tokens
                || request.offset as usize > record.tokens.len()
            {
                return Err(ServingError::InvalidDecoderOutput);
            }
            if record.state == DecoderReplayState::Complete {
                return Ok(ServingOutcome::DecoderSubmissionComplete {
                    revision: record.revision,
                    path: record.path,
                    count: record.tokens.len() as u32,
                });
            }
            if record.state != DecoderReplayState::Running || now >= record.admission_until {
                return Err(ServingError::DecoderNotAdmitted);
            }
            record.check_takeover(work, Some(retry))?;
            let replacement = path.ok_or(ServingError::InvalidDirectory)?;
            if previous.unwrap_or(replacement) != record.path {
                return Err(ServingError::DecoderSubmissionRouteChanged(record.path));
            }
            return self.takeover_decoder(
                model,
                work,
                work,
                entry,
                revision,
                record.path,
                replacement,
                now,
                request.offset,
                request.input_sha256,
                Some(retry),
            );
        }
        if self.receipt(work).is_some() {
            return Err(ServingError::OperationIdReused);
        }
        if request.offset != 0 {
            return Err(ServingError::InvalidDecoderOutput);
        }
        let path = path.ok_or(ServingError::InvalidDirectory)?;
        let result = self.admit_decoder_replay(
            model,
            revision,
            path,
            work,
            now,
            request.max_tokens,
            Some(request.input_sha256),
        )?;
        let record = self
            .decoder
            .replays
            .get_mut(&key)
            .and_then(|records| records.get_mut(&work))
            .expect("admitted decoder replay");
        record.entry = Some(entry);
        record.entry_retry = Some((work, retry));
        self.decoder.submissions.insert(work, key);
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::serving) fn takeover_decoder(
        &mut self,
        model: &str,
        work: OpId,
        expected_entry: OpId,
        replacement_entry: OpId,
        revision: u64,
        previous: PdRoute,
        replacement: PdRoute,
        now: u64,
        offset: u32,
        input_sha256: [u8; 32],
        retry: Option<DecoderEntryRetry>,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let record = self
            .decoder
            .replays
            .get(&key)
            .and_then(|records| records.get(&work))
            .ok_or(ServingError::DecoderNotAdmitted)?;
        record.check_takeover(expected_entry, retry)?;
        if record.input_sha256 != Some(input_sha256) || offset as usize > record.tokens.len() {
            return Err(ServingError::InvalidDecoderOutput);
        }
        let current = self
            .decoder
            .work
            .get(&key)
            .and_then(|jobs| jobs.get(&work))
            .filter(|job| job.expires_at_ms > now && job.route.path() == previous)
            .ok_or(ServingError::DecoderNotAdmitted)?;
        if record.state != DecoderReplayState::Running {
            return Err(ServingError::DecoderNotAdmitted);
        }
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
            .contains(&replacement)
            .then(|| self.resolve_decoder(model, replacement, None))
            .flatten()
            .filter(|route| route.prefill.fingerprint == current.route.prefill.fingerprint)
            .ok_or(ServingError::InvalidRoute(replacement))?;
        let standby = directory.standbys.get(&replacement);
        let recoverable =
            standby.is_some_and(|standby| standby.has_decode_alternative(replacement));
        let retry_prefill =
            standby.is_some_and(|standby| standby.has_prefill_alternative(replacement));
        let committed = record.tokens.len() as u32;
        let job = self
            .decoder
            .work
            .get_mut(&key)
            .and_then(|jobs| jobs.get_mut(&work))
            .expect("checked decoder work");
        job.route = route;
        job.stage = Stage::Admitted;
        job.recoverable = recoverable;
        job.retry_prefill = retry_prefill;
        job.checkpoints.clear();
        let record = self
            .decoder
            .replays
            .get_mut(&key)
            .and_then(|records| records.get_mut(&work))
            .expect("checked decoder replay");
        record.entry = Some(replacement_entry);
        record.entry_retry = retry.map(|retry| (expected_entry, retry));
        record.revision = revision;
        record.path = replacement;
        self.revoke_drained_decoder(previous.prefill);
        self.revoke_drained_decoder(previous.decode);
        Ok(ServingOutcome::DecoderTakenOver {
            revision,
            route,
            committed,
            recoverable: recoverable || retry_prefill,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::serving) fn append_decoder_output(
        &mut self,
        model: &str,
        work: OpId,
        path: PdRoute,
        grant: WorkerGrant,
        now: u64,
        owner: OpId,
        offset: u32,
        tokens: &[u32],
        checkpoint: Option<DecoderCheckpoint>,
        operation: OpId,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let record = self
            .decoder
            .replays
            .get(&key)
            .and_then(|records| records.get(&work))
            .ok_or(ServingError::DecoderNotAdmitted)?;
        let overlap = record.suffix(offset, tokens)?;
        let step = if let Some(checkpoint) = checkpoint {
            if offset.checked_add(tokens.len() as u32) != Some(checkpoint.end_offset)
                || tokens.last() != Some(&checkpoint.last_token)
            {
                return Err(ServingError::InvalidDecoderOutput);
            }
            DecoderStep::CheckOutputCheckpoint { owner, checkpoint }
        } else {
            DecoderStep::CheckOutput { owner }
        };
        let result = self.advance_decoder(model, work, path, grant, now, step, operation)?;
        self.decoder
            .replays
            .get_mut(&key)
            .and_then(|records| records.get_mut(&work))
            .expect("checked replay record")
            .tokens
            .extend_from_slice(&tokens[overlap..]);
        Ok(result)
    }

    pub(crate) fn decoder_replay(
        &self,
        model: RouteSetId,
        work: OpId,
        offset: u32,
        now: u64,
    ) -> Option<DecoderReplayPage> {
        let record = self.decoder.replays.get(&model)?.get(&work)?;
        if record.retain_until <= now {
            return None;
        }
        let state = if record.state == DecoderReplayState::Running && record.admission_until <= now
        {
            DecoderReplayState::Failed
        } else {
            record.state
        };
        let tokens = record.tokens.get(offset as usize..).unwrap_or_default();
        Some(DecoderReplayPage {
            tokens: tokens[..tokens.len().min(READ_PAGE)].to_vec(),
            available: record.tokens.len() as u32,
            state,
            revision: record.revision,
            path: record.path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::NodeId;
    use plait_core::serving::{GroupId, GroupVersion};

    #[test]
    fn durable_suffix_rejects_holes_divergence_and_capacity_overrun() {
        let group = GroupVersion {
            group: GroupId(NodeId::derive("replay-test")),
            epoch: 1,
        };
        let path = PdRoute {
            prefill: group,
            decode: group,
        };
        let mut record = Replay {
            path,
            revision: 1,
            state: DecoderReplayState::Running,
            tokens: Vec::new(),
            entry: None,
            entry_retry: None,
            input_sha256: None,
            max_tokens: 4,
            admission_until: ADMISSION_MS,
            retain_until: RETENTION_MS,
        };
        record.first(7).unwrap();
        let new = &[8, 9];
        let overlap = record.suffix(1, new).unwrap();
        record.tokens.extend_from_slice(&new[overlap..]);
        assert_eq!(record.suffix(2, &[9, 10]), Ok(1));
        assert_eq!(
            record.suffix(2, &[9, 10, 11]),
            Err(ServingError::InvalidDecoderOutput)
        );
        assert_eq!(
            record.suffix(2, &[99, 10]),
            Err(ServingError::InvalidDecoderOutput)
        );
        assert_eq!(
            record.suffix(4, &[10]),
            Err(ServingError::InvalidDecoderOutput)
        );
        assert_eq!(record.first(99), Err(ServingError::InvalidDecoderOutput));
        assert_eq!(record.tokens, [7, 8, 9]);
    }
}
