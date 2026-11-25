use std::collections::BTreeMap;

use plait_core::OpId;
use plait_core::serving::{
    DecoderAdmissionRoutes, DecoderReplayInput, DecoderReplayState, DecoderRouteWeight,
    DecoderSubmission, PdRoute, RouteSetId, ServingError, ServingOutcome, ServingResult,
};

use super::{RecoveryPolicy, ServingState};

pub(super) fn decoder_weights(
    paths: &[PdRoute],
    weights: &[DecoderRouteWeight],
    allow_zero: bool,
) -> Result<BTreeMap<PdRoute, u32>, ServingError> {
    let by_path: BTreeMap<_, _> = weights
        .iter()
        .map(|entry| (entry.path, entry.weight))
        .collect();
    if weights.len() != paths.len()
        || by_path.len() != weights.len()
        || !weights.iter().any(|entry| entry.weight > 0)
        || paths.iter().any(|path| {
            by_path
                .get(path)
                .is_none_or(|&weight| weight == 0 && !allow_zero)
        })
    {
        return Err(ServingError::InvalidDirectory);
    }
    Ok(by_path)
}

impl ServingState {
    pub(crate) fn decoder_admission_routes(&self, key: RouteSetId) -> DecoderAdmissionRoutes {
        self.decoder.routes.get(&key).map_or_else(
            || DecoderAdmissionRoutes {
                revision: 0,
                paths: Vec::new(),
            },
            |directory| DecoderAdmissionRoutes {
                revision: directory.revision,
                paths: directory.paths.clone(),
            },
        )
    }

    pub(in crate::serving) fn configure_decoder_weights(
        &mut self,
        model: &str,
        expected: u64,
        weights: &[DecoderRouteWeight],
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let directory = self
            .decoder
            .routes
            .get_mut(&key)
            .ok_or(ServingError::InvalidDirectory)?;
        if directory.revision != expected {
            return Err(ServingError::StaleDirectory {
                current: directory.revision,
            });
        }
        let weights_by_path = decoder_weights(&directory.paths, weights, false)?;
        let revision = expected
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?;
        directory.admission_weights = weights_by_path;
        directory.revision = revision;
        Ok(ServingOutcome::RoutesPublished { revision })
    }

    pub(in crate::serving) fn admit_decoder_routed(
        &mut self,
        model: &str,
        routes: &DecoderAdmissionRoutes,
        work: OpId,
        now: u64,
        replay: Option<DecoderReplayInput>,
    ) -> ServingResult {
        let path = self
            .select_decoder_path(model, routes, now, None)?
            .ok_or(ServingError::InvalidDirectory)?;
        if let Some(replay) = replay {
            self.admit_decoder_replay(
                model,
                routes.revision,
                path,
                work,
                now,
                replay.max_tokens,
                Some(replay.input_sha256),
            )
        } else {
            self.admit_decoder(
                model,
                routes.revision,
                path,
                work,
                now,
                RecoveryPolicy::Pipeline,
            )
        }
    }

    pub(in crate::serving) fn submit_decoder_routed(
        &mut self,
        model: &str,
        routes: &DecoderAdmissionRoutes,
        previous: Option<PdRoute>,
        now: u64,
        request: DecoderSubmission,
        entry: OpId,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let work = DecoderSubmission::work(request.id);
        let has_terminal = self
            .decoder
            .replays
            .get(&key)
            .and_then(|records| records.get(&work))
            .is_some_and(|record| record.state != DecoderReplayState::Running);
        let path = if has_terminal {
            None
        } else {
            let prefer = self
                .decoder
                .work
                .get(&key)
                .and_then(|jobs| jobs.get(&work))
                .map(|job| (job.route.path(), job.route.prefill.fingerprint));
            self.select_decoder_path(model, routes, now, prefer)?
        };
        self.submit_decoder(model, entry, routes.revision, path, previous, now, request)
    }

    fn select_decoder_path(
        &self,
        model: &str,
        routes: &DecoderAdmissionRoutes,
        now: u64,
        prefer: Option<(PdRoute, [u8; 32])>,
    ) -> Result<Option<PdRoute>, ServingError> {
        let key = RouteSetId::for_model(model);
        let directory = self.decoder.routes.get(&key);
        let current = directory.map_or(0, |directory| directory.revision);
        if routes.revision != current {
            return Err(ServingError::StaleDirectory { current });
        }
        let declared = directory.map_or(&[][..], |directory| directory.paths.as_slice());
        if declared != routes.paths.as_slice() {
            return Err(ServingError::InvalidDirectory);
        }
        let Some(directory) = directory else {
            return Ok(None);
        };
        let mut eligible = Vec::new();
        let mut capacity = BTreeMap::<_, u64>::new();
        for &path in declared {
            let Some(route) = self.resolve_decoder(model, path, None) else {
                continue;
            };
            if let Some((prior, fingerprint)) = prefer {
                if route.prefill.fingerprint != fingerprint {
                    continue;
                }
                if prior == path {
                    return Ok(Some(path));
                }
            }
            let weight = u64::from(directory.admission_weights.get(&path).copied().unwrap_or(1));
            if weight == 0 {
                continue;
            }
            for version in [path.prefill, path.decode] {
                let total = capacity.entry(version).or_default();
                *total = total.saturating_add(weight);
            }
            eligible.push((path, weight));
        }
        if eligible.len() <= 1 {
            return Ok(eligible.first().map(|&(path, _)| path));
        }
        let mut groups = BTreeMap::<_, u64>::new();
        let mut paths = BTreeMap::<_, u64>::new();
        if let Some(jobs) = self.decoder.work.get(&key) {
            for job in jobs.values().filter(|job| job.expires_at_ms > now) {
                let path = job.route.path();
                *paths.entry(path).or_default() += 1;
                *groups.entry(path.decode).or_default() += 1;
                if job.depends_on(path.prefill) {
                    *groups.entry(path.prefill).or_default() += 1;
                }
            }
        }
        let mut choice = None;
        for (path, weight) in eligible {
            let pressure = [
                (paths.get(&path).copied().unwrap_or(0) + 1, weight),
                (
                    groups.get(&path.prefill).copied().unwrap_or(0) + 1,
                    capacity[&path.prefill],
                ),
                (
                    groups.get(&path.decode).copied().unwrap_or(0) + 1,
                    capacity[&path.decode],
                ),
            ]
            .into_iter()
            .max_by(|&a, &b| compare(a, b))
            .expect("three pressure inputs");
            if choice.is_none_or(|(_, best)| compare(pressure, best).is_lt()) {
                choice = Some((path, pressure));
            }
        }
        Ok(choice.map(|(path, _)| path))
    }
}

fn compare(a: (u64, u64), b: (u64, u64)) -> std::cmp::Ordering {
    (u128::from(a.0) * u128::from(b.1)).cmp(&(u128::from(b.0) * u128::from(a.1)))
}
