use std::io;
use std::time::Duration;

use plait_core::serving::{
    DecoderRoute, EXECUTION_HOST_TIMEOUT_MS, GroupPhase, GroupVersion, PdRoute, RouteSetId,
    ServingAction, ServingError, ServingLookup, ServingOperation, ServingOutcome, ServingView,
};
use tokio::time::Instant;

use super::stream::Delivery;
use super::{Admission, Generation, GroupControl, RPC, Reply, Request, call, id, now_ms};

const POLL: Duration = Duration::from_millis(250);
const RECOVERY_WINDOW: Duration = Duration::from_millis(EXECUTION_HOST_TIMEOUT_MS + 5_000);

#[derive(Clone, Copy)]
enum Stage {
    Prefill,
    Decode { first_token: u32 },
}

pub(super) async fn generate(
    control: &mut GroupControl,
    model: &str,
    admission: &mut Admission,
    input: Generation,
    mut delivery: Option<&mut Delivery>,
) -> io::Result<Vec<u32>> {
    let mut stage = Stage::Prefill;
    let mut deadline = None;
    loop {
        let (endpoint, request) = match stage {
            Stage::Prefill => (
                admission.route.prefill,
                Request::Prefill {
                    job: admission.job,
                    input: input.clone(),
                },
            ),
            Stage::Decode { .. } => (
                admission.route.decode,
                Request::Decode {
                    job: admission.job,
                    max_new_tokens: input.max_new_tokens - 1,
                    stop_tokens: input.stop_tokens.clone(),
                },
            ),
        };
        let attempt = *admission;
        let is_decode = matches!(stage, Stage::Decode { .. });
        let reply = {
            let rpc = async {
                if is_decode && let Some(delivery) = delivery.as_deref_mut() {
                    delivery.receive(model, &attempt, &input).await
                } else {
                    call(endpoint.addr, request, RPC).await
                }
            };
            tokio::pin!(rpc);
            let response = if admission.recoverable {
                tokio::select! {
                    biased;
                    response = &mut rpc => Some(response),
                    () = wait_revoked(control, endpoint.grant.version, None) => None,
                }
            } else {
                Some(rpc.as_mut().await)
            };
            match response {
                Some(Ok(reply)) => reply,
                Some(Err(error))
                    if !admission.recoverable
                        || matches!(
                            error.kind(),
                            io::ErrorKind::InvalidData | io::ErrorKind::InvalidInput
                        ) =>
                {
                    return Err(error);
                }
                response => {
                    let failure = response.and_then(Result::err);
                    let until = *deadline.get_or_insert_with(|| Instant::now() + RECOVERY_WINDOW);
                    let prefill =
                        matches!(stage, Stage::Decode { .. }).then_some(admission.job.path.prefill);
                    // A failed Decode RPC can be a failed pull from Prefill. While
                    // Decode is still running, only its own revocation can interrupt it.
                    let revoked = match &failure {
                        None => true,
                        Some(error) if error.kind() == io::ErrorKind::Other => {
                            any_revoked(control, endpoint.grant.version, prefill).await
                        }
                        Some(_) => tokio::time::timeout_at(
                            until,
                            wait_revoked(control, endpoint.grant.version, prefill),
                        )
                        .await
                        .is_ok(),
                    };
                    if revoked
                        && let Some(next) = recover(control, model, admission, stage, until).await?
                    {
                        stage = next;
                        continue;
                    }
                    match failure {
                        Some(error) => return Err(error),
                        // A completion ordered before revocation can still have a reply.
                        None => rpc.await?,
                    }
                }
            }
        };
        match (stage, reply) {
            (Stage::Prefill, Reply::Prefilled { token, handoff }) => {
                if let Some(delivery) = delivery.as_deref_mut() {
                    delivery.first(token)?;
                }
                if !handoff {
                    return Ok(vec![token]);
                }
                stage = Stage::Decode { first_token: token };
            }
            (Stage::Decode { first_token }, Reply::Generated { tokens }) if delivery.is_none() => {
                let mut result = Vec::with_capacity(1 + tokens.len());
                result.push(first_token);
                result.extend(tokens);
                return Ok(result);
            }
            (Stage::Decode { .. }, Reply::StreamComplete { .. }) => {
                return delivery
                    .as_deref_mut()
                    .map(Delivery::finish)
                    .ok_or_else(|| io::Error::other("unexpected streaming completion"));
            }
            _ => return Err(io::Error::other("reply did not match generation stage")),
        }
    }
}

async fn revoked(control: &mut GroupControl, version: GroupVersion) -> bool {
    matches!(control.query(ServingLookup::Group(version)).await,
        Ok(ServingView::Group(Some(view))) if matches!(view.phase, GroupPhase::Revoked | GroupPhase::Retired))
}

async fn any_revoked(
    control: &mut GroupControl,
    version: GroupVersion,
    other: Option<GroupVersion>,
) -> bool {
    revoked(control, version).await
        || match other {
            Some(other) => revoked(control, other).await,
            None => false,
        }
}

async fn wait_revoked(
    control: &mut GroupControl,
    version: GroupVersion,
    other: Option<GroupVersion>,
) {
    while !any_revoked(control, version, other).await {
        tokio::time::sleep(POLL).await;
    }
}

async fn recover(
    control: &mut GroupControl,
    model: &str,
    admission: &mut Admission,
    stage: Stage,
    until: Instant,
) -> io::Result<Option<Stage>> {
    let lookup = ServingLookup::DecoderRoutes(RouteSetId::for_model(model));
    while Instant::now() < until {
        let current = admission.job.path;
        let routes = tokio::time::timeout_at(until, control.query(lookup)).await;
        if let Ok(Ok(ServingView::DecoderRoutes(Some(directory)))) = routes
            && directory.revision > admission.revision
            && let Ok((lost_prefill, lost_decode)) = tokio::time::timeout_at(until, async {
                (
                    revoked(control, current.prefill).await,
                    revoked(control, current.decode).await,
                )
            })
            .await
            && let Some(route) = directory
                .paths
                .into_iter()
                .find(|route| candidate(admission.route, *route, stage, lost_prefill, lost_decode))
        {
            let next = if lost_prefill { Stage::Prefill } else { stage };
            let operation = ServingOperation {
                id: id(),
                action: recovery_action(
                    model,
                    admission,
                    directory.revision,
                    route.path(),
                    lost_prefill,
                )?,
            };
            match control.apply(&operation).await? {
                Ok(ServingOutcome::DecoderRecoveryAdmitted { revision, route }) => {
                    admission.revision = revision;
                    admission.route = route;
                    admission.job.path = route.path();
                    return Ok(Some(next));
                }
                Err(ServingError::DecoderNotAdmitted | ServingError::InvalidDecoderStage) => {
                    return Ok(None);
                }
                Err(ServingError::StaleDirectory { .. } | ServingError::InvalidRoute(_)) => {}
                other => {
                    return Err(io::Error::other(format!(
                        "decoder recovery rejected: {other:?}"
                    )));
                }
            }
        }
        tokio::time::sleep_until(until.min(Instant::now() + POLL)).await;
    }
    Ok(None)
}

fn candidate(
    old: DecoderRoute,
    new: DecoderRoute,
    stage: Stage,
    lost_prefill: bool,
    lost_decode: bool,
) -> bool {
    (lost_prefill || (lost_decode && matches!(stage, Stage::Decode { .. })))
        && (new.prefill != old.prefill) == lost_prefill
        && (new.decode != old.decode) == lost_decode
        && new.prefill.fingerprint == old.prefill.fingerprint
}

fn recovery_action(
    model: &str,
    admission: &Admission,
    revision: u64,
    replacement: PdRoute,
    restart: bool,
) -> io::Result<ServingAction> {
    let model = model.to_owned();
    let work = admission.job.work;
    let failed = admission.job.path;
    let now_ms = now_ms()?;
    Ok(if let Some(entry) = admission.job.entry {
        ServingAction::RebindDecoderEntry {
            model,
            work,
            entry,
            revision,
            failed,
            replacement,
            now_ms,
            restart_prefill: restart,
        }
    } else if restart {
        ServingAction::RestartDecoderWork {
            model,
            work,
            revision,
            failed,
            replacement,
            now_ms,
        }
    } else {
        ServingAction::RecoverDecoderWork {
            model,
            work,
            revision,
            failed,
            replacement,
            now_ms,
        }
    })
}
