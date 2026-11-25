use std::io;
use std::time::Duration;

use plait_core::serving::{
    DecoderEntryRetry, DecoderReplayPage, DecoderReplayState, DecoderSubmission, PdRoute,
    RouteSetId, ServingAction, ServingError, ServingLookup, ServingOperation, ServingOutcome,
    ServingView,
};
use plait_core::{NodeId, OpId};
use serde::Deserialize;
use serde_json::json;
use tokio::time::Instant;

use super::{
    Admission, ControlPeers, Generation, GroupControl, Job, admission_routes, execute_admission,
    id, now_ms,
    stream::{Delivery, write_event},
};

#[derive(Deserialize)]
struct Follow {
    model: String,
    work: OpId,
    offset: u32,
}

#[derive(Deserialize)]
struct Resume {
    model: String,
    work: OpId,
    entry: OpId,
    offset: u32,
    retry: Option<Retry>,
    #[serde(flatten)]
    input: Generation,
}

#[derive(Deserialize)]
struct Retry {
    series: String,
    sequence: u64,
}

#[derive(Deserialize)]
pub(super) struct Submission {
    id: String,
    sequence: u64,
    #[serde(default)]
    offset: u32,
}

impl TryFrom<&Retry> for DecoderEntryRetry {
    type Error = io::Error;

    fn try_from(retry: &Retry) -> io::Result<Self> {
        let series = NodeId::from_hex(&retry.series)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        if retry.sequence == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "retry sequence must be positive",
            ));
        }
        Ok(Self {
            series,
            sequence: retry.sequence,
        })
    }
}

impl Resume {
    fn takeover_action(
        &self,
        revision: u64,
        previous: PdRoute,
        replacement: PdRoute,
        now_ms: u64,
        input_sha256: [u8; 32],
        retry: Option<DecoderEntryRetry>,
    ) -> ServingAction {
        let model = self.model.clone();
        let work = self.work;
        let expected_entry = self.entry;
        let offset = self.offset;
        if let Some(retry) = retry {
            ServingAction::TakeoverDecoderWithRetry {
                model,
                work,
                expected_entry,
                revision,
                previous,
                replacement,
                now_ms,
                offset,
                input_sha256,
                retry,
            }
        } else {
            ServingAction::TakeoverDecoder {
                model,
                work,
                expected_entry,
                revision,
                previous,
                replacement,
                now_ms,
                offset,
                input_sha256,
            }
        }
    }
}

pub(in crate::tensor_cli) async fn resume(
    config: ControlPeers,
) -> Result<(), Box<dyn std::error::Error>> {
    let request: Resume = serde_json::from_reader(io::stdin().lock())?;
    let retry = request
        .retry
        .as_ref()
        .map(DecoderEntryRetry::try_from)
        .transpose()?;
    let mut delivery = Delivery::new(GroupControl::connect(config.clone()).await?, true);
    let mut control = GroupControl::connect(config).await?;
    let model = RouteSetId::for_model(&request.model);
    let input_sha256 = request.input.digest()?;
    let (admission, committed) =
        takeover(&mut control, &request, model, input_sha256, retry).await?;
    continue_generation(
        &mut control,
        &mut delivery,
        &request.model,
        admission,
        request.input,
        committed,
        request.offset,
    )
    .await
}

pub(super) async fn submit(
    control: &mut GroupControl,
    model: &str,
    input: Generation,
    submission: Submission,
    delivery: &mut Delivery,
) -> Result<(), Box<dyn std::error::Error>> {
    let request_id = NodeId::from_hex(&submission.id)?;
    if submission.sequence == 0 {
        return Err("submission sequence must be positive".into());
    }
    let request = DecoderSubmission {
        id: request_id,
        sequence: submission.sequence,
        offset: submission.offset,
        max_tokens: input.max_new_tokens,
        input_sha256: input.digest()?,
    };
    let work = DecoderSubmission::work(request_id);
    let key = RouteSetId::for_model(model);
    let mut previous = None;
    for _ in 0..3 {
        let entry = id();
        let routes = admission_routes(control, key).await?;
        let operation = ServingOperation {
            id: entry,
            action: ServingAction::SubmitDecoderRouted {
                model: model.to_owned(),
                routes,
                previous,
                now_ms: now_ms()?,
                request,
            },
        };
        let (revision, route, committed, recoverable) = match control.apply(&operation).await? {
            Ok(ServingOutcome::DecoderAdmitted { revision, route }) => (revision, route, 0, false),
            Ok(ServingOutcome::DecoderRecoveryAdmitted { revision, route }) => {
                (revision, route, 0, true)
            }
            Ok(ServingOutcome::DecoderTakenOver {
                revision,
                route,
                committed,
                recoverable,
            }) => (revision, route, committed, recoverable),
            Ok(ServingOutcome::DecoderSubmissionComplete {
                revision,
                path,
                count,
            }) => {
                seed_delivery(control, delivery, key, work, count, request.offset).await?;
                delivery.event(
                    &json!({"event":"complete", "count":count, "revision":revision, "path":path}),
                )?;
                return Ok(());
            }
            Err(ServingError::DecoderSubmissionRouteChanged(path)) => {
                previous = Some(path);
                continue;
            }
            Err(ServingError::StaleDirectory { .. } | ServingError::InvalidRoute(_)) => continue,
            other => {
                return Err(
                    io::Error::other(format!("decoder submission rejected: {other:?}")).into(),
                );
            }
        };
        let job = Job {
            work,
            path: route.path(),
            entry: Some(entry),
        };
        let admission = Admission {
            job,
            revision,
            route,
            recoverable,
        };
        return continue_generation(
            control,
            delivery,
            model,
            admission,
            input,
            committed,
            request.offset,
        )
        .await;
    }
    Err("no compatible active route for decoder submission".into())
}

async fn seed_delivery(
    control: &mut GroupControl,
    delivery: &mut Delivery,
    model: RouteSetId,
    work: OpId,
    committed: u32,
    offset: u32,
) -> io::Result<()> {
    let mut prefix = Vec::with_capacity(committed as usize);
    while prefix.len() < committed as usize {
        let page = read_page(control, model, work, prefix.len() as u32).await?;
        if page.tokens.is_empty() || page.available < committed {
            return Err(io::Error::other("committed decoder prefix is unavailable"));
        }
        let remaining = committed as usize - prefix.len();
        prefix.extend_from_slice(&page.tokens[..page.tokens.len().min(remaining)]);
    }
    delivery.seed_prefix(&prefix, offset)
}

async fn continue_generation(
    control: &mut GroupControl,
    delivery: &mut Delivery,
    model: &str,
    admission: Admission,
    input: Generation,
    committed: u32,
    offset: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let Job { work, entry, path } = admission.job;
    delivery.event(&json!({ "event": "admitted", "work": work, "entry": entry, "path": path }))?;
    seed_delivery(
        control,
        delivery,
        RouteSetId::for_model(model),
        work,
        committed,
        offset,
    )
    .await?;
    execute_admission(control, model, admission, input, Some(delivery)).await
}

async fn takeover(
    control: &mut GroupControl,
    request: &Resume,
    model: RouteSetId,
    input_sha256: [u8; 32],
    retry: Option<DecoderEntryRetry>,
) -> io::Result<(Admission, u32)> {
    for _ in 0..2 {
        let page = read_page(control, model, request.work, 0).await?;
        if page.state != DecoderReplayState::Running {
            return Err(io::Error::other("generation has reached a terminal state"));
        }
        let ServingView::DecoderRoutes(Some(mut directory)) =
            control.query(ServingLookup::DecoderRoutes(model)).await?
        else {
            return Err(io::Error::other("decoder route directory is unavailable"));
        };
        directory
            .paths
            .sort_by_key(|route| route.path() != page.path);
        for route in directory.paths {
            let entry = id();
            let path = route.path();
            let operation = ServingOperation {
                id: entry,
                action: request.takeover_action(
                    directory.revision,
                    page.path,
                    path,
                    now_ms()?,
                    input_sha256,
                    retry,
                ),
            };
            match control.apply(&operation).await? {
                Ok(ServingOutcome::DecoderTakenOver {
                    revision,
                    route,
                    committed,
                    recoverable,
                }) => {
                    let job = Job {
                        work: request.work,
                        path,
                        entry: Some(entry),
                    };
                    return Ok((
                        Admission {
                            job,
                            revision,
                            route,
                            recoverable,
                        },
                        committed,
                    ));
                }
                Err(ServingError::InvalidRoute(_)) => continue,
                Err(ServingError::StaleDirectory { .. } | ServingError::DecoderNotAdmitted) => {
                    break;
                }
                other => {
                    return Err(io::Error::other(format!(
                        "decoder takeover rejected: {other:?}"
                    )));
                }
            }
        }
    }
    Err(io::Error::other(
        "no compatible active route for decoder takeover",
    ))
}

async fn read_page(
    control: &mut GroupControl,
    model: RouteSetId,
    work: OpId,
    offset: u32,
) -> io::Result<DecoderReplayPage> {
    match control
        .query(ServingLookup::DecoderReplay {
            model,
            work,
            offset,
        })
        .await?
    {
        ServingView::DecoderReplay(Some(page)) => Ok(page),
        _ => Err(io::Error::other("decoder replay record is unavailable")),
    }
}

pub(in crate::tensor_cli) async fn follow(
    config: ControlPeers,
) -> Result<(), Box<dyn std::error::Error>> {
    let request: Follow = serde_json::from_reader(io::stdin().lock())?;
    let mut control = GroupControl::connect(config).await?;
    let model = RouteSetId::for_model(&request.model);
    let mut offset = request.offset;
    let started = Instant::now();
    let mut observed = false;
    loop {
        let lookup = ServingLookup::DecoderReplay {
            model,
            work: request.work,
            offset,
        };
        let response =
            tokio::time::timeout_at(started + Duration::from_secs(125), control.query(lookup))
                .await;
        if let Ok(Ok(ServingView::DecoderReplay(Some(page)))) = response {
            observed = true;
            if !page.tokens.is_empty() {
                write_event(
                    &mut io::stdout().lock(),
                    &json!({ "event": "tokens", "offset": offset, "tokens": page.tokens }),
                )?;
                offset += page.tokens.len() as u32;
                continue;
            }
            if page.state != DecoderReplayState::Running {
                if offset != page.available {
                    return Err("replay offset exceeds the recorded output".into());
                }
                if page.state == DecoderReplayState::Failed {
                    return Err("generation stopped before completion".into());
                }
                write_event(
                    &mut io::stdout().lock(),
                    &json!({
                        "event": "complete", "count": page.available, "revision": page.revision, "path": page.path,
                    }),
                )?;
                return Ok(());
            }
        } else if matches!(response, Ok(Ok(ServingView::DecoderReplay(None))))
            && (observed || started.elapsed() >= Duration::from_secs(3))
        {
            return Err("replay record is unavailable".into());
        }
        if started.elapsed() >= Duration::from_secs(125) {
            return Err("replay timed out waiting for the remaining output".into());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
