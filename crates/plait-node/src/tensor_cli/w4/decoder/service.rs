use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use plait_core::serving::{
    DecoderAdmissionRoutes, DecoderCheckpoint, DecoderEndpoint, DecoderKvSource, DecoderOffer,
    DecoderReplayInput, DecoderRoute, DecoderStep, GroupPhase, PdRoute, RouteSetId, ServingAction,
    ServingError, ServingLookup, ServingOperation, ServingOutcome, ServingRole, ServingView,
    WorkerGrant,
};
use plait_core::{NodeId, OpId};
use plait_ops::decoder::{DecoderStack, DecoderStackKv, LinearProjection, TokenHead};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::net::{TcpListener, TcpStream};

use super::{transfer, write_json};
use crate::tensor_cli::w4::now_ms;
use crate::tensor_cli::{
    ControlPeers, GroupControl,
    frame::{read_frame, write_frame},
};

mod recovery;
mod replay;
mod stream;
pub(in crate::tensor_cli) use replay::follow as replay_pd;
pub(in crate::tensor_cli) use replay::resume as resume_pd;

const RPC: Duration = Duration::from_secs(45);
const SERVICE_EXPORTS: usize = 8;

#[derive(Deserialize)]
pub(super) struct Config {
    listen: SocketAddr,
    device_lock: PathBuf,
    control: ControlPeers,
    grant: WorkerGrant,
    launch_digest: [u8; 32],
    #[serde(default)]
    managed: bool,
}

impl Config {
    pub(super) fn grant(&self) -> WorkerGrant {
        self.grant
    }

    pub(super) fn lock(&self, ordinal: i32) -> io::Result<File> {
        if self.listen.ip().is_unspecified()
            || u32::try_from(ordinal).ok() != Some(self.grant.device.ordinal)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "decoder service needs a concrete listen IP and its granted ordinal",
            ));
        }
        crate::tensor_cli::runtime::enter_worker(&self.device_lock)
    }
}

pub(super) struct KvLimits {
    pub(super) layers: u32,
    pub(super) bytes: usize,
    pub(super) tokens: u32,
}

#[derive(Clone, Deserialize, Serialize)]
struct Generation {
    tokens: Vec<u32>,
    max_new_tokens: u32,
    #[serde(default)]
    stop_tokens: Vec<u32>,
}

impl Generation {
    fn digest(&self) -> io::Result<[u8; 32]> {
        Ok(Sha256::digest(serde_json::to_vec(self)?).into())
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct Job {
    work: OpId,
    path: PdRoute,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    entry: Option<OpId>,
}

#[derive(Clone, Copy)]
struct Admission {
    job: Job,
    revision: u64,
    route: DecoderRoute,
    recoverable: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum Request {
    Prefill {
        job: Job,
        input: Generation,
    },
    Decode {
        job: Job,
        max_new_tokens: u32,
        stop_tokens: Vec<u32>,
    },
    DecodeStream {
        job: Job,
        max_new_tokens: u32,
        stop_tokens: Vec<u32>,
        #[serde(default)]
        checkpoint: bool,
        #[serde(default)]
        through_offset: u32,
    },
    Release {
        work: OpId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entry: Option<OpId>,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "response", rename_all = "snake_case")]
enum Reply {
    Prefilled {
        token: u32,
        handoff: bool,
    },
    Generated {
        tokens: Vec<u32>,
    },
    Chunk {
        owner: OpId,
        offset: u32,
        tokens: Vec<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        checkpoint: Option<DecoderCheckpoint>,
    },
    StreamStart {
        owner: OpId,
        offset: u32,
        last_token: u32,
    },
    StreamComplete {
        end_offset: u32,
    },
    Released,
    KvUnavailable {
        message: String,
    },
    Error {
        message: String,
    },
}

struct Worker<'a, P> {
    stack: &'a DecoderStack<P>,
    head: &'a TokenHead<P>,
    endpoint: DecoderEndpoint,
    limits: KvLimits,
    control: GroupControl,
    model: String,
    role: ServingRole,
    exports: BTreeMap<(OpId, Option<OpId>), transfer::Publication>,
}

struct DecodeSession {
    cache: DecoderStackKv,
    first_token: u32,
    owner: OpId,
    source: DecoderKvSource,
    offset: u32,
}

pub(super) async fn serve<P: LinearProjection>(
    config: Config,
    stack: &DecoderStack<P>,
    head: &TokenHead<P>,
    fingerprint: [u8; 32],
    limits: KvLimits,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut control = GroupControl::connect(config.control).await?;
    let group = control.version(config.grant.version).await?;
    let listener = TcpListener::bind(config.listen).await?;
    let endpoint = DecoderEndpoint {
        grant: config.grant,
        addr: listener.local_addr()?,
        fingerprint,
    };
    let ready = if config.managed {
        ServingAction::DecoderReadyManaged {
            endpoint,
            launch_digest: config.launch_digest,
        }
    } else {
        ServingAction::DecoderReady {
            endpoint,
            launch_digest: config.launch_digest,
        }
    };
    apply(&mut control, id(), ready).await?;
    write_json(&json!({ "ready": endpoint }))?;
    let mut worker = Worker {
        stack,
        head,
        endpoint,
        limits,
        control,
        model: group.plan.model,
        role: group.plan.role,
        exports: BTreeMap::new(),
    };
    let mut retirement = tokio::time::interval(Duration::from_millis(250));
    retirement.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        let (mut stream, _) = tokio::select! {
            incoming = listener.accept() => incoming?,
            signal = tokio::signal::ctrl_c() => { signal?; return Ok(()); }
            _ = retirement.tick(), if config.managed => {
                let lookup = ServingLookup::Group(config.grant.version);
                if let Ok(ServingView::Group(Some(view))) = worker.control.query(lookup).await
                    && matches!(view.phase, GroupPhase::Revoked | GroupPhase::Retired)
                {
                    return Ok(());
                }
                continue;
            }
        };
        let Ok(Ok(request)) =
            tokio::time::timeout(Duration::from_secs(3), read_frame(&mut stream)).await
        else {
            continue;
        };
        let reply = match request {
            Request::DecodeStream {
                job,
                max_new_tokens,
                stop_tokens,
                checkpoint,
                through_offset,
            } if worker.role == ServingRole::Decode => {
                worker
                    .stream_decode(
                        &mut stream,
                        job,
                        max_new_tokens,
                        &stop_tokens,
                        checkpoint,
                        through_offset,
                    )
                    .await
            }
            request => worker.handle(request).await,
        }
        .unwrap_or_else(|error| Reply::Error {
            message: error.to_string(),
        });
        let _ =
            tokio::time::timeout(Duration::from_secs(3), write_frame(&mut stream, &reply)).await;
    }
}

impl<P: LinearProjection> Worker<'_, P> {
    async fn handle(&mut self, request: Request) -> io::Result<Reply> {
        self.exports.retain(|_, publication| publication.active());
        match request {
            Request::Prefill { job, input } if self.role == ServingRole::Prefill => {
                self.prefill(job, input).await
            }
            Request::Decode {
                job,
                max_new_tokens,
                stop_tokens,
            } if self.role == ServingRole::Decode => {
                self.decode(job, max_new_tokens, stop_tokens).await
            }
            Request::Release { work, entry } if self.role == ServingRole::Prefill => {
                self.exports.remove(&(work, entry));
                Ok(Reply::Released)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "request does not match the decoder role",
            )),
        }
    }

    async fn advance(
        &mut self,
        operation: OpId,
        job: Job,
        step: DecoderStep,
    ) -> io::Result<ServingOutcome> {
        apply(
            &mut self.control,
            operation,
            ServingAction::AdvanceDecoder {
                model: self.model.clone(),
                work: job.work,
                path: job.path,
                grant: self.endpoint.grant,
                now_ms: now_ms()?,
                step,
            },
        )
        .await
    }

    async fn prefill(&mut self, job: Job, input: Generation) -> io::Result<Reply> {
        if input.max_new_tokens == 0
            || (self.exports.len() >= SERVICE_EXPORTS
                && !self.exports.keys().any(|(work, _)| *work == job.work))
        {
            return Err(io::Error::other(
                "invalid generation limit or prefill export capacity exhausted",
            ));
        }
        let owner = id();
        let step = job.entry.map_or(DecoderStep::StartPrefill, |entry| {
            DecoderStep::StartPrefillEntry { entry }
        });
        self.advance(owner, job, step).await?;
        self.exports.retain(|(work, _), _| *work != job.work);
        let mut cache = self
            .stack
            .empty_cache(self.limits.tokens)
            .map_err(io::Error::other)?;
        let token = self
            .head
            .generate(self.stack, &mut cache, &input.tokens, 1, &input.stop_tokens)
            .map_err(io::Error::other)?[0];
        let handoff = input.max_new_tokens > 1 && !input.stop_tokens.contains(&token);
        if handoff {
            let mut publication = transfer::Publication::default();
            let listen = SocketAddr::new(self.endpoint.addr.ip(), 0);
            let source = publication
                .publish_with_capacity(listen, 120_000, &cache, self.limits.bytes)
                .await?;
            let kv = DecoderOffer {
                source,
                first_token: token,
                prefix_tokens: cache.tokens() as u32,
            };
            self.advance(id(), job, DecoderStep::Offer { owner, kv })
                .await?;
            self.exports.insert((job.work, job.entry), publication);
        } else {
            self.advance(id(), job, DecoderStep::FinishPrefillToken { owner, token })
                .await?;
        }
        Ok(Reply::Prefilled { token, handoff })
    }

    async fn decode(
        &mut self,
        job: Job,
        max_new_tokens: u32,
        stop_tokens: Vec<u32>,
    ) -> io::Result<Reply> {
        let result = self.claim_decode(job, None).await?;
        let DecodeSession {
            mut cache,
            first_token,
            owner,
            ..
        } = match result {
            Ok(session) => session,
            Err(reply) => return Ok(reply),
        };
        let tokens = self
            .head
            .generate(
                self.stack,
                &mut cache,
                &[first_token],
                max_new_tokens,
                &stop_tokens,
            )
            .map_err(io::Error::other)?;
        self.advance(id(), job, DecoderStep::Finish { owner })
            .await?;
        Ok(Reply::Generated { tokens })
    }

    async fn claim_decode(
        &mut self,
        job: Job,
        through_offset: Option<u32>,
    ) -> io::Result<Result<DecodeSession, Reply>> {
        let step = job.entry.map_or(DecoderStep::ReadOffer, |entry| {
            DecoderStep::ReadOfferEntry { entry }
        });
        let ServingOutcome::DecoderOffered(kv) = self.advance(id(), job, step).await? else {
            return Err(io::Error::other(
                "control did not return an admitted KV offer",
            ));
        };
        let mut snapshot = match transfer::pull(
            kv.source,
            self.limits.bytes,
            self.limits.layers,
            Duration::from_secs(15),
        )
        .await
        {
            Ok(snapshot) => snapshot,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::InvalidData | io::ErrorKind::Other
                ) =>
            {
                return Err(error);
            }
            Err(error) => {
                return Ok(Err(Reply::KvUnavailable {
                    message: error.to_string(),
                }));
            }
        };
        if snapshot.tokens != kv.prefix_tokens {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "prefill snapshot has an unexpected length",
            ));
        }
        let (offset, first_token) = if let Some(through) = through_offset {
            self.restore_checkpoints(job, kv, through, &mut snapshot)
                .await
        } else {
            (1, kv.first_token)
        };
        let cache = self
            .stack
            .restore_cache(snapshot.layers, self.limits.tokens)
            .map_err(io::Error::other)?;
        if cache.tokens() != snapshot.tokens as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "admitted decoder prefix length differs from the checkpoint",
            ));
        }
        let owner = id();
        let step = if through_offset.is_some() {
            DecoderStep::ClaimDecodeCheckpoint {
                transfer: kv.source.transfer,
                offset,
            }
        } else {
            DecoderStep::ClaimDecode {
                transfer: kv.source.transfer,
            }
        };
        self.advance(owner, job, step).await?;
        Ok(Ok(DecodeSession {
            cache,
            first_token,
            owner,
            source: kv.source,
            offset,
        }))
    }
}

#[derive(Deserialize)]
struct Submit {
    model: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    replay: bool,
    submission: Option<replay::Submission>,
    #[serde(flatten)]
    input: Generation,
}

pub(in crate::tensor_cli) async fn submit(
    config: ControlPeers,
) -> Result<(), Box<dyn std::error::Error>> {
    let request: Submit = serde_json::from_reader(io::stdin().lock())?;
    if request.input.tokens.is_empty() || request.input.max_new_tokens == 0 {
        return Err("generate-pd needs input tokens and a positive generation limit".into());
    }
    if request.replay && !request.stream {
        return Err("generate-pd replay requires streaming output".into());
    }
    if request.submission.is_some() && !request.replay {
        return Err("generate-pd submission requires streaming replay".into());
    }
    let mut stream = if request.stream {
        Some(stream::Delivery::new(
            GroupControl::connect(config.clone()).await?,
            request.replay,
        ))
    } else {
        None
    };
    let mut control = GroupControl::connect(config).await?;
    if let Some(submission) = request.submission {
        return replay::submit(
            &mut control,
            &request.model,
            request.input,
            submission,
            stream
                .as_mut()
                .expect("submission requires streaming replay"),
        )
        .await;
    }
    let replay_input = request.replay.then_some(&request.input);
    let admission = admit(&mut control, &request.model, replay_input).await?;
    if let Some(stream) = &mut stream
        && let Some(entry) = admission.job.entry
    {
        stream
            .event(&json!({ "event": "admitted", "work": admission.job.work, "entry": entry, "path": admission.job.path }))?;
    }
    execute_admission(
        &mut control,
        &request.model,
        admission,
        request.input,
        stream.as_mut(),
    )
    .await
}

async fn execute_admission(
    control: &mut GroupControl,
    model: &str,
    mut admission: Admission,
    input: Generation,
    mut stream: Option<&mut stream::Delivery>,
) -> Result<(), Box<dyn std::error::Error>> {
    let result =
        recovery::generate(control, model, &mut admission, input, stream.as_deref_mut()).await;
    let Admission {
        job,
        revision,
        route,
        ..
    } = admission;
    if result.is_err() {
        let action = if let Some(entry) = job.entry {
            ServingAction::CancelDecoderEntry {
                model: model.to_owned(),
                work: job.work,
                entry,
                path: job.path,
            }
        } else {
            ServingAction::CancelDecoder {
                model: model.to_owned(),
                work: job.work,
                path: job.path,
            }
        };
        let _ = apply(control, id(), action).await;
    }
    let _ = call(
        route.prefill.addr,
        Request::Release {
            work: job.work,
            entry: job.entry,
        },
        Duration::from_secs(2),
    )
    .await;
    let tokens = result?;
    if let Some(stream) = &mut stream {
        stream.event(&json!({ "event": "complete", "count": tokens.len(), "revision": revision, "path": job.path }))?;
    } else {
        write_json(&json!({ "tokens": tokens, "revision": revision, "path": job.path }))?;
    }
    Ok(())
}

async fn admit(
    control: &mut GroupControl,
    model: &str,
    replay_input: Option<&Generation>,
) -> io::Result<Admission> {
    let replay = replay_input
        .map(|input| -> io::Result<_> {
            Ok(DecoderReplayInput {
                max_tokens: input.max_new_tokens,
                input_sha256: input.digest()?,
            })
        })
        .transpose()?;
    for _ in 0..2 {
        let routes = admission_routes(control, RouteSetId::for_model(model)).await?;
        if routes.paths.is_empty() {
            return Err(io::Error::other("no published decoder route"));
        }
        let work = id();
        let action = ServingAction::AdmitDecoderRouted {
            model: model.to_owned(),
            routes,
            now_ms: now_ms()?,
            replay,
        };
        let operation = ServingOperation { id: work, action };
        let (revision, route, recoverable) = match control.apply(&operation).await? {
            Ok(ServingOutcome::DecoderAdmitted { revision, route }) => (revision, route, false),
            Ok(ServingOutcome::DecoderRecoveryAdmitted { revision, route }) => {
                (revision, route, true)
            }
            Err(ServingError::StaleDirectory { .. } | ServingError::InvalidRoute(_)) => continue,
            other => {
                return Err(io::Error::other(format!(
                    "decoder admission rejected: {other:?}"
                )));
            }
        };
        return Ok(Admission {
            job: Job {
                work,
                path: route.path(),
                entry: replay_input.map(|_| work),
            },
            revision,
            route,
            recoverable,
        });
    }
    Err(io::Error::other("decoder routes changed during admission"))
}

async fn admission_routes(
    control: &mut GroupControl,
    model: RouteSetId,
) -> io::Result<DecoderAdmissionRoutes> {
    match control
        .query(ServingLookup::DecoderAdmissionRoutes(model))
        .await?
    {
        ServingView::DecoderAdmissionRoutes(routes) => Ok(routes),
        _ => Err(io::Error::other("expected decoder admission routes")),
    }
}

async fn call(addr: SocketAddr, request: Request, deadline: Duration) -> io::Result<Reply> {
    tokio::time::timeout(deadline, async {
        let mut stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        write_frame(&mut stream, &request).await?;
        match read_frame(&mut stream).await? {
            Reply::Error { message } => Err(io::Error::other(message)),
            Reply::KvUnavailable { message } => {
                Err(io::Error::new(io::ErrorKind::ConnectionAborted, message))
            }
            reply => Ok(reply),
        }
    })
    .await?
}

fn id() -> OpId {
    OpId::new(NodeId::random(), 1)
}

async fn apply(
    control: &mut GroupControl,
    id: OpId,
    action: ServingAction,
) -> io::Result<ServingOutcome> {
    control
        .apply(&ServingOperation { id, action })
        .await?
        .map_err(|error| io::Error::other(format!("decoder control rejected: {error:?}")))
}
