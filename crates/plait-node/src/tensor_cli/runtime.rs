//! Local execution and network dispatch within a Plait node.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use plait_core::serving::{
    DecoderRouteWork, EXECUTION_PULSE_MS, GroupPhase, HostMemory, ResidentOwner, RouteSetId,
    ServingAction, ServingError, ServingLookup, ServingOperation, ServingView, W4ShardLayout,
    WorkerGrant,
};
use plait_core::tensor::KPartial;
use plait_core::{DeviceId, NodeId, OpId};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::task::JoinSet;
use tokio::time::Instant;

use super::control::{ControlPeers, GroupControl};
use super::frame::{read_frame, read_payload, write_frame, write_payload};
use super::w4::worker::{Invocation as WorkerInvocation, Request};

mod process;
mod submit;
pub(super) use process::enter_worker;
use process::{SlotLock, WorkerProcess, lock_device, lock_identity};
pub(super) use submit::run as submit_w4;

const CALL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Deserialize)]
pub(crate) struct Config {
    listen: SocketAddr,
    host_memory: HostMemory,
    lock_dir: PathBuf,
    devices: Vec<Device>,
}

#[derive(Deserialize)]
struct Device {
    ordinal: u32,
    incarnation: u64,
    device_bytes: u64,
    release_check: Option<PathBuf>,
    artifacts: Vec<Artifact>,
}

#[derive(Deserialize)]
struct Artifact {
    launch_digest: [u8; 32],
    #[serde(flatten)]
    program: Program,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Program {
    W4(W4Artifact),
    Decoder {
        decoder_config: PathBuf,
        weights: PathBuf,
    },
}

#[derive(Deserialize)]
struct W4Artifact {
    k_range: [u32; 2],
    group_size: u32,
    output_columns: u32,
    library: PathBuf,
    weights: PathBuf,
}

struct Invocation {
    request: Request,
    response: oneshot::Sender<Value>,
}

type Routes = BTreeMap<DeviceId, mpsc::Sender<Invocation>>;

struct DispatchServer {
    addr: SocketAddr,
    control: ControlPeers,
    routes: Routes,
    submissions: Arc<Semaphore>,
}

#[derive(Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum HostRequest {
    Execute(Box<WorkerInvocation>),
    Release { partial: KPartial },
    Submit(Box<submit::Operator>),
}

enum State {
    Idle,
    Running(WorkerGrant, WorkerProcess),
    Exited(WorkerGrant),
    Recovering(Recovery),
}

struct Recovery {
    grant: WorkerGrant,
    owner: ResidentOwner,
    needs_check: bool,
    next_check: Instant,
}

impl Recovery {
    fn new(grant: WorkerGrant, owner: ResidentOwner, needs_check: bool) -> Self {
        Self {
            grant,
            owner,
            needs_check,
            next_check: Instant::now(),
        }
    }
}

pub(crate) async fn run(
    host: NodeId,
    config: Config,
    peers: BTreeMap<NodeId, String>,
    seeds: Vec<SocketAddr>,
) -> io::Result<()> {
    if config.listen.ip().is_unspecified() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "execution needs a concrete listen IP",
        ));
    }
    let mut ordinals = BTreeSet::new();
    for device in &config.devices {
        let mut plans = BTreeSet::new();
        if !ordinals.insert(device.ordinal)
            || device.ordinal > i32::MAX as u32
            || device.device_bytes == 0
            || device.artifacts.is_empty()
            || device
                .artifacts
                .iter()
                .any(|a| !plans.insert(a.launch_digest))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid local W4 device or duplicate artifact",
            ));
        }
    }
    if config.devices.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "execution needs local devices",
        ));
    }
    tokio::fs::create_dir_all(&config.lock_dir).await?;
    let supervisor_lock = lock_device(&config.lock_dir.join("host-supervisor.lock"))?;
    let pulse = ServingAction::PulseExecutionHost {
        host,
        supervisor_lock: lock_identity(&supervisor_lock)?,
    };
    let listen = SocketAddr::new(config.listen.ip(), 0);
    let control_config = ControlPeers::new(
        peers
            .iter()
            .map(|(id, addr)| format!("{}@{addr}", id.to_hex()))
            .collect(),
        seeds,
        listen.to_string(),
    );
    let mut control = GroupControl::connect(control_config.clone()).await?;
    let listener = TcpListener::bind(config.listen).await?;
    let dispatch_addr = listener.local_addr()?;
    println!(
        "plait W4 execution on {} listening on {}",
        host, dispatch_addr
    );
    register(
        &mut control,
        ServingAction::SetHostMemory {
            host,
            capacity: config.host_memory,
        },
    )
    .await?;
    register(&mut control, pulse.clone()).await?;

    let mut workers = JoinSet::new();
    workers.spawn(renew_execution_host(control, pulse));
    let mut routes = BTreeMap::new();
    for device in config.devices {
        let id = DeviceId::new(host, device.ordinal);
        let lock = config
            .lock_dir
            .join(format!("device-{}.lock", device.ordinal));
        let (sender, inbox) = mpsc::channel(4);
        routes.insert(id, sender);
        workers.spawn(run_device(
            id,
            device,
            dispatch_addr,
            lock,
            control_config.clone(),
            inbox,
        ));
    }
    let server = Arc::new(DispatchServer {
        addr: dispatch_addr,
        control: control_config,
        routes,
        submissions: Arc::new(Semaphore::new(8)),
    });
    let slots = Arc::new(Semaphore::new(64));
    loop {
        tokio::select! {
            result = workers.join_next() => {
                return match result {
                    Some(Ok(result)) => result,
                    Some(Err(error)) => Err(io::Error::other(error)),
                    None => Ok(()),
                };
            }
            incoming = listener.accept() => {
                let (stream, _) = incoming?;
                let Ok(slot) = slots.clone().try_acquire_owned() else { continue };
                let server = server.clone();
                tokio::spawn(async move {
                    let _ = tokio::time::timeout(CALL_TIMEOUT, serve(stream, &server, slot)).await;
                });
            }
        }
    }
}

async fn renew_execution_host(mut control: GroupControl, pulse: ServingAction) -> io::Result<()> {
    let mut tick = tokio::time::interval(Duration::from_millis(EXECUTION_PULSE_MS));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    tick.tick().await;
    loop {
        tick.tick().await;
        if let Err(error) = apply(&mut control, pulse.clone()).await {
            eprintln!("execution host pulse: {error}");
        }
    }
}

async fn run_device(
    id: DeviceId,
    device: Device,
    dispatch_addr: SocketAddr,
    lock: PathBuf,
    config: ControlPeers,
    mut inbox: mpsc::Receiver<Invocation>,
) -> io::Result<()> {
    let supervisor_lock = lock_device(&lock.with_extension("supervisor.lock"))?;
    let lock = SlotLock::open(lock)?;
    let supervisor = ResidentOwner {
        session: NodeId::random(),
        supervisor_lock: lock_identity(&supervisor_lock)?,
        device_lock: lock.identity,
    };
    let mut control = GroupControl::connect(config.clone()).await?;
    register(
        &mut control,
        ServingAction::RegisterWorker {
            device: id,
            incarnation: device.incarnation,
            device_bytes: device.device_bytes,
        },
    )
    .await?;
    let mut state = State::Idle;
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            next = inbox.recv() => {
                let Some(invocation) = next else { return Ok(()) };
                if invocation.response.is_closed() && matches!(invocation.request, Request::Execute(_)) {
                    continue;
                }
                let reply = match &mut state {
                    State::Running(grant, process) if invocation.request.partial().grant == *grant => {
                        process.call(&invocation.request).await
                    }
                    _ => Err(io::Error::other("node has no resident worker for this grant")),
                };
                let _ = invocation.response.send(reply.unwrap_or_else(|error| json!({ "error": error.to_string() })));
            }
            _ = tick.tick() => {
                if let Err(error) = reconcile(&mut state, &device, id, supervisor, &mut control, dispatch_addr, &lock, &config).await {
                    eprintln!("{id:?}: W4 execution reconciliation: {error}");
                }
            }
        }
    }
}

async fn reconcile(
    state: &mut State,
    device: &Device,
    id: DeviceId,
    supervisor: ResidentOwner,
    control: &mut GroupControl,
    dispatch_addr: SocketAddr,
    lock: &SlotLock,
    config: &ControlPeers,
) -> io::Result<()> {
    let grant = match state {
        State::Idle => {
            let ServingView::Resources(resources) = control.query(ServingLookup::Resources).await?
            else {
                return Err(io::Error::other("expected a Plait resource view"));
            };
            let Some(grant) = resources
                .workers
                .into_iter()
                .find(|worker| worker.device == id && worker.incarnation == device.incarnation)
                .and_then(|worker| worker.grant)
            else {
                return Ok(());
            };
            grant
        }
        State::Running(grant, _) | State::Exited(grant) => *grant,
        State::Recovering(recovery) => recovery.grant,
    };
    let ServingView::Group(Some(view)) = control.query(ServingLookup::Group(grant.version)).await?
    else {
        return Ok(());
    };
    if view.grants.get(&id) != Some(&grant) || view.launch_owner.is_some() {
        return Ok(());
    }
    if view.phase == GroupPhase::Retired {
        *state = State::Idle;
        return Ok(());
    }
    if matches!(state, State::Idle) {
        let ServingView::ResidentOwner(owner) =
            control.query(ServingLookup::ResidentOwner(grant)).await?
        else {
            return Err(io::Error::other("expected a resident owner"));
        };
        if let Some(owner) = owner.filter(|owner| *owner != supervisor) {
            if owner.supervisor_lock != supervisor.supervisor_lock
                || owner.device_lock != supervisor.device_lock
            {
                return Err(io::Error::other(
                    "resident grant belongs to a different lock directory or replaced lock file",
                ));
            }
            *state = if view.stopped.contains(&id) {
                State::Exited(grant)
            } else {
                State::Recovering(Recovery::new(grant, owner, true))
            };
        }
    }
    if let State::Running(_, process) = state
        && let Some(status) = process.try_wait()?
    {
        if !status.success() {
            eprintln!("{id:?}: resident worker {grant:?} exited with {status}");
        }
        *state = State::Recovering(Recovery::new(grant, supervisor, !status.success()));
    }
    if let State::Recovering(recovery) = state {
        if recover(
            recovery,
            view.phase,
            device.release_check.as_deref(),
            lock,
            control,
        )
        .await?
        {
            *state = State::Exited(grant);
        }
        return Ok(());
    }
    if matches!(view.phase, GroupPhase::Ready | GroupPhase::Active)
        && matches!(state, State::Running(_, process) if process.is_decoder())
    {
        reconcile_decoder_routes(control, &view.plan.model).await?;
    }
    match view.phase {
        GroupPhase::Preparing | GroupPhase::Ready => {
            if view.ready.contains(&id) {
                return Ok(());
            }
            let Some(artifact) = device
                .artifacts
                .iter()
                .find(|a| a.launch_digest == view.plan.launch_digest)
            else {
                return Ok(());
            };
            if matches!(state, State::Idle) {
                if matches!(artifact.program, Program::Decoder { .. })
                    && (view.grants.len() != 1
                        || !matches!(
                            view.plan.role,
                            plait_core::serving::ServingRole::Prefill
                                | plait_core::serving::ServingRole::Decode
                        ))
                {
                    return Err(io::Error::other(
                        "native decoder needs a single-device Prefill or Decode group",
                    ));
                }
                apply(
                    control,
                    ServingAction::ClaimResident {
                        grant,
                        owner: supervisor,
                        launch_digest: artifact.launch_digest,
                    },
                )
                .await?;
                let listen = SocketAddr::new(dispatch_addr.ip(), 0);
                match WorkerProcess::start(grant, artifact, listen, lock, config).await {
                    Ok(process) => *state = State::Running(grant, process),
                    Err(error) => {
                        *state = State::Recovering(Recovery::new(grant, supervisor, true));
                        return Err(error);
                    }
                }
            }
            if let (State::Running(_, process), Program::W4(layout)) = (state, &artifact.program) {
                process.check_running()?;
                apply(
                    control,
                    ServingAction::WorkerReadyW4 {
                        grant,
                        launch_digest: artifact.launch_digest,
                        addr: dispatch_addr,
                        shard: W4ShardLayout {
                            k_range: layout.k_range,
                            group_size: layout.group_size,
                            output_columns: layout.output_columns,
                        },
                    },
                )
                .await?;
            }
        }
        GroupPhase::Revoked => {
            if matches!(state, State::Running(..)) {
                let State::Running(_, process) = std::mem::replace(state, State::Idle) else {
                    unreachable!()
                };
                match process.stop().await {
                    Ok(()) => *state = State::Exited(grant),
                    Err(error) => {
                        *state = State::Recovering(Recovery::new(grant, supervisor, true));
                        return Err(error);
                    }
                }
            }
            if matches!(state, State::Exited(_)) {
                if !view.stopped.contains(&id) {
                    apply(control, ServingAction::WorkerStopped(grant)).await?;
                }
                if view
                    .grants
                    .keys()
                    .all(|device| *device == id || view.stopped.contains(device))
                {
                    let version = grant.version;
                    // Multiple hosts can observe the last stop concurrently.
                    let origin = NodeId::derive(&format!("plait/w4/retire/{}", version.group.0));
                    let operation = ServingOperation {
                        id: OpId::new(origin, version.epoch),
                        action: ServingAction::Retire {
                            version,
                            workers: view.grants.keys().copied().collect(),
                        },
                    };
                    control
                        .apply(&operation)
                        .await?
                        .map_err(|error| io::Error::other(format!("{error:?}")))?;
                    *state = State::Idle;
                }
            }
        }
        GroupPhase::Retired => unreachable!("retired grant resets the local slot"),
        GroupPhase::Draining => {
            if matches!(state, State::Running(_, process) if process.is_decoder()) {
                expire_decoder(control, grant.version, &view.plan.model).await?;
            } else {
                expire_w4(control, grant.version).await?;
            }
        }
        GroupPhase::Active => {}
    }
    Ok(())
}

async fn reconcile_decoder_routes(control: &mut GroupControl, model: &str) -> io::Result<()> {
    let lookup = ServingLookup::DecoderRouteWork(RouteSetId::for_model(model));
    let ServingView::DecoderRouteWork(work) = control.query(lookup).await? else {
        return Err(io::Error::other("expected decoder route work"));
    };
    let action = match work {
        Some(DecoderRouteWork::Failover(intent)) => ServingAction::FailoverDecoder {
            model: model.to_owned(),
            intent,
        },
        Some(DecoderRouteWork::Switch(intent)) => ServingAction::CommitDecoderSwitch {
            model: model.to_owned(),
            intent,
        },
        None => return Ok(()),
    };
    let operation = ServingOperation {
        id: OpId::new(NodeId::random(), 1),
        action,
    };
    match control.apply(&operation).await? {
        Ok(_)
        | Err(
            ServingError::StaleDirectory { .. }
            | ServingError::InvalidDirectory
            | ServingError::InvalidRoute(_)
            | ServingError::InvalidPhase(_)
            | ServingError::StaleEpoch { .. }
            | ServingError::ActiveEpochChanged { .. },
        ) => Ok(()),
        Err(error) => Err(io::Error::other(format!("{error:?}"))),
    }
}

async fn recover(
    recovery: &mut Recovery,
    phase: GroupPhase,
    release_check: Option<&Path>,
    lock: &SlotLock,
    control: &mut GroupControl,
) -> io::Result<bool> {
    let grant = recovery.grant;
    if phase != GroupPhase::Revoked {
        apply(
            control,
            ServingAction::LoseResident {
                grant,
                owner: recovery.owner,
            },
        )
        .await?;
    }
    if recovery.needs_check && release_check.is_none() {
        return Ok(false);
    }
    if Instant::now() < recovery.next_check {
        return Ok(false);
    }
    recovery.next_check = Instant::now() + Duration::from_secs(1);
    let _lease = match lock.acquire() {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
        Err(error) => return Err(error),
    };
    if recovery.needs_check
        && let Some(check) = release_check
    {
        check_release(check, grant).await?;
    }
    apply(control, ServingAction::WorkerStopped(grant)).await?;
    Ok(true)
}

async fn check_release(program: &Path, grant: WorkerGrant) -> io::Result<()> {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let mut child = Command::new(program)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    let status = tokio::time::timeout(Duration::from_secs(15), async {
        let mut input = child.stdin.take().expect("piped release checker stdin");
        input.write_all(&serde_json::to_vec(&grant)?).await?;
        drop(input);
        child.wait().await
    })
    .await??;
    if !status.success() {
        return Err(io::Error::other(format!(
            "device release check exited with {status}"
        )));
    }
    Ok(())
}

async fn expire_decoder(
    control: &mut GroupControl,
    version: plait_core::serving::GroupVersion,
    model: &str,
) -> io::Result<()> {
    let ServingView::DecoderDrain(Some(status)) =
        control.query(ServingLookup::DecoderDrain(version)).await?
    else {
        return Ok(());
    };
    if let Some((deadline, path)) = status.next_expiry
        && deadline <= super::w4::now_ms()?
    {
        let origin = NodeId::derive(&format!(
            "plait/decoder/expire/{}/{}/{}/{}",
            path.prefill.group.0, path.prefill.epoch, path.decode.group.0, path.decode.epoch
        ));
        let operation = ServingOperation {
            id: OpId::new(origin, deadline),
            action: ServingAction::ExpireDecoder {
                model: model.to_owned(),
                path,
                through_ms: deadline,
            },
        };
        control
            .apply(&operation)
            .await?
            .map_err(|error| io::Error::other(format!("{error:?}")))?;
    }
    Ok(())
}

async fn expire_w4(
    control: &mut GroupControl,
    version: plait_core::serving::GroupVersion,
) -> io::Result<()> {
    let ServingView::W4Drain(Some(status)) = control.query(ServingLookup::W4Drain(version)).await?
    else {
        return Ok(());
    };
    let Some(deadline) = status.next_expiry_ms else {
        return Ok(());
    };
    if deadline <= super::w4::now_ms()? {
        let origin = NodeId::derive(&format!(
            "plait/w4/expire/{}/{}",
            version.group.0, version.epoch
        ));
        let operation = ServingOperation {
            id: OpId::new(origin, deadline),
            action: ServingAction::ExpireW4 {
                version,
                through_ms: deadline,
            },
        };
        control
            .apply(&operation)
            .await?
            .map_err(|error| io::Error::other(format!("{error:?}")))?;
    }
    Ok(())
}

async fn register(control: &mut GroupControl, action: ServingAction) -> io::Result<()> {
    let operation = ServingOperation {
        id: OpId::new(NodeId::random(), 1),
        action,
    };
    loop {
        match control.apply(&operation).await {
            Ok(result) => {
                return result
                    .map(|_| ())
                    .map_err(|error| io::Error::other(format!("{error:?}")));
            }
            Err(error) => eprintln!("waiting for execution registration: {error}"),
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn apply(control: &mut GroupControl, action: ServingAction) -> io::Result<()> {
    let operation = ServingOperation {
        id: OpId::new(NodeId::random(), 1),
        action,
    };
    control
        .apply(&operation)
        .await?
        .map(|_| ())
        .map_err(|error| io::Error::other(format!("{error:?}")))
}

async fn serve(
    mut stream: TcpStream,
    server: &Arc<DispatchServer>,
    slot: OwnedSemaphorePermit,
) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let request = match read_frame(&mut stream).await? {
        HostRequest::Execute(invocation) => Request::Execute(invocation),
        HostRequest::Release { partial } => Request::Release { partial },
        HostRequest::Submit(operator) => {
            drop(slot);
            let Ok(_submission) = server.submissions.clone().try_acquire_owned() else {
                return write_frame(
                    &mut stream,
                    &json!({ "error": "W4 submit endpoint is busy" }),
                )
                .await;
            };
            return submit::serve(&mut stream, server, *operator).await;
        }
    };
    write_frame(&mut stream, &route_local(&server.routes, request).await).await
}

async fn route_local(routes: &Routes, request: Request) -> Value {
    if let Some(worker) = routes.get(&request.partial().grant.device) {
        let (response, pending) = oneshot::channel();
        match worker.try_send(Invocation { request, response }) {
            Ok(()) => pending
                .await
                .unwrap_or_else(|_| json!({ "error": "resident worker disconnected" })),
            Err(_) => json!({ "error": "resident worker busy or unavailable" }),
        }
    } else {
        json!({ "error": "unknown local device" })
    }
}

impl DispatchServer {
    async fn route(&self, addr: SocketAddr, request: Request) -> io::Result<Value> {
        if addr == self.addr {
            return Ok(route_local(&self.routes, request).await);
        }
        request_at(addr, &request).await
    }
}

async fn request_at(addr: SocketAddr, request: &Request) -> io::Result<Value> {
    let mut stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    write_frame(&mut stream, request).await?;
    read_frame(&mut stream).await
}

pub(super) async fn dispatch(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let request: Request = serde_json::from_reader(io::stdin().lock())?;
    let reply: Value = tokio::time::timeout(CALL_TIMEOUT, request_at(addr, &request)).await??;
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &reply)?;
    writeln!(out)?;
    Ok(())
}
