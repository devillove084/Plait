use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use plait_core::serving::{
    GroupId, GroupPhase, GroupPlan, GroupVersion, GroupView, HostMemory, RouteDirectory,
    RouteSetId, ServingAction, ServingEndpoint, ServingLookup, ServingOperation, ServingResources,
    ServingRole, ServingView, W4GroupLayout, WorkerDemand, WorkerGrant,
};
use plait_core::{DeviceId, NodeId, OpId};
use plait_node::{Client, discover_voters};
use plait_parallel::serving::{
    ServingCandidate, ServingForecast, SloProfile, select_serving_candidate,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

mod replanning;

#[derive(Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum Request {
    Execute {
        operation: ServingOperation,
    },
    Prepare {
        id: OpId,
        group: GroupId,
        expected_epoch: u64,
        plan: NativePlanInput,
    },
    Inspect {
        version: GroupVersion,
    },
    W4Drain {
        version: GroupVersion,
    },
    DecoderDrain {
        version: GroupVersion,
    },
    DecoderClaims {
        version: GroupVersion,
    },
    Resources,
    Routes {
        model: String,
        #[serde(default)]
        minimum_revision: u64,
    },
    DecoderRoutes {
        model: String,
    },
    DecoderFailover {
        model: String,
    },
    DecoderSwitch {
        model: String,
    },
    ExecutionHost {
        host: NodeId,
    },
    Select {
        forecast: ServingForecast,
        candidates: Vec<CandidateInput>,
    },
    ReplanW4 {
        id: OpId,
        forecast: ServingForecast,
        candidates: Vec<W4CandidateInput>,
    },
    ReplanPd {
        id: OpId,
        expected_revision: u64,
        forecast: ServingForecast,
        #[serde(default)]
        candidates: Vec<replanning::Input>,
        pool: Option<plait_parallel::serving::DecoderPool>,
    },
}

#[derive(Deserialize)]
struct W4CandidateInput {
    #[serde(flatten)]
    candidate: CandidateInput,
    layout: W4LayoutInput,
}

#[derive(Deserialize)]
struct W4LayoutInput {
    k: u32,
    group_size: u32,
    output_columns: u32,
    shards: Vec<(DeviceId, [u32; 2])>,
}

impl TryFrom<W4CandidateInput> for (ServingCandidate, W4GroupLayout) {
    type Error = io::Error;

    fn try_from(input: W4CandidateInput) -> Result<Self, Self::Error> {
        let candidate = ServingCandidate::try_from(input.candidate)?;
        let count = input.layout.shards.len();
        let layout = W4GroupLayout {
            k: input.layout.k,
            group_size: input.layout.group_size,
            output_columns: input.layout.output_columns,
            shards: input.layout.shards.into_iter().collect(),
        };
        if candidate.plan.role != ServingRole::Aggregated
            || count != layout.shards.len()
            || !layout.covers(&candidate.plan)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid native W4 candidate layout",
            ));
        }
        Ok((candidate, layout))
    }
}

#[derive(Deserialize)]
struct CandidateInput {
    group: GroupId,
    expected_epoch: u64,
    plan: NativePlanInput,
    capacity: SloProfile,
    ready_after_ms: u64,
    transition_loss_requests: f64,
}

impl TryFrom<CandidateInput> for ServingCandidate {
    type Error = io::Error;

    fn try_from(input: CandidateInput) -> Result<Self, Self::Error> {
        Ok(Self {
            group: input.group,
            expected_epoch: input.expected_epoch,
            plan: input.plan.try_into()?,
            capacity: input.capacity,
            ready_after_ms: input.ready_after_ms,
            transition_loss_requests: input.transition_loss_requests,
        })
    }
}

#[derive(Deserialize, Serialize)]
struct NativePlanInput {
    role: ServingRole,
    model: String,
    tensor_parallel: u32,
    pipeline_parallel: u32,
    launch_digest: [u8; 32],
    workers: Vec<(DeviceId, WorkerDemand)>,
    hosts: Vec<(NodeId, HostMemory)>,
}

impl TryFrom<NativePlanInput> for GroupPlan {
    type Error = io::Error;

    fn try_from(input: NativePlanInput) -> Result<Self, Self::Error> {
        let devices = input.workers.len();
        let host_count = input.hosts.len();
        let workers: BTreeMap<_, _> = input.workers.into_iter().collect();
        let hosts: BTreeMap<_, _> = input.hosts.into_iter().collect();
        if workers.len() != devices || hosts.len() != host_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "duplicate worker or host in serving plan",
            ));
        }
        Ok(Self {
            role: input.role,
            model: input.model,
            tensor_parallel: input.tensor_parallel,
            pipeline_parallel: input.pipeline_parallel,
            launch_digest: input.launch_digest,
            workers,
            hosts,
        })
    }
}

#[derive(Serialize)]
struct GroupInspection {
    version: GroupVersion,
    phase: GroupPhase,
    model: String,
    role: ServingRole,
    tensor_parallel: u32,
    pipeline_parallel: u32,
    launch_digest: [u8; 32],
    launch_owner: Option<NodeId>,
    endpoint: Option<ServingEndpoint>,
    grants: Vec<WorkerGrant>,
}

impl From<GroupView> for GroupInspection {
    fn from(view: GroupView) -> Self {
        Self {
            version: view.version,
            phase: view.phase,
            model: view.plan.model,
            role: view.plan.role,
            tensor_parallel: view.plan.tensor_parallel,
            pipeline_parallel: view.plan.pipeline_parallel,
            launch_digest: view.plan.launch_digest,
            launch_owner: view.launch_owner,
            endpoint: view.endpoint,
            grants: view.grants.into_values().collect(),
        }
    }
}

pub(super) async fn run(
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut listen = "127.0.0.1:0".to_owned();
    let mut peers = Vec::new();
    let mut seeds = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen" => listen = args.next().ok_or("--listen needs an address")?,
            "--peer" => peers.push(super::parse_peer(
                &args.next().ok_or("--peer needs id@addr")?,
            )?),
            "--seed" => seeds.push(super::parse_seed(
                &args.next().ok_or("--seed needs an IP:port")?,
            )?),
            _ => return Err(format!("unknown serving argument {arg}").into()),
        }
    }
    if peers.is_empty() == seeds.is_empty() {
        return Err(
            "serving requires either --peer <hex-id>@<host:port> or --seed <IP:port>".into(),
        );
    }
    if !seeds.is_empty() {
        peers.extend(discover_voters(&seeds).await?);
    }
    let mut client = Client::new_on(&listen).await?;
    for (id, address) in &peers {
        client.add_peer(*id, address);
    }
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let response = match serde_json::from_str::<Request>(&line?) {
            Ok(request) => handle(&mut client, peers.iter().map(|(id, _)| *id), request)
                .await
                .unwrap_or_else(|error| json!({ "error": error.to_string() })),
            Err(error) => json!({ "error": error.to_string() }),
        };
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}

async fn handle(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId> + Clone,
    request: Request,
) -> io::Result<Value> {
    match request {
        Request::Resources => {
            let resources = query_resources(client, peers).await?;
            Ok(json!({ "resources": resources }))
        }
        Request::Routes {
            model,
            minimum_revision,
        } => {
            let directory = query_routes(client, peers, &model, minimum_revision).await?;
            Ok(json!({ "directory": directory.map(directory_json) }))
        }
        Request::DecoderRoutes { model } => {
            let ServingView::DecoderRoutes(directory) = query(
                client,
                peers,
                ServingLookup::DecoderRoutes(RouteSetId::for_model(&model)),
            )
            .await?
            else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected decoder route directory",
                ));
            };
            Ok(json!({ "directory": directory }))
        }
        Request::Select {
            forecast,
            candidates,
        } => {
            let candidates = candidates
                .into_iter()
                .map(ServingCandidate::try_from)
                .collect::<io::Result<Vec<_>>>()?;
            let resources = query_resources(client, peers).await?;
            let result = select_serving_candidate(&resources, &forecast, &candidates)
                .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
            Ok(json!({ "selection": result }))
        }
        Request::ReplanW4 {
            id,
            forecast,
            candidates,
        } => replan_w4(client, peers, id, forecast, candidates).await,
        Request::ReplanPd {
            id,
            expected_revision,
            forecast,
            candidates,
            pool,
        } => {
            replanning::run(
                client,
                peers,
                id,
                expected_revision,
                forecast,
                candidates,
                pool,
            )
            .await
        }
        Request::Inspect { version } => {
            let ServingView::Group(view) =
                query(client, peers, ServingLookup::Group(version)).await?
            else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected serving group",
                ));
            };
            Ok(json!({ "group": view.map(GroupInspection::from) }))
        }
        Request::W4Drain { version } => {
            let ServingView::W4Drain(status) =
                query(client, peers, ServingLookup::W4Drain(version)).await?
            else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected W4 drain status",
                ));
            };
            Ok(json!({ "w4_drain": status }))
        }
        Request::DecoderDrain { version } => {
            let ServingView::DecoderDrain(status) =
                query(client, peers, ServingLookup::DecoderDrain(version)).await?
            else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected decoder drain status",
                ));
            };
            Ok(json!({ "decoder_drain": status }))
        }
        Request::DecoderFailover { model } => {
            let lookup = ServingLookup::DecoderFailover(RouteSetId::for_model(&model));
            let ServingView::DecoderFailover(intent) = query(client, peers, lookup).await? else {
                return Err(io::Error::other("expected decoder failover intent"));
            };
            Ok(json!({ "decoder_failover": intent }))
        }
        Request::DecoderSwitch { model } => {
            let lookup = ServingLookup::DecoderSwitch(RouteSetId::for_model(&model));
            let ServingView::DecoderSwitch(intent) = query(client, peers, lookup).await? else {
                return Err(io::Error::other("expected staged decoder switch"));
            };
            Ok(json!({ "decoder_switch": intent }))
        }
        Request::DecoderClaims { version } => {
            let ServingView::DecoderClaims(claims) =
                query(client, peers, ServingLookup::DecoderClaims(version)).await?
            else {
                return Err(io::Error::other("expected decoder claims"));
            };
            Ok(json!({ "decoder_claims": claims }))
        }
        Request::ExecutionHost { host } => {
            let ServingView::ExecutionHostLiveness(liveness) =
                query(client, peers, ServingLookup::ExecutionHostLiveness(host)).await?
            else {
                return Err(io::Error::other("expected execution host liveness"));
            };
            Ok(json!({ "execution_host": liveness }))
        }
        Request::Execute { operation } => execute(client, peers, operation).await,
        Request::Prepare {
            id,
            group,
            expected_epoch,
            plan,
        } => {
            let action = ServingAction::Prepare {
                group,
                expected_epoch,
                plan: plan.try_into()?,
            };
            execute(client, peers, ServingOperation { id, action }).await
        }
    }
}

async fn replan_w4(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId> + Clone,
    id: OpId,
    mut forecast: ServingForecast,
    inputs: Vec<W4CandidateInput>,
) -> io::Result<Value> {
    let (candidates, layouts): (Vec<_>, Vec<_>) = inputs
        .into_iter()
        .map(TryInto::<(ServingCandidate, W4GroupLayout)>::try_into)
        .collect::<io::Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    let ServingView::Operation(receipt) =
        query(client, peers.clone(), ServingLookup::Operation(id)).await?
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected a serving operation receipt",
        ));
    };
    if let Some(receipt) = receipt {
        if let ServingAction::PrepareW4Switch {
            group,
            expected_epoch,
            plan,
            layout,
            ..
        }
        | ServingAction::PrepareManagedW4 {
            group,
            expected_epoch,
            plan,
            layout,
            ..
        } = &receipt.operation.action
            && candidates
                .iter()
                .zip(&layouts)
                .any(|(candidate, proposed)| {
                    candidate.group == *group
                        && candidate.expected_epoch == *expected_epoch
                        && candidate.plan == *plan
                        && proposed == layout
                })
        {
            return Ok(json!({ "replayed": true, "result": receipt.result }));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "operation ID was used for a different serving action",
        ));
    }
    let resources = query_resources(client, peers.clone()).await?;
    forecast.now_ms = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_millis(),
    )
    .map_err(io::Error::other)?;
    let selection = select_serving_candidate(&resources, &forecast, &candidates)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let Some(index) = selection.selected else {
        return Ok(json!({ "selection": selection, "result": null }));
    };
    let candidate = &candidates[index];
    let expected_active = resources
        .groups
        .iter()
        .find(|group| group.latest.group == candidate.group)
        .and_then(|group| group.active.as_ref())
        .map(|active| active.version.epoch);
    let operation = ServingOperation {
        id,
        action: ServingAction::PrepareManagedW4 {
            group: candidate.group,
            expected_epoch: candidate.expected_epoch,
            expected_active,
            plan: candidate.plan.clone(),
            layout: layouts[index].clone(),
        },
    };
    let result = apply(client, peers, &operation).await?;
    Ok(json!({ "selection": selection, "result": result }))
}

fn directory_json(directory: RouteDirectory) -> Value {
    let version = |value: GroupVersion| format!("{}@{}", value.group.0.to_hex(), value.epoch);
    let paths: Vec<_> = directory
        .paths
        .into_iter()
        .map(|path| {
            json!({
                "prefill": {
                    "url": format!("http://{}", path.prefill_http),
                    "version": version(path.route.prefill),
                    "kv_host": path.prefill_kv.ip().to_string(),
                    "kv_port": path.prefill_kv.port(),
                },
                "decode": {
                    "url": format!("http://{}", path.decode_http),
                    "version": version(path.route.decode),
                },
            })
        })
        .collect();
    json!({ "model": directory.model, "revision": directory.revision, "paths": paths })
}

async fn query_routes(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
    model: &str,
    minimum_revision: u64,
) -> io::Result<Option<RouteDirectory>> {
    let lookup = ServingLookup::Routes(RouteSetId::for_model(model));
    let mut last_error = None;
    let mut unanswered_revision = false;
    for node in peers {
        match tokio::time::timeout(Duration::from_secs(2), client.query_serving(node, lookup)).await
        {
            Ok(Ok(ServingView::Routes(Some(directory)))) => {
                if directory.model != model {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "directory model mismatch",
                    ));
                }
                if directory.revision >= minimum_revision {
                    return Ok(Some(directory));
                }
                unanswered_revision = true;
            }
            Ok(Ok(ServingView::Routes(None))) => unanswered_revision = true,
            Ok(Ok(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected route directory",
                ));
            }
            Ok(Err(error)) => last_error = Some(error),
            Err(error) => last_error = Some(io::Error::new(io::ErrorKind::TimedOut, error)),
        }
    }
    if unanswered_revision {
        return Ok(None);
    }
    Err(last_error.expect("at least one peer"))
}

async fn query_resources(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
) -> io::Result<ServingResources> {
    match query(client, peers, ServingLookup::Resources).await? {
        ServingView::Resources(resources) => Ok(resources),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected serving resources",
        )),
    }
}

async fn query(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
    lookup: ServingLookup,
) -> io::Result<ServingView> {
    let mut last_error = None;
    let mut missing = None;
    for node in peers {
        match tokio::time::timeout(Duration::from_secs(2), client.query_serving(node, lookup)).await
        {
            Ok(Ok(view @ (ServingView::Group(None) | ServingView::Operation(None)))) => {
                missing = Some(view);
            }
            Ok(Ok(view)) => return Ok(view),
            Ok(Err(error)) => last_error = Some(error),
            Err(error) => last_error = Some(io::Error::new(io::ErrorKind::TimedOut, error)),
        }
    }
    if let Some(view) = missing {
        return Ok(view);
    }
    Err(last_error.expect("at least one peer"))
}

async fn execute(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
    operation: ServingOperation,
) -> io::Result<Value> {
    let result = apply(client, peers, &operation).await?;
    Ok(json!({ "result": result }))
}

async fn apply(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
    operation: &ServingOperation,
) -> io::Result<plait_core::serving::ServingResult> {
    let mut last_error = None;
    for node in peers {
        // The same operation ID follows redirects and uncertain submissions.
        match tokio::time::timeout(
            Duration::from_secs(2),
            client.apply_serving(node, operation),
        )
        .await
        {
            Ok(Ok(result)) => return Ok(result),
            Ok(Err(error)) if error.kind() == io::ErrorKind::InvalidData => return Err(error),
            Ok(Err(error)) => last_error = Some(error),
            Err(error) => last_error = Some(io::Error::new(io::ErrorKind::TimedOut, error)),
        }
    }
    Err(last_error.expect("at least one peer"))
}
