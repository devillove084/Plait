use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::time::Duration;

use bytes::Bytes;
use plait_core::NodeId;
use plait_core::serving::{ServingLookup, ServingView};
use plait_core::tensor::KPartial;
use plait_fabric::reduction::KSum;
use plait_net::tensor::{PartialSource, fetch_partial, partial_size};
use plait_node::Client;
use serde::Deserialize;
use tokio::task::JoinSet;

pub(crate) mod control;
mod frame;
pub(super) mod runtime;
mod w4;

use control::{ControlPeers, GroupControl};

const OPERATIONS: &str = "tensor requires publish, publish-w4, worker-w4, dispatch-w4, submit-w4, profile-w4, select-w4, plan-w4, or sum";

#[derive(Deserialize)]
struct Publish {
    listen: SocketAddr,
    partial: KPartial,
    lifetime_ms: u64,
}

#[derive(Deserialize)]
struct Source {
    addr: SocketAddr,
    partial: KPartial,
}

#[derive(Deserialize)]
struct Sum {
    k: u32,
    group_size: u32,
    timeout_ms: u64,
    sources: Vec<Source>,
    control: Option<ControlPeers>,
    control_peer: Option<String>,
}

enum SumControl {
    Ordered(GroupControl),
    Local(Client, NodeId),
}

pub(super) async fn run(
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let operation = args.next().ok_or(OPERATIONS)?;
    if matches!(operation.as_str(), "select-w4" | "plan-w4") {
        if args.next().is_some() {
            return Err(format!("tensor {operation} reads its request from stdin").into());
        }
        return if operation == "plan-w4" {
            let request = serde_json::from_reader(io::stdin().lock())?;
            w4::plan(request).await
        } else {
            w4::select(serde_json::from_reader(io::stdin().lock())?)
        };
    }
    let config = args.next().ok_or("tensor requires a JSON configuration")?;
    if args.next().is_some() {
        return Err("unexpected tensor argument".into());
    }
    match operation.as_str() {
        "publish" => publish(serde_json::from_str(&config)?).await,
        "publish-w4" | "publish-w4a16" => w4::publish(serde_json::from_str(&config)?).await,
        "worker-w4" => w4::run_worker(serde_json::from_str(&config)?).await,
        "dispatch-w4" => runtime::dispatch(config.parse()?).await,
        "submit-w4" => runtime::submit_w4(config.parse()?).await,
        "profile-w4" => w4::profile(serde_json::from_str(&config)?),
        "sum" => sum(serde_json::from_str(&config)?).await,
        _ => Err(OPERATIONS.into()),
    }
}

async fn publish(command: Publish) -> Result<(), Box<dyn std::error::Error>> {
    let size = partial_size(command.partial)?;
    let mut partial = Vec::with_capacity(size);
    io::stdin()
        .take(size as u64 + 1)
        .read_to_end(&mut partial)?;
    serve_partial(command, partial).await
}

async fn serve_partial(
    command: Publish,
    payload: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = PartialSource::bind(command.listen, command.partial, payload).await?;
    eprintln!("tensor source listening on {}", source.local_addr()?);
    let lifetime = Duration::from_millis(command.lifetime_ms);
    tokio::time::timeout(lifetime, async {
        loop {
            if let Err(error) = source.serve_next(Duration::from_secs(3)).await {
                if retryable_pull_error(&error) {
                    eprintln!("tensor source request failed: {error}");
                    continue;
                }
                return Err(error);
            }
        }
    })
    .await
    .unwrap_or(Ok(()))?;
    Ok(())
}

fn retryable_pull_error(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::InvalidData
            | io::ErrorKind::UnexpectedEof
            | io::ErrorKind::TimedOut
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::BrokenPipe
    )
}

async fn sum(command: Sum) -> Result<(), Box<dyn std::error::Error>> {
    let plan = KSum::new(
        command.k,
        command.group_size,
        command
            .sources
            .iter()
            .map(|source| source.partial)
            .collect(),
    )?;
    let mut control = match (command.control, command.control_peer) {
        (Some(_), Some(_)) => return Err("provide either control or control_peer".into()),
        (Some(peers), None) => Some(SumControl::Ordered(GroupControl::connect(peers).await?)),
        (None, Some(peer)) => {
            let (node, addr) = super::parse_peer(&peer)?;
            let client = Client::new().await?;
            client.add_peer(node, addr);
            Some(SumControl::Local(client, node))
        }
        (None, None) => None,
    };
    let first = command.sources[0].partial;
    check_control(&mut control, &plan, first).await?;
    let output = fetch_and_reduce(
        &plan,
        command.sources,
        Duration::from_millis(command.timeout_ms),
    )
    .await?;
    check_control(&mut control, &plan, first).await?;
    io::stdout().write_all(&output)?;
    Ok(())
}

async fn fetch_and_reduce(
    plan: &KSum,
    sources: Vec<Source>,
    deadline: Duration,
) -> io::Result<Vec<u8>> {
    let mut transfers = JoinSet::new();
    for source in sources {
        partial_size(source.partial)?;
        transfers.spawn(pull_contribution(source, deadline));
    }
    reduce_transfers(plan, &mut transfers).await
}

async fn pull_contribution(source: Source, deadline: Duration) -> io::Result<(KPartial, Bytes)> {
    let bytes = fetch_partial(source.addr, source.partial, deadline)
        .await
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "shard {} from {}: {error}",
                    source.partial.shard.0, source.addr
                ),
            )
        })?;
    Ok((source.partial, bytes))
}

async fn reduce_transfers(
    plan: &KSum,
    transfers: &mut JoinSet<io::Result<(KPartial, Bytes)>>,
) -> io::Result<Vec<u8>> {
    let mut reduction = plan.reduce();
    while let Some(transfer) = transfers.join_next().await {
        let (partial, payload) = transfer??;
        reduction.push(partial, payload).map_err(io::Error::other)?;
    }
    reduction.finish().map_err(io::Error::other)
}

async fn check_control(
    control: &mut Option<SumControl>,
    plan: &KSum,
    first: KPartial,
) -> Result<(), Box<dyn std::error::Error>> {
    let group = match control {
        Some(SumControl::Ordered(control)) => {
            control.check_worker(first.grant, Some(first.op)).await?;
            control.version(plan.version()).await?
        }
        Some(SumControl::Local(client, node)) => {
            let ServingView::Group(Some(group)) = client
                .query_serving(*node, ServingLookup::Group(plan.version()))
                .await?
            else {
                return Err("group version is absent from the control view".into());
            };
            if matches!(
                client
                    .query_serving(*node, ServingLookup::W4Drain(plan.version()))
                    .await?,
                ServingView::W4Drain(Some(_))
            ) {
                return Err("managed W4 reduction requires ordered control".into());
            }
            group
        }
        None => return Ok(()),
    };
    plan.check_grants(&group)?;
    Ok(())
}
