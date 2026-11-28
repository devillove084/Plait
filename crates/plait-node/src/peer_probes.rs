//! Resident link sampling for native group placement.

use std::collections::BTreeMap;
use std::future::pending;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use plait_core::group::{GroupAction, GroupLookup, GroupOperation, GroupOutcome, GroupReply};
use plait_core::link::{ProbeEndpoint, ProbeRegistration};
use plait_core::{NodeId, OpId};
use tokio::net::TcpListener;
use tokio::sync::RwLock;

use crate::hosted_probe::{self, Samples};
use crate::tensor_cli::control::{ControlPeers, GroupControl};

const ROUND: Duration = Duration::from_secs(10);
const RETRY: Duration = Duration::from_secs(1);

pub(super) async fn run(
    host: NodeId,
    address: Option<SocketAddr>,
    peers: BTreeMap<NodeId, String>,
    seeds: Vec<SocketAddr>,
    auxiliary: bool,
) -> io::Result<()> {
    let Some(address) = address else {
        return pending().await;
    };
    let result = run_listener(host, address, peers, seeds).await;
    if auxiliary {
        if let Err(error) = result {
            eprintln!("plait-node link probe: {error}");
        }
        return pending().await;
    }
    result
}

async fn run_listener(
    host: NodeId,
    address: SocketAddr,
    mut peers: BTreeMap<NodeId, String>,
    seeds: Vec<SocketAddr>,
) -> io::Result<()> {
    let listener = TcpListener::bind(address).await?;
    let endpoint = ProbeEndpoint {
        host,
        session: NodeId::random(),
        address: listener.local_addr()?,
    };
    if !endpoint.valid() {
        return Err(io::Error::other(
            "link probe requires a concrete unicast address",
        ));
    }
    // A voter may listen on all interfaces. Its own RPC still needs a destination.
    if let Some(local) = peers.get_mut(&host)
        && let Ok(address) = local.parse::<SocketAddr>()
        && address.ip().is_unspecified()
    {
        *local = SocketAddr::new(endpoint.address.ip(), address.port()).to_string();
    }
    let peers = peers
        .into_iter()
        .map(|(node, addr)| format!("{node}@{addr}"))
        .collect();
    let listen = SocketAddr::new(endpoint.address.ip(), 0).to_string();
    let mut control = GroupControl::connect(ControlPeers::new(peers, seeds, listen)).await?;
    let samples = Arc::new(RwLock::new(Vec::new()));
    tokio::select! {
        result = hosted_probe::serve(listener, endpoint.session, samples.clone()) => result,
        result = sample_links(&mut control, endpoint, samples) => result,
    }
}

async fn sample_links(
    control: &mut GroupControl,
    endpoint: ProbeEndpoint,
    samples: Samples,
) -> io::Result<()> {
    let observe = GroupOperation {
        id: OpId::new(endpoint.session, 1),
        action: GroupAction::ObserveProbe(endpoint.host),
    };
    let expected_revision = match apply(control, &observe).await? {
        GroupOutcome::Probe(current) => current.map_or(0, |entry| entry.revision),
        _ => return Err(io::Error::other("expected link probe observation")),
    };
    let publish = GroupOperation {
        id: OpId::new(endpoint.session, 2),
        action: GroupAction::PublishProbe {
            endpoint,
            expected_revision,
        },
    };
    let registered = match apply(control, &publish).await? {
        GroupOutcome::Probe(Some(entry)) if entry.endpoint == endpoint => entry,
        _ => return Err(io::Error::other("expected link probe registration")),
    };
    let jitter_ms = u64::from(endpoint.session.as_bytes()[0]) * 10;
    tokio::time::sleep(Duration::from_millis(jitter_ms)).await;
    let mut interval = tokio::time::interval(ROUND);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        let Ok(GroupReply::LinkProbes(directory)) =
            control.query_group(GroupLookup::LinkProbes).await
        else {
            continue;
        };
        let Some(local) = directory
            .iter()
            .find(|entry| entry.endpoint.host == endpoint.host)
        else {
            continue;
        };
        if local.revision > registered.revision {
            return Err(io::Error::other("link probe was replaced"));
        }
        if *local != registered {
            continue;
        }
        sample_directory(endpoint, directory, &samples).await;
    }
}

async fn sample_directory(
    local: ProbeEndpoint,
    directory: Vec<ProbeRegistration>,
    samples: &Samples,
) {
    let targets: BTreeMap<_, _> = directory
        .into_iter()
        .map(|entry| (entry.endpoint.host, entry.endpoint))
        .filter(|(host, endpoint)| {
            *host != local.host && endpoint.address.is_ipv4() == local.address.is_ipv4()
        })
        .collect();
    samples.write().await.retain(|sample| {
        targets
            .get(&sample.target)
            .is_some_and(|target| target.session == sample.target_session)
    });
    for target in targets.into_values() {
        let sample = hosted_probe::measure(local.address.ip(), target).await;
        let mut current = samples.write().await;
        current.retain(|entry| entry.target != target.host);
        if let Ok(sample) = sample {
            current.push(sample);
        }
    }
}

async fn apply(control: &mut GroupControl, operation: &GroupOperation) -> io::Result<GroupOutcome> {
    loop {
        if let Ok(result) = control.apply_group(operation).await {
            return result
                .map_err(|error| io::Error::other(format!("link probe registration: {error:?}")));
        }
        tokio::time::sleep(RETRY).await;
    }
}
