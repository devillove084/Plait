use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use plait_core::NodeId;
use plait_core::serving::{HostedProbeEndpoint, ServingAction, ServingLookup, ServingView};
use tokio::net::TcpListener;

use crate::hosted_probe::{self, Samples};

use super::{GroupControl, apply, register};

pub(super) async fn run(
    host: NodeId,
    address: SocketAddr,
    mut control: GroupControl,
) -> io::Result<()> {
    let listener = TcpListener::bind(address).await?;
    let endpoint = HostedProbeEndpoint {
        host,
        session: NodeId::random(),
        address: listener.local_addr()?,
    };
    register(&mut control, ServingAction::RegisterHostedProbe(endpoint)).await?;
    let samples = Samples::default();
    tokio::select! {
        result = hosted_probe::serve(listener, endpoint.session, samples.clone()) => result,
        result = sample_links(endpoint, &mut control, samples) => result,
    }
}

async fn sample_links(
    local: HostedProbeEndpoint,
    control: &mut GroupControl,
    samples: Samples,
) -> io::Result<()> {
    let jitter = u64::from(u16::from_le_bytes([
        local.host.as_bytes()[0],
        local.host.as_bytes()[1],
    ])) % 1000;
    tokio::time::sleep(Duration::from_millis(jitter)).await;
    let mut tick = tokio::time::interval(Duration::from_secs(10));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let endpoints = match control.query(ServingLookup::HostedProbeEndpoints).await {
            Ok(ServingView::HostedProbeEndpoints(endpoints)) => endpoints,
            Ok(_) => return Err(io::Error::other("expected hosted link probe endpoints")),
            Err(error) => {
                eprintln!("hosted probe lookup: {error}");
                continue;
            }
        };
        if !endpoints.contains(&local)
            && let Err(error) = apply(control, ServingAction::RegisterHostedProbe(local)).await
        {
            eprintln!("hosted probe registration: {error}");
            continue;
        }
        let mut round = Vec::new();
        for target in endpoints
            .into_iter()
            .filter(|endpoint| endpoint.host != local.host)
        {
            if let Ok(sample) = hosted_probe::measure(local.address.ip(), target).await {
                round.push(sample);
            }
        }
        *samples.write().await = round;
    }
}
