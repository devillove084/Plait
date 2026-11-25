use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use plait_core::NodeId;
use plait_net::probe_voter;
use tokio::task::JoinSet;

/// Probe only the configured addresses. Discovery supplies transport routes;
/// consensus still determines admission and membership.
pub async fn discover_voters(seeds: &[SocketAddr]) -> io::Result<BTreeMap<NodeId, String>> {
    let seeds = seeds.iter().copied().collect::<BTreeSet<_>>();
    if seeds.is_empty()
        || seeds
            .iter()
            .any(|addr| addr.port() == 0 || addr.ip().is_unspecified())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "specify at least one concrete control seed IP and port",
        ));
    }
    let mut requests = JoinSet::new();
    for addr in seeds {
        requests.spawn(async move {
            let id = tokio::time::timeout(Duration::from_secs(1), probe_voter(addr)).await??;
            Ok::<_, io::Error>((id, addr))
        });
    }
    let mut voters = BTreeMap::new();
    while let Some(result) = requests.join_next().await {
        if let Ok(Ok((id, addr))) = result {
            voters
                .entry(id)
                .and_modify(|earlier: &mut SocketAddr| *earlier = (*earlier).min(addr))
                .or_insert(addr);
        }
    }
    if voters.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotConnected,
            "no control seed advertised a voter",
        ));
    }
    Ok(voters
        .into_iter()
        .map(|(id, addr)| (id, addr.to_string()))
        .collect())
}
