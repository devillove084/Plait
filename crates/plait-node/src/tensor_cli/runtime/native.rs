//! Publish host liveness for native communication groups.

use std::collections::BTreeMap;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use plait_core::NodeId;
use plait_core::serving::{HostMemory, ServingAction};
use serde::Deserialize;

use super::{
    ControlPeers, GroupControl, lock_device, lock_identity, register, renew_execution_host,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    control_ip: IpAddr,
    host_memory: HostMemory,
    lock_dir: PathBuf,
}

pub(crate) async fn run(
    host: NodeId,
    config: Config,
    mut peers: BTreeMap<NodeId, String>,
    seeds: Vec<SocketAddr>,
) -> io::Result<()> {
    if config.control_ip.is_unspecified()
        || config.control_ip.is_multicast()
        || matches!(config.control_ip, IpAddr::V4(ip) if ip.is_broadcast())
        || config.host_memory.bytes == 0
        || config.host_memory.pinned_bytes > config.host_memory.bytes
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "native execution needs a concrete control IP and valid host memory",
        ));
    }
    if let Some(address) = peers.get_mut(&host)
        && let Ok(socket) = address.parse::<SocketAddr>()
        && socket.ip().is_unspecified()
    {
        *address = SocketAddr::new(config.control_ip, socket.port()).to_string();
    }

    tokio::fs::create_dir_all(&config.lock_dir).await?;
    let lock = lock_device(&config.lock_dir.join("host-supervisor.lock"))?;
    let pulse = ServingAction::PulseExecutionHost {
        host,
        supervisor_lock: lock_identity(&lock)?,
    };
    let peers = peers
        .into_iter()
        .map(|(id, addr)| format!("{id}@{addr}"))
        .collect();
    let listen = SocketAddr::new(config.control_ip, 0).to_string();
    let mut control = GroupControl::connect(ControlPeers::new(peers, seeds, listen)).await?;
    register(
        &mut control,
        ServingAction::SetHostMemory {
            host,
            capacity: config.host_memory,
        },
    )
    .await?;
    register(&mut control, pulse.clone()).await?;
    println!("plait native execution on {host}");
    renew_execution_host(control, pulse).await
}
