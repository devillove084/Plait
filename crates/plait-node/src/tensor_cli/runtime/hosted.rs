//! Start assigned vLLM groups using installed host profiles.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use plait_core::serving::{
    GroupPhase, GroupVersion, HostMemory, HostedInventory, HostedManagedAssignment, ServingAction,
    ServingLookup, ServingView,
};
use plait_core::{DeviceId, NodeId};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, ChildStdin, Command};

use super::{
    ControlPeers, GroupControl, apply, lock_device, lock_identity, register, renew_execution_host,
};

mod probes;
mod scaling;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    listen: SocketAddr,
    probe_listen: Option<SocketAddr>,
    host_memory: HostMemory,
    lock_dir: PathBuf,
    launcher: Vec<String>,
    devices: Vec<Device>,
    profiles: BTreeMap<String, EngineProfile>,
    #[serde(default)]
    pd_scaling: Vec<scaling::Policy>,
}

#[derive(Deserialize)]
struct Device {
    ordinal: u32,
    incarnation: u64,
    device_bytes: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EngineProfile {
    argv: Vec<String>,
    model_path: String,
    device_env: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    verify_release: Option<PathBuf>,
}

struct Launch {
    child: Child,
    _lease: ChildStdin,
}

pub(crate) async fn run(
    host: NodeId,
    mut config: Config,
    peers: BTreeMap<NodeId, String>,
    seeds: Vec<SocketAddr>,
) -> io::Result<()> {
    let unique: BTreeSet<_> = config.devices.iter().map(|device| device.ordinal).collect();
    if config.listen.ip().is_unspecified()
        || config.listen.ip().is_multicast()
        || config
            .probe_listen
            .is_some_and(|address| address.ip() != config.listen.ip())
        || config.launcher.first().is_none_or(String::is_empty)
        || config.profiles.is_empty()
        || config.devices.is_empty()
        || unique.len() != config.devices.len()
        || config.devices.iter().any(|device| device.device_bytes == 0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid hosted execution configuration",
        ));
    }
    scaling::validate(&config.pd_scaling)?;
    let policies = std::mem::take(&mut config.pd_scaling);
    tokio::fs::create_dir_all(&config.lock_dir).await?;
    let lock = lock_device(&config.lock_dir.join("host-supervisor.lock"))?;
    let inventory = HostedInventory {
        host,
        address: config.listen.ip(),
        profiles: config.profiles.keys().cloned().collect(),
        devices: config
            .devices
            .iter()
            .map(|device| (device.ordinal, device.incarnation))
            .collect(),
    };
    let supervisor_lock = lock_identity(&lock)?;
    let release_profiles: BTreeSet<_> = config
        .profiles
        .iter()
        .filter(|(_, profile)| {
            use std::os::unix::fs::PermissionsExt;
            profile.verify_release.as_ref().is_some_and(|path| {
                std::fs::metadata(path).is_ok_and(|metadata| {
                    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
                })
            })
        })
        .map(|(name, _)| name.clone())
        .collect();
    let pulse = if release_profiles.is_empty() {
        ServingAction::AdvertiseHostedInventory {
            inventory,
            supervisor_lock,
        }
    } else {
        ServingAction::AdvertiseHostedInventoryWithRelease {
            inventory,
            supervisor_lock,
            release_profiles,
        }
    };
    let peers: Vec<_> = peers
        .into_iter()
        .map(|(id, addr)| format!("{id}@{addr}"))
        .collect();
    let listen = SocketAddr::new(config.listen.ip(), 0);
    let peer_config = ControlPeers::new(peers.clone(), seeds.clone(), listen.to_string());
    let mut control = GroupControl::connect(peer_config.clone()).await?;
    register(
        &mut control,
        ServingAction::SetHostMemory {
            host,
            capacity: config.host_memory,
        },
    )
    .await?;
    for device in &config.devices {
        register(
            &mut control,
            ServingAction::RegisterWorker {
                device: DeviceId::new(host, device.ordinal),
                incarnation: device.incarnation,
                device_bytes: device.device_bytes,
            },
        )
        .await?;
    }
    register(&mut control, pulse.clone()).await?;
    println!("plait hosted execution on {host}");
    let heartbeat = GroupControl::connect(peer_config.clone()).await?;
    let network_probes = async {
        if let Some(address) = config.probe_listen {
            loop {
                let result = match GroupControl::connect(peer_config.clone()).await {
                    Ok(control) => probes::run(host, address, control).await,
                    Err(error) => Err(error),
                };
                if let Err(error) = result {
                    eprintln!("hosted probe service: {error}");
                }
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }
        std::future::pending::<io::Result<()>>().await
    };
    tokio::select! {
        result = renew_execution_host(heartbeat, pulse) => result,
        result = supervise(host, &config, &peers, &seeds, &lock, control) => result,
        result = network_probes => result,
        result = scaling::run(policies, peer_config.clone()) => result,
    }
}

async fn supervise(
    host: NodeId,
    config: &Config,
    peers: &[String],
    seeds: &[SocketAddr],
    lock: &File,
    mut control: GroupControl,
) -> io::Result<()> {
    let mut running: BTreeMap<GroupVersion, Launch> = BTreeMap::new();
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let mut stopped = Vec::new();
        for (version, launch) in &mut running {
            if let Some(status) = launch.child.try_wait()? {
                eprintln!(
                    "hosted group {}@{} launcher exited with {status}",
                    version.group.0, version.epoch
                );
                stopped.push(*version);
            }
        }
        for version in stopped {
            running.remove(&version);
        }
        let assignments = match control
            .query(ServingLookup::HostedManagedAssignments(host))
            .await
        {
            Ok(ServingView::HostedManagedAssignments(assignments)) => assignments,
            Ok(_) => return Err(io::Error::other("expected hosted assignments")),
            Err(error) => {
                eprintln!("hosted assignment lookup: {error}");
                continue;
            }
        };
        for deployment in assignments {
            let assignment = &deployment.assignment;
            let version = assignment.version;
            if running.contains_key(&version) {
                continue;
            }
            // The local supervisor lock survives in every live launcher. A
            // claim without a child under this lock cannot restart its epoch.
            if assignment.holder.is_some() {
                if let Err(error) = apply(&mut control, ServingAction::Revoke(version)).await {
                    eprintln!(
                        "retiring lost hosted launcher {}@{}: {error}",
                        version.group.0, version.epoch
                    );
                }
                continue;
            }
            if assignment.phase != GroupPhase::Preparing {
                continue;
            }
            let Some(profile) = config.profiles.get(&assignment.recipe.profile) else {
                continue;
            };
            if !assignment
                .recipe
                .hosts
                .iter()
                .any(|entry| entry.host == host && entry.address == config.listen.ip())
            {
                continue;
            }
            let holder = NodeId::random();
            if let Err(error) = apply(
                &mut control,
                ServingAction::ClaimHostLaunch {
                    version,
                    host,
                    holder,
                    launch_digest: deployment.launch_digest(),
                },
            )
            .await
            {
                eprintln!(
                    "claiming hosted group {}@{}: {error}",
                    version.group.0, version.epoch
                );
                continue;
            }
            match launch(
                host,
                holder,
                config,
                profile,
                &deployment,
                peers,
                seeds,
                lock,
            )
            .await
            {
                Ok(launch) => {
                    running.insert(version, launch);
                }
                Err(error) => {
                    eprintln!(
                        "starting hosted group {}@{}: {error}",
                        version.group.0, version.epoch
                    );
                    let _ = apply(&mut control, ServingAction::Revoke(version)).await;
                }
            }
        }
    }
}

async fn launch(
    host: NodeId,
    holder: NodeId,
    config: &Config,
    profile: &EngineProfile,
    deployment: &HostedManagedAssignment,
    peers: &[String],
    seeds: &[SocketAddr],
    lock: &File,
) -> io::Result<Launch> {
    let assignment = &deployment.assignment;
    let recipe = &assignment.recipe;
    let version = assignment.version;
    let mut command = Command::new(&config.launcher[0]);
    command
        .args(&config.launcher[1..])
        .arg("--resident-hosted")
        .args([
            "--model",
            &assignment.model,
            "--served-model-name",
            &recipe.api_model,
        ])
        .args(["--tp", &assignment.tensor_parallel.to_string()])
        .args([
            "--port-base",
            &recipe.http_port.to_string(),
            "--backend-port",
            &recipe.backend_port.to_string(),
        ])
        .args(["--max-model-len", &recipe.max_model_len.to_string()])
        .args([
            "--gpu-memory-utilization",
            &(f64::from(recipe.gpu_memory_bps) / 10_000.0).to_string(),
        ])
        .args([
            "--group-id",
            &version.group.0.to_hex(),
            "--group-epoch",
            &version.epoch.to_string(),
        ])
        .args([
            "--host-id",
            &host.to_hex(),
            "--control-listen",
            &SocketAddr::new(config.listen.ip(), 0).to_string(),
        ])
        .arg("--control-bin")
        .arg(std::env::current_exe()?)
        .stdin(Stdio::piped());
    if !recipe.prefix_caching {
        command.arg("--no-prefix-caching");
    }
    for peer in peers {
        command.args(["--control-peer", peer]);
    }
    for seed in seeds {
        command.args(["--control-seed", &seed.to_string()]);
    }
    if let Some(path) = &profile.verify_release {
        command.arg("--verify-release").arg(path);
    }
    hold_lock(&mut command, lock)?;
    let mut bootstrap = serde_json::to_vec(&json!({
        "holder": holder, "launch_digest": deployment.launch_digest(),
        "recipe": recipe, "engine": profile, "pd": deployment.pd,
    }))?;
    bootstrap.push(b'\n');
    if bootstrap.len() > 65_536 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "hosted launch record exceeds 64 KiB",
        ));
    }
    let mut child = command.spawn()?;
    let mut input = child.stdin.take().expect("piped launcher stdin");
    tokio::time::timeout(Duration::from_secs(5), input.write_all(&bootstrap)).await??;
    Ok(Launch {
        child,
        _lease: input,
    })
}

#[cfg(unix)]
fn hold_lock(command: &mut Command, lock: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let fd = lock.as_raw_fd();
    command.env("PLAIT_HOSTED_LOCK_FD", fd.to_string());
    // SAFETY: The borrowed file stays open through spawn. fcntl is safe after fork.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn hold_lock(_command: &mut Command, _lock: &File) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "hosted supervisor lock requires Unix",
    ))
}
