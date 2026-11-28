//! Detect disappearance of an enrolled Linux PCI function outside model workers.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::future::pending;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use plait_core::group::{GroupAction, GroupLookup, GroupOperation, GroupReply};
use plait_core::{DeviceId, NodeId, OpId};
use serde::Deserialize;
use tokio::time::{MissedTickBehavior, interval, sleep, timeout};

use crate::tensor_cli::control::{ControlPeers, GroupControl};

const PCI_DEVICES: &str = "/sys/bus/pci/devices";
const PERIOD: Duration = Duration::from_millis(500);
const REQUIRED_ABSENCES: u8 = 2;
const RPC_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    control_ip: IpAddr,
    devices: Vec<Device>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Device {
    ordinal: u32,
    incarnation: u64,
    bdf: String,
}

impl Config {
    fn parse(bytes: &[u8]) -> io::Result<Self> {
        let mut config: Self = serde_json::from_slice(bytes)?;
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if config.control_ip.is_unspecified()
            || config.control_ip.is_multicast()
            || matches!(config.control_ip, IpAddr::V4(ip) if ip.is_broadcast())
        {
            return Err(invalid("PCI watch requires a concrete control IP"));
        }
        let mut ordinals = BTreeSet::new();
        for device in &mut config.devices {
            if !ordinals.insert(device.ordinal) {
                return Err(invalid("PCI watch requires distinct device ordinals"));
            }
            device.bdf =
                canonical_bdf(&device.bdf).ok_or_else(|| invalid("invalid PCI address"))?;
        }
        Ok(config)
    }
}

pub(super) struct Inventory {
    path: PathBuf,
    control_ip: IpAddr,
    contents: Vec<u8>,
    // Keep the last accepted binding even when the ordinal leaves the watch list.
    registrations: BTreeMap<u32, Device>,
    observations: BTreeMap<u32, Observation>,
}

impl Inventory {
    pub(super) fn load(path: PathBuf) -> io::Result<Self> {
        let contents = std::fs::read(&path)?;
        let config = Config::parse(&contents)?;
        let registrations: BTreeMap<_, _> = config
            .devices
            .into_iter()
            .map(|device| (device.ordinal, device))
            .collect();
        let observations = registrations
            .keys()
            .map(|&ordinal| (ordinal, Observation::default()))
            .collect();
        Ok(Self {
            path,
            control_ip: config.control_ip,
            contents,
            registrations,
            observations,
        })
    }

    async fn refresh(&mut self) -> io::Result<bool> {
        let contents = tokio::fs::read(&self.path).await?;
        if contents == self.contents {
            return Ok(false);
        }
        let config = Config::parse(&contents)?;
        if config.control_ip != self.control_ip {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "PCI watch control IP requires a restart to change",
            ));
        }
        for device in &config.devices {
            if let Some(previous) = self.registrations.get(&device.ordinal)
                && (device.incarnation < previous.incarnation
                    || (device.incarnation == previous.incarnation && device != previous))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "PCI watch ordinal {} requires a newer incarnation",
                        device.ordinal
                    ),
                ));
            }
        }
        let mut previous = std::mem::take(&mut self.observations);
        for device in config.devices {
            let ordinal = device.ordinal;
            let observation = if self.registrations.get(&ordinal) == Some(&device) {
                previous.remove(&ordinal).unwrap_or_default()
            } else {
                self.registrations.insert(ordinal, device);
                Observation::default()
            };
            self.observations.insert(ordinal, observation);
        }
        self.contents = contents;
        Ok(true)
    }
}

fn canonical_bdf(value: &str) -> Option<String> {
    fn hex(value: &str, width: usize) -> Option<u32> {
        if value.len() != width || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        u32::from_str_radix(value, 16).ok()
    }
    let (domain, rest) = value.split_once(':')?;
    let (bus, rest) = rest.split_once(':')?;
    let (slot, function) = rest.split_once('.')?;
    if !(4..=8).contains(&domain.len()) {
        return None;
    }
    let domain = hex(domain, domain.len())?;
    let bus = hex(bus, 2)?;
    let slot = hex(slot, 2)?;
    let function = hex(function, 1)?;
    (slot <= 31 && function <= 7).then(|| format!("{domain:04x}:{bus:02x}:{slot:02x}.{function}"))
}

#[derive(Default)]
struct Observation {
    absent: u8,
    reported: bool,
    submitted: Option<GroupOperation>,
}

impl Observation {
    fn sample(&mut self, present: bool) -> bool {
        if present {
            *self = Self::default();
            return false;
        }
        self.absent = self.absent.saturating_add(1);
        self.absent >= REQUIRED_ABSENCES && !self.reported
    }

    async fn reconcile(
        &mut self,
        control: &mut GroupControl,
        device: DeviceId,
        incarnation: u64,
    ) -> io::Result<()> {
        if self.submitted.is_none() {
            let GroupReply::Worker(state) =
                control.query_group(GroupLookup::Worker(device)).await?
            else {
                return Err(io::Error::other("expected worker state"));
            };
            let Some(state) = state.filter(|state| state.incarnation == incarnation) else {
                return Ok(());
            };
            let action = match state.grant {
                None => GroupAction::ReportIdleWorkerUnavailable {
                    device,
                    incarnation,
                },
                Some(grant) if state.has_native_grant => {
                    GroupAction::ReportWorkerUnavailable { grant }
                }
                Some(_) => return Ok(()),
            };
            self.submitted = Some(GroupOperation {
                id: OpId::new(NodeId::random(), 1),
                action,
            });
        }
        let operation = self
            .submitted
            .as_ref()
            .expect("worker observation awaiting a receipt");
        let result = control.apply_group(operation).await?;
        self.submitted = None;
        self.reported = result.is_ok();
        if self.reported {
            println!("plait-node PCI watch withdrew {device} incarnation {incarnation}");
        }
        Ok(())
    }
}

pub(super) async fn run(
    host: NodeId,
    inventory: Option<Inventory>,
    mut peers: BTreeMap<NodeId, String>,
    seeds: Vec<SocketAddr>,
    local_voter: bool,
) -> io::Result<()> {
    let Some(inventory) = inventory else {
        return pending().await;
    };
    if local_voter
        && let Some(address) = peers.get_mut(&host)
        && let Ok(socket) = address.parse::<SocketAddr>()
        && socket.ip().is_unspecified()
    {
        *address = SocketAddr::new(inventory.control_ip, socket.port()).to_string();
    }
    let peers = peers
        .into_iter()
        .map(|(id, address)| format!("{id}@{address}"))
        .collect();
    let listen = SocketAddr::new(inventory.control_ip, 0).to_string();
    let endpoints = ControlPeers::new(peers, seeds, listen);
    let mut last_connect_error = None;
    let control = loop {
        match GroupControl::connect(endpoints.clone()).await {
            Ok(control) => break control,
            Err(error) => {
                let message = error.to_string();
                if last_connect_error.as_ref() != Some(&message) {
                    eprintln!("plait-node PCI watch control connection: {message}");
                    last_connect_error = Some(message);
                }
                sleep(PERIOD).await;
            }
        }
    };
    watch(host, inventory, control, Path::new(PCI_DEVICES)).await
}

fn snapshot(root: &Path) -> io::Result<BTreeSet<OsString>> {
    // A failed directory read is unknown evidence, even if some entries were read.
    std::fs::read_dir(root)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect()
}

async fn watch(
    host: NodeId,
    mut inventory: Inventory,
    mut control: GroupControl,
    root: &Path,
) -> io::Result<()> {
    let mut ticks = interval(PERIOD);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut last_config_error = None;
    let mut last_snapshot_error = None;
    loop {
        ticks.tick().await;
        match inventory.refresh().await {
            Ok(reloaded) => {
                last_config_error = None;
                if reloaded {
                    println!(
                        "plait-node PCI watch reloaded {} enrollments",
                        inventory.observations.len()
                    );
                }
            }
            Err(error) => {
                let message = error.to_string();
                if last_config_error.as_ref() != Some(&message) {
                    eprintln!("plait-node PCI watch could not reload enrollments: {message}");
                    last_config_error = Some(message);
                }
            }
        }
        let current = match snapshot(root) {
            Ok(current) => {
                last_snapshot_error = None;
                current
            }
            Err(error) => {
                let message = error.to_string();
                if last_snapshot_error.as_ref() != Some(&message) {
                    eprintln!(
                        "plait-node PCI watch could not read the device directory: {message}"
                    );
                    last_snapshot_error = Some(message);
                }
                for observation in inventory.observations.values_mut() {
                    observation.absent = 0;
                }
                continue;
            }
        };
        for (&ordinal, observation) in &mut inventory.observations {
            let device = &inventory.registrations[&ordinal];
            if observation.sample(current.contains(OsStr::new(&device.bdf))) {
                let id = DeviceId::new(host, ordinal);
                let _ = timeout(
                    RPC_TIMEOUT,
                    observation.reconcile(&mut control, id, device.incarnation),
                )
                .await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::group::{
        GroupId, GroupMember, GroupOutcome, GroupPhase, GroupVersion, HostMemory, PeerEndpoint,
        ResourcePlan, WorkerDemand,
    };
    use plait_core::serving::{ExecutionDrainTarget, ServingLookup, ServingView};
    use plait_core::{AcceleratorKind, DeviceProfile};
    use plait_node::{Client, NodeConfig, RunningNode};
    use serde_json::json;
    use std::fs;

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("plait-pci-{}", NodeId::random()));
            fs::create_dir(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn replace_config(path: &Path, registrations: &[(u32, u64, &str)]) {
        let devices = registrations
            .iter()
            .map(|(ordinal, incarnation, bdf)| {
                json!({"ordinal": ordinal, "incarnation": incarnation, "bdf": bdf})
            })
            .collect::<Vec<_>>();
        let draft = path.with_extension("next");
        fs::write(
            &draft,
            json!({"control_ip":"127.0.0.1", "devices":devices}).to_string(),
        )
        .unwrap();
        fs::rename(draft, path).unwrap();
    }

    #[test]
    fn addresses_are_canonical_and_cannot_name_another_path() {
        assert_eq!(
            canonical_bdf("00000000:AF:1F.7").as_deref(),
            Some("0000:af:1f.7")
        );
        for invalid in [
            "../../device",
            "0000:af:20.0",
            "0000:af:00.8",
            "+000:af:00.0",
        ] {
            assert_eq!(canonical_bdf(invalid), None);
        }
    }

    #[tokio::test]
    async fn reload_preserves_only_unchanged_active_observations() {
        let temp = TestRoot::new();
        let path = temp.0.join("pci.json");
        let first = "0000:01:00.0";
        let second = "0000:01:01.0";
        replace_config(&path, &[(0, 1, first), (1, 2, second)]);
        let mut inventory = Inventory::load(path.clone()).unwrap();

        let id = OpId::new(NodeId::random(), 1);
        let observation = inventory.observations.get_mut(&0).unwrap();
        observation.sample(false);
        observation.submitted = Some(GroupOperation {
            id,
            action: GroupAction::ReportIdleWorkerUnavailable {
                device: DeviceId::new(id.origin, 0),
                incarnation: 1,
            },
        });
        inventory.observations.get_mut(&1).unwrap().sample(false);

        // One invalid binding rejects a simultaneous valid change.
        replace_config(&path, &[(0, 2, first), (1, 2, first)]);
        assert!(inventory.refresh().await.is_err());
        assert_eq!(inventory.registrations[&0].incarnation, 1);
        assert_eq!(inventory.observations[&1].absent, 1);
        fs::write(&path, "{").unwrap();
        assert!(inventory.refresh().await.is_err());

        replace_config(&path, &[(1, 3, second), (0, 1, "00000000:01:00.0")]);
        assert!(inventory.refresh().await.unwrap());
        assert!(!inventory.refresh().await.unwrap());
        assert_eq!(inventory.observations[&0].absent, 1);
        assert_eq!(
            inventory.observations[&0].submitted.as_ref().unwrap().id,
            id
        );
        assert!(!inventory.observations.get_mut(&1).unwrap().sample(false));
        assert!(inventory.observations.get_mut(&1).unwrap().sample(false));

        replace_config(&path, &[]);
        assert!(inventory.refresh().await.unwrap());
        assert!(inventory.observations.is_empty());
        replace_config(&path, &[(1, 2, second)]);
        assert!(inventory.refresh().await.is_err());
        replace_config(&path, &[(1, 3, second)]);
        assert!(inventory.refresh().await.unwrap());
        assert_eq!(inventory.observations[&1].absent, 0);
    }

    #[tokio::test]
    async fn watcher_withdraws_a_readmitted_worker_after_live_reload() {
        let temp = TestRoot::new();
        let root = temp.0.join("devices");
        fs::create_dir(&root).unwrap();
        let host = NodeId::random();
        let names = [
            "0000:01:00.0",
            "0000:01:01.0",
            "0000:01:02.0",
            "0000:01:03.0",
        ];
        let mut enrollments = [0, 1, 2, 3].map(|i| (i as u32, 1, names[i]));
        for address in &names {
            fs::create_dir(root.join(address)).unwrap();
        }
        let path = temp.0.join("pci.json");
        replace_config(&path, &enrollments);
        let inventory = Inventory::load(path.clone()).unwrap();
        let mut config = NodeConfig::bootstrap(host, "127.0.0.1:0", BTreeMap::new());
        config.advertise_devices = false;
        let node = RunningNode::start(config).await.unwrap();
        let endpoints = ControlPeers::new(
            vec![format!("{host}@{}", node.addr())],
            Vec::new(),
            "127.0.0.1:0".to_owned(),
        );
        let mut control = GroupControl::connect(endpoints.clone()).await.unwrap();
        let memory = HostMemory {
            bytes: 64,
            pinned_bytes: 0,
        };
        apply(
            &mut control,
            GroupAction::SetHostMemory {
                host,
                capacity: memory,
            },
        )
        .await;
        let devices = [0, 1, 2, 3].map(|ordinal| DeviceId::new(host, ordinal));
        for (index, device) in devices.into_iter().enumerate() {
            apply(
                &mut control,
                GroupAction::RegisterWorker {
                    device,
                    incarnation: if index == 3 { 2 } else { 1 },
                    device_bytes: 16,
                },
            )
            .await;
        }
        let incumbent = active(&mut control, devices[0], 1, 0).await;
        let neighbor = active(&mut control, devices[1], 1, 1).await;
        let monitor_root = root.clone();
        let monitor = tokio::spawn(async move {
            let control = GroupControl::connect(endpoints).await.unwrap();
            watch(host, inventory, control, &monitor_root).await
        });
        let hidden = temp.0.join("hidden");
        fs::rename(&root, &hidden).unwrap();
        sleep(PERIOD * 3).await;
        assert_eq!(
            view(&mut control, incumbent).await.phase,
            GroupPhase::Active
        );
        fs::rename(&hidden, &root).unwrap();

        for missing in [0, 2, 3] {
            fs::remove_dir(root.join(&names[missing])).unwrap();
        }
        timeout(Duration::from_secs(6), async {
            loop {
                let GroupReply::Placement(inventory) = control
                    .query_group(GroupLookup::Placement(incumbent.group))
                    .await
                    .unwrap()
                else {
                    unreachable!()
                };
                let eligibility = inventory
                    .workers
                    .iter()
                    .map(|worker| worker.eligible)
                    .collect::<Vec<_>>();
                if eligibility == [false, true, false, true] {
                    break;
                }
                sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            view(&mut control, incumbent).await.phase,
            GroupPhase::Revoked
        );
        assert_eq!(view(&mut control, neighbor).await.phase, GroupPhase::Active);
        let failed_view = view(&mut control, incumbent).await;
        assert!(failed_view.released.is_empty());
        assert_eq!(
            control
                .query_group(GroupLookup::Worker(devices[0]))
                .await
                .unwrap(),
            GroupReply::Worker(Some(plait_core::group::WorkerStateView {
                incarnation: 1,
                grant: Some(failed_view.grants[0]),
                has_native_grant: true,
            })),
        );

        let mut client = Client::new_on("127.0.0.1:0").await.unwrap();
        client.add_peer(host, node.addr());
        let target = ExecutionDrainTarget::Device(devices[0]);
        let ServingView::ExecutionDrain(old_drain) = client
            .query_serving(host, ServingLookup::ExecutionDrain(target))
            .await
            .unwrap()
        else {
            unreachable!()
        };
        apply(&mut control, GroupAction::Release(failed_view.grants[0])).await;
        let replacement = "0000:01:04.0";
        fs::create_dir(root.join(replacement)).unwrap();
        apply(
            &mut control,
            GroupAction::ReadmitWorker {
                profile: DeviceProfile {
                    id: devices[0],
                    name: "test".into(),
                    arch: AcceleratorKind::Nvidia,
                    memory_bytes: 32,
                    flops: 1,
                    compute_units: 1,
                    supports_fp16: true,
                    supports_bf16: true,
                    supports_fp64: false,
                    rdma_capable: false,
                    numa_node: None,
                },
                expected_incarnation: 1,
                incarnation: 2,
                device_bytes: 16,
                drain: old_drain.drain.unwrap(),
            },
        )
        .await;
        enrollments[0] = (0, 2, replacement);
        replace_config(&path, &enrollments);
        let recovered = active(&mut control, devices[0], 2, 2).await;
        let fresh_grant = view(&mut control, recovered).await.grants[0];
        assert!(fresh_grant.fence > failed_view.grants[0].fence);
        sleep(PERIOD * 3).await;
        assert_eq!(
            view(&mut control, recovered).await.phase,
            GroupPhase::Active
        );

        fs::remove_dir(root.join(replacement)).unwrap();
        timeout(Duration::from_secs(6), async {
            while view(&mut control, recovered).await.phase != GroupPhase::Revoked {
                sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(view(&mut control, neighbor).await.phase, GroupPhase::Active);
        let ServingView::ExecutionDrain(new_drain) = client
            .query_serving(host, ServingLookup::ExecutionDrain(target))
            .await
            .unwrap()
        else {
            unreachable!()
        };
        assert!(new_drain.drain.is_some());
        assert_ne!(new_drain.drain, old_drain.drain);
        monitor.abort();
        let _ = monitor.await;
    }

    async fn apply(control: &mut GroupControl, action: GroupAction) -> GroupOutcome {
        control
            .apply_group(&GroupOperation {
                id: OpId::new(NodeId::random(), 1),
                action,
            })
            .await
            .unwrap()
            .unwrap()
    }

    async fn view(
        control: &mut GroupControl,
        version: GroupVersion,
    ) -> plait_core::group::GroupView {
        let GroupReply::Version(Some(view)) = control
            .query_group(GroupLookup::Version(version))
            .await
            .unwrap()
        else {
            unreachable!()
        };
        view
    }

    async fn active(
        control: &mut GroupControl,
        device: DeviceId,
        incarnation: u64,
        slot: u16,
    ) -> GroupVersion {
        let GroupOutcome::Version { version, .. } = apply(
            control,
            GroupAction::Prepare {
                group: GroupId(NodeId::random()),
                expected_epoch: 0,
                resources: ResourcePlan {
                    workers: BTreeMap::from([(
                        device,
                        WorkerDemand {
                            incarnation,
                            device_bytes: 8,
                        },
                    )]),
                    hosts: BTreeMap::from([(
                        device.node,
                        HostMemory {
                            bytes: 8,
                            pinned_bytes: 0,
                        },
                    )]),
                },
            },
        )
        .await
        else {
            unreachable!()
        };
        let grant = view(control, version).await.grants[0];
        apply(
            control,
            GroupAction::PublishEndpoint(GroupMember {
                grant,
                endpoint: PeerEndpoint {
                    address: ([127, 0, 0, 1], 9230 + slot).into(),
                    ticket: [slot as u8 + 1; 16],
                },
            }),
        )
        .await;
        let manifest = view(control, version).await.manifest.unwrap().digest();
        apply(control, GroupAction::Ready { grant, manifest }).await;
        apply(control, GroupAction::Activate(version)).await;
        version
    }
}
