use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use plait_core::serving::{
    GroupId, GroupPhase, GroupPlan, GroupVersion, GroupView, HostMemory, PdRoute, RouteSetId,
    ServingAction, ServingEndpoint, ServingError, ServingLookup, ServingOperation, ServingOutcome,
    ServingResult, ServingRole, ServingView, WorkerDemand, WorkerGrant,
};
use plait_core::{
    ClusterConfig, Command, DeviceId, FuseKind, NodeId, OpId, OpSpec, ShardId, TensorId, TensorKey,
};
use plait_net::TcpTransport;
use plait_node::{Client, JoinTarget, NodeConfig, RunningNode};

fn ids(prefix: &str, count: usize) -> Vec<NodeId> {
    (0..count)
        .map(|index| NodeId::derive(&format!("{prefix}-{index}")))
        .collect()
}

async fn start_bootstrap(ids: &[NodeId]) -> (Vec<RunningNode>, Vec<String>) {
    let mut transports = Vec::new();
    for id in ids {
        transports.push(
            TcpTransport::bind(*id, "127.0.0.1:0", BTreeMap::new())
                .await
                .unwrap(),
        );
    }
    let addrs: Vec<String> = transports
        .iter()
        .map(|transport| transport.local_addr().to_owned())
        .collect();
    let peers: BTreeMap<NodeId, String> = ids.iter().copied().zip(addrs.iter().cloned()).collect();
    let mut nodes = Vec::new();
    for (id, transport) in ids.iter().zip(transports) {
        transport.set_peers(peers.clone());
        let config = NodeConfig {
            id: *id,
            listen: String::new(),
            peers: peers.clone(),
            join: None,
            options: Default::default(),
            advertise_devices: false,
        };
        let cluster = ClusterConfig::new(ids.iter().copied()).unwrap();
        nodes.push(RunningNode::start_with_transport(config, transport, cluster).unwrap());
    }
    (nodes, addrs)
}

macro_rules! wait_until {
    ($what:expr, $check:expr) => {{
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if $check {
                break;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {}", $what);
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }};
}

macro_rules! wait_value {
    ($what:expr, $check:expr) => {{
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(value) = $check {
                break value;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {}", $what);
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }};
}

#[cfg(unix)]
#[tokio::test]
async fn device_discovery_does_not_block_group_control() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::process::Stdio;

    use plait_core::group::{GroupAction, GroupOperation};
    use plait_core::{ComputeRequirement, ComputeTask};
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command as ProcessCommand;

    struct ProbeDirectory(PathBuf);
    impl Drop for ProbeDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    let dir =
        ProbeDirectory(std::env::temp_dir().join(format!("plait-probe-{}", NodeId::random())));
    fs::create_dir(&dir.0).unwrap();
    let probe = dir.0.join("nvidia-smi");
    fs::write(
        &probe,
        concat!(
            "#!/bin/sh\n",
            ": > \"$PLAIT_TEST_PROBE_DIR/started\"\n",
            "while [ -d \"$PLAIT_TEST_PROBE_DIR\" ]; do /bin/sleep 0.05; done\n",
            "exit 1\n",
        ),
    )
    .unwrap();
    fs::set_permissions(probe, fs::Permissions::from_mode(0o700)).unwrap();

    let id = NodeId::random();
    let mut child = ProcessCommand::new(env!("CARGO_BIN_EXE_plait-node"))
        .args(["run", "--listen", "127.0.0.1:0", "--id", &id.to_hex()])
        .env("PATH", &dir.0)
        .env("PLAIT_TEST_PROBE_DIR", &dir.0)
        .env("TOKIO_WORKER_THREADS", "2")
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap()).lines();
    let line = tokio::time::timeout(Duration::from_secs(5), stdout.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let address = line.rsplit_once(" listening on ").unwrap().1;
    wait_until!("vendor probe to block", dir.0.join("started").exists());

    let mut client = Client::new().await.unwrap();
    client.add_peer(id, address);
    let operation = GroupOperation {
        id: OpId::new(id, 1),
        action: GroupAction::SetHostMemory {
            host: id,
            capacity: HostMemory {
                bytes: 64,
                pinned_bytes: 8,
            },
        },
    };
    let receipt = tokio::time::timeout(Duration::from_secs(5), client.apply_group(id, &operation))
        .await
        .expect("control stopped while the vendor probe was running")
        .unwrap();
    assert!(receipt.is_ok());
    drop(dir);

    let task = Command::Compute(ComputeTask {
        task: OpId::new(id, 2),
        kernel: "probe-test".into(),
        device: DeviceId::new(id, u32::MAX),
        requirement: ComputeRequirement::default(),
        inputs: Vec::new(),
        output: TensorKey::new(TensorId(1), ShardId(0)),
        version: 1,
        values: vec![1.0],
    });
    wait_until!(
        "discovered CPU profile to be registered",
        client.submit(id, task.clone()).await.is_ok()
    );
    child.kill().await.unwrap();
}

#[tokio::test]
async fn tcp_cluster_proposes_and_fuses_from_every_node() {
    let ids = ids("tcp-fuse", 3);
    let (_nodes, addrs) = start_bootstrap(&ids).await;
    let mut client = Client::new().await.unwrap();
    for (id, addr) in ids.iter().zip(&addrs) {
        client.add_peer(*id, addr.clone());
    }
    let op = OpId::new(NodeId::derive("tcp-op"), 7);
    let spec = OpSpec {
        id: op,
        kind: FuseKind::Sum,
        contributors: ids.iter().copied().collect(),
        tensor: TensorKey::new(TensorId(9), ShardId(0)),
        epoch: 2,
    };
    client
        .submit(ids[0], Command::DeclareOp(spec))
        .await
        .unwrap();
    wait_until!("declaration", {
        client
            .query(ids[0], op)
            .await
            .ok()
            .is_some_and(|reply| reply.declared)
    });
    let partials = [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]];
    for (index, id) in ids.iter().enumerate() {
        client
            .submit(
                *id,
                Command::Contribute {
                    op,
                    contributor: *id,
                    values: partials[index].to_vec(),
                },
            )
            .await
            .unwrap();
    }
    let values = wait_value!("fused result", {
        client
            .query(ids[2], op)
            .await
            .ok()
            .and_then(|reply| reply.values)
    });
    assert_eq!(values, vec![7.0, 7.0]);
}

#[tokio::test]
async fn a_joining_node_receives_a_snapshot_and_becomes_a_voter() {
    let ids = ids("join", 3);
    let bootstrap = [ids[0], ids[1]];
    let (mut nodes, addrs) = start_bootstrap(&bootstrap).await;
    let mut client = Client::new().await.unwrap();
    for (id, addr) in bootstrap.iter().zip(&addrs) {
        client.add_peer(*id, addr.clone());
    }

    let joining_config = NodeConfig {
        id: ids[2],
        listen: "127.0.0.1:0".to_owned(),
        peers: BTreeMap::new(),
        join: Some(JoinTarget {
            node: ids[0],
            addr: addrs[0].clone(),
        }),
        options: Default::default(),
        advertise_devices: false,
    };
    let joining = RunningNode::start(joining_config).await.unwrap();
    client.add_peer(ids[2], joining.addr().to_owned());

    let op = OpId::new(NodeId::derive("join-op"), 1);
    wait_until!("three voters", {
        client
            .query(ids[2], op)
            .await
            .ok()
            .is_some_and(|reply| reply.voters.len() == 3)
    });
    let spec = OpSpec {
        id: op,
        kind: FuseKind::Max,
        contributors: ids.iter().copied().collect(),
        tensor: TensorKey::new(TensorId(10), ShardId(0)),
        epoch: 3,
    };
    client
        .submit(ids[2], Command::DeclareOp(spec))
        .await
        .unwrap();
    wait_until!("declaration", {
        client
            .query(ids[0], op)
            .await
            .ok()
            .is_some_and(|reply| reply.declared)
    });
    for (index, id) in ids.iter().enumerate() {
        client
            .submit(
                *id,
                Command::Contribute {
                    op,
                    contributor: *id,
                    values: vec![(index + 1) as f64],
                },
            )
            .await
            .unwrap();
    }
    let values = wait_value!("joined result", {
        client
            .query(ids[1], op)
            .await
            .ok()
            .and_then(|reply| reply.values)
    });
    assert_eq!(values, vec![3.0]);

    drop(joining);
    nodes.clear();
}

async fn serving_step(
    client: &mut Client,
    replica: NodeId,
    counter: u64,
    action: ServingAction,
) -> ServingResult {
    let operation = ServingOperation {
        id: OpId::new(NodeId::derive("serving-test-client"), counter),
        action,
    };
    client.apply_serving(replica, &operation).await.unwrap()
}

async fn serving_group(client: &mut Client, replica: NodeId, version: GroupVersion) -> GroupView {
    wait_value!("applied group version", {
        match client
            .query_serving(replica, ServingLookup::Group(version))
            .await
            .unwrap()
        {
            ServingView::Group(group) => group,
            _ => unreachable!("group query returns a view"),
        }
    })
}

fn plan(device: DeviceId, incarnation: u64) -> GroupPlan {
    GroupPlan {
        role: ServingRole::Decode,
        model: "test-model".to_owned(),
        tensor_parallel: 1,
        pipeline_parallel: 1,
        launch_digest: [7; 32],
        workers: BTreeMap::from([(
            device,
            WorkerDemand {
                incarnation,
                device_bytes: 24,
            },
        )]),
        hosts: BTreeMap::from([(
            device.node,
            HostMemory {
                bytes: 32,
                pinned_bytes: 24,
            },
        )]),
    }
}

#[tokio::test]
async fn hosted_launch_switches_tp_width_after_one_control_replica_leaves() {
    let voters = ids("hosted-launch-voter", 3);
    let (mut nodes, addrs) = start_bootstrap(&voters).await;
    let mut client = Client::new().await.unwrap();
    for (id, addr) in voters.iter().zip(addrs) {
        client.add_peer(*id, addr);
    }
    let [http, follower, spare] =
        ["hosted-http", "hosted-follower", "hosted-spare"].map(NodeId::derive);
    let [http_owner, follower_owner, spare_owner] =
        ["owner-http", "owner-follower", "owner-spare"].map(NodeId::derive);
    let group = GroupId(NodeId::derive("hosted-aggregate"));
    let old = GroupVersion { group, epoch: 1 };
    let new = GroupVersion { group, epoch: 2 };
    let old_devices = [
        DeviceId::new(http, 0),
        DeviceId::new(http, 1),
        DeviceId::new(follower, 0),
    ];
    let new_device = DeviceId::new(spare, 0);
    let memory = HostMemory {
        bytes: 64,
        pinned_bytes: 16,
    };
    let make_plan = |devices: &[DeviceId]| GroupPlan {
        role: ServingRole::Aggregated,
        tensor_parallel: devices.len() as u32,
        workers: devices
            .iter()
            .map(|&device| {
                (
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 24,
                    },
                )
            })
            .collect(),
        hosts: devices.iter().map(|device| (device.node, memory)).collect(),
        ..plan(devices[0], 1)
    };
    let mut setup = vec![];
    for host in [http, follower, spare] {
        setup.push(ServingAction::SetHostMemory {
            host,
            capacity: memory,
        });
    }
    for device in old_devices.into_iter().chain([new_device]) {
        setup.push(ServingAction::RegisterWorker {
            device,
            incarnation: 1,
            device_bytes: 64,
        });
    }
    let claim = |version, host, holder| ServingAction::ClaimHostLaunch {
        version,
        host,
        holder,
        launch_digest: [7; 32],
    };
    let ready = |version, host, holder, port: Option<u16>| ServingAction::HostLaunchReady {
        version,
        host,
        holder,
        endpoint: port.map(|port| ServingEndpoint {
            api_model: "test-model".into(),
            http: ([127, 0, 0, 1], port).into(),
            prefill_kv: None,
        }),
    };
    let stop = |version, host, holder| ServingAction::HostLaunchStopped {
        version,
        host,
        holder,
    };
    setup.extend([
        ServingAction::Prepare {
            group,
            expected_epoch: 0,
            plan: make_plan(&old_devices),
        },
        claim(old, http, http_owner),
        claim(old, follower, follower_owner),
        ready(old, http, http_owner, Some(8100)),
        ready(old, follower, follower_owner, None),
        ServingAction::Activate(old),
    ]);
    let mut next = 1;
    for action in setup {
        serving_step(&mut client, voters[0], next, action)
            .await
            .unwrap();
        next += 1;
    }
    drop(nodes.remove(0));

    for action in [
        ServingAction::Prepare {
            group,
            expected_epoch: 1,
            plan: make_plan(&[new_device]),
        },
        claim(new, spare, spare_owner),
        ready(new, spare, spare_owner, Some(8200)),
        ServingAction::Activate(new),
    ] {
        serving_step(&mut client, voters[1], next, action)
            .await
            .unwrap();
        next += 1;
    }
    assert_eq!(
        serving_step(
            &mut client,
            voters[2],
            next,
            ServingAction::ObserveGroup { group, epoch: None }
        )
        .await,
        Ok(ServingOutcome::Group {
            version: new,
            phase: GroupPhase::Active
        }),
    );
    assert_eq!(
        serving_group(&mut client, voters[2], old).await.phase,
        GroupPhase::Draining
    );
    assert_eq!(
        serving_step(
            &mut client,
            voters[2],
            next + 1,
            stop(old, follower, follower_owner)
        )
        .await,
        Err(ServingError::WorkersStillRunning),
    );
    serving_step(
        &mut client,
        voters[1],
        next + 2,
        stop(old, http, http_owner),
    )
    .await
    .unwrap();
    serving_step(
        &mut client,
        voters[2],
        next + 3,
        stop(old, follower, follower_owner),
    )
    .await
    .unwrap();
    serving_step(
        &mut client,
        voters[1],
        next + 4,
        ServingAction::Retire {
            version: old,
            workers: old_devices.into_iter().collect(),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        serving_group(&mut client, voters[1], old).await.phase,
        GroupPhase::Retired
    );
    assert_eq!(
        serving_group(&mut client, voters[2], new).await.phase,
        GroupPhase::Active
    );
}

#[tokio::test]
async fn route_publication_replaces_group_epoch_atomically_through_surviving_voters() {
    let ids = ids("route-publisher", 3);
    let (mut nodes, addrs) = start_bootstrap(&ids).await;
    let mut client = Client::new().await.unwrap();
    for (id, address) in ids.iter().zip(&addrs) {
        client.add_peer(*id, address.clone());
    }
    let host = NodeId::derive("route-host");
    let holder = NodeId::derive("launcher");
    let prefill = GroupId(NodeId::derive("route-prefill"));
    let d = GroupVersion {
        group: GroupId(NodeId::derive("route-decode")),
        epoch: 1,
    };
    let p1 = GroupVersion {
        group: prefill,
        epoch: 1,
    };
    let p2 = GroupVersion {
        group: prefill,
        epoch: 2,
    };
    let before = PdRoute {
        prefill: p1,
        decode: d,
    };
    let after = PdRoute {
        prefill: p2,
        decode: d,
    };
    let model = "route-smoke";
    let lookup = ServingLookup::Routes(RouteSetId::for_model(model));
    serving_step(
        &mut client,
        ids[0],
        1,
        ServingAction::SetHostMemory {
            host,
            capacity: HostMemory {
                bytes: 128,
                pinned_bytes: 96,
            },
        },
    )
    .await
    .unwrap();
    let mut next = 2;
    for (index, (version, role)) in [
        (p1, ServingRole::Prefill),
        (d, ServingRole::Decode),
        (p2, ServingRole::Prefill),
    ]
    .into_iter()
    .enumerate()
    {
        let device = DeviceId::new(host, index as u32);
        for action in [
            ServingAction::RegisterWorker {
                device,
                incarnation: 1,
                device_bytes: 32,
            },
            ServingAction::Prepare {
                group: version.group,
                expected_epoch: version.epoch - 1,
                plan: GroupPlan {
                    role,
                    ..plan(device, 1)
                },
            },
            ServingAction::ClaimLaunch {
                version,
                host,
                holder,
                launch_digest: [7; 32],
            },
            ServingAction::LaunchReadyAt {
                version,
                holder,
                endpoint: ServingEndpoint {
                    api_model: model.into(),
                    http: format!("127.0.0.1:{}", 8100 + index).parse().unwrap(),
                    prefill_kv: (role == ServingRole::Prefill)
                        .then(|| format!("127.0.0.1:{}", 5600 + index).parse().unwrap()),
                },
            },
        ] {
            serving_step(&mut client, ids[0], next, action)
                .await
                .unwrap();
            next += 1;
        }
        if version != p2 {
            serving_step(&mut client, ids[0], next, ServingAction::Activate(version))
                .await
                .unwrap();
            next += 1;
        }
    }

    let publish = |expected_revision, paths, activate| ServingAction::PublishRoutes {
        model: model.into(),
        expected_revision,
        paths,
        activate,
    };
    assert_eq!(
        serving_step(&mut client, ids[0], next, publish(0, vec![before], None)).await,
        Ok(ServingOutcome::RoutesPublished { revision: 1 }),
    );
    assert_eq!(
        serving_step(
            &mut client,
            ids[0],
            next + 1,
            publish(1, vec![after, before], Some(p2))
        )
        .await,
        Err(ServingError::InvalidRoute(before)),
    );
    assert_eq!(
        serving_group(&mut client, ids[0], p1).await.phase,
        GroupPhase::Active
    );
    assert_eq!(
        serving_group(&mut client, ids[0], p2).await.phase,
        GroupPhase::Ready
    );

    drop(nodes.remove(0));
    assert_eq!(
        serving_step(
            &mut client,
            ids[1],
            next + 2,
            publish(1, vec![after], Some(p2))
        )
        .await,
        Ok(ServingOutcome::RoutesPublished { revision: 2 }),
    );
    assert_eq!(
        serving_step(
            &mut client,
            ids[2],
            next + 3,
            publish(1, vec![before], None)
        )
        .await,
        Err(ServingError::StaleDirectory { current: 2 }),
    );
    let ServingView::Routes(Some(directory)) = client.query_serving(ids[2], lookup).await.unwrap()
    else {
        panic!("published directory is missing");
    };
    assert_eq!(directory.revision, 2);
    assert_eq!(directory.paths[0].route, after);
    assert_eq!(directory.paths[0].prefill_kv.port(), 5602);
    assert_eq!(
        serving_group(&mut client, ids[2], p1).await.phase,
        GroupPhase::Draining
    );
    assert_eq!(
        serving_group(&mut client, ids[2], p2).await.phase,
        GroupPhase::Active
    );

    let mut competitor = Client::new().await.unwrap();
    competitor.add_peer(ids[2], addrs[2].clone());
    let (left, right) = tokio::join!(
        serving_step(&mut client, ids[1], next + 4, publish(2, vec![after], None)),
        serving_step(
            &mut competitor,
            ids[2],
            next + 5,
            publish(2, vec![after], None)
        ),
    );
    let published = Ok(ServingOutcome::RoutesPublished { revision: 3 });
    let rejected = Err(ServingError::StaleDirectory { current: 3 });
    assert!((left == published && right == rejected) || (left == rejected && right == published));
    let observe = || ServingAction::ObserveRoutes {
        model: model.into(),
    };
    assert_eq!(
        serving_step(&mut client, ids[2], next + 6, observe()).await,
        Ok(ServingOutcome::RoutesObserved { revision: 3 }),
    );

    serving_step(&mut client, ids[2], next + 7, ServingAction::Drain(d))
        .await
        .unwrap();
    let ServingView::Routes(Some(directory)) = client.query_serving(ids[2], lookup).await.unwrap()
    else {
        panic!("published directory is missing");
    };
    assert_eq!(directory.revision, 3);
    assert!(directory.paths.is_empty());

    // A surviving minority voter can still answer ordinary views, but it
    // cannot renew the router's directory observation.
    drop(nodes.remove(0));
    let rejected_observation = tokio::time::timeout(
        Duration::from_millis(500),
        serving_step(&mut client, ids[2], next + 8, observe()),
    );
    assert!(rejected_observation.await.is_err());
    assert!(matches!(
        client.query_serving(ids[2], lookup).await.unwrap(),
        ServingView::Routes(Some(_))
    ));
}

#[tokio::test]
async fn tensor_admission_follows_the_active_epoch_and_fences_draining_results() {
    use std::io::Write;
    use std::process::{Command as ProcessCommand, Stdio};
    use std::time::{SystemTime, UNIX_EPOCH};

    use plait_net::tensor::PartialSource;
    use plait_parallel::precision::{W4Decision, W4Profile, W4Scope};
    use serde_json::{Value, json};

    async fn tensor(operation: &'static str, request: Value) -> std::process::Output {
        tokio::task::spawn_blocking(move || {
            let mut command = ProcessCommand::new(env!("CARGO_BIN_EXE_plait-node"));
            command
                .args(["tensor", operation])
                .env("TOKIO_WORKER_THREADS", "2");
            if operation == "sum" {
                return command.arg(request.to_string()).output().unwrap();
            }
            let mut child = command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(request.to_string().as_bytes())
                .unwrap();
            child.wait_with_output().unwrap()
        })
        .await
        .unwrap()
    }

    let ids = ids("tensor-voter", 3);
    let (mut nodes, addrs) = start_bootstrap(&ids).await;
    let mut client = Client::new().await.unwrap();
    for (id, address) in ids.iter().zip(&addrs) {
        client.add_peer(*id, address.clone());
    }
    let host = NodeId::derive("tensor-host");
    let group = GroupId(NodeId::derive("tensor-group"));
    let op = OpId::new(NodeId::derive("tensor-request"), 1);
    let first = GroupVersion { group, epoch: 1 };
    let replacement = GroupVersion { group, epoch: 2 };
    let mut counter = 1;
    serving_step(
        &mut client,
        ids[0],
        counter,
        ServingAction::SetHostMemory {
            host,
            capacity: HostMemory {
                bytes: 128,
                pinned_bytes: 96,
            },
        },
    )
    .await
    .unwrap();

    let mut profiles = Vec::new();
    let expiration = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
        + 60_000;
    for (ordinal, version) in [first, replacement].into_iter().enumerate() {
        let device = DeviceId::new(host, ordinal as u32);
        for action in [
            ServingAction::RegisterWorker {
                device,
                incarnation: 1,
                device_bytes: 32,
            },
            ServingAction::Prepare {
                group,
                expected_epoch: version.epoch - 1,
                plan: plan(device, 1),
            },
        ] {
            counter += 1;
            serving_step(&mut client, ids[0], counter, action)
                .await
                .unwrap();
        }
        let grant = serving_group(&mut client, ids[0], version).await.grants[&device];
        counter += 1;
        serving_step(
            &mut client,
            ids[0],
            counter,
            ServingAction::WorkerReady {
                grant,
                launch_digest: [7; 32],
            },
        )
        .await
        .unwrap();
        if version == first {
            counter += 1;
            serving_step(
                &mut client,
                ids[0],
                counter,
                ServingAction::Activate(version),
            )
            .await
            .unwrap();
        }
        profiles.push(W4Profile {
            scope: W4Scope {
                device,
                incarnation: 1,
                ordinal: ordinal as i32,
                library_sha256: [1; 32],
                weights_sha256: [2; 32],
                k_range: [0, 128],
                output_shape: [1, 1],
                group_size: 128,
            },
            calibration_id: [3; 32],
            activation_bits: 16,
            quantization: None,
            kernel: None,
            latency_ns: 100 + ordinal as u64,
            error_ppm: 0,
            valid_until_ms: expiration,
        });
    }
    let peers: Vec<String> = ids
        .iter()
        .zip(&addrs)
        .map(|(id, addr)| format!("{}@{addr}", id.to_hex()))
        .collect();
    let request = || {
        json!({
            "k":128, "group_size":128, "output_shape":[1,1], "op":op,
            "now_ms":0, "calibration_id":([3u8;32]), "max_error_ppm":0,
            "scopes":profiles.iter().map(|profile| profile.scope).collect::<Vec<_>>(),
            "profiles":profiles, "control":{"group":group, "peers":peers}
        })
    };
    let admitted = |value: &std::process::Output| {
        assert!(
            value.status.success(),
            "{}",
            String::from_utf8_lossy(&value.stderr)
        );
        let reply: Value = serde_json::from_slice(&value.stdout).unwrap();
        serde_json::from_value::<W4Decision>(reply["decisions"][0].clone()).unwrap()
    };
    let old = admitted(&tensor("plan-w4", request()).await);
    assert_eq!(old.shard.partial.grant.version, first);

    drop(nodes.remove(0));
    counter += 1;
    serving_step(
        &mut client,
        ids[1],
        counter,
        ServingAction::Activate(replacement),
    )
    .await
    .unwrap();
    let new = admitted(&tensor("plan-w4", request()).await);
    assert_eq!(new.shard.partial.grant.version, replacement);
    assert_eq!(new.profile, profiles[1]);

    let source = PartialSource::bind(
        "127.0.0.1:0".parse().unwrap(),
        old.shard.partial,
        42f32.to_le_bytes().to_vec(),
    )
    .await
    .unwrap();
    let sum = json!({
        "k":128,"group_size":128,"timeout_ms":2000,
        "sources":[{"addr":source.local_addr().unwrap(), "partial":old.shard.partial}],
        "control":{"peers":peers}
    });
    let sender =
        tokio::spawn(async move { source.serve_next(Duration::from_secs(10)).await.unwrap() });
    let drained = tensor("sum", sum.clone()).await;
    assert!(
        drained.status.success(),
        "{}",
        String::from_utf8_lossy(&drained.stderr)
    );
    assert_eq!(drained.stdout, 42f32.to_le_bytes());
    sender.await.unwrap();

    // Hold an already admitted TCP transfer until the revoke is committed.
    let source = PartialSource::bind(
        "127.0.0.1:0".parse().unwrap(),
        old.shard.partial,
        42f32.to_le_bytes().to_vec(),
    )
    .await
    .unwrap();
    let upstream = source.local_addr().unwrap();
    let relay = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut delayed = sum.clone();
    delayed["sources"][0]["addr"] = json!(relay.local_addr().unwrap());
    delayed["timeout_ms"] = json!(6000);
    let (accepted, wait_for_accept) = tokio::sync::oneshot::channel();
    let (release, wait_for_release) = tokio::sync::oneshot::channel();
    let relay_task = tokio::spawn(async move {
        let (mut client, _) = relay.accept().await.unwrap();
        accepted.send(()).unwrap();
        wait_for_release.await.unwrap();
        let mut upstream = tokio::net::TcpStream::connect(upstream).await.unwrap();
        tokio::io::copy_bidirectional(&mut client, &mut upstream)
            .await
            .unwrap();
    });
    let sender =
        tokio::spawn(async move { source.serve_next(Duration::from_secs(10)).await.unwrap() });
    let in_flight = tokio::spawn(tensor("sum", delayed));
    tokio::time::timeout(Duration::from_secs(8), wait_for_accept)
        .await
        .unwrap()
        .unwrap();
    counter += 1;
    serving_step(&mut client, ids[1], counter, ServingAction::Revoke(first))
        .await
        .unwrap();
    release.send(()).unwrap();
    let blocked = in_flight.await.unwrap();
    assert!(!blocked.status.success());
    assert!(blocked.stdout.is_empty());
    assert!(String::from_utf8_lossy(&blocked.stderr).contains("no live worker grant"));
    sender.await.unwrap();
    relay_task.await.unwrap();

    let revoked = tensor("sum", sum).await;
    assert!(!revoked.status.success());
    assert!(String::from_utf8_lossy(&revoked.stderr).contains("no live worker grant"));

    drop(nodes.remove(0));
    let mut minority_request = request();
    minority_request["control"]["peers"] = json!([peers[2]]);
    let minority = tensor("plan-w4", minority_request).await;
    assert!(!minority.status.success());
    assert!(minority.stdout.is_empty());
}

#[tokio::test]
async fn serving_groups_hold_resources_through_fenced_replacement_without_the_original_replica() {
    let ids = ids("serving-voter", 3);
    let (mut nodes, addrs) = start_bootstrap(&ids).await;
    let mut client = Client::new().await.unwrap();
    for (id, addr) in ids.iter().zip(&addrs) {
        client.add_peer(*id, addr.clone());
    }
    let host_a = NodeId::derive("serving-host-a");
    let host_b = NodeId::derive("serving-host-b");
    let a = DeviceId::new(host_a, 0);
    let spare = DeviceId::new(host_a, 1);
    let b = DeviceId::new(host_b, 0);
    let group_a = GroupId(NodeId::derive("serving-group-a"));
    let group_b = GroupId(NodeId::derive("serving-group-b"));
    let a1 = GroupVersion {
        group: group_a,
        epoch: 1,
    };
    let a2 = GroupVersion {
        group: group_a,
        epoch: 2,
    };
    let b1 = GroupVersion {
        group: group_b,
        epoch: 1,
    };
    for (step, host) in [host_a, host_b].into_iter().enumerate() {
        serving_step(
            &mut client,
            ids[0],
            step as u64 + 1,
            ServingAction::SetHostMemory {
                host,
                capacity: HostMemory {
                    bytes: 128,
                    pinned_bytes: 40,
                },
            },
        )
        .await
        .unwrap();
    }
    for (step, device) in [a, spare, b].into_iter().enumerate() {
        serving_step(
            &mut client,
            ids[0],
            step as u64 + 3,
            ServingAction::RegisterWorker {
                device,
                incarnation: 1,
                device_bytes: 48,
            },
        )
        .await
        .unwrap();
    }
    let mut next = 6;
    for (version, device) in [(a1, a), (b1, b)] {
        let prepare = ServingAction::Prepare {
            group: version.group,
            expected_epoch: 0,
            plan: plan(device, 1),
        };
        let expected = Ok(ServingOutcome::Group {
            version,
            phase: GroupPhase::Preparing,
        });
        assert_eq!(
            serving_step(&mut client, ids[0], next, prepare.clone()).await,
            expected
        );
        assert_eq!(
            serving_step(&mut client, ids[0], next, prepare).await,
            expected
        );
        let grant = serving_group(&mut client, ids[0], version).await.grants[&device];
        assert_eq!(grant.fence, 1);
        serving_step(
            &mut client,
            ids[0],
            next + 1,
            ServingAction::WorkerReady {
                grant,
                launch_digest: [7; 32],
            },
        )
        .await
        .unwrap();
        serving_step(
            &mut client,
            ids[0],
            next + 2,
            ServingAction::Activate(version),
        )
        .await
        .unwrap();
        next += 3;
    }

    let old_grant = serving_group(&mut client, ids[2], a1).await.grants[&a];
    assert_eq!(
        serving_step(
            &mut client,
            ids[0],
            next,
            ServingAction::Prepare {
                group: group_a,
                expected_epoch: 1,
                plan: plan(a, 1),
            }
        )
        .await,
        Err(ServingError::WorkerBusy(a))
    );
    assert_eq!(
        serving_step(
            &mut client,
            ids[0],
            next + 1,
            ServingAction::Prepare {
                group: group_a,
                expected_epoch: 1,
                plan: plan(spare, 1),
            }
        )
        .await,
        Err(ServingError::HostCapacityExceeded(host_a))
    );

    // The same transaction protocol can proceed through either surviving voter.
    drop(nodes.remove(0));
    serving_step(&mut client, ids[1], next + 2, ServingAction::Revoke(a1))
        .await
        .unwrap();
    let retire = ServingAction::Retire {
        version: a1,
        workers: BTreeSet::from([a]),
    };
    assert_eq!(
        serving_step(&mut client, ids[1], next + 3, retire.clone()).await,
        Err(ServingError::WorkersStillRunning)
    );
    let stale_stop = WorkerGrant {
        fence: old_grant.fence + 1,
        ..old_grant
    };
    assert_eq!(
        serving_step(
            &mut client,
            ids[1],
            next + 4,
            ServingAction::WorkerStopped(stale_stop)
        )
        .await,
        Err(ServingError::GrantMismatch)
    );
    serving_step(
        &mut client,
        ids[1],
        next + 5,
        ServingAction::WorkerStopped(old_grant),
    )
    .await
    .unwrap();
    serving_step(&mut client, ids[1], next + 6, retire)
        .await
        .unwrap();
    serving_step(
        &mut client,
        ids[1],
        next + 7,
        ServingAction::RegisterWorker {
            device: a,
            incarnation: 2,
            device_bytes: 48,
        },
    )
    .await
    .unwrap();
    serving_step(
        &mut client,
        ids[1],
        next + 8,
        ServingAction::Prepare {
            group: group_a,
            expected_epoch: 1,
            plan: plan(a, 2),
        },
    )
    .await
    .unwrap();
    let new_grant = serving_group(&mut client, ids[1], a2).await.grants[&a];
    assert_eq!(new_grant.fence, old_grant.fence + 1);
    assert_eq!(new_grant.incarnation, 2);
    assert_eq!(
        serving_step(
            &mut client,
            ids[1],
            next + 9,
            ServingAction::WorkerReady {
                grant: old_grant,
                launch_digest: [7; 32],
            }
        )
        .await,
        Err(ServingError::InvalidPhase(GroupPhase::Retired))
    );
    assert_eq!(
        serving_step(
            &mut client,
            ids[1],
            next + 10,
            ServingAction::WorkerReady {
                grant: new_grant,
                launch_digest: [0; 32],
            }
        )
        .await,
        Err(ServingError::LaunchDigestMismatch)
    );
    serving_step(
        &mut client,
        ids[1],
        next + 11,
        ServingAction::WorkerReady {
            grant: new_grant,
            launch_digest: [7; 32],
        },
    )
    .await
    .unwrap();
    serving_step(&mut client, ids[1], next + 12, ServingAction::Activate(a2))
        .await
        .unwrap();
    wait_until!("new group active on the other voter", {
        serving_group(&mut client, ids[2], a2).await.phase == GroupPhase::Active
    });
    assert_eq!(
        serving_group(&mut client, ids[2], a1).await.phase,
        GroupPhase::Retired
    );
    assert_eq!(
        serving_group(&mut client, ids[2], b1).await.phase,
        GroupPhase::Active
    );
    let ServingView::Resources(resources) = client
        .query_serving(ids[2], ServingLookup::Resources)
        .await
        .unwrap()
    else {
        panic!("resource query must return a snapshot");
    };
    let worker = resources
        .workers
        .iter()
        .find(|worker| worker.device == a)
        .unwrap();
    assert_eq!(worker.grant, Some(new_grant));
    assert_eq!(worker.incarnation, 2);
    let host = resources
        .hosts
        .iter()
        .find(|host| host.host == host_a)
        .unwrap();
    assert_eq!(host.reserved, plan(a, 2).hosts[&host_a]);
    let group = resources
        .groups
        .iter()
        .find(|entry| entry.latest == a2)
        .unwrap();
    assert_eq!(group.active.as_ref().unwrap().version, a2);
}

#[tokio::test]
async fn concurrent_group_prepares_cannot_reserve_the_same_worker() {
    let ids = ids("serving-compete", 3);
    let (_nodes, addrs) = start_bootstrap(&ids).await;
    let mut first = Client::new().await.unwrap();
    let mut second = Client::new().await.unwrap();
    for (id, addr) in ids.iter().zip(&addrs) {
        first.add_peer(*id, addr.clone());
        second.add_peer(*id, addr.clone());
    }
    let host = NodeId::derive("shared-serving-host");
    let device = DeviceId::new(host, 0);
    serving_step(
        &mut first,
        ids[0],
        1,
        ServingAction::SetHostMemory {
            host,
            capacity: HostMemory {
                bytes: 128,
                pinned_bytes: 128,
            },
        },
    )
    .await
    .unwrap();
    serving_step(
        &mut first,
        ids[0],
        2,
        ServingAction::RegisterWorker {
            device,
            incarnation: 1,
            device_bytes: 48,
        },
    )
    .await
    .unwrap();
    let groups = [
        GroupId(NodeId::derive("competing-first")),
        GroupId(NodeId::derive("competing-second")),
    ];
    let prepare = |group| ServingAction::Prepare {
        group,
        expected_epoch: 0,
        plan: plan(device, 1),
    };
    let (left, right) = tokio::join!(
        serving_step(&mut first, ids[0], 3, prepare(groups[0])),
        serving_step(&mut second, ids[1], 4, prepare(groups[1])),
    );
    let won = match (left, right) {
        (
            Ok(ServingOutcome::Group {
                version,
                phase: GroupPhase::Preparing,
            }),
            Err(ServingError::WorkerBusy(failed)),
        )
        | (
            Err(ServingError::WorkerBusy(failed)),
            Ok(ServingOutcome::Group {
                version,
                phase: GroupPhase::Preparing,
            }),
        ) => {
            assert_eq!(failed, device);
            version
        }
        results => panic!("expected one successful reservation, got {results:?}"),
    };
    let grant = serving_group(&mut first, ids[2], won).await.grants[&device];
    assert_eq!(grant.fence, 1);
    let owner = NodeId::derive("first host supervisor");
    let other = NodeId::derive("second host supervisor");
    let claim = |holder| ServingAction::ClaimLaunch {
        version: won,
        host,
        holder,
        launch_digest: [7; 32],
    };
    serving_step(&mut first, ids[0], 5, claim(owner))
        .await
        .unwrap();
    assert_eq!(
        serving_step(&mut second, ids[1], 6, claim(other)).await,
        Err(ServingError::LaunchOwnerMismatch)
    );
    assert_eq!(
        serving_step(
            &mut first,
            ids[0],
            7,
            ServingAction::WorkerReady {
                grant,
                launch_digest: [7; 32],
            }
        )
        .await,
        Err(ServingError::LaunchOwnerMismatch)
    );
    serving_step(
        &mut first,
        ids[0],
        8,
        ServingAction::LaunchReady {
            version: won,
            holder: owner,
        },
    )
    .await
    .unwrap();
    serving_step(&mut first, ids[0], 9, ServingAction::Activate(won))
        .await
        .unwrap();
    assert_eq!(
        serving_step(
            &mut second,
            ids[1],
            10,
            ServingAction::ObserveLaunch {
                version: won,
                holder: owner,
            }
        )
        .await,
        Ok(ServingOutcome::Group {
            version: won,
            phase: GroupPhase::Active
        })
    );
    serving_step(&mut first, ids[0], 11, ServingAction::Revoke(won))
        .await
        .unwrap();
    assert_eq!(
        serving_step(
            &mut second,
            ids[1],
            12,
            ServingAction::LaunchStopped {
                version: won,
                holder: other,
            }
        )
        .await,
        Err(ServingError::LaunchOwnerMismatch)
    );
    serving_step(
        &mut first,
        ids[0],
        13,
        ServingAction::LaunchStopped {
            version: won,
            holder: owner,
        },
    )
    .await
    .unwrap();
    serving_step(
        &mut first,
        ids[0],
        14,
        ServingAction::Retire {
            version: won,
            workers: BTreeSet::from([device]),
        },
    )
    .await
    .unwrap();
}
