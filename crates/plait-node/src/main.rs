use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;

use plait_core::NodeId;
use plait_node::{
    NodeConfig, RunningNode, StorageMode, run_collectives_demo, run_demo, run_devices,
    run_dsa_demo, run_gpu_bench, run_gpu_demo, run_gpu_rdma_probe, run_hetero_demo,
    run_hetero_plan, run_matrix_node, run_parallel_demo, run_rdma_selftest,
};

mod kv_store;
mod serving_cli;
mod tensor_cli;

#[tokio::main]
async fn main() {
    if let Err(error) = real_main().await {
        eprintln!("plait-node: {error}");
        std::process::exit(1);
    }
}

async fn real_main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "demo".to_owned());
    match command.as_str() {
        "demo" => run_demo().await,
        "dsa" => run_dsa_demo(),
        "parallel" => run_parallel_demo(),
        "rdma-selftest" => run_rdma_selftest().await,
        "devices" => run_devices(),
        "hetero" => run_hetero_demo(),
        "hetero-plan" => {
            let mut local_gflops = 1.0;
            let mut remote_gflops = 4.0;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--local-gflops" => {
                        local_gflops = args.next().ok_or("--local-gflops needs a number")?.parse()?
                    }
                    "--remote-gflops" => {
                        remote_gflops = args.next().ok_or("--remote-gflops needs a number")?.parse()?
                    }
                    other => return Err(format!("unknown hetero-plan argument {other}").into()),
                }
            }
            run_hetero_plan(local_gflops, remote_gflops)
        }
        "gpu-demo" => run_gpu_demo(),
        "gpu-bench" => run_gpu_bench(),
        "gpu-rdma" => run_gpu_rdma_probe(),
        "collectives" => {
            let mut gpus = 0usize;
            let mut bytes = 1usize << 20;
            let mut iters = 20usize;
            let mut algorithm = "ring".to_owned();
            let mut channels = 1usize;
            let mut op = "allreduce".to_owned();
            let mut resident = false;
            let mut device_ring = false;
            let mut protocol = "simple".to_owned();
            let mut trace = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--gpus" | "--ranks" => {
                        gpus = args.next().ok_or("--ranks needs a number")?.parse()?
                    }
                    "--bytes" => bytes = args.next().ok_or("--bytes needs a number")?.parse()?,
                    "--iters" => iters = args.next().ok_or("--iters needs a number")?.parse()?,
                    "--algorithm" => algorithm = args.next().ok_or("--algorithm needs a value")?,
                    "--channels" => {
                        channels = args.next().ok_or("--channels needs a number")?.parse()?
                    }
                    "--op" => op = args.next().ok_or("--op needs a value")?,
                    "--resident" => resident = true,
                    "--device-ring" => device_ring = true,
                    "--protocol" => protocol = args.next().ok_or("--protocol needs a value")?,
                    "--trace" => {
                        trace = Some(args.next().ok_or("--trace needs a path")?);
                    }
                    other => return Err(format!("unknown collectives argument {other}").into()),
                }
            }
            run_collectives_demo(
                gpus,
                bytes,
                iters,
                &algorithm,
                &op,
                channels,
                resident,
                device_ring,
                &protocol,
                trace.as_deref(),
            )
        }
        "matrix" => {
            let mut manifest = None;
            let mut start = 0usize;
            let mut count = 6usize;
            let mut driver = false;
            let mut hold = 40u64;
            let mut m = 8usize;
            let mut n = 8usize;
            let mut k = 6usize;
            let mut asymmetric = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--manifest" => manifest = Some(args.next().ok_or("--manifest needs a path")?),
                    "--start" => start = args.next().ok_or("--start needs a number")?.parse()?,
                    "--count" => count = args.next().ok_or("--count needs a number")?.parse()?,
                    "--driver" => driver = true,
                    "--hold" => hold = args.next().ok_or("--hold needs seconds")?.parse()?,
                    "--m" => m = args.next().ok_or("--m needs a number")?.parse()?,
                    "--n" => n = args.next().ok_or("--n needs a number")?.parse()?,
                    "--k" => k = args.next().ok_or("--k needs a number")?.parse()?,
                    "--asymmetric" => asymmetric = true,
                    other => return Err(format!("unknown matrix argument {other}").into()),
                }
            }
            let manifest = manifest.ok_or("--manifest is required")?;
            run_matrix_node(&manifest, start, count, driver, hold, (m, n, k), asymmetric).await
        }
        "id" => {
            println!("{}", NodeId::random());
            Ok(())
        }
        "serving" => serving_cli::run(args).await,
        "kv-store" => kv_store::run(args).await,
        "tensor" => tensor_cli::run(args).await,
        "run" => {
            let mut listen = "127.0.0.1:7001".to_owned();
            let mut id: Option<NodeId> = None;
            let mut peers = BTreeMap::new();
            let mut seeds = Vec::new();
            let mut join = None;
            let mut state_dir = None;
            let mut initialize = false;
            let mut advertise_devices = true;
            let mut execution = None;
            let mut compute_only = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--listen" => listen = args.next().ok_or("--listen needs a value")?,
                    "--id" => {
                        id = Some(NodeId::from_hex(&args.next().ok_or("--id needs a value")?)?)
                    }
                    "--peer" => {
                        let value = args.next().ok_or("--peer needs id@addr")?;
                        let (node, addr) = parse_peer(&value)?;
                        peers.insert(node, addr);
                    }
                    "--seed" => {
                        seeds.push(parse_seed(&args.next().ok_or("--seed needs an IP:port")?)?);
                    }
                    "--join" => {
                        let value = args.next().ok_or("--join needs id@addr")?;
                        let (node, addr) = parse_peer(&value)?;
                        join = Some(plait_node::JoinTarget { node, addr });
                    }
                    "--state-dir" => state_dir = Some(PathBuf::from(args.next().ok_or("--state-dir needs a path")?)),
                    "--init-state" => initialize = true,
                    "--no-advertise-devices" => advertise_devices = false,
                    "--w4-execution" => {
                        let path = args.next().ok_or("--w4-execution needs a path")?;
                        execution = Some(serde_json::from_reader::<_, tensor_cli::runtime::Config>(
                            std::fs::File::open(path)?,
                        )?);
                    }
                    "--compute-only" => compute_only = true,
                    other => return Err(format!("unknown argument {other}").into()),
                }
            }
            if initialize && state_dir.is_none() {
                return Err("--init-state requires --state-dir".into());
            }
            if state_dir.is_some() && id.is_none() {
                return Err("durable replicas require an explicit --id".into());
            }
            if execution.is_some() && id.is_none() {
                return Err("--w4-execution requires a stable --id".into());
            }
            let id = id.unwrap_or_else(NodeId::random);
            if compute_only {
                if join.is_some() || state_dir.is_some() || peers.is_empty() == seeds.is_empty() {
                    return Err("--compute-only requires either --peer or --seed and no voting membership or journal flags".into());
                }
                let config = execution.ok_or("--compute-only requires --w4-execution")?;
                tokio::select! {
                    result = tokio::signal::ctrl_c() => result?,
                    result = tensor_cli::runtime::run(id, config, peers, seeds) => result?,
                }
                return Ok(());
            }
            if !seeds.is_empty() {
                return Err("run --seed requires --compute-only".into());
            }
            let mut execution_peers = peers.clone();
            if let Some(target) = &join {
                execution_peers.insert(target.node, target.addr.clone());
            }
            let config = NodeConfig {
                id,
                listen: listen.clone(),
                peers,
                join,
                options: Default::default(),
                advertise_devices,
            };
            let mut node = match state_dir {
                Some(path) if initialize => RunningNode::start_durable(config, StorageMode::Initialize(path)).await?,
                Some(path) => RunningNode::start_durable(config, StorageMode::Resume(path)).await?,
                None => RunningNode::start(config).await?,
            };
            println!("plait node {} listening on {}", node.id(), node.addr());
            execution_peers.insert(id, node.addr().to_owned());
            let execute = async {
                if let Some(config) = execution {
                    tensor_cli::runtime::run(id, config, execution_peers, Vec::new()).await
                } else {
                    std::future::pending::<std::io::Result<()>>().await
                }
            };
            tokio::select! {
                result = tokio::signal::ctrl_c() => result?,
                result = node.wait() => result?,
                result = execute => result?,
            }
            drop(node);
            Ok(())
        }
        other => Err(format!(
            "unknown command {other}; expected run, serving, kv-store, tensor, demo, dsa, parallel, rdma-selftest, devices, hetero, gpu-demo, gpu-bench, gpu-rdma, or id"
        )
        .into()),
    }
}

fn parse_peer(value: &str) -> Result<(NodeId, String), Box<dyn std::error::Error>> {
    let (id, addr) = value
        .split_once('@')
        .ok_or("peer must be <hex-id>@<host:port>")?;
    Ok((NodeId::from_hex(id)?, addr.to_owned()))
}

fn parse_seed(value: &str) -> Result<SocketAddr, Box<dyn std::error::Error>> {
    let addr = value.parse::<SocketAddr>()?;
    if addr.ip().is_unspecified() || addr.port() == 0 {
        return Err("control seed requires a concrete IP and nonzero port".into());
    }
    Ok(addr)
}
