use std::collections::BTreeMap;

use plait_core::NodeId;
use plait_node::{
    NodeConfig, RunningNode, run_collectives_demo, run_demo, run_devices, run_dsa_demo,
    run_gpu_demo, run_hetero_demo, run_matrix_node, run_parallel_demo, run_rdma_selftest,
};

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
        "gpu-demo" => run_gpu_demo(),
        "collectives" => {
            let mut gpus = 0usize;
            let mut bytes = 1usize << 20;
            let mut iters = 20usize;
            let mut algorithm = "ring".to_owned();
            let mut trace = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--gpus" | "--ranks" => {
                        gpus = args.next().ok_or("--ranks needs a number")?.parse()?
                    }
                    "--bytes" => bytes = args.next().ok_or("--bytes needs a number")?.parse()?,
                    "--iters" => iters = args.next().ok_or("--iters needs a number")?.parse()?,
                    "--algorithm" => algorithm = args.next().ok_or("--algorithm needs a value")?,
                    "--trace" => {
                        trace = Some(args.next().ok_or("--trace needs a path")?);
                    }
                    other => return Err(format!("unknown collectives argument {other}").into()),
                }
            }
            run_collectives_demo(gpus, bytes, iters, &algorithm, trace.as_deref())
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
                    other => return Err(format!("unknown matrix argument {other}").into()),
                }
            }
            let manifest = manifest.ok_or("--manifest is required")?;
            run_matrix_node(&manifest, start, count, driver, hold, (m, n, k)).await
        }
        "id" => {
            println!("{}", NodeId::random());
            Ok(())
        }
        "run" => {
            let mut listen = "127.0.0.1:7001".to_owned();
            let mut id: Option<NodeId> = None;
            let mut peers = BTreeMap::new();
            let mut join = None;
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
                    "--join" => {
                        let value = args.next().ok_or("--join needs id@addr")?;
                        let (node, addr) = parse_peer(&value)?;
                        join = Some(plait_node::JoinTarget { node, addr });
                    }
                    other => return Err(format!("unknown argument {other}").into()),
                }
            }
            let id = id.unwrap_or_else(NodeId::random);
            let config = NodeConfig {
                id,
                listen: listen.clone(),
                peers,
                join,
                options: Default::default(),
                advertise_devices: true,
            };
            let node = RunningNode::start(config).await?;
            println!("plait node {} listening on {}", node.id(), node.addr());
            tokio::signal::ctrl_c().await?;
            drop(node);
            Ok(())
        }
        other => Err(format!(
            "unknown command {other}; expected run, demo, dsa, parallel, rdma-selftest, devices, hetero, gpu-demo, or id"
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
