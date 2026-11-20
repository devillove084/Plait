use std::collections::BTreeMap;

use plait_core::NodeId;
use plait_node::{
    NodeConfig, RunningNode, run_demo, run_devices, run_dsa_demo, run_gpu_demo, run_hetero_demo,
    run_parallel_demo, run_rdma_selftest,
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
