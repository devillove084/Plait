use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

use plait_core::group::{
    GroupAction, GroupId, GroupLookup, GroupOperation, GroupReply, GroupVersion, HostMemory,
    ResourcePlan, WorkerDemand,
};
use plait_core::{DeviceId, NodeId, OpId};
use plait_node::{Client, discover_voters};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum Request {
    Apply {
        operation: GroupOperation,
    },
    Prepare {
        id: OpId,
        group: GroupId,
        expected_epoch: u64,
        workers: Vec<(DeviceId, WorkerDemand)>,
        hosts: Vec<(NodeId, HostMemory)>,
    },
    View {
        version: GroupVersion,
    },
    Observe {
        id: OpId,
        group: GroupId,
        epoch: Option<u64>,
    },
}

pub(super) async fn run(
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut listen = "127.0.0.1:0".to_owned();
    let mut peers = BTreeMap::new();
    let mut seeds = Vec::new();
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
        match arg.as_str() {
            "--listen" => listen = value,
            "--peer" => {
                let (id, address) = crate::parse_peer(&value)?;
                peers.insert(id, address);
            }
            "--seed" => seeds.push(crate::parse_seed(&value)?),
            _ => return Err(format!("unknown group argument {arg}").into()),
        }
    }
    if peers.is_empty() == seeds.is_empty() {
        return Err("configure either --peer <id>@<address> or --seed <IP:port>".into());
    }
    if !seeds.is_empty() {
        peers = discover_voters(&seeds).await?;
    }
    let mut client = Client::new_on(&listen).await?;
    for (&id, address) in &peers {
        client.add_peer(id, address);
    }
    let voters: Vec<_> = peers.keys().copied().collect();
    for line in io::stdin().lock().lines() {
        let command = serde_json::from_str::<Request>(&line?);
        let result = match command {
            Ok(request) => handle(&mut client, &voters, request).await,
            Err(error) => Err(io::Error::other(error)),
        };
        let response = result.unwrap_or_else(|error| json!({ "error": error.to_string() }));
        let mut stdout = io::stdout().lock();
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}

async fn handle(client: &mut Client, voters: &[NodeId], request: Request) -> io::Result<Value> {
    let operation = match request {
        Request::Apply { operation } => operation,
        Request::Prepare {
            id,
            group,
            expected_epoch,
            workers,
            hosts,
        } => {
            let counts = (workers.len(), hosts.len());
            let resources = ResourcePlan {
                workers: workers.into_iter().collect(),
                hosts: hosts.into_iter().collect(),
            };
            if counts != (resources.workers.len(), resources.hosts.len()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate group device or host",
                ));
            }
            GroupOperation {
                id,
                action: GroupAction::Prepare {
                    group,
                    expected_epoch,
                    resources,
                },
            }
        }
        Request::View { version } => {
            let mut missing = false;
            let mut last = io::Error::other("no reachable voter");
            for &peer in voters {
                match client
                    .query_group(peer, GroupLookup::Version(version))
                    .await
                {
                    Ok(GroupReply::Version(Some(view))) => return Ok(json!({ "group": view })),
                    Ok(GroupReply::Version(None)) => missing = true,
                    Ok(_) => last = io::Error::other("expected group version"),
                    Err(error) => last = error,
                }
            }
            return if missing {
                Ok(json!({ "group": null }))
            } else {
                Err(last)
            };
        }
        Request::Observe { id, group, epoch } => {
            let mut last = io::Error::other("no reachable voter");
            for &peer in voters {
                match client.observe_group(peer, id, group, epoch).await {
                    Ok(result) => return Ok(json!({ "result": result })),
                    Err(error) => last = error,
                }
            }
            return Err(last);
        }
    };
    let mut last = io::Error::other("no reachable voter");
    for &peer in voters {
        match client.apply_group(peer, &operation).await {
            Ok(result) => return Ok(json!({ "result": result })),
            Err(error) => last = error,
        }
    }
    Err(last)
}
