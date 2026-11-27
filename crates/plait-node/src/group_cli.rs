use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

use plait_core::group::{
    GroupAction, GroupId, GroupLookup, GroupOperation, GroupReceipt, GroupReply, GroupVersion,
    HostMemory, PlacementInventory, ResourcePlan, WorkerDemand,
};
use plait_core::{DeviceId, NodeId, OpId};
use plait_node::{Client, discover_voters};
use plait_parallel::group_placement::{self, Decision, DirectedLink, Workload};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum Request {
    Placement {
        group: GroupId,
    },
    Plan {
        group: GroupId,
        workload: Workload,
        #[serde(default)]
        links: Vec<DirectedLink>,
        prepare: Option<OpId>,
    },
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
        Request::Placement { group } => {
            let available = placement_inventory(client, voters, group).await?;
            return Ok(json!({ "placement": available }));
        }
        Request::Plan {
            group,
            workload,
            links,
            prepare,
        } => {
            if let Some(id) = prepare
                && let Some(receipt) = group_receipt(client, voters, id).await?
            {
                let GroupAction::PreparePlanned {
                    group: selected,
                    resources,
                    ..
                } = receipt.operation.action
                else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "operation id already used for another action",
                    ));
                };
                if selected != group {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "operation id already used for another group",
                    ));
                }
                return Ok(json!({ "decision": "prepare", "replayed": true,
                    "workers": resources.workers.iter().collect::<Vec<_>>(),
                    "hosts": resources.hosts.iter().collect::<Vec<_>>(),
                    "result": receipt.result,
                }));
            }
            let available = placement_inventory(client, voters, group).await?;
            let now = crate::hosted_probe::now_ms()?;
            let decision = group_placement::select(&available, &workload, &links, now)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
            let response = match decision {
                Decision::Pending(version) => json!({ "decision": "pending", "version": version }),
                Decision::NoPlacement => json!({ "decision": "no_placement" }),
                Decision::AwaitLinks(version) => {
                    json!({ "decision": "await_links", "version": version })
                }
                Decision::Keep { version, cost } => {
                    json!({ "decision": "keep", "version": version, "cost": cost })
                }
                Decision::Prepare {
                    resources,
                    cost,
                    incumbent,
                } => {
                    let mut response = json!({
                        "decision": "prepare", "cost": cost, "incumbent": incumbent,
                        "workers": resources.workers.iter().collect::<Vec<_>>(),
                        "hosts": resources.hosts.iter().collect::<Vec<_>>(),
                    });
                    if let Some(id) = prepare {
                        let expected_epoch =
                            available.latest.map_or(0, |(version, _)| version.epoch);
                        let expected_active = available.active.map(|active| active.version.epoch);
                        let operation = GroupOperation {
                            id,
                            action: GroupAction::PreparePlanned {
                                group,
                                expected_epoch,
                                expected_active,
                                resources,
                            },
                        };
                        response["result"] = json!(apply(client, voters, &operation).await?);
                    }
                    response
                }
            };
            return Ok(response);
        }
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
    Ok(json!({ "result": apply(client, voters, &operation).await? }))
}

async fn apply(
    client: &mut Client,
    voters: &[NodeId],
    operation: &GroupOperation,
) -> io::Result<plait_core::group::GroupResult> {
    let mut last = io::Error::other("no reachable voter");
    for &peer in voters {
        match client.apply_group(peer, operation).await {
            Ok(result) => return Ok(result),
            Err(error) => last = error,
        }
    }
    Err(last)
}

async fn placement_inventory(
    client: &mut Client,
    voters: &[NodeId],
    group: GroupId,
) -> io::Result<PlacementInventory> {
    let mut last = io::Error::other("no reachable voter");
    for &peer in voters {
        match client
            .query_group(peer, GroupLookup::Placement(group))
            .await
        {
            Ok(GroupReply::Placement(view)) => return Ok(view),
            Ok(_) => last = io::Error::other("expected group placement inventory"),
            Err(error) => last = error,
        }
    }
    Err(last)
}

async fn group_receipt(
    client: &mut Client,
    voters: &[NodeId],
    id: OpId,
) -> io::Result<Option<GroupReceipt>> {
    let mut observed = false;
    let mut last = io::Error::other("no reachable voter");
    for &peer in voters {
        match client.query_group(peer, GroupLookup::Receipt(id)).await {
            Ok(GroupReply::Receipt(Some(receipt))) => return Ok(Some(receipt)),
            Ok(GroupReply::Receipt(None)) => observed = true,
            Ok(_) => last = io::Error::other("expected group operation receipt"),
            Err(error) => last = error,
        }
    }
    if observed { Ok(None) } else { Err(last) }
}
