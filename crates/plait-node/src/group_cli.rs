use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

use plait_core::group::{
    GroupAction, GroupId, GroupLookup, GroupOperation, GroupReceipt, GroupRecipe, GroupReply,
    GroupVersion, HostMemory, ResourcePlan, WorkerDemand,
};
use plait_core::link::ProbeRegistration;
use plait_core::{DeviceId, NodeId, OpId};
use plait_node::{Client, discover_voters};
use plait_parallel::group_placement::{Decision, DirectedLink, Workload};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::group_plan::{Snapshot, collect_links};

#[derive(Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
enum Request {
    Placement {
        group: GroupId,
        calibration: Option<NodeId>,
    },
    Standby {
        group: GroupId,
    },
    Links,
    Plan {
        group: GroupId,
        workload: Workload,
        links: Option<Vec<DirectedLink>>,
        prepare: Option<OpId>,
        calibration: Option<NodeId>,
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
    Recipe {
        version: GroupVersion,
    },
    Observe {
        id: OpId,
        group: GroupId,
        epoch: Option<u64>,
        #[serde(default)]
        with_recipe: bool,
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
        Request::Placement { group, calibration } => {
            let snapshot = placement_inventory(client, voters, group, calibration).await?;
            return Ok(match snapshot.rates {
                Some(rates) => json!({ "placement": snapshot.inventory, "rates": rates }),
                None => json!({ "placement": snapshot.inventory }),
            });
        }
        Request::Standby { group } => {
            let mut last = io::Error::other("no reachable voter");
            for &peer in voters {
                match client.query_group(peer, GroupLookup::Standby(group)).await {
                    Ok(GroupReply::Standby(version)) => return Ok(json!({ "standby": version })),
                    Ok(_) => last = io::Error::other("expected group standby"),
                    Err(error) => last = error,
                }
            }
            return Err(last);
        }
        Request::Links => {
            let probes = link_probes(client, voters).await?;
            let links = collect_links(&probes).await;
            return Ok(json!({ "probes": probes, "links": links }));
        }
        Request::Plan {
            group,
            workload,
            links,
            prepare,
            calibration,
        } => {
            if let Some(id) = prepare
                && let Some(receipt) = group_receipt(client, voters, id).await?
            {
                let recipe = match &receipt.operation.action {
                    GroupAction::PrepareRecipe { recipe, .. } => Some(*recipe),
                    _ => None,
                };
                let (GroupAction::PreparePlanned {
                    group: selected,
                    resources,
                    ..
                }
                | GroupAction::PrepareMeasured {
                    group: selected,
                    resources,
                    ..
                }
                | GroupAction::PrepareWithRates {
                    group: selected,
                    resources,
                    ..
                }
                | GroupAction::PrepareRecipe {
                    group: selected,
                    resources,
                    ..
                }) = receipt.operation.action
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
                let mut response = json!({ "decision": "prepare", "replayed": true,
                    "workers": resources.workers.iter().collect::<Vec<_>>(),
                    "hosts": resources.hosts.iter().collect::<Vec<_>>(),
                    "result": receipt.result,
                });
                if let Some(recipe) = recipe {
                    response["recipe"] = json!(recipe);
                }
                return Ok(response);
            }
            let mut snapshot = placement_inventory(client, voters, group, calibration).await?;
            if !workload.recipes.is_empty()
                && let Some(active) = &snapshot.inventory.active
            {
                snapshot.active_recipe = group_recipe(client, voters, active.version)
                    .await?
                    .ok_or_else(|| io::Error::other("active group recipe is not visible yet"))?
                    .recipe;
            }
            let (links, probes) = match links {
                Some(links) => (links, None),
                None if !workload.needs_links() => (Vec::new(), None),
                None => {
                    let mut probes = link_probes(client, voters).await?;
                    snapshot.restrict_probes(&mut probes);
                    (collect_links(&probes).await, Some(probes))
                }
            };
            let decision = snapshot.select(&workload, &links)?;
            let response = match decision {
                Decision::Pending(version) => json!({ "decision": "pending", "version": version }),
                Decision::NoPlacement => json!({ "decision": "no_placement" }),
                Decision::AwaitLinks(version) => {
                    json!({ "decision": "await_links", "version": version })
                }
                Decision::AwaitComputeRates(version) => {
                    json!({ "decision": "await_compute_rates", "version": version })
                }
                Decision::Keep { version, cost } => {
                    json!({ "decision": "keep", "version": version, "cost": cost })
                }
                Decision::Prepare {
                    resources,
                    cost,
                    incumbent,
                    recipe,
                } => {
                    let mut response = json!({
                        "decision": "prepare", "cost": cost, "incumbent": incumbent,
                        "workers": resources.workers.iter().collect::<Vec<_>>(),
                        "hosts": resources.hosts.iter().collect::<Vec<_>>(),
                    });
                    if let Some(recipe) = recipe {
                        response["recipe"] = json!(recipe);
                    }
                    if let Some(id) = prepare {
                        let action =
                            snapshot.prepare(resources, incumbent.is_some(), probes, recipe);
                        let operation = GroupOperation { id, action };
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
        Request::Recipe { version } => {
            return Ok(json!({ "recipe": group_recipe(client, voters, version).await? }));
        }
        Request::Observe {
            id,
            group,
            epoch,
            with_recipe,
        } => {
            let mut last = io::Error::other("no reachable voter");
            for &peer in voters {
                match client.observe_group(peer, id, group, epoch).await {
                    Ok(result) => {
                        if with_recipe && let Ok(ref view) = result {
                            let recipe = group_recipe(client, voters, view.version)
                                .await?
                                .ok_or_else(|| {
                                    io::Error::other("observed group recipe is not visible yet")
                                })?;
                            return Ok(json!({ "result": result, "recipe": recipe }));
                        }
                        return Ok(json!({ "result": result }));
                    }
                    Err(error) => last = error,
                }
            }
            return Err(last);
        }
    };
    Ok(json!({ "result": apply(client, voters, &operation).await? }))
}

async fn group_recipe(
    client: &mut Client,
    voters: &[NodeId],
    version: GroupVersion,
) -> io::Result<Option<GroupRecipe>> {
    let mut missing = false;
    let mut last = io::Error::other("no reachable voter");
    for &peer in voters {
        match client.query_group(peer, GroupLookup::Recipe(version)).await {
            Ok(GroupReply::Recipe(Some(recipe))) if recipe.version == version => {
                return Ok(Some(recipe));
            }
            Ok(GroupReply::Recipe(None)) => missing = true,
            Ok(_) => last = io::Error::other("expected group recipe"),
            Err(error) => last = error,
        }
    }
    if missing { Ok(None) } else { Err(last) }
}

async fn link_probes(client: &mut Client, voters: &[NodeId]) -> io::Result<Vec<ProbeRegistration>> {
    let mut last = io::Error::other("no reachable voter");
    for &peer in voters {
        match client.query_group(peer, GroupLookup::LinkProbes).await {
            Ok(GroupReply::LinkProbes(probes)) => return Ok(probes),
            Ok(_) => last = io::Error::other("expected link probe directory"),
            Err(error) => last = error,
        }
    }
    Err(last)
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
    calibration: Option<NodeId>,
) -> io::Result<Snapshot> {
    let lookup = Snapshot::lookup(group, calibration);
    let mut last = io::Error::other("no reachable voter");
    for &peer in voters {
        match client
            .query_group(peer, lookup)
            .await
            .and_then(|reply| Snapshot::from_reply(reply, calibration))
        {
            Ok(snapshot) => return Ok(snapshot),
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
