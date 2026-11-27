use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Write};

use plait_core::group::{
    GroupAction, GroupId, GroupLookup, GroupOperation, GroupPhase, GroupPreparation, GroupReceipt,
    GroupRecipe, GroupReply, GroupVersion, HostMemory, ResourcePlan, WorkerDemand,
};
use plait_core::link::ProbeRegistration;
use plait_core::{DeviceId, NodeId, OpId};
use plait_node::{Client, discover_voters};
use plait_parallel::group_placement::{
    self, Cost, Decision, DirectedLink, PlacementGroup, Workload,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::group_plan::{Snapshot, collect_links};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetMember {
    group: GroupId,
    workload: Workload,
    calibration: Option<NodeId>,
}

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
    PlanSet {
        groups: Vec<SetMember>,
        #[serde(default)]
        guards: Vec<(GroupId, Option<u64>)>,
        links: Option<Vec<DirectedLink>>,
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
        Request::PlanSet {
            groups,
            guards,
            links,
            prepare,
        } => {
            return plan_set(client, voters, groups, guards, links, prepare).await;
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

async fn plan_set(
    client: &mut Client,
    voters: &[NodeId],
    groups: Vec<SetMember>,
    guards: Vec<(GroupId, Option<u64>)>,
    links: Option<Vec<DirectedLink>>,
    prepare: Option<OpId>,
) -> io::Result<Value> {
    if groups.is_empty()
        || groups
            .iter()
            .map(|member| member.group)
            .collect::<BTreeSet<_>>()
            .len()
            != groups.len()
        || guards
            .iter()
            .map(|&(group, _)| group)
            .collect::<BTreeSet<_>>()
            .len()
            != guards.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "empty set or duplicate group or guard",
        ));
    }
    if let Some(id) = prepare
        && let Some(receipt) = group_receipt(client, voters, id).await?
    {
        if let GroupAction::PrepareSet {
            preparations,
            guards: previous,
        } = receipt.operation.action
            && guards == previous
            && preparations
                .iter()
                .map(|item| {
                    (
                        item.group,
                        item.rates.as_ref().map(|rates| rates.calibration),
                    )
                })
                .eq(groups.iter().map(|item| (item.group, item.calibration)))
        {
            return Ok(json!({ "decision": "prepare_set", "replayed": true,
                "plans": preparations.iter().map(|plan| preparation_json(plan, None)).collect::<Vec<_>>(),
                "result": receipt.result,
            }));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "operation id already used for another action or group set",
        ));
    }

    let mut snapshots = Vec::with_capacity(groups.len());
    for member in &groups {
        snapshots
            .push(placement_inventory(client, voters, member.group, member.calibration).await?);
    }
    let pending: Vec<_> = snapshots
        .iter()
        .filter_map(|snapshot| match snapshot.inventory.latest {
            Some((version, GroupPhase::Preparing | GroupPhase::Ready)) => Some(version),
            _ => None,
        })
        .collect();
    if !pending.is_empty() {
        return Ok(json!({ "decision": "pending", "versions": pending }));
    }
    let (links, probes) = match links {
        Some(links) => (links, Vec::new()),
        None if !groups.iter().any(|member| member.workload.needs_links()) => {
            (Vec::new(), Vec::new())
        }
        None => {
            let hosts: BTreeSet<_> = snapshots
                .iter()
                .flat_map(|snapshot| &snapshot.inventory.hosts)
                .filter(|host| host.eligible)
                .map(|host| host.host)
                .collect();
            let mut probes = link_probes(client, voters).await?;
            probes.retain(|probe| hosts.contains(&probe.endpoint.host));
            (collect_links(&probes).await, probes)
        }
    };
    let inputs: Vec<_> = groups
        .iter()
        .zip(&snapshots)
        .map(|(member, snapshot)| PlacementGroup {
            inventory: &snapshot.inventory,
            workload: &member.workload,
            rates: snapshot.rates.as_deref(),
        })
        .collect();
    let Some(choices) =
        group_placement::select_set(&inputs, &links, crate::hosted_probe::now_ms()?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
    else {
        return Ok(json!({ "decision": "no_placement" }));
    };
    let mut preparations = Vec::with_capacity(choices.len());
    let mut plans = Vec::with_capacity(choices.len());
    for (choice, snapshot) in choices.into_iter().zip(snapshots) {
        let plan = snapshot.preparation(choice.resources, false, probes.clone(), choice.recipe);
        plans.push(preparation_json(&plan, Some(choice.cost)));
        preparations.push(plan);
    }
    let mut response = json!({ "decision": "prepare_set", "plans": plans });
    if let Some(id) = prepare {
        let action = GroupAction::PrepareSet {
            preparations,
            guards,
        };
        response["result"] = json!(apply(client, voters, &GroupOperation { id, action }).await?);
    }
    Ok(response)
}

fn preparation_json(plan: &GroupPreparation, cost: Option<Cost>) -> Value {
    json!({
        "group": plan.group,
        "expected_epoch": plan.expected_epoch,
        "expected_active": plan.expected_active,
        "workers": plan.resources.workers.iter().collect::<Vec<_>>(),
        "hosts": plan.resources.hosts.iter().collect::<Vec<_>>(),
        "recipe": plan.recipe,
        "cost": cost,
    })
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
