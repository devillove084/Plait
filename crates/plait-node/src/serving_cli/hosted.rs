use std::collections::BTreeSet;
use std::io;

use plait_core::serving::{
    GroupId, GroupPlan, HostedActivation, HostedPlanning, HostedProbeEndpoint, HostedRecipe,
    ServingAction, ServingError, ServingLookup, ServingOperation, ServingRole, ServingView,
};
use plait_core::{NodeId, OpId};
use plait_node::Client;
use plait_parallel::serving::{
    HostedLinkLimit, HostedLinkPolicy, HostedPool, ServingCandidate, ServingForecast,
    generate_hosted_candidates, limit_hosted_links, select_serving_candidate,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{CandidateInput, apply, query, query_resources};
use crate::hosted_probe;

#[derive(Deserialize, Serialize)]
struct Input {
    #[serde(flatten)]
    candidate: CandidateInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    recipe: Option<HostedRecipe>,
}

#[derive(Deserialize)]
pub(super) struct Replan {
    id: OpId,
    forecast: ServingForecast,
    #[serde(default)]
    candidates: Vec<Input>,
    #[serde(default)]
    activation: HostedActivation,
    link_policy: Option<HostedLinkPolicy>,
    pool: Option<HostedPool>,
}

pub(super) fn prepare(
    group: GroupId,
    expected_epoch: u64,
    expected_active: Option<u64>,
    mut plan: GroupPlan,
    activation: HostedActivation,
    recipe: Option<HostedRecipe>,
    request_sha256: Option<[u8; 32]>,
) -> ServingAction {
    if let Some(recipe) = recipe {
        plan.launch_digest = recipe.launch_digest();
        ServingAction::PrepareHostedDeployment {
            group,
            expected_epoch,
            expected_active,
            plan,
            activation,
            request_sha256,
            recipe,
        }
    } else {
        ServingAction::PrepareHostedSwitch {
            group,
            expected_epoch,
            expected_active,
            plan,
            activation,
            request_sha256,
        }
    }
}

pub(super) async fn replan(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId> + Clone,
    input: Replan,
) -> io::Result<Value> {
    let Replan {
        id,
        mut forecast,
        candidates: inputs,
        activation,
        link_policy,
        pool,
    } = input;
    if pool.is_some() && (!inputs.is_empty() || link_policy.is_none()) {
        return Err(invalid_input(
            "hosted pool requires link_policy and no explicit candidates",
        ));
    }
    forecast.now_ms = 0;
    let request = match (&pool, link_policy) {
        (Some(pool), Some(policy)) => {
            serde_json::to_vec(&("hosted-pool-v1", &forecast, pool, activation, policy))?
        }
        (None, Some(policy)) => serde_json::to_vec(&(&forecast, &inputs, activation, policy))?,
        (None, None) => serde_json::to_vec(&(&forecast, &inputs, activation))?,
        (Some(_), None) => unreachable!("checked pool link policy"),
    };
    let request_sha256: [u8; 32] = Sha256::digest(request).into();
    if let Some(result) = replay(client, peers.clone(), id, request_sha256).await? {
        return Ok(result);
    }
    let resources = query_resources(client, peers.clone()).await?;
    let (mut candidates, recipes, samples): (Vec<_>, Vec<_>, _) =
        if let (Some(pool), Some(policy)) = (&pool, link_policy) {
            let planning = planning(client, peers.clone()).await?;
            let hosts = planning
                .hosts
                .iter()
                .map(|inv| inv.host)
                .collect::<BTreeSet<_>>();
            let samples =
                hosted_probe::collect(probe_endpoints(client, peers.clone()).await?, Some(&hosts))
                    .await;
            forecast.now_ms = hosted_probe::now_ms()?;
            let (candidates, recipes) = generate_hosted_candidates(
                &resources, &planning, &forecast, pool, policy, &samples,
            )
            .map_err(invalid_input)?
            .into_iter()
            .map(|(candidate, recipe)| (candidate, Some(recipe)))
            .unzip();
            (candidates, recipes, samples)
        } else {
            let (candidates, recipes): (Vec<_>, Vec<_>) = inputs
                .into_iter()
                .map(manual_candidate)
                .collect::<io::Result<Vec<_>>>()?
                .into_iter()
                .unzip();
            let samples = if link_policy.is_some() {
                let hosts = candidates
                    .iter()
                    .flat_map(|candidate| candidate.plan.hosts.keys().copied())
                    .collect::<BTreeSet<_>>();
                hosted_probe::collect(probe_endpoints(client, peers.clone()).await?, Some(&hosts))
                    .await
            } else {
                Default::default()
            };
            (candidates, recipes, samples)
        };
    forecast.now_ms = hosted_probe::now_ms()?;
    let link_limits = if let Some(policy) = link_policy {
        candidates
            .iter_mut()
            .map(|candidate| {
                let limit = limit_hosted_links(&candidate.plan, policy, forecast.now_ms, &samples)
                    .map_err(invalid_input)?;
                match limit {
                    HostedLinkLimit::Local => {}
                    HostedLinkLimit::MissingLink { .. } => candidate.capacity.valid_until_ms = 0,
                    HostedLinkLimit::Measured {
                        rps,
                        valid_until_ms,
                    } => {
                        candidate.capacity.rps = candidate.capacity.rps.min(rps);
                        candidate.capacity.valid_until_ms =
                            candidate.capacity.valid_until_ms.min(valid_until_ms);
                    }
                }
                Ok(limit)
            })
            .collect::<io::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let selection =
        select_serving_candidate(&resources, &forecast, &candidates).map_err(invalid_input)?;
    let mut response =
        json!({ "selection": selection, "link_limits": link_limits, "result": null });
    if pool.is_some() {
        let placements: Vec<_> = candidates
            .iter()
            .zip(&recipes)
            .map(|(candidate, recipe)| {
                json!({
                    "tensor_parallel": candidate.plan.tensor_parallel,
                    "workers": candidate.plan.workers.keys().collect::<Vec<_>>(),
                    "recipe": recipe,
                })
            })
            .collect();
        response["placements"] = json!(placements);
    }
    let Some(index) = selection.selected else {
        return Ok(response);
    };
    let candidate = &candidates[index];
    let expected_active = resources
        .groups
        .iter()
        .find(|group| group.latest.group == candidate.group)
        .and_then(|group| group.active.as_ref())
        .map(|active| active.version.epoch);
    let action = if pool.is_some() {
        ServingAction::PrepareHostedPoolDeployment {
            group: candidate.group,
            expected_epoch: candidate.expected_epoch,
            expected_active,
            plan: candidate.plan.clone(),
            activation,
            request_sha256,
            recipe: recipes[index].clone().expect("pool candidate has a recipe"),
        }
    } else {
        prepare(
            candidate.group,
            candidate.expected_epoch,
            expected_active,
            candidate.plan.clone(),
            activation,
            recipes[index].clone(),
            Some(request_sha256),
        )
    };
    let operation = ServingOperation { id, action };
    let result = apply(client, peers.clone(), &operation).await?;
    if result == Err(ServingError::OperationIdReused)
        && let Some(previous) = replay(client, peers, id, request_sha256).await?
    {
        return Ok(previous);
    }
    response["result"] = json!(result);
    Ok(response)
}

fn manual_candidate(input: Input) -> io::Result<(ServingCandidate, Option<HostedRecipe>)> {
    let mut candidate = ServingCandidate::try_from(input.candidate)?;
    if let Some(recipe) = &input.recipe {
        candidate.plan.launch_digest = recipe.launch_digest();
        recipe
            .validate(&candidate.plan)
            .map_err(|error| invalid_input(format!("invalid hosted recipe: {error:?}")))?;
    }
    if candidate.plan.role != ServingRole::Aggregated || candidate.plan.pipeline_parallel != 1 {
        return Err(invalid_input("expected hosted TP candidates"));
    }
    Ok((candidate, input.recipe))
}

fn invalid_input(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub(super) async fn planning(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
) -> io::Result<HostedPlanning> {
    let ServingView::HostedPlanning(planning) =
        query(client, peers, ServingLookup::HostedPlanning).await?
    else {
        return Err(io::Error::other("expected hosted planning information"));
    };
    Ok(planning)
}

pub(super) async fn links(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
) -> io::Result<Value> {
    let endpoints = probe_endpoints(client, peers).await?;
    let links = hosted_probe::collect(endpoints.clone(), None).await;
    let samples: Vec<_> = links
        .into_iter()
        .map(|((source, _), sample)| {
            json!({
                "source": source, "sample": sample,
            })
        })
        .collect();
    Ok(json!({ "endpoints": endpoints, "samples": samples }))
}

async fn probe_endpoints(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
) -> io::Result<Vec<HostedProbeEndpoint>> {
    let ServingView::HostedProbeEndpoints(endpoints) =
        query(client, peers, ServingLookup::HostedProbeEndpoints).await?
    else {
        return Err(io::Error::other("expected hosted link probe endpoints"));
    };
    Ok(endpoints)
}
async fn replay(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId>,
    id: OpId,
    request_sha256: [u8; 32],
) -> io::Result<Option<Value>> {
    let ServingView::Operation(receipt) =
        query(client, peers, ServingLookup::Operation(id)).await?
    else {
        return Err(io::Error::other("expected a serving operation receipt"));
    };
    let Some(receipt) = receipt else {
        return Ok(None);
    };
    if let ServingAction::PrepareHostedSwitch {
        request_sha256: Some(prior),
        ..
    }
    | ServingAction::PrepareHostedDeployment {
        request_sha256: Some(prior),
        ..
    }
    | ServingAction::PrepareHostedPoolDeployment {
        request_sha256: prior,
        ..
    } = receipt.operation.action
        && prior == request_sha256
    {
        return Ok(Some(json!({ "replayed": true, "result": receipt.result })));
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "operation ID was used for a different serving action",
    ))
}
