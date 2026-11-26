use std::collections::BTreeSet;
use std::io;

use plait_core::serving::{
    GroupId, GroupPlan, HostedActivation, HostedProbeEndpoint, HostedRecipe, ServingAction,
    ServingError, ServingLookup, ServingOperation, ServingRole, ServingView,
};
use plait_core::{NodeId, OpId};
use plait_node::Client;
use plait_parallel::serving::{
    HostedLinkLimit, HostedLinkPolicy, ServingCandidate, ServingForecast, limit_hosted_links,
    select_serving_candidate,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{CandidateInput, apply, query, query_resources};
use crate::hosted_probe;

#[derive(Deserialize, Serialize)]
pub(super) struct Input {
    #[serde(flatten)]
    candidate: CandidateInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    recipe: Option<HostedRecipe>,
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
    id: OpId,
    mut forecast: ServingForecast,
    inputs: Vec<Input>,
    activation: HostedActivation,
    link_policy: Option<HostedLinkPolicy>,
) -> io::Result<Value> {
    forecast.now_ms = 0;
    let request = match link_policy {
        Some(policy) => serde_json::to_vec(&(&forecast, &inputs, activation, policy))?,
        None => serde_json::to_vec(&(&forecast, &inputs, activation))?,
    };
    let request_sha256: [u8; 32] = Sha256::digest(request).into();
    if let Some(result) = replay(client, peers.clone(), id, request_sha256).await? {
        return Ok(result);
    }
    let (mut candidates, recipes): (Vec<_>, Vec<_>) = inputs
        .into_iter()
        .map(|input| {
            let mut candidate = ServingCandidate::try_from(input.candidate)?;
            if let Some(recipe) = &input.recipe {
                candidate.plan.launch_digest = recipe.launch_digest();
                recipe.validate(&candidate.plan).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("invalid hosted recipe: {error:?}"),
                    )
                })?;
            }
            Ok((candidate, input.recipe))
        })
        .collect::<io::Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    if candidates.iter().any(|candidate| {
        candidate.plan.role != ServingRole::Aggregated || candidate.plan.pipeline_parallel != 1
    }) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected hosted TP candidates",
        ));
    }
    let resources = query_resources(client, peers.clone()).await?;
    let link_limits = if let Some(policy) = link_policy {
        let hosts = candidates
            .iter()
            .flat_map(|candidate| candidate.plan.hosts.keys().copied())
            .collect::<BTreeSet<_>>();
        let samples =
            hosted_probe::collect(probe_endpoints(client, peers.clone()).await?, Some(&hosts))
                .await;
        let now = hosted_probe::now_ms()?;
        candidates
            .iter_mut()
            .map(|candidate| {
                let limit = limit_hosted_links(&candidate.plan, policy, now, &samples)
                    .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
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
    forecast.now_ms = hosted_probe::now_ms()?;
    let selection = select_serving_candidate(&resources, &forecast, &candidates)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let Some(index) = selection.selected else {
        return Ok(json!({ "selection": selection, "link_limits": link_limits, "result": null }));
    };
    let candidate = &candidates[index];
    let expected_active = resources
        .groups
        .iter()
        .find(|group| group.latest.group == candidate.group)
        .and_then(|group| group.active.as_ref())
        .map(|active| active.version.epoch);
    let operation = ServingOperation {
        id,
        action: prepare(
            candidate.group,
            candidate.expected_epoch,
            expected_active,
            candidate.plan.clone(),
            activation,
            recipes[index].clone(),
            Some(request_sha256),
        ),
    };
    let result = apply(client, peers.clone(), &operation).await?;
    if result == Err(ServingError::OperationIdReused)
        && let Some(previous) = replay(client, peers, id, request_sha256).await?
    {
        return Ok(previous);
    }
    Ok(json!({ "selection": selection, "link_limits": link_limits, "result": result }))
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
