use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use plait_core::serving::{
    HostedActivation, ServingAction, ServingError, ServingLookup, ServingOperation, ServingRole,
    ServingView,
};
use plait_core::{NodeId, OpId};
use plait_node::Client;
use plait_parallel::serving::{ServingCandidate, ServingForecast, select_serving_candidate};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{CandidateInput, apply, query, query_resources};

pub(super) async fn replan(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId> + Clone,
    id: OpId,
    mut forecast: ServingForecast,
    inputs: Vec<CandidateInput>,
    activation: HostedActivation,
) -> io::Result<Value> {
    forecast.now_ms = 0;
    let request_sha256: [u8; 32] =
        Sha256::digest(serde_json::to_vec(&(&forecast, &inputs, activation))?).into();
    if let Some(result) = replay(client, peers.clone(), id, request_sha256).await? {
        return Ok(result);
    }
    let candidates = inputs
        .into_iter()
        .map(ServingCandidate::try_from)
        .collect::<io::Result<Vec<_>>>()?;
    if candidates.iter().any(|candidate| {
        candidate.plan.role != ServingRole::Aggregated || candidate.plan.pipeline_parallel != 1
    }) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected hosted TP candidates",
        ));
    }
    let resources = query_resources(client, peers.clone()).await?;
    forecast.now_ms = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_millis(),
    )
    .map_err(io::Error::other)?;
    let selection = select_serving_candidate(&resources, &forecast, &candidates)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let Some(index) = selection.selected else {
        return Ok(json!({ "selection": selection, "result": null }));
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
        action: ServingAction::PrepareHostedSwitch {
            group: candidate.group,
            expected_epoch: candidate.expected_epoch,
            expected_active,
            plan: candidate.plan.clone(),
            activation,
            request_sha256: Some(request_sha256),
        },
    };
    let result = apply(client, peers.clone(), &operation).await?;
    if result == Err(ServingError::OperationIdReused)
        && let Some(previous) = replay(client, peers, id, request_sha256).await?
    {
        return Ok(previous);
    }
    Ok(json!({ "selection": selection, "result": result }))
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
