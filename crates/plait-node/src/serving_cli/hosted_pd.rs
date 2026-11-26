use std::{collections::BTreeSet, io};

use plait_core::serving::{
    HostedPdReconfiguration, RouteSetId, ServingAction, ServingError, ServingLookup,
    ServingOperation, ServingView,
};
use plait_core::{NodeId, OpId};
use plait_node::Client;
use plait_parallel::serving::{
    HostedLinkPolicy, HostedPdPool, ServingForecast, plan_hosted_pd_with_releases,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{apply, hosted::probe_endpoints, query};
use crate::hosted_probe;

#[derive(Deserialize)]
pub(super) struct Replan {
    id: OpId,
    model: String,
    expected_revision: u64,
    forecast: ServingForecast,
    pool: HostedPdPool,
    link_policy: HostedLinkPolicy,
}

pub(super) async fn replan(
    client: &mut Client,
    peers: impl Iterator<Item = NodeId> + Clone,
    input: Replan,
) -> io::Result<Value> {
    let Replan {
        id,
        model,
        expected_revision,
        mut forecast,
        pool,
        link_policy,
    } = input;
    forecast.now_ms = 0;
    let request = serde_json::to_vec(&(
        "hosted-pd-plan-v1",
        &model,
        expected_revision,
        &forecast,
        &pool,
        link_policy,
    ))?;
    let request_sha256 = Sha256::digest(request).into();
    if let Some(previous) = replay(client, peers.clone(), id, request_sha256).await? {
        return Ok(previous);
    }
    let key = RouteSetId::for_model(&model);
    let lookup = if pool.can_reconfigure() {
        ServingLookup::HostedPdReconfiguration(key)
    } else {
        ServingLookup::HostedPdPlanning(key)
    };
    let view = query(client, peers.clone(), lookup).await?;
    let HostedPdReconfiguration {
        planning: snapshot,
        releases,
    } = match view {
        ServingView::HostedPdPlanning(planning) => HostedPdReconfiguration {
            planning,
            releases: Vec::new(),
        },
        ServingView::HostedPdReconfiguration(snapshot) => snapshot,
        _ => return Err(io::Error::other("expected a hosted P/D planning snapshot")),
    };
    if snapshot.revision != expected_revision {
        let result: Result<(), _> = Err(ServingError::StaleDirectory {
            current: snapshot.revision,
        });
        return Ok(json!({ "selection": null, "result": result }));
    }
    if let Some(staged) = snapshot.pending {
        let result: Result<(), _> = Err(ServingError::HostedPdInProgress { staged });
        return Ok(json!({ "selection": null, "result": result }));
    }
    let hosts: BTreeSet<_> = snapshot
        .inventory
        .hosts
        .iter()
        .map(|entry| entry.host)
        .collect();
    let links =
        hosted_probe::collect(probe_endpoints(client, peers.clone()).await?, Some(&hosts)).await;
    forecast.now_ms = hosted_probe::now_ms()?;
    let selection = plan_hosted_pd_with_releases(
        &model,
        &snapshot,
        &forecast,
        &pool,
        link_policy,
        &links,
        &releases,
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let mut reply = json!({ "selection": selection, "result": null });
    if let Some(decision) = selection.selected {
        let operation = ServingOperation {
            id,
            action: decision.into_action(model, expected_revision, pool.kv, Some(request_sha256)),
        };
        let result = apply(client, peers.clone(), &operation).await?;
        if result == Err(ServingError::OperationIdReused)
            && let Some(previous) = replay(client, peers, id, request_sha256).await?
        {
            return Ok(previous);
        }
        reply["result"] = json!(result);
    }
    Ok(reply)
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
    let prior = match receipt.operation.action {
        ServingAction::PreparePlannedHostedPd { request_sha256, .. } => Some(request_sha256),
        ServingAction::PreparePlannedHostedPdTransition { request_sha256, .. } => request_sha256,
        _ => None,
    };
    if prior == Some(request_sha256) {
        return Ok(Some(json!({ "replayed": true, "result": receipt.result })));
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "operation ID was used for a different serving action",
    ))
}
