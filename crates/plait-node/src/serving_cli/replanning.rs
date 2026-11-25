use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use plait_core::OpId;
use plait_core::serving::{
    DecoderPreparation, DecoderStandby, GroupId, PdRoute, RouteSetId, ServingAction, ServingLookup,
    ServingOperation, ServingView,
};
use plait_node::Client;
use plait_parallel::serving::{
    DecoderLayoutCandidate, DecoderPlacement, DecoderPool, DecoderSelection, ServingForecast,
    SloProfile, generate_decoder_layouts, select_decoder_layout,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{NativePlanInput, apply, query, query_resources};

#[derive(Deserialize, Serialize)]
pub(super) struct Input {
    prepare: Vec<PlacementInput>,
    paths: Vec<PdRoute>,
    #[serde(default)]
    standbys: Vec<DecoderStandby>,
    transition_loss_requests: f64,
}

#[derive(Deserialize, Serialize)]
struct PlacementInput {
    group: GroupId,
    expected_epoch: u64,
    plan: NativePlanInput,
    capacity: SloProfile,
    ready_after_ms: u64,
}

impl TryFrom<Input> for DecoderLayoutCandidate {
    type Error = io::Error;

    fn try_from(input: Input) -> io::Result<Self> {
        let placements = input
            .prepare
            .into_iter()
            .map(|placement| {
                Ok(DecoderPlacement {
                    group: placement.group,
                    expected_epoch: placement.expected_epoch,
                    plan: placement.plan.try_into()?,
                    capacity: placement.capacity,
                    ready_after_ms: placement.ready_after_ms,
                })
            })
            .collect::<io::Result<_>>()?;
        Ok(Self {
            placements,
            paths: input.paths,
            standbys: input.standbys,
            transition_loss_requests: input.transition_loss_requests,
        })
    }
}

pub(super) async fn run(
    client: &mut Client,
    peers: impl Iterator<Item = plait_core::NodeId> + Clone,
    id: OpId,
    expected_revision: u64,
    mut forecast: ServingForecast,
    inputs: Vec<Input>,
    pool: Option<DecoderPool>,
) -> io::Result<Value> {
    if pool.is_some() && !inputs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "provide a decoder pool or explicit candidates",
        ));
    }
    forecast.now_ms = 0;
    let from_pool = pool.is_some();
    let request_bytes = match &pool {
        Some(pool) => serde_json::to_vec(&(&forecast, pool))?,
        None => serde_json::to_vec(&(&forecast, &inputs))?,
    };
    let request_sha256: [u8; 32] = Sha256::digest(request_bytes).into();
    let mut candidates = inputs
        .into_iter()
        .map(DecoderLayoutCandidate::try_from)
        .collect::<io::Result<Vec<_>>>()?;
    let ServingView::Operation(receipt) =
        query(client, peers.clone(), ServingLookup::Operation(id)).await?
    else {
        return Err(io::Error::other("expected a serving operation receipt"));
    };
    if let Some(receipt) = receipt {
        if let ServingAction::PrepareWeightedDecoderSwitch {
            model,
            expected_revision: prior,
            request_sha256: prior_digest,
            from_pool: prior_from_pool,
            ..
        } = &receipt.operation.action
            && *model == forecast.model
            && *prior == expected_revision
            && *prior_digest == request_sha256
            && *prior_from_pool == from_pool
        {
            return Ok(json!({"replayed":true, "result":receipt.result}));
        }
        if let ServingAction::PrepareDecoderSwitchFromPool {
            model,
            expected_revision: prior,
            request_sha256: prior_digest,
            ..
        } = &receipt.operation.action
            && *model == forecast.model
            && *prior == expected_revision
            && from_pool
            && *prior_digest == request_sha256
        {
            return Ok(json!({"replayed":true, "result":receipt.result}));
        }
        if !from_pool
            && let ServingAction::PrepareDecoderSwitch {
                model,
                expected_revision: prior,
                prepare,
                paths,
                standbys,
            } = &receipt.operation.action
            && *model == forecast.model
            && *prior == expected_revision
            && candidates.iter().any(|candidate| {
                candidate.paths == *paths
                    && candidate.standbys == *standbys
                    && candidate.placements.len() == prepare.len()
                    && candidate.placements.iter().zip(prepare).all(|(a, b)| {
                        a.group == b.group
                            && a.expected_epoch == b.expected_epoch
                            && a.plan == b.plan
                    })
            })
        {
            return Ok(json!({"replayed":true, "result":receipt.result}));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "operation ID was used for a different serving action",
        ));
    }

    let key = RouteSetId::for_model(&forecast.model);
    let ServingView::DecoderRoutes(directory) =
        query(client, peers.clone(), ServingLookup::DecoderRoutes(key)).await?
    else {
        return Err(io::Error::other("expected a decoder directory"));
    };
    let current = directory.as_ref().map_or(0, |directory| directory.revision);
    if current != expected_revision {
        return Err(io::Error::other(format!(
            "decoder directory revision is {current}"
        )));
    }
    let active = directory.map_or_else(Vec::new, |directory| directory.paths);
    let incumbent: Vec<_> = active.iter().map(|route| route.path()).collect();
    let resources = query_resources(client, peers.clone()).await?;
    forecast.now_ms = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_millis(),
    )
    .map_err(io::Error::other)?;
    if let Some(pool) = &pool {
        candidates = generate_decoder_layouts(&resources, &active, &mut forecast, pool)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    }
    let selection = select_decoder_layout(&resources, &forecast, &incumbent, &candidates)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let generated = from_pool.then(|| {
        candidates
            .iter()
            .map(|candidate| candidate.paths.as_slice())
            .collect::<Vec<_>>()
    });
    let Some(index) = selection.summary.selected else {
        return Ok(response(selection, None, generated));
    };
    let candidate = &candidates[index];
    let prepare = candidate
        .placements
        .iter()
        .map(|placement| DecoderPreparation {
            group: placement.group,
            expected_epoch: placement.expected_epoch,
            expected_active: resources
                .groups
                .iter()
                .find(|group| group.latest.group == placement.group)
                .and_then(|group| group.active.as_ref())
                .map(|active| active.version.epoch),
            plan: placement.plan.clone(),
        })
        .collect();
    let action = ServingAction::PrepareWeightedDecoderSwitch {
        model: forecast.model,
        expected_revision,
        request_sha256,
        from_pool,
        prepare,
        paths: candidate.paths.clone(),
        standbys: candidate.standbys.clone(),
        weights: selection.route_weights.clone(),
    };
    let result = apply(client, peers, &ServingOperation { id, action }).await?;
    Ok(response(selection, Some(result), generated))
}

fn response(
    selection: DecoderSelection,
    result: Option<plait_core::serving::ServingResult>,
    generated: Option<Vec<&[PdRoute]>>,
) -> Value {
    let mut result = json!({"selection":selection, "result":result});
    if let Some(generated) = generated {
        result["generated_paths"] = json!(generated);
    }
    result
}
