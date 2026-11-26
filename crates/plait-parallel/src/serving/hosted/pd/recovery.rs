use std::collections::BTreeMap;

use plait_core::NodeId;
use plait_core::serving::{
    GroupVersion, HostedLinkSample, HostedPdPlacement, HostedPdRecovery, PdRoute, ServingAction,
};

use super::super::pool::PoolTarget;

use super::{
    EndpointContext, HostedLinkPolicy, HostedPdPool, LAYOUTS_PER_SHAPE, LinkEvidence, Traffic,
    available, generate_layouts, validate_placement,
};

#[derive(Clone)]
struct RecoveryLayout {
    placements: Vec<HostedPdPlacement>,
    versions: BTreeMap<GroupVersion, GroupVersion>,
    ready_after_ms: u64,
    compute_rps: f64,
    traffic: Traffic,
}

impl RecoveryLayout {
    fn rate(&self) -> f64 {
        self.compute_rps.min(self.traffic.rate_limit())
    }
}

/// Replace unavailable endpoints of the committed graph using currently grantable resources.
pub fn plan_hosted_pd_recovery(
    model: &str,
    checkpoint: &str,
    now_ms: u64,
    recovery: &HostedPdRecovery,
    pool: &HostedPdPool,
    policy: HostedLinkPolicy,
    links: &BTreeMap<(NodeId, NodeId), HostedLinkSample>,
) -> Result<Option<ServingAction>, &'static str> {
    let snapshot = &recovery.planning;
    if snapshot.pending.is_some() || recovery.failed.is_empty() {
        return Ok(None);
    }
    if !snapshot.managed || recovery.paths.is_empty() {
        return Err("hosted P/D recovery requires committed paths");
    }
    if checkpoint != recovery.checkpoint || pool.kv != recovery.kv {
        return Err("hosted P/D recovery requires the committed checkpoint and KV contract");
    }
    policy.validate()?;
    pool.validate(model, now_ms)?;
    let store_host = pool.store_host(&snapshot.inventory)?;
    let evidence = LinkEvidence {
        policy,
        now_ms,
        links,
    };
    let context = EndpointContext {
        evidence,
        store_host: Some(store_host),
        kv_bytes: pool.kv_bytes_per_request,
    };
    let mut layouts = vec![RecoveryLayout {
        placements: Vec::new(),
        versions: BTreeMap::new(),
        ready_after_ms: 0,
        compute_rps: pool.store_bytes_per_second as f64 / (2.0 * pool.kv_bytes_per_request as f64),
        traffic: Traffic::new(),
    }];
    for (&failed, &role) in &recovery.failed {
        let Some(slot) = pool
            .slots
            .iter()
            .find(|slot| slot.pool.group() == failed.group && slot.role == role)
        else {
            return Ok(None);
        };
        if snapshot
            .resources
            .groups
            .iter()
            .any(|history| history.latest.group == failed.group && history.active.is_some())
        {
            return Ok(None);
        }
        let affected_paths = recovery
            .paths
            .iter()
            .filter(|path| path.prefill == failed || path.decode == failed)
            .count();
        if affected_paths == 0 {
            return Err("hosted P/D recovery requires failures in the committed graph");
        }
        let share = affected_paths as f64 / recovery.paths.len() as f64;
        let mut next = Vec::new();
        for layout in &layouts {
            let (resources, inventory) = available(snapshot, &layout.placements);
            for (candidate, recipe) in generate_layouts(
                &resources,
                &inventory,
                checkpoint,
                &slot.pool,
                evidence,
                PoolTarget::Split(role, &pool.kv),
            )? {
                let Ok(version) = validate_placement(
                    &resources,
                    checkpoint,
                    failed.group,
                    candidate.expected_epoch,
                    &candidate.plan,
                ) else {
                    continue;
                };
                let Some(endpoint) = context.candidate(version, &candidate, &recipe, &slot.pool)
                else {
                    continue;
                };
                if !endpoint.capacity.is_valid(now_ms) || endpoint.capacity.rps == 0.0 {
                    continue;
                }
                let mut trial = layout.clone();
                trial.ready_after_ms = trial.ready_after_ms.max(candidate.ready_after_ms);
                trial.compute_rps = trial.compute_rps.min(endpoint.capacity.rps / share);
                trial.traffic.include(&endpoint.traffic, share);
                trial.versions.insert(failed, version);
                trial.placements.push(HostedPdPlacement {
                    group: failed.group,
                    expected_epoch: candidate.expected_epoch,
                    expected_active: None,
                    plan: candidate.plan,
                    recipe,
                });
                next.push(trial);
            }
        }
        if next.is_empty() {
            return Ok(None);
        }
        next.sort_by(|left, right| {
            left.ready_after_ms
                .cmp(&right.ready_after_ms)
                .then_with(|| right.rate().total_cmp(&left.rate()))
        });
        next.truncate(LAYOUTS_PER_SHAPE);
        layouts = next;
    }
    Ok(layouts.into_iter().next().map(|layout| {
        let successor = |version| layout.versions.get(&version).copied().unwrap_or(version);
        let paths = recovery
            .paths
            .iter()
            .map(|path| PdRoute {
                prefill: successor(path.prefill),
                decode: successor(path.decode),
            })
            .collect();
        ServingAction::PrepareHostedPd {
            model: model.to_owned(),
            expected_revision: snapshot.revision,
            kv: pool.kv.clone(),
            placements: layout.placements,
            paths,
        }
    }))
}
