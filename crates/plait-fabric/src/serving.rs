//! Deterministic serving control state applied by the replicated runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;

use plait_core::serving::{
    ActiveGroup, ExecutionHostLiveness, GroupId, GroupPhase, GroupPlan, GroupResources,
    GroupVersion, GroupView, HostMemory, HostResources, HostedRecipe, PdRoute, ResidentOwner,
    ResolvedPdRoute, RouteDirectory, RouteSetId, ServingAction, ServingEndpoint, ServingError,
    ServingOperation, ServingOutcome, ServingReceipt, ServingResources, ServingResult, ServingRole,
    W4_MAX_ADMISSION_MS, W4DrainStatus, W4GroupLayout, W4ShardLayout, WorkerDispatch, WorkerGrant,
    WorkerResources,
};
use plait_core::{DeviceId, NodeId, OpId};

mod decoder;
mod host;
mod host_launch;
use decoder::{DecoderState, RecoveryPolicy};
use host_launch::{HostLaunch, HostedSwitch};

#[derive(Clone)]
struct HostState {
    capacity: HostMemory,
    reserved: HostMemory,
}

#[derive(Clone)]
struct WorkerState {
    incarnation: u64,
    device_bytes: u64,
    fence: u64,
    grant: Option<WorkerGrant>,
}

#[derive(Default)]
struct Reservations {
    devices: BTreeSet<DeviceId>,
    hosts: BTreeMap<NodeId, HostMemory>,
}

#[derive(Clone, Default)]
struct GroupHistory {
    versions: BTreeMap<u64, GroupView>,
    active: Option<u64>,
}

#[derive(Clone)]
struct RouteSet {
    model: String,
    revision: u64,
    paths: Vec<PdRoute>,
}

#[derive(Clone, Copy)]
struct DispatchEndpoint {
    addr: SocketAddr,
    w4: Option<W4ShardLayout>,
}

#[derive(Clone)]
struct W4Switch {
    incumbent: Option<u64>,
    layout: W4GroupLayout,
}

#[derive(Clone, Default)]
pub(crate) struct ServingState {
    groups: BTreeMap<GroupId, GroupHistory>,
    hosts: BTreeMap<NodeId, HostState>,
    workers: BTreeMap<DeviceId, WorkerState>,
    receipts: BTreeMap<OpId, ServingReceipt>,
    routes: BTreeMap<RouteSetId, RouteSet>,
    dispatch: BTreeMap<GroupVersion, BTreeMap<DeviceId, DispatchEndpoint>>,
    w4_switches: BTreeMap<GroupVersion, W4Switch>,
    w4_work: BTreeMap<GroupVersion, BTreeMap<OpId, u64>>,
    resident: BTreeMap<GroupVersion, BTreeMap<DeviceId, ResidentOwner>>,
    host_launches: BTreeMap<GroupVersion, HostLaunch>,
    hosted_switches: BTreeMap<GroupVersion, HostedSwitch>,
    hosted_recipes: BTreeMap<GroupVersion, HostedRecipe>,
    execution_hosts: BTreeMap<NodeId, ExecutionHostLiveness>,
    decoder: DecoderState,
}

impl ServingState {
    pub(crate) fn group(&self, version: GroupVersion) -> Option<&GroupView> {
        self.groups
            .get(&version.group)?
            .versions
            .get(&version.epoch)
    }

    pub(crate) fn receipt(&self, id: OpId) -> Option<&ServingReceipt> {
        self.receipts.get(&id)
    }

    pub(crate) fn resident_owner(&self, grant: WorkerGrant) -> Option<ResidentOwner> {
        if self.group(grant.version)?.grants.get(&grant.device) != Some(&grant) {
            return None;
        }
        self.resident
            .get(&grant.version)?
            .get(&grant.device)
            .copied()
    }

    pub(crate) fn dispatch(&self, version: GroupVersion) -> Vec<WorkerDispatch> {
        let Some(view) = self.group(version) else {
            return Vec::new();
        };
        self.dispatch
            .get(&version)
            .into_iter()
            .flatten()
            .filter_map(|(device, endpoint)| {
                let grant = *view.grants.get(device)?;
                view.holds_grant(grant).then_some(WorkerDispatch {
                    grant,
                    addr: endpoint.addr,
                })
            })
            .collect()
    }

    pub(crate) fn w4_drain(&self, version: GroupVersion) -> Option<W4DrainStatus> {
        let work = self.w4_work.get(&version)?;
        Some(W4DrainStatus {
            pending: work.len() as u64,
            next_expiry_ms: work.values().min().copied(),
        })
    }

    pub(crate) fn routes(&self, id: RouteSetId) -> Option<RouteDirectory> {
        let routes = self.routes.get(&id)?;
        Some(RouteDirectory {
            model: routes.model.clone(),
            revision: routes.revision,
            paths: routes
                .paths
                .iter()
                .filter_map(|&path| self.resolve_route(&routes.model, path, None))
                .collect(),
        })
    }

    pub(crate) fn resources(&self) -> ServingResources {
        ServingResources {
            hosts: self
                .hosts
                .iter()
                .map(|(&host, state)| HostResources {
                    host,
                    capacity: state.capacity,
                    reserved: state.reserved,
                })
                .collect(),
            workers: self
                .workers
                .iter()
                .map(|(&device, worker)| WorkerResources {
                    device,
                    incarnation: worker.incarnation,
                    device_bytes: worker.device_bytes,
                    grant: worker.grant,
                })
                .collect(),
            groups: self
                .groups
                .values()
                .filter_map(|history| {
                    let (_, latest) = history.versions.last_key_value()?;
                    let active = history.active.map(|epoch| {
                        let view = &history.versions[&epoch];
                        ActiveGroup {
                            version: view.version,
                            role: view.plan.role,
                            model: view.plan.model.clone(),
                        }
                    });
                    Some(GroupResources {
                        latest: latest.version,
                        phase: latest.phase,
                        active,
                    })
                })
                .collect(),
        }
    }

    pub(crate) fn apply(&mut self, operation: &ServingOperation) -> ServingResult {
        if let Some(receipt) = self.receipt(operation.id) {
            return if &receipt.operation == operation {
                receipt.result
            } else {
                Err(ServingError::OperationIdReused)
            };
        }
        let result = self.transition(operation);
        self.receipts.insert(
            operation.id,
            ServingReceipt {
                operation: operation.clone(),
                result,
            },
        );
        result
    }

    fn transition(&mut self, operation: &ServingOperation) -> ServingResult {
        match &operation.action {
            ServingAction::SetHostMemory { host, capacity } => {
                self.set_host_memory(*host, *capacity)
            }
            ServingAction::RegisterWorker {
                device,
                incarnation,
                device_bytes,
            } => self.register_worker(*device, *incarnation, *device_bytes),
            ServingAction::Prepare {
                group,
                expected_epoch,
                plan,
            } => self.prepare(*group, *expected_epoch, plan),
            ServingAction::PrepareHostedSwitch {
                group,
                expected_epoch,
                expected_active,
                plan,
                activation,
                ..
            } => self.prepare_hosted_switch(
                *group,
                *expected_epoch,
                *expected_active,
                plan,
                *activation,
            ),
            ServingAction::PrepareHostedDeployment {
                group,
                expected_epoch,
                expected_active,
                plan,
                activation,
                recipe,
                ..
            } => {
                recipe.validate(plan)?;
                if let Some(epoch) = expected_active
                    && let Some(endpoint) = self
                        .group(GroupVersion {
                            group: *group,
                            epoch: *epoch,
                        })
                        .and_then(|view| view.endpoint.as_ref())
                    && endpoint.api_model != recipe.api_model
                {
                    return Err(ServingError::InvalidEndpoint);
                }
                let result = self.prepare_hosted_switch(
                    *group,
                    *expected_epoch,
                    *expected_active,
                    plan,
                    *activation,
                )?;
                let ServingOutcome::Group { version, .. } = result else {
                    unreachable!("hosted preparation creates a group version");
                };
                self.hosted_recipes.insert(version, recipe.clone());
                Ok(result)
            }
            ServingAction::PrepareW4Switch {
                group,
                expected_epoch,
                expected_active,
                plan,
                layout,
            } => self.prepare_w4_switch(*group, *expected_epoch, *expected_active, plan, layout),
            ServingAction::PrepareManagedW4 {
                group,
                expected_epoch,
                expected_active,
                plan,
                layout,
            } => {
                let result = self.prepare_w4_switch(
                    *group,
                    *expected_epoch,
                    *expected_active,
                    plan,
                    layout,
                )?;
                let ServingOutcome::Group { version, .. } = result else {
                    unreachable!("prepare creates a group version");
                };
                self.w4_work.insert(version, BTreeMap::new());
                Ok(result)
            }
            ServingAction::AdmitW4 {
                group,
                now_ms,
                expires_at_ms,
            } => self.admit_w4(*group, operation.id, *now_ms, *expires_at_ms),
            ServingAction::CheckW4 { grant, work } => {
                self.check_w4(grant.version, std::slice::from_ref(grant), *work)
            }
            ServingAction::FinishW4 { version, work } => self.finish_w4(*version, *work),
            ServingAction::CompleteW4 {
                version,
                work,
                grants,
            } => self.complete_w4(*version, *work, grants),
            ServingAction::ExpireW4 {
                version,
                through_ms,
            } => self.expire_w4(*version, *through_ms),
            ServingAction::DecoderReady {
                endpoint,
                launch_digest,
            } => self.decoder_ready(*endpoint, *launch_digest, false),
            ServingAction::DecoderReadyManaged {
                endpoint,
                launch_digest,
            } => self.decoder_ready(*endpoint, *launch_digest, true),
            ServingAction::ExpireDecoder {
                model,
                path,
                through_ms,
            } => self.expire_decoder(model, *path, *through_ms),
            ServingAction::PublishDecoderRoutes {
                model,
                expected_revision,
                paths,
                activate,
            } => self.publish_decoder_routes(model, *expected_revision, paths, *activate, &[]),
            ServingAction::PublishDecoderStandbys {
                model,
                expected_revision,
                paths,
                activate,
                standbys,
            } => self.publish_decoder_routes(model, *expected_revision, paths, *activate, standbys),
            ServingAction::FailoverDecoder { model, intent } => {
                self.failover_decoder(model, intent)
            }
            ServingAction::StageDecoderSwitch {
                model,
                expected_revision,
                paths,
                standbys,
            } => {
                self.stage_decoder_switch(model, *expected_revision, paths, standbys, operation.id)
            }
            ServingAction::CommitDecoderSwitch { model, intent } => {
                self.commit_decoder_switch(model, intent)
            }
            ServingAction::CancelDecoderSwitch { model, staged } => {
                self.cancel_decoder_switch(model, *staged)
            }
            ServingAction::PrepareDecoderSwitch {
                model,
                expected_revision,
                prepare,
                paths,
                standbys,
            } => self.prepare_decoder_switch(
                model,
                *expected_revision,
                prepare,
                paths,
                standbys,
                operation.id,
                None,
            ),
            ServingAction::PrepareDecoderSwitchFromPool {
                model,
                expected_revision,
                prepare,
                paths,
                standbys,
                ..
            } => {
                if paths.len() != 1
                    || !standbys.is_empty()
                    || !self.supports_decoder_pool(model, false)
                {
                    return Err(ServingError::InvalidDirectory);
                }
                self.prepare_decoder_switch(
                    model,
                    *expected_revision,
                    prepare,
                    paths,
                    standbys,
                    operation.id,
                    None,
                )
            }
            ServingAction::PrepareWeightedDecoderSwitch {
                model,
                expected_revision,
                from_pool,
                prepare,
                paths,
                standbys,
                weights,
                ..
            } => {
                if *from_pool && (!standbys.is_empty() || !self.supports_decoder_pool(model, true))
                {
                    return Err(ServingError::InvalidDirectory);
                }
                self.prepare_decoder_switch(
                    model,
                    *expected_revision,
                    prepare,
                    paths,
                    standbys,
                    operation.id,
                    Some(weights),
                )
            }
            ServingAction::AdmitDecoder {
                model,
                revision,
                path,
                now_ms,
            } => self.admit_decoder(
                model,
                *revision,
                *path,
                operation.id,
                *now_ms,
                RecoveryPolicy::None,
            ),
            ServingAction::ConfigureDecoderWeights {
                model,
                expected_revision,
                weights,
            } => self.configure_decoder_weights(model, *expected_revision, weights),
            ServingAction::AdmitDecoderRouted {
                model,
                routes,
                now_ms,
                replay,
            } => self.admit_decoder_routed(model, routes, operation.id, *now_ms, *replay),
            ServingAction::SubmitDecoderRouted {
                model,
                routes,
                previous,
                now_ms,
                request,
            } => self.submit_decoder_routed(
                model,
                routes,
                *previous,
                *now_ms,
                *request,
                operation.id,
            ),
            ServingAction::AdmitDecoderWithRecovery {
                model,
                revision,
                path,
                now_ms,
            } => self.admit_decoder(
                model,
                *revision,
                *path,
                operation.id,
                *now_ms,
                RecoveryPolicy::Decode,
            ),
            ServingAction::AdmitDecoderWithRetry {
                model,
                revision,
                path,
                now_ms,
            } => self.admit_decoder(
                model,
                *revision,
                *path,
                operation.id,
                *now_ms,
                RecoveryPolicy::Pipeline,
            ),
            ServingAction::AdmitDecoderReplay {
                model,
                revision,
                path,
                now_ms,
                max_tokens,
            } => self.admit_decoder_replay(
                model,
                *revision,
                *path,
                operation.id,
                *now_ms,
                *max_tokens,
                None,
            ),
            ServingAction::AdmitResumableDecoder {
                model,
                revision,
                path,
                now_ms,
                max_tokens,
                input_sha256,
            } => self.admit_decoder_replay(
                model,
                *revision,
                *path,
                operation.id,
                *now_ms,
                *max_tokens,
                Some(*input_sha256),
            ),
            ServingAction::TakeoverDecoder {
                model,
                work,
                expected_entry,
                revision,
                previous,
                replacement,
                now_ms,
                offset,
                input_sha256,
            } => self.takeover_decoder(
                model,
                *work,
                *expected_entry,
                operation.id,
                *revision,
                *previous,
                *replacement,
                *now_ms,
                *offset,
                *input_sha256,
                None,
            ),
            ServingAction::TakeoverDecoderWithRetry {
                model,
                work,
                expected_entry,
                revision,
                previous,
                replacement,
                now_ms,
                offset,
                input_sha256,
                retry,
            } => self.takeover_decoder(
                model,
                *work,
                *expected_entry,
                operation.id,
                *revision,
                *previous,
                *replacement,
                *now_ms,
                *offset,
                *input_sha256,
                Some(*retry),
            ),
            ServingAction::SubmitDecoder {
                model,
                revision,
                path,
                previous,
                now_ms,
                request,
            } => self.submit_decoder(
                model,
                operation.id,
                *revision,
                *path,
                *previous,
                *now_ms,
                *request,
            ),
            ServingAction::AppendDecoderOutput {
                model,
                work,
                path,
                grant,
                now_ms,
                owner,
                offset,
                tokens,
                checkpoint,
            } => self.append_decoder_output(
                model,
                *work,
                *path,
                *grant,
                *now_ms,
                *owner,
                *offset,
                tokens,
                *checkpoint,
                operation.id,
            ),
            ServingAction::RecoverDecoderWork {
                model,
                work,
                revision,
                failed,
                replacement,
                now_ms,
            } => {
                self.check_decoder_entry(model, *work, None)?;
                self.recover_decoder_work(
                    model,
                    *work,
                    *revision,
                    *failed,
                    *replacement,
                    *now_ms,
                    false,
                )
            }
            ServingAction::RestartDecoderWork {
                model,
                work,
                revision,
                failed,
                replacement,
                now_ms,
            } => {
                self.check_decoder_entry(model, *work, None)?;
                self.recover_decoder_work(
                    model,
                    *work,
                    *revision,
                    *failed,
                    *replacement,
                    *now_ms,
                    true,
                )
            }
            ServingAction::RebindDecoderEntry {
                model,
                work,
                entry,
                revision,
                failed,
                replacement,
                now_ms,
                restart_prefill,
            } => {
                self.check_decoder_entry(model, *work, Some(*entry))?;
                self.recover_decoder_work(
                    model,
                    *work,
                    *revision,
                    *failed,
                    *replacement,
                    *now_ms,
                    *restart_prefill,
                )
            }
            ServingAction::AdvanceDecoder {
                model,
                work,
                path,
                grant,
                now_ms,
                step,
            } => self.advance_decoder(model, *work, *path, *grant, *now_ms, *step, operation.id),
            ServingAction::CancelDecoder { model, work, path } => {
                self.check_decoder_entry(model, *work, None)?;
                self.cancel_decoder(model, *work, *path)
            }
            ServingAction::CancelDecoderEntry {
                model,
                work,
                entry,
                path,
            } => {
                self.check_decoder_entry(model, *work, Some(*entry))?;
                self.cancel_decoder(model, *work, *path)
            }
            ServingAction::WorkerReady {
                grant,
                launch_digest,
            } => {
                if self.w4_switches.contains_key(&grant.version) {
                    return Err(ServingError::InvalidEndpoint);
                }
                self.worker_ready(*grant, *launch_digest)
            }
            ServingAction::WorkerReadyAt {
                grant,
                launch_digest,
                addr,
            } => self.worker_ready_at(*grant, *launch_digest, *addr, None),
            ServingAction::WorkerReadyW4 {
                grant,
                launch_digest,
                addr,
                shard,
            } => self.worker_ready_at(*grant, *launch_digest, *addr, Some(*shard)),
            ServingAction::Activate(version) => self.activate(*version),
            ServingAction::Drain(version) => self.quiesce(*version, GroupPhase::Draining),
            ServingAction::Revoke(version) => self.quiesce(*version, GroupPhase::Revoked),
            ServingAction::WorkerStopped(grant) => self.worker_stopped(*grant),
            ServingAction::ReclaimResident {
                grant,
                owner,
                workers,
            } => self.reclaim_resident(*grant, *owner, workers),
            ServingAction::ClaimResident {
                grant,
                owner,
                launch_digest,
            } => self.claim_resident(*grant, *owner, *launch_digest),
            ServingAction::LoseResident { grant, owner } => self.lose_resident(*grant, *owner),
            ServingAction::PulseExecutionHost {
                host,
                supervisor_lock,
            } => self.pulse_execution_host(*host, operation.id, *supervisor_lock),
            ServingAction::ExpireExecutionHost {
                host,
                pulse,
                groups,
            } => self.expire_execution_host(*host, *pulse, groups),
            ServingAction::Retire { version, workers } => self.retire(*version, workers),
            ServingAction::ClaimLaunch {
                version,
                host,
                holder,
                launch_digest,
            } => self.claim_launch(*version, *host, *holder, *launch_digest),
            ServingAction::LaunchReady { version, holder } => {
                self.launch_ready(*version, *holder, None)
            }
            ServingAction::LaunchReadyAt {
                version,
                holder,
                endpoint,
            } => self.launch_ready(*version, *holder, Some(endpoint)),
            ServingAction::ObserveLaunch { version, holder } => {
                let view = self
                    .group(*version)
                    .ok_or(ServingError::UnknownVersion(*version))?;
                check_launch_owner(view, Some(*holder))?;
                Ok(group_outcome(view.version, view.phase))
            }
            ServingAction::LaunchStopped { version, holder } => {
                self.launch_stopped(*version, *holder)
            }
            ServingAction::ClaimHostLaunch {
                version,
                host,
                holder,
                launch_digest,
            } => self.claim_host_launch(*version, *host, *holder, *launch_digest),
            ServingAction::HostLaunchReady {
                version,
                host,
                holder,
                endpoint,
            } => self.host_launch_ready(*version, *host, *holder, endpoint.as_ref()),
            ServingAction::ObserveHostLaunch {
                version,
                host,
                holder,
            } => self.observe_host_launch(*version, *host, *holder),
            ServingAction::HostLaunchStopped {
                version,
                host,
                holder,
            } => self.host_launch_stopped(*version, *host, *holder),
            ServingAction::PublishRoutes {
                model,
                expected_revision,
                paths,
                activate,
            } => self.publish_routes(model, *expected_revision, paths, *activate),
            ServingAction::ObserveRoutes { model } => {
                let revision = self
                    .routes
                    .get(&RouteSetId::for_model(model))
                    .map_or(0, |directory| directory.revision);
                Ok(ServingOutcome::RoutesObserved { revision })
            }
            ServingAction::ObserveGroup { group, epoch } => {
                let epoch = epoch
                    .or_else(|| self.groups.get(group)?.active)
                    .ok_or(ServingError::NoActiveGroup(*group))?;
                let version = GroupVersion {
                    group: *group,
                    epoch,
                };
                let view = self
                    .group(version)
                    .ok_or(ServingError::UnknownVersion(version))?;
                Ok(group_outcome(version, view.phase))
            }
        }
    }

    fn set_host_memory(&mut self, host: NodeId, capacity: HostMemory) -> ServingResult {
        if capacity.pinned_bytes > capacity.bytes {
            return Err(ServingError::InvalidCapacity);
        }
        let reserved = self
            .hosts
            .get(&host)
            .map_or(HostMemory::default(), |old| old.reserved);
        if !fits(reserved, capacity) {
            return Err(ServingError::HostCapacityExceeded(host));
        }
        self.hosts.insert(host, HostState { capacity, reserved });
        Ok(ServingOutcome::HostConfigured)
    }

    fn register_worker(
        &mut self,
        device: DeviceId,
        incarnation: u64,
        device_bytes: u64,
    ) -> ServingResult {
        if device_bytes == 0 {
            return Err(ServingError::InvalidCapacity);
        }
        let mut fence = 0;
        if let Some(old) = self.workers.get(&device) {
            if old.incarnation == incarnation && old.device_bytes == device_bytes {
                return Ok(ServingOutcome::WorkerRegistered);
            }
            if incarnation <= old.incarnation {
                return Err(ServingError::StaleIncarnation(device));
            }
            if old.grant.is_some() {
                return Err(ServingError::WorkerBusy(device));
            }
            fence = old.fence;
        }
        self.workers.insert(
            device,
            WorkerState {
                incarnation,
                device_bytes,
                fence,
                grant: None,
            },
        );
        Ok(ServingOutcome::WorkerRegistered)
    }

    fn prepare(&mut self, group: GroupId, expected_epoch: u64, plan: &GroupPlan) -> ServingResult {
        let mut reservations = Reservations::default();
        let view = self.reserve_group(group, expected_epoch, plan, &mut reservations)?;
        let version = view.version;
        self.install_groups([view], reservations);
        Ok(group_outcome(version, GroupPhase::Preparing))
    }

    fn reserve_group(
        &self,
        group: GroupId,
        expected_epoch: u64,
        plan: &GroupPlan,
        reservations: &mut Reservations,
    ) -> Result<GroupView, ServingError> {
        plan.validate()?;
        let latest = self
            .groups
            .get(&group)
            .and_then(|history| history.versions.last_key_value());
        let current = latest.map_or(0, |(epoch, _)| *epoch);
        if expected_epoch != current {
            return Err(ServingError::StaleEpoch { current });
        }
        if let Some((_, view)) = latest
            .filter(|(_, view)| matches!(view.phase, GroupPhase::Preparing | GroupPhase::Ready))
        {
            return Err(ServingError::CandidateExists(view.version));
        }
        let version = GroupVersion {
            group,
            epoch: current
                .checked_add(1)
                .ok_or(ServingError::SequenceExhausted)?,
        };

        // Build the complete set of grants and new counters before mutating any reservation.
        let mut grants = BTreeMap::new();
        for (&device, demand) in &plan.workers {
            let worker = self
                .workers
                .get(&device)
                .ok_or(ServingError::UnknownWorker(device))?;
            if worker.incarnation != demand.incarnation {
                return Err(ServingError::StaleIncarnation(device));
            }
            if worker.grant.is_some() || !reservations.devices.insert(device) {
                return Err(ServingError::WorkerBusy(device));
            }
            if demand.device_bytes > worker.device_bytes {
                return Err(ServingError::DeviceCapacityExceeded(device));
            }
            let fence = worker
                .fence
                .checked_add(1)
                .ok_or(ServingError::SequenceExhausted)?;
            grants.insert(
                device,
                WorkerGrant {
                    version,
                    device,
                    incarnation: worker.incarnation,
                    fence,
                },
            );
        }
        for (&host, demand) in &plan.hosts {
            self.require_execution_host(host)?;
            let state = self
                .hosts
                .get(&host)
                .ok_or(ServingError::UnknownHost(host))?;
            let prior = reservations
                .hosts
                .get(&host)
                .copied()
                .unwrap_or(state.reserved);
            let reserved = add_memory(prior, *demand)
                .filter(|total| fits(*total, state.capacity))
                .ok_or(ServingError::HostCapacityExceeded(host))?;
            reservations.hosts.insert(host, reserved);
        }

        Ok(GroupView {
            version,
            phase: GroupPhase::Preparing,
            plan: plan.clone(),
            grants,
            ready: BTreeSet::new(),
            stopped: BTreeSet::new(),
            launch_owner: None,
            endpoint: None,
        })
    }

    fn install_groups(
        &mut self,
        views: impl IntoIterator<Item = GroupView>,
        reservations: Reservations,
    ) {
        for (host, reserved) in reservations.hosts {
            self.hosts.get_mut(&host).expect("validated host").reserved = reserved;
        }
        for view in views {
            let version = view.version;
            for (&device, &grant) in &view.grants {
                let worker = self.workers.get_mut(&device).expect("validated worker");
                worker.fence = grant.fence;
                worker.grant = Some(grant);
            }
            self.groups
                .entry(version.group)
                .or_default()
                .versions
                .insert(version.epoch, view);
        }
    }

    fn worker_ready(&mut self, grant: WorkerGrant, launch_digest: [u8; 32]) -> ServingResult {
        self.require_unhosted(grant.version)?;
        let view = self.group_mut(grant.version)?;
        check_grant(view, grant)?;
        check_launch_owner(view, None)?;
        if !matches!(view.phase, GroupPhase::Preparing | GroupPhase::Ready) {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if view.plan.launch_digest != launch_digest {
            return Err(ServingError::LaunchDigestMismatch);
        }
        view.ready.insert(grant.device);
        if view.ready.len() == view.grants.len() {
            view.phase = GroupPhase::Ready;
        }
        Ok(group_outcome(view.version, view.phase))
    }

    fn prepare_w4_switch(
        &mut self,
        group: GroupId,
        expected_epoch: u64,
        expected_active: Option<u64>,
        plan: &GroupPlan,
        layout: &W4GroupLayout,
    ) -> ServingResult {
        if plan.role != ServingRole::Aggregated || !layout.covers(plan) {
            return Err(ServingError::InvalidPlan);
        }
        let incumbent = self.groups.get(&group).and_then(|history| history.active);
        if incumbent != expected_active {
            return Err(ServingError::ActiveEpochChanged { current: incumbent });
        }
        if let Some(epoch) = incumbent {
            let active = self
                .group(GroupVersion { group, epoch })
                .expect("active version");
            if active.plan.role != ServingRole::Aggregated || active.plan.model != plan.model {
                return Err(ServingError::InvalidPlan);
            }
            let endpoints = self
                .dispatch
                .get(&active.version)
                .ok_or(ServingError::InvalidPlan)?;
            let shards = endpoints
                .iter()
                .filter_map(|(&device, endpoint)| {
                    let shard = endpoint.w4?;
                    (shard.group_size == layout.group_size
                        && shard.output_columns == layout.output_columns)
                        .then_some((device, shard.k_range))
                })
                .collect();
            let previous = W4GroupLayout {
                shards,
                ..layout.clone()
            };
            if !previous.covers(&active.plan) {
                return Err(ServingError::InvalidPlan);
            }
        }
        let result = self.prepare(group, expected_epoch, plan)?;
        let ServingOutcome::Group { version, .. } = result else {
            unreachable!("prepare creates a group version");
        };
        self.w4_switches.insert(
            version,
            W4Switch {
                incumbent,
                layout: layout.clone(),
            },
        );
        Ok(result)
    }

    fn worker_ready_at(
        &mut self,
        grant: WorkerGrant,
        launch_digest: [u8; 32],
        addr: SocketAddr,
        shard: Option<W4ShardLayout>,
    ) -> ServingResult {
        let previous = self
            .dispatch
            .get(&grant.version)
            .and_then(|map| map.get(&grant.device));
        let switch = self.w4_switches.get(&grant.version);
        if addr.port() == 0
            || addr.ip().is_unspecified()
            || previous.is_some_and(|old| {
                old.addr != addr || old.w4.zip(shard).is_some_and(|(a, b)| a != b)
            })
            || switch
                .is_some_and(|switch| shard.is_none() || switch.layout.shard(grant.device) != shard)
        {
            return Err(ServingError::InvalidEndpoint);
        }
        let auto_activate = switch
            .is_some_and(|switch| self.groups[&grant.version.group].active == switch.incumbent);
        let endpoint = DispatchEndpoint {
            addr,
            w4: shard.or(previous.and_then(|old| old.w4)),
        };
        let result = self.worker_ready(grant, launch_digest)?;
        self.dispatch
            .entry(grant.version)
            .or_default()
            .insert(grant.device, endpoint);
        if auto_activate
            && matches!(
                result,
                ServingOutcome::Group {
                    phase: GroupPhase::Ready,
                    ..
                }
            )
        {
            // A Preparing version prevents any successor from being prepared until it is activated or revoked.
            return self.activate(grant.version);
        }
        Ok(result)
    }

    fn activate(&mut self, version: GroupVersion) -> ServingResult {
        self.check_activation(version)?;
        let history = self.groups.get_mut(&version.group).expect("checked group");
        let view = history
            .versions
            .get_mut(&version.epoch)
            .expect("checked version");
        view.phase = GroupPhase::Active;
        if let Some(previous) = history.active.replace(version.epoch) {
            history
                .versions
                .get_mut(&previous)
                .expect("active version")
                .phase = GroupPhase::Draining;
            let previous = GroupVersion {
                group: version.group,
                epoch: previous,
            };
            self.revoke_drained_w4(previous);
            self.revoke_drained_decoder(previous);
        }
        Ok(group_outcome(version, GroupPhase::Active))
    }

    fn check_activation(&self, version: GroupVersion) -> Result<(), ServingError> {
        let history = self
            .groups
            .get(&version.group)
            .ok_or(ServingError::UnknownVersion(version))?;
        let current = history
            .versions
            .last_key_value()
            .map_or(0, |(epoch, _)| *epoch);
        if current != version.epoch {
            return Err(ServingError::StaleEpoch { current });
        }
        let view = history
            .versions
            .get(&version.epoch)
            .ok_or(ServingError::UnknownVersion(version))?;
        if view.phase != GroupPhase::Ready {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        Ok(())
    }

    fn publish_routes(
        &mut self,
        model: &str,
        expected_revision: u64,
        paths: &[PdRoute],
        activate: Option<GroupVersion>,
    ) -> ServingResult {
        let id = RouteSetId::for_model(model);
        let previous = self.routes.get(&id);
        if model.is_empty() || previous.is_some_and(|directory| directory.model != model) {
            return Err(ServingError::InvalidDirectory);
        }
        let current = previous.map_or(0, |directory| directory.revision);
        if current != expected_revision {
            return Err(ServingError::StaleDirectory { current });
        }
        let revision = current
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?;
        if let Some(version) = activate {
            self.check_activation(version)?;
            if !paths
                .iter()
                .any(|path| path.prefill == version || path.decode == version)
            {
                return Err(ServingError::InvalidDirectory);
            }
        }
        let mut unique = BTreeSet::new();
        for &path in paths {
            if !unique.insert(path) || self.resolve_route(model, path, activate).is_none() {
                return Err(ServingError::InvalidRoute(path));
            }
        }
        if let Some(version) = activate {
            self.activate(version)?;
        }
        self.routes.insert(
            id,
            RouteSet {
                model: model.to_owned(),
                revision,
                paths: paths.to_vec(),
            },
        );
        Ok(ServingOutcome::RoutesPublished { revision })
    }

    fn resolve_route(
        &self,
        model: &str,
        path: PdRoute,
        activate: Option<GroupVersion>,
    ) -> Option<ResolvedPdRoute> {
        let prefill = self.group(path.prefill)?;
        let decode = self.group(path.decode)?;
        if prefill.plan.role != ServingRole::Prefill
            || decode.plan.role != ServingRole::Decode
            || prefill.plan.model != decode.plan.model
            || !routable(prefill, activate)
            || !routable(decode, activate)
        {
            return None;
        }
        let source = prefill.endpoint.as_ref()?;
        let target = decode.endpoint.as_ref()?;
        if source.api_model != model || target.api_model != model {
            return None;
        }
        Some(ResolvedPdRoute {
            route: path,
            prefill_http: source.http,
            prefill_kv: source.prefill_kv?,
            decode_http: target.http,
        })
    }

    fn quiesce(&mut self, version: GroupVersion, phase: GroupPhase) -> ServingResult {
        let view = self.group_mut(version)?;
        let allowed = match phase {
            GroupPhase::Draining => view.phase == GroupPhase::Active,
            GroupPhase::Revoked => matches!(
                view.phase,
                GroupPhase::Preparing
                    | GroupPhase::Ready
                    | GroupPhase::Active
                    | GroupPhase::Draining
            ),
            _ => unreachable!("quiesce only supports drain and revoke"),
        };
        if !allowed {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        view.phase = phase;
        let history = self.groups.get_mut(&version.group).expect("known group");
        let was_active = history.active == Some(version.epoch);
        if was_active {
            history.active = None;
        }
        let phase = if phase == GroupPhase::Draining {
            self.revoke_drained_w4(version);
            self.revoke_drained_decoder(version);
            self.group(version).expect("known group").phase
        } else {
            if let Some(work) = self.w4_work.get_mut(&version) {
                work.clear();
            }
            if was_active {
                self.hosted_incumbent_revoked(version);
            }
            phase
        };
        Ok(group_outcome(version, phase))
    }

    fn admit_w4(
        &mut self,
        group: GroupId,
        id: OpId,
        now_ms: u64,
        expires_at_ms: u64,
    ) -> ServingResult {
        if expires_at_ms <= now_ms || expires_at_ms - now_ms > W4_MAX_ADMISSION_MS {
            return Err(ServingError::W4NotAdmitted);
        }
        let epoch = self
            .groups
            .get(&group)
            .and_then(|history| history.active)
            .ok_or(ServingError::NoActiveGroup(group))?;
        let version = GroupVersion { group, epoch };
        let tracked = if let Some(work) = self.w4_work.get_mut(&version) {
            work.retain(|_, deadline| *deadline > now_ms);
            work.insert(id, expires_at_ms);
            true
        } else {
            false
        };
        Ok(ServingOutcome::W4Admitted { version, tracked })
    }

    fn check_w4(
        &self,
        version: GroupVersion,
        grants: &[WorkerGrant],
        work: Option<OpId>,
    ) -> ServingResult {
        let view = self
            .group(version)
            .ok_or(ServingError::UnknownVersion(version))?;
        if grants.is_empty() || grants.iter().any(|&grant| !view.allows_grant(grant)) {
            return Err(ServingError::GrantMismatch);
        }
        if let Some(pending) = self.w4_work.get(&version)
            && !work.is_some_and(|id| pending.contains_key(&id))
        {
            return Err(ServingError::W4NotAdmitted);
        }
        Ok(group_outcome(view.version, view.phase))
    }

    fn complete_w4(
        &mut self,
        version: GroupVersion,
        work: OpId,
        grants: &[WorkerGrant],
    ) -> ServingResult {
        let authorized = self.check_w4(version, grants, Some(work))?;
        if self.w4_work.contains_key(&version) {
            return self.finish_w4(version, work);
        }
        Ok(authorized)
    }

    fn finish_w4(&mut self, version: GroupVersion, work: OpId) -> ServingResult {
        if !self
            .w4_work
            .get_mut(&version)
            .is_some_and(|pending| pending.remove(&work).is_some())
        {
            return Err(ServingError::W4NotAdmitted);
        }
        Ok(group_outcome(version, self.revoke_drained_w4(version)))
    }

    fn expire_w4(&mut self, version: GroupVersion, through_ms: u64) -> ServingResult {
        let view = self
            .group(version)
            .ok_or(ServingError::UnknownVersion(version))?;
        if view.phase != GroupPhase::Draining {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        let pending = self
            .w4_work
            .get_mut(&version)
            .ok_or(ServingError::InvalidPlan)?;
        if pending
            .values()
            .min()
            .is_some_and(|earliest| through_ms > *earliest)
        {
            return Err(ServingError::W4NotAdmitted);
        }
        pending.retain(|_, deadline| *deadline > through_ms);
        Ok(group_outcome(version, self.revoke_drained_w4(version)))
    }

    fn revoke_drained_w4(&mut self, version: GroupVersion) -> GroupPhase {
        let empty = self.w4_work.get(&version).is_some_and(BTreeMap::is_empty);
        let view = self.group_mut(version).expect("draining version exists");
        if empty && view.phase == GroupPhase::Draining {
            view.phase = GroupPhase::Revoked;
        }
        view.phase
    }

    fn worker_stopped(&mut self, grant: WorkerGrant) -> ServingResult {
        if self.host_has_launch(grant.version, grant.device.node) {
            return Err(ServingError::LaunchOwnerMismatch);
        }
        let managed = self.w4_work.contains_key(&grant.version)
            || self.decoder.managed.contains(&grant.version)
            || self.resident_owner(grant).is_some();
        let view = self.group_mut(grant.version)?;
        check_grant(view, grant)?;
        check_launch_owner(view, None)?;
        check_quiescing(view)?;
        if managed && view.phase != GroupPhase::Revoked {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        view.stopped.insert(grant.device);
        Ok(group_outcome(view.version, view.phase))
    }

    fn reclaim_resident(
        &mut self,
        grant: WorkerGrant,
        owner: Option<ResidentOwner>,
        workers: &BTreeSet<DeviceId>,
    ) -> ServingResult {
        let version = grant.version;
        self.require_unhosted(version)?;
        let view = self
            .group(version)
            .ok_or(ServingError::UnknownVersion(version))?;
        check_grant(view, grant)?;
        check_launch_owner(view, None)?;
        if !workers.iter().eq(view.grants.keys()) {
            return Err(ServingError::InvalidPlan);
        }
        if view.phase == GroupPhase::Retired {
            return Ok(group_outcome(version, GroupPhase::Retired));
        }
        if view.phase != GroupPhase::Revoked {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if self.resident_owner(grant) != owner
            || (owner.is_none() && view.ready.contains(&grant.device))
        {
            return Err(ServingError::LaunchOwnerMismatch);
        }

        let already_stopped = view.stopped.contains(&grant.device);
        let worker = self
            .workers
            .get_mut(&grant.device)
            .expect("reserved worker");
        if worker.grant == Some(grant) {
            worker.grant = None;
        } else if !already_stopped {
            return Err(ServingError::GrantMismatch);
        }
        let view = self.group_mut(version)?;
        view.stopped.insert(grant.device);
        if &view.stopped == workers {
            return self.retire(version, workers);
        }
        Ok(group_outcome(version, GroupPhase::Revoked))
    }

    fn claim_resident(
        &mut self,
        grant: WorkerGrant,
        owner: ResidentOwner,
        launch_digest: [u8; 32],
    ) -> ServingResult {
        self.require_unhosted(grant.version)?;
        self.require_execution_host(grant.device.node)?;
        let view = self
            .group(grant.version)
            .ok_or(ServingError::UnknownVersion(grant.version))?;
        check_grant(view, grant)?;
        check_launch_owner(view, None)?;
        if view.phase != GroupPhase::Preparing || view.ready.contains(&grant.device) {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if view.plan.launch_digest != launch_digest {
            return Err(ServingError::LaunchDigestMismatch);
        }
        if let Some(previous) = self.resident_owner(grant) {
            return if previous == owner {
                Ok(group_outcome(grant.version, view.phase))
            } else {
                Err(ServingError::LaunchOwnerMismatch)
            };
        }
        self.resident
            .entry(grant.version)
            .or_default()
            .insert(grant.device, owner);
        Ok(group_outcome(grant.version, GroupPhase::Preparing))
    }

    fn lose_resident(&mut self, grant: WorkerGrant, owner: ResidentOwner) -> ServingResult {
        if self.resident_owner(grant) != Some(owner) {
            return Err(ServingError::LaunchOwnerMismatch);
        }
        let phase = self.group(grant.version).expect("known resident").phase;
        if phase == GroupPhase::Revoked {
            return Ok(group_outcome(grant.version, GroupPhase::Revoked));
        }
        self.fail_resident_version(grant.version)
    }

    fn fail_resident_version(&mut self, version: GroupVersion) -> ServingResult {
        let was_active = self
            .group(version)
            .is_some_and(|view| view.phase == GroupPhase::Active);
        let result = self.quiesce(version, GroupPhase::Revoked)?;
        if was_active && self.decoder.endpoints.contains_key(&version) {
            self.decoder.faulted.insert(version);
        }
        Ok(result)
    }

    fn claim_launch(
        &mut self,
        version: GroupVersion,
        host: NodeId,
        holder: NodeId,
        launch_digest: [u8; 32],
    ) -> ServingResult {
        self.require_unhosted(version)?;
        if self.w4_switches.contains_key(&version) || self.resident.contains_key(&version) {
            return Err(ServingError::InvalidPlan);
        }
        let view = self.group_mut(version)?;
        if view.phase != GroupPhase::Preparing || !view.ready.is_empty() {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if view.plan.hosts.len() != 1 || !view.plan.hosts.contains_key(&host) {
            return Err(ServingError::InvalidPlan);
        }
        if view.plan.launch_digest != launch_digest {
            return Err(ServingError::LaunchDigestMismatch);
        }
        check_launch_owner(view, None)?;
        view.launch_owner = Some(holder);
        Ok(group_outcome(version, GroupPhase::Preparing))
    }

    fn launch_ready(
        &mut self,
        version: GroupVersion,
        holder: NodeId,
        endpoint: Option<&ServingEndpoint>,
    ) -> ServingResult {
        let view = self.group_mut(version)?;
        check_launch_owner(view, Some(holder))?;
        if !matches!(view.phase, GroupPhase::Preparing | GroupPhase::Ready) {
            return Err(ServingError::InvalidPhase(view.phase));
        }
        if let Some(endpoint) = endpoint {
            check_launch_endpoint(view, endpoint)?;
            view.endpoint = Some(endpoint.clone());
        }
        view.ready.extend(view.grants.keys().copied());
        view.phase = GroupPhase::Ready;
        Ok(group_outcome(version, view.phase))
    }

    fn launch_stopped(&mut self, version: GroupVersion, holder: NodeId) -> ServingResult {
        let view = self.group_mut(version)?;
        check_launch_owner(view, Some(holder))?;
        check_quiescing(view)?;
        view.stopped.extend(view.grants.keys().copied());
        Ok(group_outcome(version, view.phase))
    }

    fn retire(&mut self, version: GroupVersion, workers: &BTreeSet<DeviceId>) -> ServingResult {
        let view = self
            .groups
            .get_mut(&version.group)
            .and_then(|history| history.versions.get_mut(&version.epoch))
            .ok_or(ServingError::UnknownVersion(version))?;
        check_quiescing(view)?;
        if !workers.iter().eq(view.grants.keys()) {
            return Err(ServingError::InvalidPlan);
        }
        if &view.stopped != workers {
            return Err(ServingError::WorkersStillRunning);
        }

        for (device, &grant) in &view.grants {
            let worker = self.workers.get_mut(device).expect("reserved worker");
            if worker.grant == Some(grant) {
                worker.grant = None;
            }
        }
        for (host, demand) in &view.plan.hosts {
            let reserved = &mut self.hosts.get_mut(host).expect("reserved host").reserved;
            reserved.bytes -= demand.bytes;
            reserved.pinned_bytes -= demand.pinned_bytes;
        }
        view.phase = GroupPhase::Retired;
        self.dispatch.remove(&version);
        self.w4_switches.remove(&version);
        self.w4_work.remove(&version);
        self.resident.remove(&version);
        self.host_launches.remove(&version);
        self.hosted_switches.remove(&version);
        self.hosted_recipes.remove(&version);
        self.decoder.endpoints.remove(&version);
        self.decoder.managed.remove(&version);
        Ok(group_outcome(version, GroupPhase::Retired))
    }

    fn group_mut(&mut self, version: GroupVersion) -> Result<&mut GroupView, ServingError> {
        self.groups
            .get_mut(&version.group)
            .and_then(|history| history.versions.get_mut(&version.epoch))
            .ok_or(ServingError::UnknownVersion(version))
    }
}

fn routable(group: &GroupView, activate: Option<GroupVersion>) -> bool {
    match activate {
        Some(version) if version.group == group.version.group => {
            version == group.version && group.phase == GroupPhase::Ready
        }
        _ => group.phase == GroupPhase::Active,
    }
}

fn check_grant(view: &GroupView, grant: WorkerGrant) -> Result<(), ServingError> {
    if view.grants.get(&grant.device) != Some(&grant) {
        return Err(ServingError::GrantMismatch);
    }
    Ok(())
}

fn check_launch_owner(view: &GroupView, holder: Option<NodeId>) -> Result<(), ServingError> {
    if view.launch_owner != holder {
        return Err(ServingError::LaunchOwnerMismatch);
    }
    Ok(())
}

fn check_launch_endpoint(view: &GroupView, endpoint: &ServingEndpoint) -> Result<(), ServingError> {
    let kv_valid = endpoint.prefill_kv.is_some_and(|addr| {
        addr.port() != 0 && !addr.ip().is_unspecified() && !addr.ip().is_multicast()
    });
    let wrong_kv_role = match view.plan.role {
        ServingRole::Prefill => !kv_valid,
        ServingRole::Decode | ServingRole::Aggregated => endpoint.prefill_kv.is_some(),
    };
    if endpoint.api_model.is_empty()
        || endpoint.http.port() == 0
        || endpoint.http.ip().is_unspecified()
        || endpoint.http.ip().is_multicast()
        || wrong_kv_role
        || view
            .endpoint
            .as_ref()
            .is_some_and(|previous| previous != endpoint)
    {
        return Err(ServingError::InvalidEndpoint);
    }
    Ok(())
}

fn check_quiescing(view: &GroupView) -> Result<(), ServingError> {
    if !matches!(view.phase, GroupPhase::Draining | GroupPhase::Revoked) {
        return Err(ServingError::InvalidPhase(view.phase));
    }
    Ok(())
}

fn group_outcome(version: GroupVersion, phase: GroupPhase) -> ServingOutcome {
    ServingOutcome::Group { version, phase }
}

fn fits(demand: HostMemory, capacity: HostMemory) -> bool {
    demand.bytes <= capacity.bytes && demand.pinned_bytes <= capacity.pinned_bytes
}

fn add_memory(current: HostMemory, demand: HostMemory) -> Option<HostMemory> {
    Some(HostMemory {
        bytes: current.bytes.checked_add(demand.bytes)?,
        pinned_bytes: current.pinned_bytes.checked_add(demand.pinned_bytes)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::serving::WorkerDemand;

    #[test]
    fn resident_loss_requires_the_exact_grant_and_owner() {
        let mut state = ServingState::default();
        let host = NodeId::derive("resident-host");
        let group = GroupId(NodeId::derive("resident-group"));
        let device = DeviceId::new(host, 0);
        let memory = HostMemory {
            bytes: 16,
            pinned_bytes: 4,
        };
        let plan = GroupPlan {
            role: ServingRole::Aggregated,
            model: "resident-model".into(),
            tensor_parallel: 1,
            pipeline_parallel: 1,
            launch_digest: [4; 32],
            workers: BTreeMap::from([(
                device,
                WorkerDemand {
                    incarnation: 1,
                    device_bytes: 16,
                },
            )]),
            hosts: BTreeMap::from([(host, memory)]),
        };
        let owner = ResidentOwner {
            session: NodeId::derive("original-supervisor"),
            supervisor_lock: [1, 2],
            device_lock: [1, 3],
        };
        let different_lock = ResidentOwner {
            device_lock: [1, 4],
            ..owner
        };
        let workers = BTreeSet::from([device]);
        state.set_host_memory(host, memory).unwrap();
        state.register_worker(device, 1, 32).unwrap();
        state.prepare(group, 0, &plan).unwrap();
        let version = GroupVersion { group, epoch: 1 };
        let grant = state.group(version).unwrap().grants[&device];
        state
            .claim_resident(grant, owner, plan.launch_digest)
            .unwrap();
        assert_eq!(
            state.claim_resident(grant, different_lock, plan.launch_digest),
            Err(ServingError::LaunchOwnerMismatch)
        );
        state.worker_ready(grant, plan.launch_digest).unwrap();
        state.activate(version).unwrap();
        assert_eq!(
            state.lose_resident(grant, different_lock),
            Err(ServingError::LaunchOwnerMismatch)
        );
        assert_eq!(
            state.lose_resident(
                WorkerGrant {
                    fence: grant.fence + 1,
                    ..grant
                },
                owner
            ),
            Err(ServingError::LaunchOwnerMismatch)
        );
        assert!(state.group(version).unwrap().allows_grant(grant));
        state.lose_resident(grant, owner).unwrap();
        assert!(!state.group(version).unwrap().allows_grant(grant));
        assert_eq!(
            state.retire(version, &workers),
            Err(ServingError::WorkersStillRunning)
        );
        state.worker_stopped(grant).unwrap();
        state.retire(version, &workers).unwrap();
        assert_eq!(state.resident_owner(grant), None);
    }

    #[test]
    fn w4_switch_stays_ready_if_the_expected_active_epoch_drains_during_prepare() {
        let mut state = ServingState::default();
        let host = NodeId::derive("switch-host");
        let group = GroupId(NodeId::derive("switch-group"));
        let incumbent = DeviceId::new(host, 0);
        let candidate = DeviceId::new(host, 1);
        let address = "127.0.0.1:7001".parse().unwrap();
        let demand = HostMemory {
            bytes: 16,
            pinned_bytes: 4,
        };
        state
            .set_host_memory(
                host,
                HostMemory {
                    bytes: 64,
                    pinned_bytes: 16,
                },
            )
            .unwrap();
        for device in [incumbent, candidate] {
            state.register_worker(device, 1, 32).unwrap();
        }
        let plan = |device| GroupPlan {
            role: ServingRole::Aggregated,
            model: "switch-model".into(),
            tensor_parallel: 1,
            pipeline_parallel: 1,
            launch_digest: [4; 32],
            workers: BTreeMap::from([(
                device,
                WorkerDemand {
                    incarnation: 1,
                    device_bytes: 16,
                },
            )]),
            hosts: BTreeMap::from([(host, demand)]),
        };
        let shard = W4ShardLayout {
            k_range: [0, 256],
            group_size: 128,
            output_columns: 512,
        };
        let old = GroupVersion { group, epoch: 1 };
        state.prepare(group, 0, &plan(incumbent)).unwrap();
        let old_grant = state.group(old).unwrap().grants[&incumbent];
        state
            .worker_ready_at(old_grant, [4; 32], address, Some(shard))
            .unwrap();
        state.activate(old).unwrap();

        let layout = W4GroupLayout {
            k: 256,
            group_size: 128,
            output_columns: 512,
            shards: BTreeMap::from([(candidate, [0, 256])]),
        };
        assert_eq!(
            state.prepare_w4_switch(group, 1, None, &plan(candidate), &layout),
            Err(ServingError::ActiveEpochChanged { current: Some(1) })
        );
        state
            .prepare_w4_switch(group, 1, Some(1), &plan(candidate), &layout)
            .unwrap();
        let new = GroupVersion { group, epoch: 2 };
        let new_grant = state.group(new).unwrap().grants[&candidate];
        assert_eq!(
            state.worker_ready_at(new_grant, [4; 32], address, None),
            Err(ServingError::InvalidEndpoint)
        );
        state.quiesce(old, GroupPhase::Draining).unwrap();
        assert_eq!(
            state.worker_ready_at(new_grant, [4; 32], address, Some(shard)),
            Ok(group_outcome(new, GroupPhase::Ready))
        );
        assert_eq!(state.groups[&group].active, None);
        assert!(state.group(old).unwrap().allows_grant(old_grant));
    }

    #[test]
    fn w4_completion_orders_all_grants_and_admission_against_expiry_and_revocation() {
        let mut state = ServingState::default();
        let host = NodeId::derive("w4-completion");
        let group = GroupId(host);
        let version = GroupVersion { group, epoch: 1 };
        let [first, second] = [0, 1].map(|ordinal| DeviceId::new(host, ordinal));
        let address = "127.0.0.1:7001".parse().unwrap();
        let demand = WorkerDemand {
            incarnation: 1,
            device_bytes: 16,
        };
        let memory = HostMemory {
            bytes: 16,
            pinned_bytes: 4,
        };
        let id = |counter| OpId::new(host, counter);
        state.set_host_memory(host, memory).unwrap();
        for device in [first, second] {
            state.register_worker(device, 1, 32).unwrap();
        }
        let layout = W4GroupLayout {
            k: 16,
            group_size: 8,
            output_columns: 4,
            shards: BTreeMap::from([(first, [0, 8]), (second, [8, 16])]),
        };
        state
            .apply(&ServingOperation {
                id: id(1),
                action: ServingAction::PrepareManagedW4 {
                    group,
                    expected_epoch: 0,
                    expected_active: None,
                    plan: GroupPlan {
                        role: ServingRole::Aggregated,
                        model: "completion-model".into(),
                        tensor_parallel: 2,
                        pipeline_parallel: 1,
                        launch_digest: [4; 32],
                        workers: BTreeMap::from([(first, demand), (second, demand)]),
                        hosts: BTreeMap::from([(host, memory)]),
                    },
                    layout: layout.clone(),
                },
            })
            .unwrap();
        let grants = [first, second].map(|device| state.group(version).unwrap().grants[&device]);
        for grant in grants {
            state
                .worker_ready_at(grant, [4; 32], address, layout.shard(grant.device))
                .unwrap();
        }
        for (counter, deadline) in [(2, 20), (3, 30), (4, 30), (5, 30)] {
            assert_eq!(
                state.admit_w4(group, id(counter), 10, deadline),
                Ok(ServingOutcome::W4Admitted {
                    version,
                    tracked: true
                })
            );
        }
        state.quiesce(version, GroupPhase::Draining).unwrap();
        state.expire_w4(version, 20).unwrap();

        let complete = |counter, work, grants| ServingOperation {
            id: id(counter),
            action: ServingAction::CompleteW4 {
                version,
                work: id(work),
                grants,
            },
        };
        assert_eq!(
            state.apply(&complete(6, 2, grants.to_vec())),
            Err(ServingError::W4NotAdmitted)
        );
        let mut stale = grants;
        stale[1].fence += 1;
        assert_eq!(
            state.apply(&complete(7, 3, stale.to_vec())),
            Err(ServingError::GrantMismatch)
        );
        assert_eq!(state.w4_drain(version).unwrap().pending, 3);

        // A later admission on the same workers can commit while earlier work is pending.
        let receipt = Ok(group_outcome(version, GroupPhase::Draining));
        assert_eq!(state.apply(&complete(8, 5, grants.to_vec())), receipt);
        assert_eq!(state.w4_drain(version).unwrap().pending, 2);

        let committed = complete(9, 3, grants.to_vec());
        assert_eq!(state.apply(&committed), receipt);
        assert_eq!(state.w4_drain(version).unwrap().pending, 1);
        assert_eq!(
            state.apply(&complete(10, 3, grants.to_vec())),
            Err(ServingError::W4NotAdmitted)
        );

        state.quiesce(version, GroupPhase::Revoked).unwrap();
        assert_eq!(
            state.apply(&complete(11, 4, grants.to_vec())),
            Err(ServingError::GrantMismatch)
        );
        assert_eq!(state.apply(&committed), receipt);
    }

    #[test]
    fn a_revoked_group_reclaims_each_verified_device_without_clearing_new_grants() {
        let mut state = ServingState::default();
        let host = NodeId::derive("reclaim-test");
        let group = GroupId(host);
        let old = GroupVersion { group, epoch: 1 };
        let new = GroupVersion { group, epoch: 2 };
        let [healthy, failed] = [0, 1].map(|index| DeviceId::new(host, index));
        let workers = BTreeSet::from([healthy, failed]);
        let memory = HostMemory {
            bytes: 16,
            pinned_bytes: 4,
        };
        let demand = WorkerDemand {
            incarnation: 1,
            device_bytes: 16,
        };
        let owner = |index: u64| ResidentOwner {
            session: NodeId::derive(&format!("reclaim-owner-{index}")),
            supervisor_lock: [1, 2],
            device_lock: [1, 3 + index],
        };
        let plan = |workers: &BTreeSet<_>| GroupPlan {
            role: ServingRole::Aggregated,
            model: "reclaim-model".into(),
            tensor_parallel: workers.len() as u32,
            pipeline_parallel: 1,
            launch_digest: [4; 32],
            workers: workers.iter().map(|&device| (device, demand)).collect(),
            hosts: BTreeMap::from([(host, memory)]),
        };
        state
            .set_host_memory(host, add_memory(memory, memory).unwrap())
            .unwrap();
        for &device in &workers {
            state.register_worker(device, 1, 32).unwrap();
        }
        state.prepare(group, 0, &plan(&workers)).unwrap();
        let first = state.group(old).unwrap().grants[&healthy];
        let second = state.group(old).unwrap().grants[&failed];
        for (grant, owner) in [(first, owner(0)), (second, owner(1))] {
            state.claim_resident(grant, owner, [4; 32]).unwrap();
            state.worker_ready(grant, [4; 32]).unwrap();
        }
        state.activate(old).unwrap();
        assert_eq!(
            state.reclaim_resident(first, Some(owner(0)), &workers),
            Err(ServingError::InvalidPhase(GroupPhase::Active))
        );
        state.lose_resident(second, owner(1)).unwrap();
        assert_eq!(
            state.reclaim_resident(second, Some(owner(0)), &workers),
            Err(ServingError::LaunchOwnerMismatch)
        );
        assert_eq!(
            state.reclaim_resident(first, Some(owner(0)), &BTreeSet::from([healthy])),
            Err(ServingError::InvalidPlan)
        );
        state
            .reclaim_resident(first, Some(owner(0)), &workers)
            .unwrap();
        assert_eq!(state.hosts[&host].reserved, memory);
        assert_eq!(state.workers[&failed].grant, Some(second));
        assert_eq!(
            state.prepare(group, 1, &plan(&BTreeSet::from([failed]))),
            Err(ServingError::WorkerBusy(failed))
        );
        state
            .prepare(group, 1, &plan(&BTreeSet::from([healthy])))
            .unwrap();
        let replacement = state.group(new).unwrap().grants[&healthy];
        assert_eq!(replacement.fence, first.fence + 1);
        state.worker_ready(replacement, [4; 32]).unwrap();
        state.activate(new).unwrap();
        assert_eq!(
            state.reclaim_resident(first, Some(owner(0)), &workers),
            Ok(group_outcome(old, GroupPhase::Revoked))
        );
        assert_eq!(
            state.reclaim_resident(second, Some(owner(1)), &workers),
            Ok(group_outcome(old, GroupPhase::Retired))
        );
        assert_eq!(state.workers[&healthy].grant, Some(replacement));
        assert_eq!(state.hosts[&host].reserved, memory);
        assert_eq!(state.workers[&failed].grant, None);
        assert!(!state.group(old).unwrap().allows_grant(first));
        assert!(state.group(new).unwrap().allows_grant(replacement));
        state
            .reclaim_resident(second, Some(owner(1)), &workers)
            .unwrap();
        assert_eq!(state.hosts[&host].reserved, memory);
    }
}
