use std::collections::{BTreeMap, BTreeSet};

use plait_core::OpId;
use plait_core::serving::{
    GroupPhase, GroupVersion, HostedPdInstance, HostedPdIntent, HostedPdKv, HostedPdLaunch,
    HostedPdPlacement, HostedPdPlanning, HostedPortUse, PdRoute, RouteSetId, ServingError,
    ServingOutcome, ServingResult, ServingRole,
};

use super::{Reservations, RouteSet, ServingState};

enum Resolution {
    Wait,
    Cancel,
    Publish(u64),
}

impl ServingState {
    pub(super) fn prepare_hosted_pd(
        &mut self,
        staged: OpId,
        model: &str,
        expected_revision: u64,
        kv: &HostedPdKv,
        placements: &[HostedPdPlacement],
        paths: &[PdRoute],
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let current = self.routes.get(&key);
        let revision = current.map_or(0, |directory| directory.revision);
        if revision != expected_revision {
            return Err(ServingError::StaleDirectory { current: revision });
        }
        if let Some(pending) = self.hosted_pd_intents.get(&key) {
            return Err(ServingError::HostedPdInProgress {
                staged: pending.staged,
            });
        }
        if model.is_empty()
            || current.is_some_and(|directory| directory.model != model)
            || (paths.is_empty() && current.is_none_or(|directory| !directory.hosted))
            || paths.iter().copied().collect::<BTreeSet<_>>().len() != paths.len()
        {
            return Err(ServingError::InvalidDirectory);
        }
        revision
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?;
        let previous = current.map_or_else(Vec::new, |directory| directory.paths.clone());
        let previous_groups = route_versions(&previous);
        let recipients = route_versions(paths);
        if recipients
            .iter()
            .map(|version| version.group)
            .collect::<BTreeSet<_>>()
            .len()
            != recipients.len()
        {
            return Err(ServingError::InvalidDirectory);
        }

        let mut reservations = Reservations::default();
        let mut occupied: Vec<_> = self
            .hosted_recipes
            .values()
            .map(HostedPortUse::from)
            .collect();
        let mut prepared = BTreeMap::new();
        for placement in placements {
            let active = self
                .groups
                .get(&placement.group)
                .and_then(|history| history.active);
            if active != placement.expected_active
                || active.is_some_and(|epoch| {
                    !previous_groups.contains(&GroupVersion {
                        group: placement.group,
                        epoch,
                    })
                })
            {
                return Err(ServingError::ActiveEpochChanged { current: active });
            }
            let launch = HostedPdLaunch {
                role: placement.plan.role,
                kv: kv.clone(),
            };
            launch.validate(&placement.recipe, &placement.plan)?;
            if placement.recipe.api_model != model {
                return Err(ServingError::InvalidEndpoint);
            }
            for host in &placement.recipe.hosts {
                self.check_hosted_inventory(host, &placement.plan, &placement.recipe.profile)?;
            }
            let ports = HostedPortUse::from(&placement.recipe);
            if occupied.iter().any(|&other| other.conflicts(ports)) {
                return Err(ServingError::HostedPortsInUse(ports.host));
            }
            occupied.push(ports);
            let view = self.reserve_group(
                placement.group,
                placement.expected_epoch,
                &placement.plan,
                &mut reservations,
            )?;
            if !recipients.contains(&view.version) || prepared.contains_key(&view.version) {
                return Err(ServingError::InvalidPlan);
            }
            prepared.insert(view.version, (view, &placement.recipe, launch));
        }

        let mut model_identity = None;
        for path in paths {
            for (version, role) in [
                (path.prefill, ServingRole::Prefill),
                (path.decode, ServingRole::Decode),
            ] {
                let matches = if let Some((view, _, _)) = prepared.get(&version) {
                    (view.plan.role == role).then_some(&view.plan.model)
                } else {
                    self.group(version)
                        .filter(|view| {
                            view.phase == GroupPhase::Active
                                && view.plan.role == role
                                && previous_groups.contains(&version)
                                && self
                                    .hosted_pd_launches
                                    .get(&version)
                                    .is_some_and(|launch| &launch.kv == kv)
                                && self.hosted_recipes[&version].api_model == model
                        })
                        .map(|view| &view.plan.model)
                };
                let Some(identity) = matches else {
                    return Err(ServingError::InvalidRoute(*path));
                };
                if model_identity
                    .replace(identity)
                    .is_some_and(|previous| previous != identity)
                {
                    return Err(ServingError::InvalidRoute(*path));
                }
            }
        }
        let intent = HostedPdIntent {
            staged,
            model: model.into(),
            expected_revision,
            previous,
            paths: paths.to_vec(),
            prepared: prepared.keys().copied().collect(),
            hosts: prepared
                .values()
                .flat_map(|(view, _, _)| view.plan.hosts.keys().copied())
                .collect(),
        };
        let mut views = Vec::new();
        for (version, (view, recipe, launch)) in prepared {
            views.push(view);
            self.hosted_recipes.insert(version, recipe.clone());
            self.hosted_pool_versions.insert(version);
            self.hosted_pd_launches.insert(version, launch);
        }
        self.install_groups(views, reservations);
        self.hosted_pd_intents.insert(key, intent);
        Ok(ServingOutcome::HostedPdStaged { revision })
    }

    pub(crate) fn hosted_pd_pending(&self, key: RouteSetId) -> Option<HostedPdIntent> {
        self.hosted_pd_intents.get(&key).cloned()
    }

    pub(crate) fn hosted_pd_planning(&self, key: RouteSetId) -> HostedPdPlanning {
        let directory = self.routes.get(&key);
        let active = directory
            .into_iter()
            .flat_map(|directory| route_versions(&directory.paths))
            .filter_map(|version| {
                let view = self.group(version)?;
                let launch = self.hosted_pd_launches.get(&version)?;
                if view.phase != GroupPhase::Active {
                    return None;
                }
                let mut ranks = BTreeMap::new();
                for device in view.grants.keys() {
                    *ranks.entry(device.node).or_default() += 1;
                }
                Some(HostedPdInstance {
                    version,
                    model: view.plan.model.clone(),
                    role: launch.role,
                    kv: launch.kv.clone(),
                    ranks,
                })
            })
            .collect();
        HostedPdPlanning {
            revision: directory.map_or(0, |directory| directory.revision),
            managed: directory.is_none_or(|directory| directory.hosted),
            pending: self.hosted_pd_intents.get(&key).map(|intent| intent.staged),
            active,
            paths: self.routes(key).map_or_else(Vec::new, |directory| {
                directory.paths.into_iter().map(|path| path.route).collect()
            }),
            resources: self.resources(),
            inventory: self.hosted_planning(),
        }
    }

    pub(crate) fn hosted_pd_work(&self) -> Vec<HostedPdIntent> {
        self.hosted_pd_intents
            .values()
            .filter(|intent| !matches!(self.hosted_pd_resolution(intent), Resolution::Wait))
            .cloned()
            .collect()
    }

    pub(super) fn resolve_hosted_pd(
        &mut self,
        intent: &HostedPdIntent,
        cancel: bool,
    ) -> ServingResult {
        let key = RouteSetId::for_model(&intent.model);
        if self.hosted_pd_intents.get(&key) != Some(intent) {
            return Err(ServingError::InvalidDirectory);
        }
        let decision = if cancel {
            Resolution::Cancel
        } else {
            self.hosted_pd_resolution(intent)
        };
        let result = match decision {
            Resolution::Wait => return Err(ServingError::InvalidDirectory),
            Resolution::Cancel => {
                for &version in &intent.prepared {
                    self.cancel_hosted_pd_candidate(version);
                }
                ServingOutcome::HostedPdCancelled
            }
            Resolution::Publish(revision) => {
                for &version in &intent.prepared {
                    self.activate(version)
                        .expect("validated independent hosted activation");
                }
                self.routes.insert(
                    key,
                    RouteSet {
                        model: intent.model.clone(),
                        revision,
                        paths: intent.paths.clone(),
                        hosted: true,
                    },
                );
                for &removed in
                    route_versions(&intent.previous).difference(&route_versions(&intent.paths))
                {
                    if self
                        .group(removed)
                        .is_some_and(|view| view.phase == GroupPhase::Active)
                    {
                        self.quiesce(removed, GroupPhase::Draining)
                            .expect("checked active group");
                    }
                }
                ServingOutcome::RoutesPublished { revision }
            }
        };
        self.hosted_pd_intents.remove(&key);
        Ok(result)
    }

    fn cancel_hosted_pd_candidate(&mut self, version: GroupVersion) {
        if self
            .group(version)
            .is_some_and(|view| matches!(view.phase, GroupPhase::Preparing | GroupPhase::Ready))
        {
            self.quiesce(version, GroupPhase::Revoked)
                .expect("unpublished candidate exists");
        }
        if let Some(view) = self
            .group(version)
            .filter(|view| view.phase == GroupPhase::Revoked)
        {
            let unclaimed: Vec<_> = view
                .grants
                .values()
                .copied()
                .filter(|grant| !self.host_has_launch(version, grant.device.node))
                .collect();
            for grant in unclaimed {
                self.worker_stopped(grant)
                    .expect("checked unclaimed hosted grant");
            }
        }
    }

    fn hosted_pd_resolution(&self, intent: &HostedPdIntent) -> Resolution {
        let directory = self.routes.get(&RouteSetId::for_model(&intent.model));
        let revision = directory.map_or(0, |current| current.revision);
        if revision != intent.expected_revision
            || directory.map_or(&[][..], |current| &current.paths[..]) != intent.previous
        {
            return Resolution::Cancel;
        }
        let Some(next_revision) = revision.checked_add(1) else {
            return Resolution::Cancel;
        };
        let mut waiting = false;
        for version in route_versions(&intent.paths) {
            let Some(view) = self.group(version) else {
                return Resolution::Cancel;
            };
            if intent.prepared.contains(&version) {
                let Some(recipe) = self.hosted_recipes.get(&version) else {
                    return Resolution::Cancel;
                };
                if recipe.hosts.iter().any(|host| {
                    self.check_hosted_inventory(host, &view.plan, &recipe.profile)
                        .is_err()
                }) {
                    return Resolution::Cancel;
                }
                match view.phase {
                    GroupPhase::Preparing => waiting = true,
                    GroupPhase::Ready if self.check_activation(version).is_ok() => {}
                    _ => return Resolution::Cancel,
                }
            } else if view.phase != GroupPhase::Active {
                return Resolution::Cancel;
            }
        }
        if waiting {
            return Resolution::Wait;
        }
        let eligible = |view: &plait_core::serving::GroupView| {
            view.phase == GroupPhase::Active || intent.prepared.contains(&view.version)
        };
        if intent.paths.iter().any(|&path| {
            self.resolve_route_with(&intent.model, path, eligible)
                .is_none()
        }) {
            return Resolution::Cancel;
        }
        Resolution::Publish(next_revision)
    }
}

fn route_versions(paths: &[PdRoute]) -> BTreeSet<GroupVersion> {
    paths
        .iter()
        .flat_map(|path| [path.prefill, path.decode])
        .collect()
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{
        GroupId, GroupPlan, HostMemory, HostedAddress, HostedInventory, HostedRecipe,
        ServingAction, ServingEndpoint, ServingOperation, WorkerDemand,
    };
    use plait_core::{DeviceId, NodeId};

    use super::*;

    fn kv() -> HostedPdKv {
        HostedPdKv {
            store: "192.0.2.90:7952".parse().unwrap(),
            namespace: "verified-checkpoint".into(),
            block_size: 128,
        }
    }

    fn placement(
        state: &mut ServingState,
        name: u8,
        host: u8,
        ordinal: u32,
        role: ServingRole,
    ) -> HostedPdPlacement {
        let host_id = NodeId::from_bytes([host; 16]);
        let device = DeviceId::new(host_id, ordinal);
        let memory = HostMemory {
            bytes: 64,
            pinned_bytes: 8,
        };
        state.set_host_memory(host_id, memory).unwrap();
        state.register_worker(device, 1, 16).unwrap();
        let address = std::net::Ipv4Addr::new(192, 0, 2, host).into();
        let profile = "engine".to_owned();
        state
            .advertise_hosted_inventory(
                &HostedInventory {
                    host: host_id,
                    address,
                    profiles: BTreeSet::from([profile.clone()]),
                    devices: state
                        .resources()
                        .workers
                        .into_iter()
                        .filter(|worker| worker.device.node == host_id)
                        .map(|worker| (worker.device.ordinal, worker.incarnation))
                        .collect(),
                },
                OpId::new(host_id, u64::from(ordinal) + 1),
                [5, 6],
            )
            .unwrap();
        let recipe = HostedRecipe {
            profile,
            api_host: host_id,
            hosts: vec![HostedAddress {
                host: host_id,
                address,
            }],
            master_port: 20000,
            http_port: 20001,
            backend_port: 20002,
            api_model: "served".into(),
            max_model_len: 1024,
            gpu_memory_bps: 5000,
            prefix_caching: false,
        };
        HostedPdPlacement {
            group: GroupId(NodeId::from_bytes([name; 16])),
            expected_epoch: 0,
            expected_active: None,
            plan: GroupPlan {
                role,
                model: "checkpoint".into(),
                tensor_parallel: 1,
                pipeline_parallel: 1,
                launch_digest: HostedPdLaunch { role, kv: kv() }.launch_digest(&recipe),
                hosts: BTreeMap::from([(
                    host_id,
                    HostMemory {
                        bytes: 8,
                        pinned_bytes: 1,
                    },
                )]),
                workers: BTreeMap::from([(
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 8,
                    },
                )]),
            },
            recipe,
        }
    }

    fn version(placement: &HostedPdPlacement) -> GroupVersion {
        GroupVersion {
            group: placement.group,
            epoch: placement.expected_epoch + 1,
        }
    }

    fn ready(state: &mut ServingState, placement: &HostedPdPlacement) {
        let group = version(placement);
        let host = placement.recipe.api_host;
        state
            .claim_host_launch(group, host, host, placement.plan.launch_digest)
            .unwrap();
        let endpoint = ServingEndpoint {
            api_model: placement.recipe.api_model.clone(),
            http: std::net::SocketAddr::new(
                placement.recipe.hosts[0].address,
                placement.recipe.http_port,
            ),
            prefill_kv: (placement.plan.role == ServingRole::Prefill).then_some(kv().store),
        };
        state
            .host_launch_ready(group, host, host, Some(&endpoint))
            .unwrap();
    }

    #[test]
    fn hosted_pd_publishes_both_ready_roles_and_drains_only_the_removed_replica() {
        let mut state = ServingState::default();
        let p = placement(&mut state, 11, 1, 0, ServingRole::Prefill);
        let d = placement(&mut state, 12, 2, 0, ServingRole::Decode);
        let other = placement(&mut state, 13, 3, 0, ServingRole::Decode);
        let path = PdRoute {
            prefill: version(&p),
            decode: version(&d),
        };
        let expansion = PdRoute {
            prefill: version(&p),
            decode: version(&other),
        };
        let key = RouteSetId::for_model("served");
        let id = |n| OpId::new(key.0, n);
        state
            .prepare_hosted_pd(id(1), "served", 0, &kv(), &[p.clone(), d.clone()], &[path])
            .unwrap();
        assert!(state.hosted_assignments(p.recipe.api_host).is_empty());
        let assignments = state.hosted_managed_assignments(p.recipe.api_host);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].pd.as_ref().unwrap().kv, kv());
        assert_eq!(assignments[0].launch_digest(), p.plan.launch_digest);
        ready(&mut state, &p);
        assert!(state.hosted_pd_work().is_empty());
        assert!(state.routes(key).is_none());
        let early = state.hosted_pd_pending(key).unwrap();
        assert_eq!(
            state.resolve_hosted_pd(&early, false),
            Err(ServingError::InvalidDirectory)
        );
        assert_eq!(state.hosted_pd_pending(key), Some(early));
        assert_eq!(
            state.apply(&ServingOperation {
                id: id(2),
                action: ServingAction::Activate(version(&p)),
            }),
            Err(ServingError::InvalidDirectory)
        );
        ready(&mut state, &d);
        let intent = state.hosted_pd_work().pop().unwrap();
        assert_eq!(
            state.resolve_hosted_pd(&intent, false),
            Ok(ServingOutcome::RoutesPublished { revision: 1 })
        );
        assert_eq!(state.routes(key).unwrap().paths[0].route, path);
        assert_eq!(state.group(version(&p)).unwrap().phase, GroupPhase::Active);
        assert_eq!(state.group(version(&d)).unwrap().phase, GroupPhase::Active);

        state
            .prepare_hosted_pd(
                id(3),
                "served",
                1,
                &kv(),
                &[other.clone()],
                &[path, expansion],
            )
            .unwrap();
        assert_eq!(state.routes(key).unwrap().paths.len(), 1);
        ready(&mut state, &other);
        let intent = state.hosted_pd_work().pop().unwrap();
        state.resolve_hosted_pd(&intent, false).unwrap();
        assert_eq!(state.routes(key).unwrap().paths.len(), 2);
        state
            .prepare_hosted_pd(id(4), "served", 2, &kv(), &[], &[expansion])
            .unwrap();
        let intent = state.hosted_pd_work().pop().unwrap();
        state.resolve_hosted_pd(&intent, false).unwrap();
        assert_eq!(state.routes(key).unwrap().paths[0].route, expansion);
        assert_eq!(
            state.group(version(&d)).unwrap().phase,
            GroupPhase::Draining
        );
        assert_eq!(state.group(version(&p)).unwrap().phase, GroupPhase::Active);
        assert_eq!(
            state.group(version(&other)).unwrap().phase,
            GroupPhase::Active
        );
    }

    #[test]
    fn hosted_pd_port_conflict_and_host_loss_do_not_publish_partial_routes() {
        let mut state = ServingState::default();
        let p = placement(&mut state, 11, 1, 0, ServingRole::Prefill);
        let conflicting = placement(&mut state, 12, 1, 1, ServingRole::Decode);
        let d = placement(&mut state, 13, 2, 0, ServingRole::Decode);
        let key = RouteSetId::for_model("served");
        let path = |decode| PdRoute {
            prefill: version(&p),
            decode,
        };
        assert_eq!(
            state.prepare_hosted_pd(
                OpId::new(key.0, 1),
                "served",
                0,
                &kv(),
                &[p.clone(), conflicting.clone()],
                &[path(version(&conflicting))]
            ),
            Err(ServingError::HostedPortsInUse(p.recipe.api_host)),
        );
        assert!(
            state
                .resources()
                .workers
                .iter()
                .all(|worker| worker.grant.is_none())
        );
        state
            .prepare_hosted_pd(
                OpId::new(key.0, 2),
                "served",
                0,
                &kv(),
                &[p.clone(), d.clone()],
                &[path(version(&d))],
            )
            .unwrap();
        ready(&mut state, &p);
        state
            .expire_execution_host(
                d.recipe.api_host,
                OpId::new(d.recipe.api_host, 1),
                &BTreeSet::from([version(&d)]),
            )
            .unwrap();
        let intent = state.hosted_pd_work().pop().unwrap();
        assert_eq!(
            state.resolve_hosted_pd(&intent, false),
            Ok(ServingOutcome::HostedPdCancelled)
        );
        assert!(state.routes(key).is_none());
        assert!(state.hosted_pd_pending(key).is_none());
        assert_eq!(state.group(version(&p)).unwrap().phase, GroupPhase::Revoked);
        let lost = state.group(version(&d)).unwrap();
        assert_eq!(lost.phase, GroupPhase::Revoked);
        assert_eq!(lost.stopped.len(), 1);
    }
}
