use std::collections::{BTreeMap, BTreeSet};

use plait_core::OpId;
use plait_core::serving::{
    GroupPhase, GroupVersion, GroupView, HostedPdKv, HostedPdLaunch, HostedPdPlacement,
    HostedPortUse, HostedRecipe, HostedRoute, HostedRouteDirectory, HostedServingIntent,
    HostedServingPlan, HostedServingPlanning, PdRoute, RouteSetId, ServingError, ServingOutcome,
    ServingResult, ServingRole,
};

use super::{Reservations, RouteSet, ServingState};

pub(super) struct PreparedGroup<'a> {
    pub(super) view: GroupView,
    recipe: &'a HostedRecipe,
    launch: Option<HostedPdLaunch>,
}

impl RouteSet {
    fn hosted_routes(&self) -> impl Iterator<Item = HostedRoute> + '_ {
        self.aggregated
            .iter()
            .copied()
            .map(HostedRoute::Aggregated)
            .chain(self.paths.iter().copied().map(HostedRoute::Split))
    }
}

impl ServingState {
    pub(crate) fn hosted_serving_planning(&self, key: RouteSetId) -> HostedServingPlanning {
        let directory = self.routes.get(&key);
        let split = self.hosted_pd_planning(key);
        let committed: Vec<_> = directory
            .into_iter()
            .flat_map(RouteSet::hosted_routes)
            .collect();
        let versions = route_groups(&committed);
        let assignments: BTreeMap<_, _> = versions
            .into_iter()
            .filter_map(|version| {
                let view = self
                    .group(version)
                    .filter(|view| view.phase == GroupPhase::Active)?;
                let recipe = self.hosted_recipes.get(&version)?;
                Some((
                    version,
                    plait_core::serving::HostedAssignment {
                        version,
                        model: view.plan.model.clone(),
                        tensor_parallel: view.plan.tensor_parallel,
                        phase: view.phase,
                        holder: None,
                        recipe: recipe.clone(),
                    },
                ))
            })
            .collect();
        let aggregate_kv = assignments
            .keys()
            .filter_map(|version| {
                let launch = self.hosted_pd_launches.get(version)?;
                (launch.role == ServingRole::Aggregated).then(|| (*version, launch.kv.clone()))
            })
            .collect();
        HostedServingPlanning {
            managed: directory.is_none_or(|current| {
                !current.aggregated.is_empty() || current.hosted_kv.is_some()
            }),
            split,
            committed,
            assignments,
            aggregate_kv,
        }
    }

    pub(crate) fn hosted_routes(&self, key: RouteSetId) -> Option<HostedRouteDirectory> {
        let directory = self.routes.get(&key)?;
        let aggregated = directory
            .aggregated
            .iter()
            .filter_map(|&version| {
                let view = self.group(version)?;
                let endpoint = view.endpoint.as_ref()?;
                (view.phase == GroupPhase::Active
                    && view.plan.role == ServingRole::Aggregated
                    && endpoint.api_model == directory.model)
                    .then_some((version, endpoint.http))
            })
            .collect();
        Some(HostedRouteDirectory {
            split: self.routes(key)?,
            aggregated,
        })
    }

    pub(crate) fn hosted_serving_pending(&self, key: RouteSetId) -> Option<HostedServingIntent> {
        self.hosted_serving_intents.get(&key).cloned()
    }

    pub(super) fn prepare_hosted_serving(
        &mut self,
        staged: OpId,
        model: &str,
        expected_revision: u64,
        plan: &HostedServingPlan,
    ) -> ServingResult {
        let key = RouteSetId::for_model(model);
        let directory = self.routes.get(&key);
        let revision = directory.map_or(0, |current| current.revision);
        if revision != expected_revision {
            return Err(ServingError::StaleDirectory { current: revision });
        }
        if let Some(staged) = self.pending_hosted_pd_id(key) {
            return Err(ServingError::HostedPdInProgress { staged });
        }
        let has_split = plan
            .routes
            .iter()
            .any(|route| matches!(route, HostedRoute::Split(_)));
        if model.is_empty()
            || plan.routes.is_empty()
            || (has_split && plan.kv.is_none())
            || directory.is_some_and(|current| {
                current.model != model
                    || (current.aggregated.is_empty() && current.hosted_kv.is_none())
            })
            || plan.routes.iter().copied().collect::<BTreeSet<_>>().len() != plan.routes.len()
        {
            return Err(ServingError::InvalidDirectory);
        }
        revision
            .checked_add(1)
            .ok_or(ServingError::SequenceExhausted)?;
        let previous: Vec<_> = directory
            .into_iter()
            .flat_map(RouteSet::hosted_routes)
            .collect();
        let previous_groups = route_groups(&previous);
        let targets = route_groups(&plan.routes);
        if targets
            .iter()
            .map(|version| version.group)
            .collect::<BTreeSet<_>>()
            .len()
            != targets.len()
        {
            return Err(ServingError::InvalidDirectory);
        }
        let (prepared, reservations) = self.reserve_hosted_route_groups(
            model,
            plan.kv.as_ref(),
            &plan.placements,
            &previous_groups,
            &targets,
        )?;
        if !has_split
            && plan.kv.is_some()
            && !targets.iter().any(|version| {
                prepared.get(version).map_or_else(
                    || self.hosted_pd_launches.contains_key(version),
                    |group| group.launch.is_some(),
                )
            })
        {
            return Err(ServingError::InvalidDirectory);
        }
        let previous_model = previous_groups
            .iter()
            .next()
            .and_then(|&version| self.group(version))
            .map(|view| &view.plan.model);
        let mut identity = previous_model;
        for route in &plan.routes {
            let roles = match *route {
                HostedRoute::Aggregated(_) => [ServingRole::Aggregated; 2],
                HostedRoute::Split(_) => [ServingRole::Prefill, ServingRole::Decode],
            };
            for (version, role) in route.groups().zip(roles) {
                let view = self
                    .hosted_route_target(
                        version,
                        role,
                        model,
                        plan.kv.as_ref(),
                        &previous_groups,
                        &prepared,
                    )
                    .ok_or(ServingError::InvalidDirectory)?;
                if identity
                    .replace(&view.plan.model)
                    .is_some_and(|last| last != &view.plan.model)
                {
                    return Err(ServingError::InvalidPlan);
                }
            }
        }
        let intent = HostedServingIntent {
            staged,
            model: model.to_owned(),
            expected_revision,
            previous,
            routes: plan.routes.clone(),
            prepared: prepared.keys().copied().collect(),
            hosts: prepared
                .values()
                .flat_map(|group| group.view.plan.hosts.keys().copied())
                .collect(),
        };
        self.install_hosted_route_groups(prepared, reservations);
        self.hosted_serving_intents.insert(key, intent);
        Ok(ServingOutcome::HostedServingStaged { revision })
    }

    pub(super) fn hosted_route_target<'a>(
        &'a self,
        version: GroupVersion,
        role: ServingRole,
        model: &str,
        kv: Option<&HostedPdKv>,
        previous: &BTreeSet<GroupVersion>,
        prepared: &'a BTreeMap<GroupVersion, PreparedGroup<'_>>,
    ) -> Option<&'a GroupView> {
        if let Some(candidate) = prepared.get(&version) {
            return (candidate.view.plan.role == role).then_some(&candidate.view);
        }
        let view = self.group(version)?;
        let recipe = self.hosted_recipes.get(&version)?;
        let matches_kv = match self.hosted_pd_launches.get(&version) {
            Some(launch) => Some(&launch.kv) == kv,
            None => role == ServingRole::Aggregated,
        };
        (previous.contains(&version)
            && view.phase == GroupPhase::Active
            && view.plan.role == role
            && recipe.api_model == model
            && matches_kv)
            .then_some(view)
    }

    pub(super) fn reserve_hosted_route_groups<'a>(
        &self,
        model: &str,
        kv: Option<&HostedPdKv>,
        placements: &'a [HostedPdPlacement],
        previous: &BTreeSet<GroupVersion>,
        targets: &BTreeSet<GroupVersion>,
    ) -> Result<(BTreeMap<GroupVersion, PreparedGroup<'a>>, Reservations), ServingError> {
        let mut reservations = Reservations::default();
        let mut occupied: Vec<_> = self
            .hosted_recipes
            .values()
            .map(HostedPortUse::from)
            .chain(self.deferred_pd_ports())
            .collect();
        let mut prepared = BTreeMap::new();
        for placement in placements {
            let active = self
                .groups
                .get(&placement.group)
                .and_then(|history| history.active);
            if active != placement.expected_active
                || active.is_some_and(|epoch| {
                    !previous.contains(&GroupVersion {
                        group: placement.group,
                        epoch,
                    })
                })
            {
                return Err(ServingError::ActiveEpochChanged { current: active });
            }
            // Mixed plans may retain the original aggregate launch format.
            // The digest distinguishes it from a launch bound to shared KV.
            let legacy_aggregate = placement.plan.role == ServingRole::Aggregated
                && placement.plan.launch_digest == placement.recipe.launch_digest();
            let launch = if legacy_aggregate {
                placement
                    .recipe
                    .validate_aggregated_route(&placement.plan)?;
                None
            } else {
                let launch = HostedPdLaunch {
                    role: placement.plan.role,
                    kv: kv.ok_or(ServingError::InvalidPlan)?.clone(),
                };
                launch.validate(&placement.recipe, &placement.plan)?;
                Some(launch)
            };
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
            if !targets.contains(&view.version) || prepared.contains_key(&view.version) {
                return Err(ServingError::InvalidPlan);
            }
            prepared.insert(
                view.version,
                PreparedGroup {
                    view,
                    recipe: &placement.recipe,
                    launch,
                },
            );
        }
        Ok((prepared, reservations))
    }

    pub(super) fn install_hosted_route_groups(
        &mut self,
        prepared: BTreeMap<GroupVersion, PreparedGroup<'_>>,
        reservations: Reservations,
    ) {
        let mut views = Vec::new();
        for (version, candidate) in prepared {
            self.hosted_recipes
                .insert(version, candidate.recipe.clone());
            self.hosted_pool_versions.insert(version);
            if let Some(launch) = candidate.launch {
                self.hosted_pd_launches.insert(version, launch);
            }
            views.push(candidate.view);
        }
        self.install_groups(views, reservations);
    }

    /// None cancels the plan. False means all groups remain eligible, but some
    /// workers or endpoints still need to report ready.
    pub(super) fn hosted_route_candidates_ready(
        &self,
        versions: impl IntoIterator<Item = GroupVersion>,
        prepared: &BTreeSet<GroupVersion>,
    ) -> Option<bool> {
        let mut ready = true;
        for version in versions {
            let view = self.group(version)?;
            if prepared.contains(&version) {
                let recipe = self.hosted_recipes.get(&version)?;
                if recipe.hosts.iter().any(|host| {
                    self.check_hosted_inventory(host, &view.plan, &recipe.profile)
                        .is_err()
                }) {
                    return None;
                }
                match view.phase {
                    GroupPhase::Preparing => ready = false,
                    GroupPhase::Ready if self.check_activation(version).is_ok() => {}
                    _ => return None,
                }
            } else if view.phase != GroupPhase::Active {
                return None;
            }
        }
        Some(ready)
    }

    fn hosted_serving_resolution(&self, intent: &HostedServingIntent) -> Option<bool> {
        let directory = self.routes.get(&RouteSetId::for_model(&intent.model));
        if directory.map_or(0, |current| current.revision) != intent.expected_revision
            || !directory
                .into_iter()
                .flat_map(RouteSet::hosted_routes)
                .eq(intent.previous.iter().copied())
        {
            return None;
        }
        if !self.hosted_route_candidates_ready(route_groups(&intent.routes), &intent.prepared)? {
            return Some(false);
        }
        let eligible = |view: &GroupView| {
            view.phase == GroupPhase::Active || intent.prepared.contains(&view.version)
        };
        for route in &intent.routes {
            match *route {
                HostedRoute::Aggregated(version) => {
                    let view = self.group(version)?;
                    if !eligible(view)
                        || view.plan.role != ServingRole::Aggregated
                        || view.endpoint.as_ref()?.api_model != intent.model
                    {
                        return None;
                    }
                }
                HostedRoute::Split(path) => {
                    self.resolve_route_with(&intent.model, path, eligible)?;
                }
            }
        }
        Some(true)
    }

    pub(crate) fn hosted_serving_work(&self) -> Vec<HostedServingIntent> {
        self.hosted_serving_intents
            .values()
            .filter(|intent| self.hosted_serving_resolution(intent) != Some(false))
            .cloned()
            .collect()
    }

    pub(super) fn resolve_hosted_serving(
        &mut self,
        intent: &HostedServingIntent,
        cancel: bool,
    ) -> ServingResult {
        let key = RouteSetId::for_model(&intent.model);
        if self.hosted_serving_intents.get(&key) != Some(intent) {
            return Err(ServingError::InvalidDirectory);
        }
        let readiness = if cancel {
            None
        } else {
            self.hosted_serving_resolution(intent)
        };
        let result = match readiness {
            Some(false) => return Err(ServingError::InvalidDirectory),
            None => {
                for &version in &intent.prepared {
                    self.cancel_hosted_pd_candidate(version);
                }
                ServingOutcome::HostedServingCancelled
            }
            Some(true) => {
                let revision = intent
                    .expected_revision
                    .checked_add(1)
                    .ok_or(ServingError::SequenceExhausted)?;
                for &version in &intent.prepared {
                    self.activate(version)
                        .expect("validated independent hosted activation");
                }
                let paths: Vec<PdRoute> = intent
                    .routes
                    .iter()
                    .filter_map(|route| match route {
                        HostedRoute::Split(path) => Some(*path),
                        _ => None,
                    })
                    .collect();
                let aggregated = intent
                    .routes
                    .iter()
                    .filter_map(|route| match route {
                        HostedRoute::Aggregated(version) => Some(*version),
                        _ => None,
                    })
                    .collect();
                let hosted_kv = paths
                    .first()
                    .map(|path| self.hosted_pd_launches[&path.prefill].kv.clone());
                self.routes.insert(
                    key,
                    RouteSet {
                        model: intent.model.clone(),
                        revision,
                        paths,
                        hosted_kv,
                        aggregated,
                    },
                );
                for &removed in
                    route_groups(&intent.previous).difference(&route_groups(&intent.routes))
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
        self.hosted_serving_intents.remove(&key);
        Ok(result)
    }
}

fn route_groups(routes: &[HostedRoute]) -> BTreeSet<GroupVersion> {
    routes.iter().flat_map(|route| route.groups()).collect()
}
