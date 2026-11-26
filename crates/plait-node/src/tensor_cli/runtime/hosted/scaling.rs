//! Drive hosted P/D planning from traffic observed at the configured ingresses.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use plait_core::serving::{
    GroupPhase, HostedPdPlanning, HostedPdReconfiguration, RouteSetId, ServingAction,
    ServingLookup, ServingOperation, ServingOutcome, ServingView,
};
use plait_core::{NodeId, OpId};
use plait_parallel::serving::{
    GroupProfile, HostedLinkPolicy, HostedPdPool, ServingForecast, plan_hosted_pd_with_releases,
};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::task::JoinSet;

use super::{ControlPeers, GroupControl};
use crate::hosted_probe;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Policy {
    model: String,
    checkpoint: String,
    ingress: Vec<SocketAddr>,
    window_ms: u64,
    stable_windows: usize,
    cooldown_ms: u64,
    horizon_ms: u64,
    minimum_gain_requests: f64,
    pool: HostedPdPool,
    link_policy: HostedLinkPolicy,
}

pub(super) fn validate(policies: &[Policy]) -> io::Result<()> {
    let mut models = BTreeSet::new();
    for policy in policies {
        let unique: BTreeSet<_> = policy.ingress.iter().collect();
        if policy.model.is_empty()
            || policy.checkpoint.is_empty()
            || !models.insert(&policy.model)
            || unique.is_empty()
            || unique.len() != policy.ingress.len()
            || policy.ingress.iter().any(|address| {
                address.port() == 0 || address.ip().is_unspecified() || address.ip().is_multicast()
            })
            || policy.window_ms == 0
            || policy.stable_windows == 0
            || policy.horizon_ms == 0
            || !policy.minimum_gain_requests.is_finite()
            || policy.minimum_gain_requests < 0.0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid hosted P/D scaling policy",
            ));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct Counters {
    model: String,
    session: String,
    uptime_ms: u64,
    arrivals: u64,
    completed: u64,
    failed: u64,
    in_flight: u64,
}

impl Counters {
    fn is_valid(&self, model: &str) -> bool {
        self.model == model
            && NodeId::from_hex(&self.session).is_ok()
            && self
                .completed
                .checked_add(self.failed)
                .and_then(|count| count.checked_add(self.in_flight))
                == Some(self.arrivals)
    }

    fn arrival_rate(&self, before: &Self, window_ms: u64) -> Option<f64> {
        let elapsed = self.uptime_ms.checked_sub(before.uptime_ms)?;
        if self.session != before.session
            || elapsed == 0
            || elapsed < window_ms / 2
            || elapsed > window_ms.saturating_mul(2)
            || self.completed < before.completed
            || self.failed < before.failed
        {
            return None;
        }
        self.arrivals
            .checked_sub(before.arrivals)
            .map(|count| count as f64 * 1000.0 / elapsed as f64)
    }
}

type Cohort = BTreeMap<SocketAddr, Counters>;

type Proposal = ServingAction;

#[derive(Default)]
struct Evidence {
    revision: Option<u64>,
    reference: Option<Cohort>,
    proposal: Option<(Proposal, usize)>,
    last_change: Option<Instant>,
}

impl Evidence {
    fn observe(
        &mut self,
        revision: u64,
        pending: bool,
        samples: Cohort,
        now: Instant,
        window_ms: u64,
        cooldown: Duration,
        horizon_ms: u64,
    ) -> Option<f64> {
        if self.revision.is_some_and(|seen| revision < seen) {
            self.reset();
            return None;
        }
        if self.revision != Some(revision) {
            if self.revision.is_some() || revision != 0 {
                self.last_change = Some(now);
            }
            self.revision = Some(revision);
            self.reset();
        }
        if pending {
            self.last_change = Some(now);
            self.reset();
            return None;
        }
        let before = self.reference.replace(samples)?;
        if self
            .last_change
            .is_some_and(|changed| now.duration_since(changed) < cooldown)
        {
            self.proposal = None;
            return None;
        }
        let measured = self
            .reference
            .as_ref()
            .expect("installed counters")
            .iter()
            .try_fold(0.0, |rate, (addr, current)| {
                let rps = current.arrival_rate(before.get(addr)?, window_ms)?;
                Some(rate + rps + current.in_flight as f64 * 1000.0 / horizon_ms as f64)
            });
        if measured.is_none() {
            self.proposal = None;
        }
        measured
    }

    fn see(&mut self, proposal: &Proposal, required: usize) -> bool {
        if let Some((previous, windows)) = &mut self.proposal
            && previous == proposal
        {
            *windows = windows.saturating_add(1);
            return *windows >= required;
        }
        self.proposal = Some((proposal.clone(), 1));
        required == 1
    }

    fn reset(&mut self) {
        self.reference = None;
        self.proposal = None;
    }
}

pub(super) async fn run(policies: Vec<Policy>, peers: ControlPeers) -> io::Result<()> {
    let mut jobs = JoinSet::new();
    for policy in policies {
        jobs.spawn(run_model(policy, peers.clone()));
    }
    while let Some(outcome) = jobs.join_next().await {
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(error)) => eprintln!("hosted P/D scaling stopped: {error}"),
            Err(error) => eprintln!("hosted P/D scaling task failed: {error}"),
        }
    }
    std::future::pending().await
}

async fn run_model(policy: Policy, peers: ControlPeers) -> io::Result<()> {
    let Policy {
        model,
        checkpoint,
        ingress,
        window_ms,
        stable_windows,
        cooldown_ms,
        horizon_ms,
        minimum_gain_requests,
        pool,
        link_policy,
    } = policy;
    let pool = Arc::new(pool);
    let cooldown = Duration::from_millis(cooldown_ms);
    let mut control = loop {
        match GroupControl::connect(peers.clone()).await {
            Ok(connection) => break connection,
            Err(error) => eprintln!("opening hosted P/D control for {model}: {error}"),
        }
        tokio::time::sleep(Duration::from_millis(window_ms)).await;
    };
    let mut evidence = Evidence::default();
    let key = RouteSetId::for_model(&model);
    let lookup = if pool.can_reconfigure() {
        ServingLookup::HostedPdReconfiguration(key)
    } else {
        ServingLookup::HostedPdPlanning(key)
    };
    let mut tick = tokio::time::interval(Duration::from_millis(window_ms));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let input = async {
            let snapshot = match control.query(lookup).await? {
                ServingView::HostedPdPlanning(planning) => HostedPdReconfiguration {
                    planning,
                    releases: Vec::new(),
                },
                ServingView::HostedPdReconfiguration(snapshot) => snapshot,
                _ => return Err(io::Error::other("expected hosted P/D planning state")),
            };
            let samples = sample_cohort(&ingress, &model).await?;
            Ok((snapshot, samples))
        }
        .await;
        let (snapshot, samples) = match input {
            Ok(input) => input,
            Err(error) => {
                evidence.reset();
                eprintln!("hosted P/D load for {model}: {error}");
                continue;
            }
        };
        let observed_at = Instant::now();
        let Some(demand_rps) = evidence.observe(
            snapshot.planning.revision,
            snapshot.planning.pending.is_some(),
            samples,
            observed_at,
            window_ms,
            cooldown,
            horizon_ms,
        ) else {
            continue;
        };
        let planned = decide(
            &mut control,
            &model,
            &checkpoint,
            demand_rps,
            horizon_ms,
            minimum_gain_requests,
            snapshot,
            pool.clone(),
            link_policy,
        )
        .await;
        let proposal = match planned {
            Ok(Some(choice)) => choice,
            Ok(None) => {
                evidence.proposal = None;
                continue;
            }
            Err(error) => {
                evidence.reset();
                eprintln!("hosted P/D planning for {model}: {error}");
                continue;
            }
        };
        if observed_at.elapsed() > Duration::from_millis(window_ms) {
            evidence.reset();
            continue;
        }
        if !evidence.see(&proposal, stable_windows) {
            continue;
        }
        evidence.last_change = Some(Instant::now());
        evidence.reset();
        let operation = ServingOperation {
            id: OpId::new(NodeId::random(), 1),
            action: proposal,
        };
        match control.apply(&operation).await {
            Ok(Ok(ServingOutcome::HostedPdStaged { revision })) => {
                println!(
                    "hosted P/D staged {model} at revision {revision} with {demand_rps:.3} observed requests/s"
                );
            }
            other => eprintln!("hosted P/D scaling proposal for {model}: {other:?}"),
        }
    }
}

async fn decide(
    control: &mut GroupControl,
    model: &str,
    checkpoint: &str,
    demand_rps: f64,
    horizon_ms: u64,
    minimum_gain_requests: f64,
    snapshot: HostedPdReconfiguration,
    pool: Arc<HostedPdPool>,
    link_policy: HostedLinkPolicy,
) -> io::Result<Option<Proposal>> {
    let HostedPdReconfiguration {
        planning: snapshot,
        releases,
    } = snapshot;
    let groups = current_profiles(control, &snapshot, &pool).await?;
    let ServingView::HostedProbeEndpoints(endpoints) =
        control.query(ServingLookup::HostedProbeEndpoints).await?
    else {
        return Err(io::Error::other("expected hosted link probes"));
    };
    let hosts = snapshot
        .inventory
        .hosts
        .iter()
        .map(|entry| entry.host)
        .collect();
    let links = hosted_probe::collect(endpoints, Some(&hosts)).await;
    let forecast = ServingForecast {
        model: checkpoint.to_owned(),
        demand_rps,
        now_ms: hosted_probe::now_ms()?,
        horizon_ms,
        minimum_gain_requests,
        groups,
        paths: Vec::new(),
        link_bytes_per_second: BTreeMap::new(),
    };
    let model = model.to_owned();
    tokio::task::spawn_blocking(move || {
        let selection = plan_hosted_pd_with_releases(
            &model,
            &snapshot,
            &forecast,
            &pool,
            link_policy,
            &links,
            &releases,
        )
        .map_err(io::Error::other)?;
        Ok(selection
            .selected
            .map(|selected| selected.into_action(model, snapshot.revision, pool.kv.clone(), None)))
    })
    .await
    .map_err(io::Error::other)?
}

async fn current_profiles(
    control: &mut GroupControl,
    snapshot: &HostedPdPlanning,
    pool: &HostedPdPool,
) -> io::Result<Vec<GroupProfile>> {
    let mut recipes = BTreeMap::new();
    let hosts: BTreeSet<_> = snapshot
        .active
        .iter()
        .filter_map(|active| active.ranks.keys().next().copied())
        .collect();
    for host in hosts {
        let ServingView::HostedManagedAssignments(assignments) = control
            .query(ServingLookup::HostedManagedAssignments(host))
            .await?
        else {
            return Err(io::Error::other("expected active hosted assignments"));
        };
        for deployment in assignments {
            if deployment.pd.is_some() && deployment.assignment.phase == GroupPhase::Active {
                recipes.insert(deployment.assignment.version, deployment.assignment);
            }
        }
    }
    Ok(snapshot
        .active
        .iter()
        .filter_map(|active| {
            let assignment = recipes.get(&active.version)?;
            let capacity = pool.calibrated_capacity(
                active.version.group,
                active.role,
                &assignment.recipe.profile,
                assignment.tensor_parallel,
            )?;
            Some(GroupProfile {
                version: active.version,
                capacity,
            })
        })
        .collect())
}

async fn sample_cohort(ingress: &[SocketAddr], model: &str) -> io::Result<Cohort> {
    let mut jobs = JoinSet::new();
    for &addr in ingress {
        jobs.spawn(async move {
            let sample =
                tokio::time::timeout(Duration::from_secs(2), read_counters(addr)).await??;
            Ok::<_, io::Error>((addr, sample))
        });
    }
    let mut cohort = BTreeMap::new();
    let mut sessions = BTreeSet::new();
    while let Some(result) = jobs.join_next().await {
        let (addr, sample) = result.map_err(io::Error::other)??;
        if !sample.is_valid(model) || !sessions.insert(sample.session.clone()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid or duplicate ingress counters",
            ));
        }
        cohort.insert(addr, sample);
    }
    Ok(cohort)
}

async fn read_counters(addr: SocketAddr) -> io::Result<Counters> {
    let mut client = TcpStream::connect(addr).await?;
    client.write_all(b"PLTPDL01").await?;
    let size = client.read_u32().await? as usize;
    if !(1..=4096).contains(&size) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid ingress counters size",
        ));
    }
    let mut body = vec![0; size];
    client.read_exact(&mut body).await?;
    Ok(serde_json::from_slice(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::serving::{GroupId, GroupVersion, HostedPdKv, PdRoute};

    fn sample(session: u8, uptime_ms: u64, arrivals: u64, in_flight: u64) -> Cohort {
        BTreeMap::from([(
            "192.0.2.1:8100".parse().unwrap(),
            Counters {
                model: "model".into(),
                session: NodeId::from_bytes([session; 16]).to_hex(),
                uptime_ms,
                arrivals,
                completed: arrivals - in_flight,
                failed: 0,
                in_flight,
            },
        )])
    }

    fn proposal(number: u8) -> Proposal {
        let version = |id| GroupVersion {
            group: GroupId(NodeId::from_bytes([id; 16])),
            epoch: 1,
        };
        ServingAction::PrepareHostedPd {
            model: "model".into(),
            expected_revision: 0,
            kv: HostedPdKv {
                store: "192.0.2.1:8000".parse().unwrap(),
                namespace: "weights".into(),
                block_size: 128,
            },
            paths: vec![PdRoute {
                prefill: version(1),
                decode: version(number),
            }],
            placements: Vec::new(),
        }
    }

    #[test]
    fn stability_resets_on_different_plan_ingress_session_and_revision() {
        let t = Instant::now();
        let mut state = Evidence::default();
        let observe =
            |state: &mut Evidence, revision, pending, session, uptime, arrivals, in_flight, now| {
                state.observe(
                    revision,
                    pending,
                    sample(session, uptime, arrivals, in_flight),
                    t + Duration::from_millis(now),
                    1000,
                    Duration::from_millis(2000),
                    10000,
                )
            };
        assert_eq!(observe(&mut state, 0, false, 1, 1000, 0, 0, 1000), None);
        assert_eq!(
            observe(&mut state, 0, false, 1, 2000, 2, 1, 2000),
            Some(2.1)
        );
        assert!(!state.see(&proposal(2), 2));
        assert_eq!(
            observe(&mut state, 0, false, 1, 3000, 4, 0, 3000),
            Some(2.0)
        );
        assert!(!state.see(&proposal(3), 2));

        assert_eq!(observe(&mut state, 0, false, 2, 1000, 1, 0, 4000), None);
        assert_eq!(
            observe(&mut state, 0, false, 2, 2000, 3, 0, 5000),
            Some(2.0)
        );
        assert!(!state.see(&proposal(3), 2));
        assert_eq!(
            observe(&mut state, 0, false, 2, 3000, 5, 0, 6000),
            Some(2.0)
        );
        assert!(state.see(&proposal(3), 2));

        assert_eq!(observe(&mut state, 0, true, 2, 4000, 8, 0, 7000), None);
        assert_eq!(observe(&mut state, 1, false, 2, 5000, 10, 0, 8000), None);
        assert_eq!(observe(&mut state, 1, false, 2, 6000, 11, 0, 9000), None);
        assert_eq!(
            observe(&mut state, 1, false, 2, 7000, 11, 0, 10000),
            Some(0.0)
        );
        assert!(!state.see(&proposal(3), 2));
    }
}
