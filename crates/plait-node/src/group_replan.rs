//! Repeatedly evaluate caller-specified workload constraints for native groups.

use std::collections::{BTreeMap, BTreeSet};
use std::future::pending;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::num::{NonZeroU32, NonZeroU64};
use std::time::Duration;

use plait_core::group::{
    GroupId, GroupLookup, GroupOperation, GroupPhase, GroupReply, GroupResult, GroupVersion,
    ResourcePlan,
};
use plait_core::{NodeId, OpId};
use plait_parallel::group_placement::{Decision, Workload};
use serde::Deserialize;
use tokio::task::JoinSet;
use tokio::time::{MissedTickBehavior, interval};

use crate::group_plan::{Snapshot, collect_links};
use crate::tensor_cli::control::{ControlPeers, GroupControl};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    control_ip: IpAddr,
    #[serde(default = "default_interval")]
    interval_ms: NonZeroU64,
    #[serde(default = "default_samples")]
    stable_samples: NonZeroU32,
    groups: Vec<Policy>,
}

fn default_interval() -> NonZeroU64 {
    NonZeroU64::new(5_000).unwrap()
}

fn default_samples() -> NonZeroU32 {
    NonZeroU32::new(2).unwrap()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    group: GroupId,
    calibration: Option<NodeId>,
    workload: Workload,
}

impl Config {
    pub(super) fn validate(&self) -> io::Result<()> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if self.control_ip.is_unspecified()
            || self.control_ip.is_multicast()
            || matches!(self.control_ip, IpAddr::V4(ip) if ip.is_broadcast())
        {
            return Err(invalid("group replanning requires a concrete control IP"));
        }
        let groups: BTreeSet<_> = self.groups.iter().map(|policy| policy.group).collect();
        if groups.is_empty() || groups.len() != self.groups.len() {
            return Err(invalid("group replanning requires distinct group policies"));
        }
        for policy in &self.groups {
            policy
                .workload
                .validate(policy.calibration.is_some())
                .map_err(invalid)?;
        }
        Ok(())
    }
}

pub(super) async fn run(
    config: Option<Config>,
    mut peers: BTreeMap<NodeId, String>,
    seeds: Vec<SocketAddr>,
    local_voter: Option<NodeId>,
) -> io::Result<()> {
    let Some(config) = config else {
        return pending().await;
    };
    if let Some(id) = local_voter
        && let Some(addr) = peers.get_mut(&id)
        && let Ok(socket) = addr.parse::<SocketAddr>()
        && socket.ip().is_unspecified()
    {
        *addr = SocketAddr::new(config.control_ip, socket.port()).to_string();
    }
    let peers = peers
        .into_iter()
        .map(|(id, addr)| format!("{id}@{addr}"))
        .collect();
    let listen = SocketAddr::new(config.control_ip, 0).to_string();
    let endpoints = ControlPeers::new(peers, seeds, listen);
    let interval = Duration::from_millis(config.interval_ms.get());
    let samples = config.stable_samples.get();
    let mut tasks = JoinSet::new();
    for policy in config.groups {
        tasks.spawn(run_policy(policy, endpoints.clone(), interval, samples));
    }
    tasks
        .join_next()
        .await
        .expect("validated group policies")
        .map_err(io::Error::other)?
}

async fn run_policy(
    policy: Policy,
    endpoints: ControlPeers,
    period: Duration,
    samples: u32,
) -> io::Result<()> {
    let mut control = GroupControl::connect(endpoints).await?;
    let mut state = Replan::default();
    let mut ticks = interval(period);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut last_error = None;
    loop {
        ticks.tick().await;
        match state.tick(&mut control, &policy, samples).await {
            Ok(result) => {
                last_error = None;
                if let Some(result) = result {
                    println!("plait-node group replan {:?}: {result:?}", policy.group);
                }
            }
            Err(error) => {
                state.seen = None;
                let message = error.to_string();
                if last_error.as_ref() != Some(&message) {
                    eprintln!("plait-node group replan {:?}: {message}", policy.group);
                    last_error = Some(message);
                }
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Proposal {
    latest: Option<(GroupVersion, GroupPhase)>,
    active: Option<GroupVersion>,
    compared_active: bool,
    recipe: Option<NodeId>,
    resources: ResourcePlan,
}

#[derive(Default)]
struct Replan {
    seen: Option<(Proposal, u32)>,
    submitted: Option<GroupOperation>,
}

impl Replan {
    fn confirm(&mut self, proposed: &Proposal, samples: u32) -> bool {
        if let Some((last, count)) = &mut self.seen
            && last == proposed
        {
            *count = count.saturating_add(1);
            return *count >= samples;
        }
        self.seen = Some((proposed.clone(), 1));
        samples == 1
    }

    async fn tick(
        &mut self,
        control: &mut GroupControl,
        policy: &Policy,
        samples: u32,
    ) -> io::Result<Option<GroupResult>> {
        if self.submitted.is_none() {
            let reply = control
                .query_group(Snapshot::lookup(policy.group, policy.calibration))
                .await?;
            let mut snapshot = Snapshot::from_reply(reply, policy.calibration)?;
            if matches!(
                snapshot.inventory.latest,
                Some((_, GroupPhase::Preparing | GroupPhase::Ready))
            ) {
                self.seen = None;
                return Ok(None);
            }
            if !policy.workload.recipes.is_empty()
                && let Some(active) = &snapshot.inventory.active
            {
                let GroupReply::Recipe(Some(recipe)) = control
                    .query_group(GroupLookup::Recipe(active.version))
                    .await?
                else {
                    return Err(io::Error::other("active group recipe is not visible yet"));
                };
                snapshot.active_recipe = recipe.recipe;
            }
            let (links, probes) = if !policy.workload.needs_links() {
                (Vec::new(), None)
            } else {
                let GroupReply::LinkProbes(mut probes) =
                    control.query_group(GroupLookup::LinkProbes).await?
                else {
                    return Err(io::Error::other("expected link probe directory"));
                };
                snapshot.restrict_probes(&mut probes);
                (collect_links(&probes).await, Some(probes))
            };
            let Decision::Prepare {
                resources,
                incumbent,
                recipe,
                ..
            } = snapshot.select(&policy.workload, &links)?
            else {
                self.seen = None;
                return Ok(None);
            };
            let proposed = Proposal {
                latest: snapshot.inventory.latest,
                active: snapshot
                    .inventory
                    .active
                    .as_ref()
                    .map(|active| active.version),
                compared_active: incumbent.is_some(),
                recipe,
                resources,
            };
            if !self.confirm(&proposed, samples) {
                return Ok(None);
            }
            self.submitted = Some(GroupOperation {
                id: OpId::new(NodeId::random(), 1),
                action: snapshot.prepare(
                    proposed.resources,
                    proposed.compared_active,
                    probes,
                    proposed.recipe,
                ),
            });
        }
        // The previous request could have committed even if its reply was lost.
        let operation = self.submitted.as_ref().expect("submitted reservation");
        let result = control.apply_group(operation).await?;
        self.submitted = None;
        self.seen = None;
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plait_core::DeviceId;
    use plait_core::group::{HostMemory, WorkerDemand};

    #[test]
    fn stability_is_scoped_to_the_same_devices_and_incumbent() {
        let node = NodeId::derive("host");
        let device = DeviceId::new(node, 0);
        let active = GroupVersion {
            group: GroupId(NodeId::derive("group")),
            epoch: 1,
        };
        let mut proposed = Proposal {
            latest: Some((active, GroupPhase::Active)),
            active: Some(active),
            compared_active: true,
            recipe: None,
            resources: ResourcePlan {
                workers: BTreeMap::from([(
                    device,
                    WorkerDemand {
                        incarnation: 1,
                        device_bytes: 1,
                    },
                )]),
                hosts: BTreeMap::from([(node, HostMemory::default())]),
            },
        };
        let mut replan = Replan::default();
        assert!(!replan.confirm(&proposed, 2));
        proposed
            .resources
            .workers
            .get_mut(&device)
            .unwrap()
            .incarnation = 2;
        assert!(!replan.confirm(&proposed, 2));
        proposed.active = None;
        proposed.compared_active = false;
        assert!(!replan.confirm(&proposed, 2));
        assert!(replan.confirm(&proposed, 2));
        proposed.recipe = Some(NodeId::derive("directed"));
        assert!(!replan.confirm(&proposed, 2));
    }
}
