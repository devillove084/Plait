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
use plait_core::link::ProbeRegistration;
use plait_core::{NodeId, OpId};
use plait_parallel::group_placement::{Decision, DirectedLink, Workload};
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

struct LinkReading {
    registrations: (ProbeRegistration, ProbeRegistration),
    sampled_at_ms: u64,
}

type LinkWindow = BTreeMap<(NodeId, NodeId), LinkReading>;

fn link_window(samples: &[DirectedLink], probes: &[ProbeRegistration]) -> io::Result<LinkWindow> {
    let by_host: BTreeMap<_, _> = probes
        .iter()
        .map(|probe| (probe.endpoint.host, *probe))
        .collect();
    samples
        .iter()
        .map(|sample| {
            let registration = |host| {
                by_host
                    .get(&host)
                    .copied()
                    .ok_or_else(|| io::Error::other("priced link has no registered probe endpoint"))
            };
            Ok((
                (sample.source, sample.target),
                LinkReading {
                    registrations: (registration(sample.source)?, registration(sample.target)?),
                    sampled_at_ms: sample.sampled_at_ms,
                },
            ))
        })
        .collect()
}

struct Observation {
    proposal: Proposal,
    links: LinkWindow,
    count: u32,
}

#[derive(Default)]
struct Replan {
    seen: Option<Observation>,
    submitted: Option<GroupOperation>,
}

impl Replan {
    fn confirm(&mut self, proposed: &Proposal, links: LinkWindow, samples: u32) -> bool {
        if let Some(last) = &mut self.seen
            && last.proposal == *proposed
            && last.links.len() == links.len()
            && last.links.iter().all(|(path, reading)| {
                links
                    .get(path)
                    .is_some_and(|present| present.registrations == reading.registrations)
            })
        {
            // Keep the last counted window while individual probe directions catch up.
            if last
                .links
                .iter()
                .any(|(path, reading)| links[path].sampled_at_ms <= reading.sampled_at_ms)
            {
                return false;
            }
            last.links = links;
            last.count = last.count.saturating_add(1);
            return last.count >= samples;
        }
        self.seen = Some(Observation {
            proposal: proposed.clone(),
            links,
            count: 1,
        });
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
                link_samples,
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
            let observed = link_window(&link_samples, probes.as_deref().unwrap_or_default())?;
            if !self.confirm(&proposed, observed, samples) {
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
    use plait_core::link::ProbeEndpoint;

    fn proposal() -> Proposal {
        let node = NodeId::derive("host");
        let device = DeviceId::new(node, 0);
        let active = GroupVersion {
            group: GroupId(NodeId::derive("group")),
            epoch: 1,
        };
        Proposal {
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
        }
    }

    #[test]
    fn stability_is_scoped_to_the_same_devices_and_incumbent() {
        let mut proposed = proposal();
        let device = *proposed.resources.workers.keys().next().unwrap();
        let mut replan = Replan::default();
        let mut confirm = |proposed: &Proposal| replan.confirm(proposed, LinkWindow::new(), 2);
        assert!(!confirm(&proposed));
        proposed
            .resources
            .workers
            .get_mut(&device)
            .unwrap()
            .incarnation = 2;
        assert!(!confirm(&proposed));
        proposed.active = None;
        proposed.compared_active = false;
        assert!(!confirm(&proposed));
        assert!(confirm(&proposed));
        proposed.recipe = Some(NodeId::derive("directed"));
        assert!(!confirm(&proposed));
    }

    #[test]
    fn stable_link_window_requires_both_directions_and_resets_on_probe_replacement() {
        let a = NodeId::derive("a");
        let b = NodeId::derive("b");
        let mut probes = [a, b].map(|host| ProbeRegistration {
            revision: 1,
            endpoint: ProbeEndpoint {
                host,
                session: host,
                address: "127.0.0.1:1234".parse().unwrap(),
            },
        });
        let sample = |source, target, sampled_at_ms| DirectedLink {
            source,
            target,
            sampled_at_ms,
            bytes_per_second: 1,
            rtt_ns: 1,
        };
        let mut replan = Replan::default();
        let proposed = proposal();
        let mut confirm = |first, second, probes: &[ProbeRegistration]| {
            let window = link_window(&[sample(a, b, first), sample(b, a, second)], probes).unwrap();
            replan.confirm(&proposed, window, 3)
        };
        assert!(!confirm(10, 11, &probes));
        assert!(!confirm(10, 11, &probes));
        assert!(!confirm(14, 11, &probes));
        assert!(!confirm(10, 16, &probes));
        assert!(!confirm(14, 16, &probes));
        assert!(confirm(18, 17, &probes));
        probes[0].revision += 1;
        probes[0].endpoint.session = NodeId::derive("restart");
        assert!(!confirm(20, 21, &probes));
        assert!(!confirm(22, 22, &probes));
        assert!(confirm(24, 24, &probes));
    }
}
