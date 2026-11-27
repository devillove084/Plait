//! Join a communicator using device grants and endpoints from the control plane.

use std::collections::BTreeSet;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

use plait_core::group::{
    GroupAction, GroupLookup, GroupOperation, GroupPhase, GroupReply, GroupVersion, GroupView,
};
use plait_core::{DeviceId, NodeId, OpId};
use plait_net::control::{Client, discover_voters};
use tokio::time::{Instant, sleep, timeout_at};

use crate::{BackendKind, Comm, RanklessComm, RanklessListener, create_runtime};

/// A caller's local device grant identity and the desired collective slot order.
pub struct JoinGroup {
    pub version: GroupVersion,
    pub member: DeviceId,
    pub layout: Vec<DeviceId>,
    pub listen: SocketAddr,
    pub seeds: Vec<SocketAddr>,
}

/// Publish the local listener, connect to the committed members, and report ready.
/// The caller decides when to activate the group and admit work to the communicator.
pub async fn join_committed_group(
    request: JoinGroup,
    backend: BackendKind,
    device: usize,
    timeout: Duration,
) -> io::Result<RanklessComm> {
    let layout: BTreeSet<_> = request.layout.iter().copied().collect();
    if layout.len() != request.layout.len()
        || !layout.contains(&request.member)
        || request.version.epoch == 0
        || request.listen.ip().is_unspecified()
        || request.seeds.is_empty()
        || request
            .seeds
            .iter()
            .any(|seed| seed.ip().is_unspecified() || seed.port() == 0)
        || timeout.is_zero()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Plait group join",
        ));
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid group join timeout"))?;
    timeout_at(
        deadline,
        connect(request, layout, backend, device, deadline, timeout),
    )
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "Plait group join timed out"))?
}

async fn connect(
    request: JoinGroup,
    layout: BTreeSet<DeviceId>,
    backend: BackendKind,
    device: usize,
    deadline: Instant,
    timeout: Duration,
) -> io::Result<RanklessComm> {
    let JoinGroup {
        version,
        member,
        layout: order,
        listen,
        seeds,
    } = request;
    let mut control = Rendezvous::new(listen, seeds).await?;
    let session = NodeId::random();
    let view = control
        .call(|client, voter| {
            Box::pin(client.observe_group(
                voter,
                OpId::new(session, 1),
                version.group,
                Some(version.epoch),
            ))
        })
        .await
        .map_err(|error| io::Error::other(format!("observe group: {error:?}")))?;
    check_open(&view)?;
    let granted = view
        .grants
        .iter()
        .map(|grant| grant.device)
        .collect::<BTreeSet<_>>();
    if layout != granted {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "slot layout differs from the committed device grants",
        ));
    }
    let grant = view
        .grants
        .into_iter()
        .find(|grant| grant.device == member)
        .ok_or_else(|| io::Error::other("this device has no grant for this group"))?;
    let runtime = create_runtime(backend, device).map_err(io::Error::other)?;
    let listener = RanklessListener::bind(listen, *session.as_bytes(), grant)?;
    control
        .apply(&GroupOperation {
            id: OpId::new(session, 2),
            action: GroupAction::PublishEndpoint(listener.member()),
        })
        .await?;

    let manifest = loop {
        if let Some(view) = control.view(version).await {
            check_open(&view)?;
            if let Some(manifest) = view.manifest {
                break manifest;
            }
        }
        sleep(POLL).await;
    };

    // Other local participants may be using this runtime to finish their joins.
    let group = tokio::task::spawn_blocking(move || {
        listener.connect(
            manifest.members,
            Some(&order),
            timeout,
            deadline.saturating_duration_since(Instant::now()),
        )
    })
    .await
    .map_err(io::Error::other)??;
    let comm = Comm::rankless_from_peers(group, device, runtime);
    let digest = comm
        .peer_manifest_digest()
        .ok_or_else(|| io::Error::other("peer group failed during startup"))?;
    control
        .apply(&GroupOperation {
            id: OpId::new(session, 3),
            action: GroupAction::Ready {
                grant,
                manifest: digest,
            },
        })
        .await?;
    Ok(comm)
}

fn check_open(view: &GroupView) -> io::Result<()> {
    if matches!(
        view.phase,
        GroupPhase::Preparing | GroupPhase::Ready | GroupPhase::Active
    ) {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "Plait group cannot be joined in phase {:?}",
            view.phase
        )))
    }
}

const POLL: Duration = Duration::from_millis(100);
type PeerCall<'a, T> = Pin<Box<dyn Future<Output = io::Result<T>> + Send + 'a>>;

struct Rendezvous {
    client: Client,
    seeds: Vec<SocketAddr>,
    voters: Vec<NodeId>,
}

impl Rendezvous {
    async fn new(listen: SocketAddr, seeds: Vec<SocketAddr>) -> io::Result<Self> {
        let reply = SocketAddr::new(listen.ip(), 0);
        Ok(Self {
            client: Client::new_on(&reply.to_string()).await?,
            seeds,
            voters: Vec::new(),
        })
    }

    async fn apply(&mut self, operation: &GroupOperation) -> io::Result<()> {
        self.call(|client, voter| {
            let operation = operation.clone();
            Box::pin(async move { client.apply_group(voter, &operation).await })
        })
        .await
        .map(|_| ())
        .map_err(|error| io::Error::other(format!("group join rejected: {error:?}")))
    }

    async fn view(&mut self, version: GroupVersion) -> Option<GroupView> {
        let reply = self
            .call(|client, voter| {
                Box::pin(client.query_group(voter, GroupLookup::Version(version)))
            })
            .await;
        // A reachable replica can be behind on publications from other workers.
        self.voters.rotate_left(1);
        match reply {
            GroupReply::Version(view) => view,
            _ => None,
        }
    }

    async fn call<T>(
        &mut self,
        invoke: impl for<'a> Fn(&'a mut Client, NodeId) -> PeerCall<'a, T>,
    ) -> T {
        loop {
            for (index, &voter) in self.voters.iter().enumerate() {
                if let Ok(result) = invoke(&mut self.client, voter).await {
                    self.voters.rotate_left(index);
                    return result;
                }
            }
            self.voters.clear();
            if let Ok(voters) = discover_voters(&self.seeds).await {
                for (id, address) in voters {
                    self.client.add_peer(id, address);
                    self.voters.push(id);
                }
            }
            sleep(POLL).await;
        }
    }
}
