use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;

use plait_core::group::{GroupLookup, GroupOperation, GroupReply, GroupResult};
use plait_core::serving::{
    GroupId, GroupVersion, GroupView, ServingAction, ServingError, ServingLookup, ServingOperation,
    ServingResult, ServingView, WorkerGrant,
};
use plait_core::{NodeId, OpId};
use plait_fabric::reduction::ReductionError;
use plait_node::{Client, discover_voters};
use serde::{Deserialize, Serialize};

type PeerCall<'a, T> = Pin<Box<dyn Future<Output = io::Result<T>> + Send + 'a>>;

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct ControlPeers {
    peers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    seeds: Vec<SocketAddr>,
    listen: Option<String>,
    reply_addr: Option<SocketAddr>,
}

impl ControlPeers {
    pub(crate) fn new(peers: Vec<String>, seeds: Vec<SocketAddr>, listen: String) -> Self {
        Self {
            peers,
            seeds,
            listen: Some(listen),
            reply_addr: None,
        }
    }
}

pub(crate) struct GroupControl {
    client: Client,
    peers: Vec<NodeId>,
    seeds: Vec<SocketAddr>,
    discovered: BTreeMap<NodeId, String>,
}

impl GroupControl {
    pub(crate) async fn connect(config: ControlPeers) -> io::Result<Self> {
        if config.peers.is_empty() == config.seeds.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "configure either control peers or control seed addresses",
            ));
        }
        let client = Client::new_with_reply_addr(
            config.listen.as_deref().unwrap_or("127.0.0.1:0"),
            config.reply_addr,
        )
        .await?;
        let mut peers = Vec::with_capacity(config.peers.len());
        for peer in config.peers {
            let (id, addr) = super::super::parse_peer(&peer)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
            client.add_peer(id, addr);
            peers.push(id);
        }
        Ok(Self {
            client,
            peers,
            seeds: config.seeds,
            discovered: BTreeMap::new(),
        })
    }

    pub(super) async fn observe(
        &mut self,
        group: GroupId,
        epoch: Option<u64>,
    ) -> io::Result<GroupView> {
        // Preserve the observation identity when retrying an ambiguous reply.
        let operation = OpId::new(NodeId::random(), 1);
        self.call(|client, peer| {
            Box::pin(client.observe_serving_group(peer, operation, group, epoch))
        })
        .await?
        .map_err(|error| io::Error::other(format!("group observation rejected: {error:?}")))
    }

    pub(super) async fn version(&mut self, version: GroupVersion) -> io::Result<GroupView> {
        self.observe(version.group, Some(version.epoch)).await
    }

    pub(super) async fn query(&mut self, lookup: ServingLookup) -> io::Result<ServingView> {
        self.call(|client, peer| Box::pin(client.query_serving(peer, lookup)))
            .await
    }

    pub(super) async fn apply(
        &mut self,
        operation: &ServingOperation,
    ) -> io::Result<ServingResult> {
        self.call(|client, peer| {
            let operation = operation.clone();
            Box::pin(async move { client.apply_serving(peer, &operation).await })
        })
        .await
    }

    pub(crate) async fn apply_group(
        &mut self,
        operation: &GroupOperation,
    ) -> io::Result<GroupResult> {
        self.call(|client, peer| {
            let operation = operation.clone();
            Box::pin(async move { client.apply_group(peer, &operation).await })
        })
        .await
    }

    pub(crate) async fn query_group(&mut self, lookup: GroupLookup) -> io::Result<GroupReply> {
        let result = self
            .call(|client, peer| Box::pin(client.query_group(peer, lookup)))
            .await;
        if result.is_ok() {
            self.peers.rotate_left(1);
        }
        result
    }

    async fn call<T>(
        &mut self,
        invoke: impl for<'a> Fn(&'a mut Client, NodeId) -> PeerCall<'a, T>,
    ) -> io::Result<T> {
        if self.peers.is_empty() {
            self.refresh().await?;
        }
        match self.try_peers(&invoke).await {
            Err(error) if !self.seeds.is_empty() => {
                if self.refresh().await.unwrap_or(false) {
                    self.try_peers(&invoke).await
                } else {
                    Err(error)
                }
            }
            result => result,
        }
    }

    async fn refresh(&mut self) -> io::Result<bool> {
        let updated = discover_voters(&self.seeds).await?;
        if updated == self.discovered {
            return Ok(false);
        }
        for (&node, addr) in &updated {
            self.client.add_peer(node, addr.clone());
        }
        self.peers = updated.keys().copied().collect();
        self.discovered = updated;
        Ok(true)
    }

    async fn try_peers<T>(
        &mut self,
        invoke: &impl for<'a> Fn(&'a mut Client, NodeId) -> PeerCall<'a, T>,
    ) -> io::Result<T> {
        let mut last = None;
        for (index, &peer) in self.peers.iter().enumerate() {
            match invoke(&mut self.client, peer).await {
                Ok(result) => {
                    self.peers.rotate_left(index);
                    return Ok(result);
                }
                Err(error) => last = Some(error),
            }
        }
        Err(last.expect("nonempty control peers"))
    }

    pub(super) async fn check_worker(
        &mut self,
        grant: WorkerGrant,
        work: Option<OpId>,
    ) -> io::Result<()> {
        let operation = ServingOperation {
            id: OpId::new(NodeId::random(), 1),
            action: ServingAction::CheckW4 { grant, work },
        };
        match self.apply(&operation).await? {
            Ok(_) => Ok(()),
            Err(ServingError::W4NotAdmitted) => {
                Err(io::Error::other("W4 work has no live admission"))
            }
            Err(_) => Err(io::Error::other(ReductionError::GrantUnavailable)),
        }
    }
}
