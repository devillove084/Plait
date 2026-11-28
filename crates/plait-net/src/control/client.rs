use std::collections::BTreeMap;
use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use plait_core::group::{
    GroupAction, GroupError, GroupId as PeerGroupId, GroupLookup, GroupOperation, GroupOutcome,
    GroupReply, GroupResult, GroupView as PeerGroupView,
};

use crate::{GroupObservation, TcpTransport, Wire};
use plait_core::serving::{
    GroupId, GroupView, ServingError, ServingLookup, ServingOperation, ServingResult, ServingView,
};
use plait_core::{Command, NodeId, OpId};

pub struct Client {
    transport: TcpTransport,
    reply_addr: String,
    next: u64,
}

impl Client {
    /// Read metadata already applied by the addressed replica.
    pub async fn query_group(
        &mut self,
        node: NodeId,
        lookup: GroupLookup,
    ) -> io::Result<GroupReply> {
        self.exchange(
            node,
            "group query timed out",
            |request, reply_to| Wire::GroupQuery {
                request,
                lookup,
                reply_to,
            },
            |request, reply| match reply {
                Wire::GroupQueryReply {
                    request: answer,
                    lookup: target,
                    view,
                } if request == answer && lookup == target => Some(Ok(*view)),
                _ => None,
            },
        )
        .await
    }

    /// Submit a group command and wait for the matching applied receipt.
    pub async fn apply_group(
        &mut self,
        node: NodeId,
        operation: &GroupOperation,
    ) -> io::Result<GroupResult> {
        self.submit(node, Command::Group(Box::new(operation.clone())))
            .await?;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let GroupReply::Receipt(Some(receipt)) = self
                    .query_group(node, GroupLookup::Receipt(operation.id))
                    .await?
                {
                    if &receipt.operation != operation {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "group operation id already used for another action",
                        ));
                    }
                    return Ok(receipt.result);
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "group receipt timed out"))?
    }

    /// Order a group observation before fetching the version from that replica.
    /// Use a fresh operation ID for each observation outside of retries.
    pub async fn observe_group(
        &mut self,
        node: NodeId,
        id: OpId,
        group: PeerGroupId,
        epoch: Option<u64>,
    ) -> io::Result<Result<PeerGroupView, GroupError>> {
        let action = GroupAction::Observe { group, epoch };
        match self
            .apply_group(node, &GroupOperation { id, action })
            .await?
        {
            Ok(GroupOutcome::Version { version, .. }) => {
                if let GroupReply::Version(Some(view)) = self
                    .query_group(node, GroupLookup::Version(version))
                    .await?
                {
                    return Ok(Ok(view));
                }
                Err(io::Error::other(
                    "group observation is missing its applied version",
                ))
            }
            Err(error) => Ok(Err(error)),
            Ok(_) => Err(io::Error::other("unexpected group observation receipt")),
        }
    }
    pub async fn new() -> io::Result<Self> {
        Self::new_on("127.0.0.1:0").await
    }

    /// Bind a listener the remote replica can reach to return replies.
    pub async fn new_on(listen: &str) -> io::Result<Self> {
        Self::new_with_reply_addr(listen, None).await
    }

    /// Advertise a reachable forwarded endpoint when it differs from the bind address.
    pub async fn new_with_reply_addr(
        listen: &str,
        reply_addr: Option<SocketAddr>,
    ) -> io::Result<Self> {
        if reply_addr.is_some_and(|addr| addr.ip().is_unspecified() || addr.port() == 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "control reply address must specify an IP and a nonzero port",
            ));
        }
        let id = NodeId::random();
        let transport = TcpTransport::bind(id, listen, BTreeMap::new()).await?;
        let reply_addr = reply_addr.map_or_else(
            || transport.local_addr().to_owned(),
            |addr| addr.to_string(),
        );
        Ok(Self {
            transport,
            reply_addr,
            next: 1,
        })
    }

    pub fn add_peer(&self, node: NodeId, addr: impl Into<String>) {
        self.transport.set_peer(node, addr);
    }

    /// Submit a proposal. An acknowledgement indicates acceptance for ordering.
    pub async fn submit(&mut self, node: NodeId, command: Command) -> io::Result<String> {
        self.exchange(
            node,
            "submit acknowledgement timed out",
            |request, reply_to| Wire::Submit {
                request,
                command,
                reply_to,
            },
            |request, wire| match wire {
                Wire::Ack {
                    request: answer,
                    ok,
                    message,
                } if answer == request => Some(if ok {
                    Ok(message)
                } else {
                    Err(io::Error::other(message))
                }),
                _ => None,
            },
        )
        .await
    }

    pub async fn query(&mut self, node: NodeId, op: OpId) -> io::Result<QueryResult> {
        self.exchange(
            node,
            "query timed out",
            |request, reply_to| Wire::Query {
                request,
                op,
                reply_to,
            },
            |request, wire| match wire {
                Wire::QueryReply {
                    request: answer,
                    op: answer_op,
                    declared,
                    values,
                    executed,
                    voters,
                } if answer == request && answer_op == op => Some(Ok(QueryResult {
                    declared,
                    values,
                    executed,
                    voters,
                })),
                _ => None,
            },
        )
        .await
    }

    /// Read a replica's applied serving state. A proposal acknowledgement can
    /// arrive before its receipt or group version is visible in this view.
    pub async fn query_serving(
        &mut self,
        node: NodeId,
        lookup: ServingLookup,
    ) -> io::Result<ServingView> {
        self.exchange(
            node,
            "serving query timed out",
            |request, reply_to| Wire::ServingQuery {
                request,
                lookup,
                reply_to,
            },
            |request, wire| match wire {
                Wire::ServingQueryReply {
                    request: answer,
                    lookup: answer_lookup,
                    view,
                } if answer == request && answer_lookup == lookup => Some(Ok(*view)),
                _ => None,
            },
        )
        .await
    }

    /// Wait for a matching applied receipt, independently of the proposal acknowledgement.
    pub async fn apply_serving(
        &mut self,
        node: NodeId,
        operation: &ServingOperation,
    ) -> io::Result<ServingResult> {
        self.submit(node, Command::Serving(Box::new(operation.clone())))
            .await?;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let ServingView::Operation(Some(receipt)) = self
                    .query_serving(node, ServingLookup::Operation(operation.id))
                    .await?
                {
                    if &receipt.operation != operation {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "serving operation id already used for another action",
                        ));
                    }
                    return Ok(receipt.result);
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "serving receipt timed out"))?
    }

    /// Order a fresh group observation and receive that version's applied view.
    /// Reuse `operation` only to retry the same observation.
    pub async fn observe_serving_group(
        &mut self,
        node: NodeId,
        operation: OpId,
        group: GroupId,
        epoch: Option<u64>,
    ) -> io::Result<Result<GroupView, ServingError>> {
        self.exchange(
            node,
            "serving receipt timed out",
            |request, reply_to| Wire::ObserveServingGroup {
                request,
                operation,
                group,
                epoch,
                reply_to,
            },
            |request, wire| {
                let Wire::ServingGroupObserved {
                    request: answer,
                    response,
                } = wire
                else {
                    return None;
                };
                if answer != request {
                    return None;
                }
                Some(match response {
                    GroupObservation::Applied(view)
                        if view.version.group == group
                            && epoch.is_none_or(|epoch| epoch == view.version.epoch) =>
                    {
                        Ok(Ok(*view))
                    }
                    GroupObservation::Applied(_) => Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "applied observation returned a different group version",
                    )),
                    GroupObservation::Rejected(error) => Ok(Err(error)),
                    GroupObservation::Unavailable(reason) => Err(io::Error::other(reason)),
                })
            },
        )
        .await
    }

    async fn exchange<T>(
        &mut self,
        node: NodeId,
        timeout_message: &'static str,
        prepare: impl FnOnce(u64, String) -> Wire,
        receive: impl Fn(u64, Wire) -> Option<io::Result<T>>,
    ) -> io::Result<T> {
        let request = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| io::Error::other("client request counter exhausted"))?;
        let mut delivery = self
            .transport
            .send_tracked(node, prepare(request, self.reply_addr.clone()))
            .await?;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                tokio::select! {
                    biased;
                    reply = self.transport.recv() => {
                        let Some((from, wire)) = reply else {
                            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "client transport closed"));
                        };
                        if from == node && let Some(result) = receive(request, wire) {
                            return result;
                        }
                    }
                    _ = delivery.disconnected() => {
                        return Err(io::Error::new(
                            io::ErrorKind::ConnectionAborted,
                            "control connection closed before reply",
                        ));
                    }
                }
            }
        })
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, timeout_message))?
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    pub declared: bool,
    pub values: Option<Vec<f64>>,
    pub executed: usize,
    pub voters: Vec<NodeId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn quits_waiting_when_the_requested_voter_disconnects() {
        let id = NodeId::derive("exiting-voter");
        let mut peer = TcpTransport::bind(id, "127.0.0.1:0", BTreeMap::new())
            .await
            .unwrap();
        let mut client = Client::new().await.unwrap();
        client.add_peer(id, peer.local_addr());
        let request = OpId::new(NodeId::derive("client-request"), 1);
        let exiting = tokio::spawn(async move {
            let (_, wire) = peer.recv().await.unwrap();
            assert!(matches!(wire, Wire::Query { op, .. } if op == request));
        });
        let error = tokio::time::timeout(Duration::from_secs(2), client.query(id, request))
            .await
            .expect("control client waited for its reply deadline after TCP closed")
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
        exiting.await.unwrap();
    }
}
