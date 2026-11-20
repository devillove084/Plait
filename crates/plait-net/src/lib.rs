//! Asynchronous, actor-owned TCP transport.
//!
//! The node owns a `TcpTransport` handle and a single connection-manager task
//! owns every outbound socket and the peer table. No application state is behind
//! a mutex: callers communicate through bounded channels, and the manager is the
//! only writer of its connection map. This is the Asio-style proactor shape:
//! completion/readiness events drive one owner, not shared locked state.

use std::collections::BTreeMap;
use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use plait_consensus::{Message, Snapshot};
use plait_core::{Command, NodeId, OpId};
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio_util::codec::{Framed, LengthDelimitedCodec};

const MAX_FRAME: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Wire {
    Hello {
        from: NodeId,
    },
    Consensus(Message),
    JoinRequest {
        node: NodeId,
        addr: String,
    },
    SnapshotRequest {
        node: NodeId,
    },
    Snapshot {
        snapshot: Snapshot,
        peers: BTreeMap<NodeId, String>,
    },
    Ready {
        node: NodeId,
    },
    Submit {
        request: u64,
        command: Command,
        reply_to: String,
    },
    Ack {
        request: u64,
        ok: bool,
        message: String,
    },
    Query {
        request: u64,
        op: OpId,
        reply_to: String,
    },
    QueryReply {
        request: u64,
        op: OpId,
        declared: bool,
        values: Option<Vec<f64>>,
        executed: usize,
        voters: Vec<NodeId>,
    },
    Ping,
    Pong,
}

#[allow(clippy::large_enum_variant)]
enum Outbound {
    Send {
        to: NodeId,
        wire: Wire,
        reply: oneshot::Sender<io::Result<()>>,
    },
    SetPeer {
        node: NodeId,
        addr: String,
    },
    Peers {
        reply: oneshot::Sender<BTreeMap<NodeId, String>>,
    },
}

/// Handle owned by the node. It is intentionally not `Clone`: a node has one
/// inbound receiver and one ordered outbound command stream.
pub struct TcpTransport {
    id: NodeId,
    local_addr: String,
    outbound: mpsc::Sender<Outbound>,
    inbound: mpsc::Receiver<(NodeId, Wire)>,
}

impl TcpTransport {
    pub async fn bind(
        id: NodeId,
        listen: &str,
        peers: BTreeMap<NodeId, String>,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind(listen).await?;
        let local_addr = listener.local_addr()?.to_string();
        let (inbound_tx, inbound_rx) = mpsc::channel(1024);
        tokio::spawn(accept_loop(listener, inbound_tx.clone()));

        let (command_tx, command_rx) = mpsc::channel(1024);
        tokio::spawn(manager(id, peers, command_rx));
        Ok(Self {
            id,
            local_addr,
            outbound: command_tx,
            inbound: inbound_rx,
        })
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn local_addr(&self) -> &str {
        &self.local_addr
    }

    /// Fire-and-forget peer update. The manager processes it in channel order,
    /// so a later `send` to the same peer sees the address.
    pub fn set_peer(&self, node: NodeId, addr: impl Into<String>) {
        let _ = self.outbound.try_send(Outbound::SetPeer {
            node,
            addr: addr.into(),
        });
    }

    pub fn set_peers(&self, peers: BTreeMap<NodeId, String>) {
        for (node, addr) in peers {
            self.set_peer(node, addr);
        }
    }

    pub async fn peers(&self) -> BTreeMap<NodeId, String> {
        let (reply, receiver) = oneshot::channel();
        if self.outbound.send(Outbound::Peers { reply }).await.is_err() {
            return BTreeMap::new();
        }
        receiver.await.unwrap_or_default()
    }

    pub async fn send(&self, to: NodeId, wire: Wire) -> io::Result<()> {
        let (reply, receiver) = oneshot::channel();
        self.outbound
            .send(Outbound::Send { to, wire, reply })
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "transport closed"))?;
        receiver
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "transport closed"))?
    }

    pub async fn recv(&mut self) -> Option<(NodeId, Wire)> {
        self.inbound.recv().await
    }

    pub fn try_recv(&mut self) -> Option<(NodeId, Wire)> {
        self.inbound.try_recv().ok()
    }
}

async fn accept_loop(listener: TcpListener, sender: mpsc::Sender<(NodeId, Wire)>) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let sender = sender.clone();
        tokio::spawn(read_connection(stream, sender));
    }
}

async fn read_connection(stream: TcpStream, sender: mpsc::Sender<(NodeId, Wire)>) {
    let mut framed = Framed::new(stream, codec());
    let from = match framed.next().await {
        Some(Ok(bytes)) => match bincode::deserialize::<Wire>(&bytes) {
            Ok(Wire::Hello { from }) => from,
            _ => return,
        },
        _ => return,
    };
    while let Some(Ok(bytes)) = framed.next().await {
        let Ok(wire) = bincode::deserialize::<Wire>(&bytes) else {
            return;
        };
        if sender.send((from, wire)).await.is_err() {
            return;
        }
    }
}

async fn manager(
    id: NodeId,
    peers: BTreeMap<NodeId, String>,
    mut commands: mpsc::Receiver<Outbound>,
) {
    let mut peers = peers;
    let mut connections: BTreeMap<NodeId, Framed<TcpStream, LengthDelimitedCodec>> =
        BTreeMap::new();
    while let Some(command) = commands.recv().await {
        match command {
            Outbound::SetPeer { node, addr } => {
                peers.insert(node, addr);
            }
            Outbound::Peers { reply } => {
                let _ = reply.send(peers.clone());
            }
            Outbound::Send { to, wire, reply } => {
                let Some(addr) = peers.get(&to).cloned() else {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("no address known for peer {to}"),
                    )));
                    continue;
                };
                if let std::collections::btree_map::Entry::Vacant(entry) = connections.entry(to) {
                    match connect(id, &addr).await {
                        Ok(connection) => {
                            entry.insert(connection);
                        }
                        Err(error) => {
                            let _ = reply.send(Err(error));
                            continue;
                        }
                    }
                }
                let result = send_frame(connections.get_mut(&to).expect("connection"), &wire).await;
                if result.is_err() {
                    connections.remove(&to);
                    if let Ok(mut connection) = connect(id, &addr).await {
                        let retry = send_frame(&mut connection, &wire).await;
                        if retry.is_ok() {
                            connections.insert(to, connection);
                        }
                        let _ = reply.send(retry);
                    } else {
                        let _ = reply.send(result);
                    }
                } else {
                    let _ = reply.send(result);
                }
            }
        }
    }
}

async fn connect(id: NodeId, addr: &str) -> io::Result<Framed<TcpStream, LengthDelimitedCodec>> {
    let resolved: SocketAddr = tokio::net::lookup_host(addr)
        .await?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "address resolved to nothing"))?;
    let stream = tokio::time::timeout(Duration::from_secs(3), TcpStream::connect(resolved))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "TCP connect timed out"))??;
    stream.set_nodelay(true)?;
    let mut framed = Framed::new(stream, codec());
    send_frame(&mut framed, &Wire::Hello { from: id }).await?;
    Ok(framed)
}

fn codec() -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .max_frame_length(MAX_FRAME)
        .new_codec()
}

async fn send_frame(
    framed: &mut Framed<TcpStream, LengthDelimitedCodec>,
    wire: &Wire,
) -> io::Result<()> {
    let bytes = bincode::serialize(wire)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    framed.send(Bytes::from(bytes)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sends_and_receives_frames() {
        let a = NodeId::derive("async-a");
        let b = NodeId::derive("async-b");
        let ta = TcpTransport::bind(a, "127.0.0.1:0", BTreeMap::new())
            .await
            .unwrap();
        let mut tb = TcpTransport::bind(b, "127.0.0.1:0", BTreeMap::new())
            .await
            .unwrap();
        ta.set_peer(b, tb.local_addr());
        tb.set_peer(a, ta.local_addr());
        ta.send(b, Wire::Ping).await.unwrap();
        let (from, wire) = tokio::time::timeout(Duration::from_secs(3), tb.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(from, a);
        assert!(matches!(wire, Wire::Ping));
    }
}
