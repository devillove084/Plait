use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use plait_consensus::{Message, Snapshot};
use plait_core::{Command, NodeId, OpId};
use serde::{Deserialize, Serialize};

const MAX_FRAME: usize = 64 * 1024 * 1024;

/// All traffic between Plait processes. Consensus messages are only one kind;
/// membership catch-up and the small demo/control surface share the same
/// framing.
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

/// A length-prefixed TCP transport. Each peer pair uses one outbound stream
/// for sending; the peer's listener delivers inbound frames to its node loop.
/// Failed writes drop the cached stream, so the next send reconnects.
pub struct TcpTransport {
    id: NodeId,
    local_addr: String,
    peers: Arc<Mutex<BTreeMap<NodeId, String>>>,
    outbound: Mutex<BTreeMap<NodeId, TcpStream>>,
    frame_buffer: Mutex<Vec<u8>>,
    inbound_tx: Sender<(NodeId, Wire)>,
    inbound_rx: Receiver<(NodeId, Wire)>,
}

impl TcpTransport {
    pub fn bind(id: NodeId, listen: &str, peers: BTreeMap<NodeId, String>) -> io::Result<Self> {
        let listener = TcpListener::bind(listen)?;
        let local_addr = listener.local_addr()?.to_string();
        let (inbound_tx, inbound_rx) = mpsc::channel();
        let sender = inbound_tx.clone();
        thread::spawn(move || accept_loop(listener, sender));
        Ok(Self {
            id,
            local_addr,
            peers: Arc::new(Mutex::new(peers)),
            outbound: Mutex::new(BTreeMap::new()),
            frame_buffer: Mutex::new(Vec::with_capacity(4096)),
            inbound_tx,
            inbound_rx,
        })
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn local_addr(&self) -> &str {
        &self.local_addr
    }

    pub fn set_peer(&self, node: NodeId, addr: impl Into<String>) {
        self.peers
            .lock()
            .expect("peer map poisoned")
            .insert(node, addr.into());
    }

    pub fn set_peers(&self, peers: BTreeMap<NodeId, String>) {
        let mut current = self.peers.lock().expect("peer map poisoned");
        current.extend(peers);
    }

    pub fn peers(&self) -> BTreeMap<NodeId, String> {
        self.peers.lock().expect("peer map poisoned").clone()
    }

    pub fn peer_addr(&self, node: NodeId) -> Option<String> {
        self.peers
            .lock()
            .expect("peer map poisoned")
            .get(&node)
            .cloned()
    }

    pub fn send(&self, to: NodeId, wire: Wire) -> io::Result<()> {
        if to == self.id {
            let _ = self.inbound_tx.send((self.id, wire));
            return Ok(());
        }
        let addr = self.peer_addr(to).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("no address known for peer {to}"),
            )
        })?;
        let mut streams = self.outbound.lock().expect("outbound map poisoned");
        let mut frame = self.frame_buffer.lock().expect("frame buffer poisoned");
        encode(&wire, &mut frame)?;
        if let std::collections::btree_map::Entry::Vacant(entry) = streams.entry(to) {
            entry.insert(connect(&addr, self.id)?);
        }
        let stream = streams.get_mut(&to).expect("stream was inserted");
        match write_frame(stream, &frame) {
            Ok(()) => Ok(()),
            Err(_) => {
                streams.remove(&to);
                let mut fresh = connect(&addr, self.id)?;
                write_frame(&mut fresh, &frame)?;
                streams.insert(to, fresh);
                Ok(())
            }
        }
    }

    pub fn recv_timeout(&self, timeout: Duration) -> Option<(NodeId, Wire)> {
        self.inbound_rx.recv_timeout(timeout).ok()
    }

    pub fn try_recv(&self) -> Option<(NodeId, Wire)> {
        self.inbound_rx.try_recv().ok()
    }
}

fn connect(addr: &str, from: NodeId) -> io::Result<TcpStream> {
    let resolved = addr
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "address resolved to nothing"))?;
    let mut stream = TcpStream::connect_timeout(&resolved, Duration::from_secs(3))?;
    stream.set_nodelay(true)?;
    let mut frame = Vec::with_capacity(64);
    encode(&Wire::Hello { from }, &mut frame)?;
    write_frame(&mut stream, &frame)?;
    Ok(stream)
}

fn accept_loop(listener: TcpListener, sender: Sender<(NodeId, Wire)>) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let sender = sender.clone();
                thread::spawn(move || read_connection(stream, sender));
            }
            Err(_) => break,
        }
    }
}

fn read_connection(mut stream: TcpStream, sender: Sender<(NodeId, Wire)>) {
    let mut payload = Vec::with_capacity(4096);
    let from = match read_frame_into(&mut stream, &mut payload) {
        Ok(Wire::Hello { from }) => from,
        Ok(_) | Err(_) => return,
    };
    loop {
        match read_frame_into(&mut stream, &mut payload) {
            Ok(wire) => {
                if sender.send((from, wire)).is_err() {
                    return;
                }
            }
            Err(_) => return,
        }
    }
}

fn encode(wire: &Wire, frame: &mut Vec<u8>) -> io::Result<()> {
    frame.clear();
    bincode::serialize_into(&mut *frame, wire)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    if frame.len() > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame exceeds the size limit",
        ));
    }
    Ok(())
}

fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    stream.write_all(&(payload.len() as u32).to_be_bytes())?;
    stream.write_all(payload)?;
    stream.flush()
}

fn read_frame_into(stream: &mut TcpStream, payload: &mut Vec<u8>) -> io::Result<Wire> {
    let mut length = [0u8; 4];
    stream.read_exact(&mut length)?;
    let length = u32::from_be_bytes(length) as usize;
    if length > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame exceeds the size limit",
        ));
    }
    payload.resize(length, 0);
    stream.read_exact(payload)?;
    bincode::deserialize(payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn sends_and_receives_frames() {
        let a = NodeId::derive("net-a");
        let b = NodeId::derive("net-b");
        let ta = TcpTransport::bind(a, "127.0.0.1:0", BTreeMap::new()).unwrap();
        let tb = TcpTransport::bind(b, "127.0.0.1:0", BTreeMap::new()).unwrap();
        ta.set_peer(b, tb.local_addr());
        tb.set_peer(a, ta.local_addr());
        ta.send(b, Wire::Ping).unwrap();

        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some((from, wire)) = tb.recv_timeout(Duration::from_millis(100)) {
                assert_eq!(from, a);
                assert!(matches!(wire, Wire::Ping));
                break;
            }
            assert!(Instant::now() < deadline, "frame did not arrive");
        }
    }

    #[test]
    fn reconnects_after_the_cached_stream_is_dropped() {
        let a = NodeId::derive("net-c");
        let b = NodeId::derive("net-d");
        let ta = TcpTransport::bind(a, "127.0.0.1:0", BTreeMap::new()).unwrap();
        let tb = TcpTransport::bind(b, "127.0.0.1:0", BTreeMap::new()).unwrap();
        ta.set_peer(b, tb.local_addr());
        tb.set_peer(a, ta.local_addr());
        ta.send(b, Wire::Ping).unwrap();
        ta.outbound.lock().unwrap().clear();
        ta.send(b, Wire::Pong).unwrap();

        let mut seen = Vec::new();
        for _ in 0..2 {
            if let Some((_, wire)) = tb.recv_timeout(Duration::from_secs(2)) {
                seen.push(matches!(wire, Wire::Ping | Wire::Pong));
            }
        }
        assert_eq!(seen.len(), 2);
    }
}
