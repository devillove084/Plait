//! Point-to-point messages over a rank-0 relay.
//!
//! NCCL's send/recv pair is not a collective: a rank may send before its peer
//! posts the matching receive, and vLLM groups several of them behind
//! `ncclGroupStart/End`. The relay keeps one full-duplex control channel per
//! rank and a router thread per source, so a send never waits for the
//! destination to enter recv. Data is host-staged, matching the rest of the
//! portable backend; the API and ordering are the stable part.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const MAGIC: &[u8; 4] = b"P2P1";
const HEADER: usize = 4 + 4 + 4 + 8;

#[derive(Debug)]
pub struct P2pError(pub String);

impl std::fmt::Display for P2pError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for P2pError {}

fn err(message: impl Into<String>) -> P2pError {
    P2pError(message.into())
}

fn write_header(
    stream: &mut TcpStream,
    source: u32,
    destination: u32,
    bytes: usize,
) -> Result<(), P2pError> {
    let mut header = [0u8; HEADER];
    header[..4].copy_from_slice(MAGIC);
    header[4..8].copy_from_slice(&source.to_le_bytes());
    header[8..12].copy_from_slice(&destination.to_le_bytes());
    header[12..20].copy_from_slice(&(bytes as u64).to_le_bytes());
    stream
        .write_all(&header)
        .map_err(|error| err(format!("p2p header: {error}")))
}

fn read_header(stream: &mut TcpStream) -> Result<(u32, u32, usize), P2pError> {
    let mut header = [0u8; HEADER];
    stream
        .read_exact(&mut header)
        .map_err(|error| err(format!("p2p header read: {error}")))?;
    if &header[..4] != MAGIC {
        return Err(err("p2p bad magic"));
    }
    let source = u32::from_le_bytes(header[4..8].try_into().expect("u32"));
    let destination = u32::from_le_bytes(header[8..12].try_into().expect("u32"));
    let bytes = u64::from_le_bytes(header[12..20].try_into().expect("u64")) as usize;
    Ok((source, destination, bytes))
}

/// A relay shared by every rank. Rank 0 owns the routers; other ranks have one
/// stream to rank 0 in each direction.
pub struct P2p {
    rank: u32,
    root: Option<Mutex<TcpStream>>,
    peers: Arc<Mutex<Vec<Option<TcpStream>>>>,
    inbox: Arc<Mutex<VecDeque<Vec<u8>>>>,
    routers: Vec<JoinHandle<()>>,
}

impl P2p {
    pub fn root(port: u16, world: usize) -> Result<Self, P2pError> {
        let listener = TcpListener::bind(("0.0.0.0", port.saturating_add(1)))
            .map_err(|error| err(format!("p2p listen: {error}")))?;
        let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        for _ in 1..world {
            let (mut stream, _) = listener
                .accept()
                .map_err(|error| err(format!("p2p accept: {error}")))?;
            stream
                .set_nodelay(true)
                .map_err(|error| err(format!("p2p nodelay: {error}")))?;
            let mut rank_bytes = [0u8; 4];
            stream
                .read_exact(&mut rank_bytes)
                .map_err(|error| err(format!("p2p rank: {error}")))?;
            let peer_rank = u32::from_le_bytes(rank_bytes) as usize;
            if peer_rank == 0 || peer_rank >= world || peers[peer_rank].is_some() {
                return Err(err("p2p duplicate or invalid rank"));
            }
            peers[peer_rank] = Some(stream);
        }
        let peers = Arc::new(Mutex::new(peers));
        let inbox = Arc::new(Mutex::new(VecDeque::new()));
        let mut routers = Vec::with_capacity(world.saturating_sub(1));
        for source in 1..world {
            let source_stream = peers.lock().unwrap()[source]
                .as_ref()
                .ok_or_else(|| err("missing p2p source"))?
                .try_clone()
                .map_err(|error| err(format!("p2p clone: {error}")))?;
            let destinations = Arc::clone(&peers);
            let local_inbox = Arc::clone(&inbox);
            routers.push(thread::spawn(move || {
                relay(source as u32, source_stream, destinations, local_inbox);
            }));
        }
        Ok(Self {
            rank: 0,
            root: None,
            peers,
            inbox,
            routers,
        })
    }

    pub fn worker(rank: u32, master: &str, port: u16) -> Result<Self, P2pError> {
        let mut last = String::new();
        let mut stream = None;
        for _ in 0..600 {
            match TcpStream::connect((master, port.saturating_add(1))) {
                Ok(mut value) => {
                    value
                        .set_nodelay(true)
                        .map_err(|error| err(format!("p2p nodelay: {error}")))?;
                    value
                        .write_all(&rank.to_le_bytes())
                        .map_err(|error| err(format!("p2p rank: {error}")))?;
                    stream = Some(value);
                    break;
                }
                Err(error) => {
                    last = error.to_string();
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
        let stream = stream.ok_or_else(|| err(format!("p2p connect: {last}")))?;
        Ok(Self {
            rank,
            root: Some(Mutex::new(stream)),
            peers: Arc::new(Mutex::new(Vec::new())),
            inbox: Arc::new(Mutex::new(VecDeque::new())),
            routers: Vec::new(),
        })
    }

    pub fn send(&self, destination: u32, bytes: &[u8]) -> Result<(), P2pError> {
        if self.rank == 0 {
            let mut peers = self.peers.lock().unwrap();
            let stream = peers
                .get_mut(destination as usize)
                .and_then(Option::as_mut)
                .ok_or_else(|| err("p2p destination is rank 0 or absent"))?;
            write_header(stream, self.rank, destination, bytes.len())?;
            stream
                .write_all(bytes)
                .map_err(|error| err(format!("p2p send: {error}")))?;
            stream
                .flush()
                .map_err(|error| err(format!("p2p flush: {error}")))?;
        } else {
            let mut root = self.root.as_ref().expect("worker root").lock().unwrap();
            write_header(&mut root, self.rank, destination, bytes.len())?;
            root.write_all(bytes)
                .map_err(|error| err(format!("p2p send: {error}")))?;
            root.flush()
                .map_err(|error| err(format!("p2p flush: {error}")))?;
        }
        Ok(())
    }

    pub fn recv(&self, source: u32, bytes: usize) -> Result<Vec<u8>, P2pError> {
        if self.rank == 0 {
            loop {
                if let Some(buffer) = self.inbox.lock().unwrap().pop_front() {
                    return Ok(buffer);
                }
                thread::yield_now();
            }
        }
        let mut root = self.root.as_ref().expect("worker root").lock().unwrap();
        let (received_source, received_destination, length) = read_header(&mut root)?;
        if received_source != source || received_destination != self.rank {
            return Err(err(format!(
                "p2p unexpected message {received_source}->{received_destination}"
            )));
        }
        if length != bytes {
            return Err(err(format!("p2p length {length} != {bytes}")));
        }
        let mut buffer = vec![0u8; length];
        root.read_exact(&mut buffer)
            .map_err(|error| err(format!("p2p recv: {error}")))?;
        Ok(buffer)
    }
}

fn relay(
    source: u32,
    mut source_stream: TcpStream,
    destinations: Arc<Mutex<Vec<Option<TcpStream>>>>,
    inbox: Arc<Mutex<VecDeque<Vec<u8>>>>,
) {
    loop {
        let (received_source, destination, length) = match read_header(&mut source_stream) {
            Ok(header) => header,
            Err(_) => return,
        };
        if received_source != source {
            return;
        }
        let mut buffer = vec![0u8; length];
        if source_stream.read_exact(&mut buffer).is_err() {
            return;
        }
        if destination == 0 {
            inbox.lock().unwrap().push_back(buffer);
            continue;
        }
        let mut guards = destinations.lock().unwrap();
        let Some(stream) = guards
            .get_mut(destination as usize)
            .and_then(Option::as_mut)
        else {
            return;
        };
        if write_header(stream, received_source, destination, buffer.len()).is_err()
            || stream.write_all(&buffer).is_err()
            || stream.flush().is_err()
        {
            return;
        }
    }
}

impl Drop for P2p {
    fn drop(&mut self) {
        // Closing the root/peer streams makes the relay threads return.
        if let Some(root) = &self.root {
            let _ = root.lock().unwrap().shutdown(std::net::Shutdown::Both);
        }
        for stream in self.peers.lock().unwrap().iter_mut().flatten() {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
        for router in self.routers.drain(..) {
            let _ = router.join();
        }
    }
}
