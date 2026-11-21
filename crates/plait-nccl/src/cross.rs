//! Parallel cross-host channels between host leaders.
//!
//! The regular collective network has one stream to rank 0. Large allreduce
//! messages split better over several independent TCP connections; this module
//! creates one small channel set per remote host leader.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

#[derive(Debug)]
pub struct CrossError(pub String);

impl std::fmt::Display for CrossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CrossError {}

fn err(message: impl Into<String>) -> CrossError {
    CrossError(message.into())
}

pub(crate) struct Peer {
    streams: Vec<Mutex<TcpStream>>,
}

pub enum Cross {
    Root(HashMap<usize, Peer>),
    Leader(Peer),
}

impl Cross {
    pub fn root(port: u16, leaders: &[usize], channels: usize) -> Result<Self, CrossError> {
        let mut peers = HashMap::new();
        for leader in leaders {
            let listener = TcpListener::bind(("0.0.0.0", port.saturating_add(2 + *leader as u16)))
                .map_err(|error| err(format!("cross listen: {error}")))?;
            let mut streams: Vec<Option<TcpStream>> = (0..channels).map(|_| None).collect();
            for _ in 0..channels {
                let (mut stream, _) = listener
                    .accept()
                    .map_err(|error| err(format!("cross accept: {error}")))?;
                stream
                    .set_nodelay(true)
                    .map_err(|error| err(format!("cross nodelay: {error}")))?;
                let mut index = [0u8; 4];
                stream
                    .read_exact(&mut index)
                    .map_err(|error| err(format!("cross index: {error}")))?;
                let index = u32::from_le_bytes(index) as usize;
                if index >= channels || streams[index].is_some() {
                    return Err(err("cross duplicate channel"));
                }
                streams[index] = Some(stream);
            }
            let streams = streams
                .into_iter()
                .map(|stream| stream.ok_or_else(|| err("cross missing channel")))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(Mutex::new)
                .collect();
            peers.insert(*leader, Peer { streams });
        }
        Ok(Self::Root(peers))
    }

    pub fn leader(port: u16, leader: usize, channels: usize) -> Result<Self, CrossError> {
        let mut streams = Vec::with_capacity(channels);
        for index in 0..channels {
            let mut last = String::new();
            let mut connected = None;
            for _ in 0..600 {
                match TcpStream::connect((
                    &*crate::master_addr(),
                    port.saturating_add(2 + leader as u16),
                )) {
                    Ok(mut stream) => {
                        stream
                            .set_nodelay(true)
                            .map_err(|error| err(format!("cross nodelay: {error}")))?;
                        stream
                            .write_all(&(index as u32).to_le_bytes())
                            .map_err(|error| err(format!("cross index: {error}")))?;
                        connected = Some(stream);
                        break;
                    }
                    Err(error) => {
                        last = error.to_string();
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            }
            streams.push(Mutex::new(
                connected.ok_or_else(|| err(format!("cross connect: {last}")))?,
            ));
        }
        Ok(Self::Leader(Peer { streams }))
    }

    fn bounds(bytes: usize, channels: usize, index: usize) -> (usize, usize) {
        let part = bytes.div_ceil(channels);
        let start = (index * part).min(bytes);
        let end = (start + part).min(bytes);
        (start, end)
    }

    fn send(peer: &Peer, data: &[u8]) -> Result<(), CrossError> {
        let channels = peer.streams.len();
        thread::scope(|scope| {
            for (index, stream) in peer.streams.iter().enumerate() {
                let (start, end) = Self::bounds(data.len(), channels, index);
                let chunk = &data[start..end];
                scope.spawn(move || {
                    let mut stream = stream.lock().unwrap();
                    stream
                        .write_all(chunk)
                        .map_err(|error| err(format!("cross send: {error}")))
                });
            }
        });
        Ok(())
    }

    fn recv(peer: &Peer, bytes: usize) -> Result<Vec<u8>, CrossError> {
        let channels = peer.streams.len();
        let mut output = vec![0u8; bytes];
        let mut failed = None;
        thread::scope(|scope| {
            let mut handles = Vec::new();
            let part = bytes.div_ceil(channels);
            for (stream, target) in peer.streams.iter().zip(output.chunks_mut(part)) {
                handles.push(scope.spawn(move || {
                    let mut stream = stream.lock().unwrap();
                    stream
                        .read_exact(target)
                        .map_err(|error| err(format!("cross recv: {error}")))
                }));
            }
            for handle in handles {
                if let Err(error) = handle.join().expect("cross thread") {
                    failed = Some(error);
                }
            }
        });
        if let Some(error) = failed {
            return Err(error);
        }
        Ok(output)
    }

    pub fn root_send(&self, leader: usize, data: &[u8]) -> Result<(), CrossError> {
        match self {
            Self::Root(peers) => Self::send(
                peers
                    .get(&leader)
                    .ok_or_else(|| err("cross leader missing"))?,
                data,
            ),
            Self::Leader(_) => Err(err("root send from leader")),
        }
    }

    pub fn root_recv(&self, leader: usize, bytes: usize) -> Result<Vec<u8>, CrossError> {
        match self {
            Self::Root(peers) => Self::recv(
                peers
                    .get(&leader)
                    .ok_or_else(|| err("cross leader missing"))?,
                bytes,
            ),
            Self::Leader(_) => Err(err("root recv from leader")),
        }
    }

    pub fn leader_send(&self, data: &[u8]) -> Result<(), CrossError> {
        match self {
            Self::Leader(peer) => Self::send(peer, data),
            Self::Root(_) => Err(err("leader send from root")),
        }
    }

    pub fn leader_recv(&self, bytes: usize) -> Result<Vec<u8>, CrossError> {
        match self {
            Self::Leader(peer) => Self::recv(peer, bytes),
            Self::Root(_) => Err(err("leader recv from root")),
        }
    }
}
