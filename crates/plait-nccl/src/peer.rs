//! Fixed-membership host transport. Every participant owns its listener and
//! one reduction shard. Slot numbers determine tensor layout and wire order.

use std::collections::HashSet;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::{DataType, ReduceOp, reduce_bytes, reduce_fp8_to_f32};

const COLLECTIVE_LANE: u8 = 0;
const DIRECT_LANE: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub address: SocketAddr,
    pub ticket: [u8; 16],
}

pub struct Listener {
    socket: TcpListener,
    endpoint: Endpoint,
}

impl Listener {
    pub fn bind(address: SocketAddr, ticket: [u8; 16]) -> io::Result<Self> {
        let socket = TcpListener::bind(address)?;
        let endpoint = Endpoint {
            address: socket.local_addr()?,
            ticket,
        };
        if endpoint.address.ip().is_unspecified() {
            return Err(io::Error::other(
                "the group endpoint needs an advertised IP",
            ));
        }
        socket.set_nonblocking(true)?;
        Ok(Self { socket, endpoint })
    }

    pub fn endpoint(&self) -> Endpoint {
        self.endpoint
    }

    pub(crate) fn connect(
        self,
        slot: usize,
        members: Vec<Endpoint>,
        timeout: Duration,
    ) -> io::Result<Arc<PeerGroup>> {
        if timeout.is_zero()
            || members.len() > i32::MAX as usize
            || members.get(slot) != Some(&self.endpoint)
            || members
                .iter()
                .map(|m| m.address)
                .collect::<HashSet<_>>()
                .len()
                != members.len()
            || members
                .iter()
                .map(|m| m.ticket)
                .collect::<HashSet<_>>()
                .len()
                != members.len()
            || members
                .iter()
                .any(|m| m.address.ip().is_unspecified() || m.address.port() == 0)
        {
            return Err(io::Error::other("invalid peer group manifest"));
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| io::Error::other("invalid peer bootstrap timeout"))?;
        let manifest = manifest_digest(&members);
        let mut connections = (0..members.len()).map(|_| [None, None]).collect::<Vec<_>>();
        for _ in 0..slot * 2 {
            let mut stream = accept(&self.socket, deadline)?;
            configure(&stream, remaining(deadline)?)?;
            let mut hello = [0; 45];
            stream.read_exact(&mut hello)?;
            let source = u32::from_le_bytes(hello[4..8].try_into().unwrap()) as usize;
            let lane = hello[44];
            if source >= slot
                || lane > DIRECT_LANE
                || connections[source][lane as usize].is_some()
                || hello != greeting(&manifest, source, slot, lane)
            {
                return Err(io::Error::other("wrong peer group greeting"));
            }
            stream.write_all(&greeting(&manifest, slot, source, lane))?;
            configure(&stream, timeout)?;
            connections[source][lane as usize] = Some(stream);
        }
        for (target, pair) in connections.iter_mut().enumerate().skip(slot + 1) {
            for lane in [COLLECTIVE_LANE, DIRECT_LANE] {
                let mut stream =
                    TcpStream::connect_timeout(&members[target].address, remaining(deadline)?)?;
                configure(&stream, remaining(deadline)?)?;
                stream.write_all(&greeting(&manifest, slot, target, lane))?;
                let mut reply = [0; 45];
                stream.read_exact(&mut reply)?;
                if reply != greeting(&manifest, target, slot, lane) {
                    return Err(io::Error::other("wrong peer group acknowledgement"));
                }
                configure(&stream, timeout)?;
                pair[lane as usize] = Some(stream);
            }
        }
        let mut links = Vec::with_capacity(members.len());
        for [collective, direct] in connections {
            let link = match (collective, direct) {
                (Some(collective), Some(direct)) => Some(Link {
                    collective,
                    direct_write: Mutex::new(None),
                    direct_read: Mutex::new(()),
                    direct,
                }),
                (None, None) => None,
                _ => return Err(io::Error::other("peer group is missing a wire lane")),
            };
            links.push(link);
        }
        Ok(Arc::new(PeerGroup {
            slot,
            links,
            sequence: Mutex::new(0),
            failed: AtomicBool::new(false),
        }))
    }
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    let time = deadline.saturating_duration_since(Instant::now());
    if time.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "peer group bootstrap",
        ));
    }
    Ok(time)
}

fn accept(listener: &TcpListener, deadline: Instant) -> io::Result<TcpStream> {
    loop {
        let time = remaining(deadline)?;
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(time.min(Duration::from_millis(2)));
            }
            Err(error) => return Err(error),
        }
    }
}

fn configure(stream: &TcpStream, timeout: Duration) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))
}

fn manifest_digest(members: &[Endpoint]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for member in members {
        match member.address {
            SocketAddr::V4(address) => {
                digest.update([4]);
                digest.update(address.ip().octets());
            }
            SocketAddr::V6(address) => {
                digest.update([6]);
                digest.update(address.ip().octets());
                digest.update(address.scope_id().to_le_bytes());
            }
        }
        digest.update(member.address.port().to_le_bytes());
        digest.update(member.ticket);
    }
    digest.finalize().into()
}

fn greeting(manifest: &[u8; 32], source: usize, target: usize, lane: u8) -> [u8; 45] {
    let mut bytes = [0; 45];
    bytes[..4].copy_from_slice(b"PLP1");
    bytes[4..8].copy_from_slice(&(source as u32).to_le_bytes());
    bytes[8..12].copy_from_slice(&(target as u32).to_le_bytes());
    bytes[12..44].copy_from_slice(manifest);
    bytes[44] = lane;
    bytes
}

struct Link {
    collective: TcpStream,
    direct: TcpStream,
    direct_write: Mutex<Option<SyncSender<DirectMessage>>>,
    direct_read: Mutex<()>,
}

struct DirectMessage {
    dtype: DataType,
    payload: Vec<u8>,
}

#[derive(Clone, Copy)]
pub(crate) enum Collective {
    AllReduce(DataType, ReduceOp),
    AllGather(DataType),
    ReduceScatter(DataType, ReduceOp),
    Broadcast(DataType, usize),
    Reduce(DataType, ReduceOp, usize),
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    AllReduceFp8ToF32,
}

impl Collective {
    fn wire(self) -> (u8, DataType, ReduceOp, usize) {
        match self {
            Self::AllReduce(dtype, op) => (1, dtype, op, 0),
            Self::AllGather(dtype) => (2, dtype, ReduceOp::Sum, 0),
            Self::ReduceScatter(dtype, op) => (3, dtype, op, 0),
            Self::Broadcast(dtype, source) => (4, dtype, ReduceOp::Sum, source),
            Self::Reduce(dtype, op, dest) => (5, dtype, op, dest),
            Self::AllReduceFp8ToF32 => (6, DataType::Float8e4m3, ReduceOp::Sum, 0),
        }
    }

    fn header(self, sequence: u64, bytes: usize, stage: u8) -> [u8; 24] {
        let (kind, dtype, op, endpoint) = self.wire();
        let mut header = [0; 24];
        header[..8].copy_from_slice(&sequence.to_le_bytes());
        header[8..16].copy_from_slice(&(bytes as u64).to_le_bytes());
        header[16..20].copy_from_slice(&(endpoint as u32).to_le_bytes());
        header[20..].copy_from_slice(&[kind, dtype as u8, op as u8, stage]);
        header
    }
}

pub(crate) struct PeerGroup {
    slot: usize,
    links: Vec<Option<Link>>,
    sequence: Mutex<u64>,
    failed: AtomicBool,
}

impl PeerGroup {
    fn live(&self) -> io::Result<()> {
        if self.failed.load(Ordering::Acquire) {
            return Err(io::Error::other("peer communicator has failed"));
        }
        Ok(())
    }

    pub fn fail(&self) {
        if !self.failed.swap(true, Ordering::AcqRel) {
            for link in self.links.iter().flatten() {
                let _ = link.collective.shutdown(Shutdown::Both);
                let _ = link.direct.shutdown(Shutdown::Both);
            }
        }
    }

    fn link(&self, slot: usize) -> &Link {
        self.links[slot].as_ref().expect("remote participant")
    }

    fn partners(&self) -> impl Iterator<Item = usize> + '_ {
        let world = self.links.len();
        let pivot = if world % 2 == 0 { world - 1 } else { world };
        (0..pivot).filter_map(move |round| {
            // Circle schedule. An odd group has one idle slot per round.
            let peer = if self.slot == pivot {
                round
            } else if self.slot == round {
                pivot
            } else {
                (2 * round + pivot - self.slot) % pivot
            };
            (peer < world).then_some(peer)
        })
    }

    fn exchange(
        &self,
        peer: usize,
        header: &[u8; 24],
        outgoing: &[u8],
        incoming: &mut [u8],
    ) -> io::Result<()> {
        let stream = &self.link(peer).collective;
        // Order complete frames so large writes cannot block each other.
        if self.slot < peer {
            write_frame(stream, header, outgoing)?;
            read_frame(stream, peer, header, incoming)
        } else {
            read_frame(stream, peer, header, incoming)?;
            write_frame(stream, header, outgoing)
        }
    }

    pub fn execute(&self, collective: Collective, input: &[u8]) -> io::Result<Vec<u8>> {
        let mut sequence = self.sequence.lock().unwrap();
        self.live()?;
        let result = self.run(collective, *sequence, input);
        if result.is_err() {
            self.fail();
        } else {
            *sequence += 1;
        }
        result
    }

    fn run(&self, collective: Collective, sequence: u64, input: &[u8]) -> io::Result<Vec<u8>> {
        let world = self.links.len();
        let bytes = input.len();
        let header = collective.header(sequence, bytes, 0);
        let (_, dtype, op, endpoint) = collective.wire();
        if matches!(
            collective,
            Collective::AllReduce(..) | Collective::ReduceScatter(..) | Collective::Reduce(..)
        ) && (dtype == DataType::Float64
            || (!matches!(
                dtype,
                DataType::Float16 | DataType::Bfloat16 | DataType::Float32 | DataType::Float8e4m3
            ) && op != ReduceOp::Sum))
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "unsupported host reduction",
            ));
        }
        match collective {
            Collective::AllReduce(..)
            | Collective::ReduceScatter(..)
            | Collective::AllReduceFp8ToF32 => {
                let ranges = shard_ranges(bytes, dtype.bytes(), world);
                let own = ranges[self.slot].clone();
                let inputs = self.contributions(&header, input, &ranges)?;
                let fp8 = matches!(collective, Collective::AllReduceFp8ToF32);
                let reduced = if fp8 {
                    reduce_fp8_to_f32(&inputs)
                } else {
                    reduce_bytes(op, dtype, &inputs)
                };
                if matches!(collective, Collective::ReduceScatter(..)) {
                    return Ok(reduced);
                }
                let output_width = if fp8 { 4 } else { 1 };
                let total = bytes
                    .checked_mul(output_width)
                    .ok_or_else(|| io::Error::other("reduction size overflow"))?;
                let mut output = vec![0; total];
                output[own.start * output_width..own.end * output_width].copy_from_slice(&reduced);
                let header = collective.header(sequence, bytes, 1);
                for peer in self.partners() {
                    let range = &ranges[peer];
                    self.exchange(
                        peer,
                        &header,
                        &reduced,
                        &mut output[range.start * output_width..range.end * output_width],
                    )?;
                }
                Ok(output)
            }
            Collective::AllGather(_) => {
                let total = bytes
                    .checked_mul(world)
                    .ok_or_else(|| io::Error::other("gather size overflow"))?;
                let mut output = vec![0; total];
                output[self.slot * bytes..(self.slot + 1) * bytes].copy_from_slice(input);
                for peer in self.partners() {
                    self.exchange(
                        peer,
                        &header,
                        input,
                        &mut output[peer * bytes..(peer + 1) * bytes],
                    )?;
                }
                Ok(output)
            }
            Collective::Broadcast(..) => {
                let mut output = input.to_vec();
                if self.slot == endpoint {
                    for peer in self.partners() {
                        self.exchange(peer, &header, input, &mut [])?;
                    }
                } else {
                    self.exchange(endpoint, &header, &[], &mut output)?;
                }
                Ok(output)
            }
            Collective::Reduce(..) => {
                if self.slot == endpoint {
                    let mut contributions = vec![vec![0; bytes]; world];
                    contributions[self.slot].copy_from_slice(input);
                    for peer in self.partners() {
                        self.exchange(peer, &header, &[], &mut contributions[peer])?;
                    }
                    Ok(reduce_bytes(op, dtype, &contributions))
                } else {
                    self.exchange(endpoint, &header, input, &mut [])?;
                    Ok(Vec::new())
                }
            }
        }
    }

    fn contributions(
        &self,
        header: &[u8; 24],
        input: &[u8],
        ranges: &[Range<usize>],
    ) -> io::Result<Vec<Vec<u8>>> {
        let own = ranges[self.slot].clone();
        let mut contributions = vec![vec![0; own.len()]; self.links.len()];
        contributions[self.slot].copy_from_slice(&input[own]);
        for peer in self.partners() {
            self.exchange(
                peer,
                header,
                &input[ranges[peer].clone()],
                &mut contributions[peer],
            )?;
        }
        Ok(contributions)
    }

    pub fn send(self: &Arc<Self>, peer: usize, dtype: DataType, input: Vec<u8>) -> io::Result<()> {
        self.live()?;
        let link = self.link(peer);
        let mut queue = link.direct_write.lock().unwrap();
        if queue.is_none() {
            let mut stream = link.direct.try_clone()?;
            let group = Arc::downgrade(self);
            let (send, recv) = sync_channel::<DirectMessage>(8);
            // An NCCL caller can enqueue a send before posting its receive.
            // A per-link writer keeps the group thread free to post that receive.
            thread::Builder::new()
                .name("plait-peer-send".to_owned())
                .spawn(move || {
                    while let Ok(DirectMessage { dtype, payload }) = recv.recv() {
                        if stream
                            .write_all(&direct_header(dtype, payload.len()))
                            .and_then(|()| stream.write_all(&payload))
                            .is_err()
                        {
                            if let Some(group) = group.upgrade() {
                                group.fail();
                            }
                            break;
                        }
                    }
                })?;
            *queue = Some(send);
        }
        let result = queue
            .as_ref()
            .unwrap()
            .try_send(DirectMessage {
                dtype,
                payload: input,
            })
            .map_err(|error| io::Error::other(format!("peer send queue: {error}")));
        if result.is_err() {
            self.fail();
        }
        result
    }

    pub fn recv(&self, peer: usize, dtype: DataType, bytes: usize) -> io::Result<Vec<u8>> {
        self.live()?;
        let link = self.link(peer);
        let _reading = link.direct_read.lock().unwrap();
        let mut stream = &link.direct;
        let result = (|| {
            let mut header = [0; 9];
            stream.read_exact(&mut header)?;
            if header != direct_header(dtype, bytes) {
                return Err(io::Error::other(
                    "peer sent a different point-to-point descriptor",
                ));
            }
            let mut payload = vec![0; bytes];
            stream.read_exact(&mut payload)?;
            Ok(payload)
        })();
        if result.is_err() {
            self.fail();
        }
        result
    }
}

impl Drop for PeerGroup {
    fn drop(&mut self) {
        self.fail();
    }
}

fn write_frame(mut stream: &TcpStream, header: &[u8; 24], data: &[u8]) -> io::Result<()> {
    stream.write_all(header)?;
    stream.write_all(data)
}

fn read_frame(
    mut stream: &TcpStream,
    peer: usize,
    expected: &[u8; 24],
    data: &mut [u8],
) -> io::Result<()> {
    let mut incoming = [0; 24];
    stream.read_exact(&mut incoming)?;
    if incoming != *expected {
        return Err(io::Error::other(format!(
            "peer {peer} submitted a different collective descriptor, expected {}, received {}",
            describe(expected),
            describe(&incoming)
        )));
    }
    stream.read_exact(data)
}

fn direct_header(dtype: DataType, bytes: usize) -> [u8; 9] {
    let mut header = [0; 9];
    header[0] = dtype as u8;
    header[1..].copy_from_slice(&(bytes as u64).to_le_bytes());
    header
}

fn describe(header: &[u8; 24]) -> String {
    let sequence = u64::from_le_bytes(header[..8].try_into().unwrap());
    let bytes = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let endpoint = u32::from_le_bytes(header[16..20].try_into().unwrap());
    let [kind, dtype, op, stage] = header[20..].try_into().unwrap();
    format!(
        "[seq {sequence}, bytes {bytes}, op {kind}/{op}, dtype {dtype}, endpoint {endpoint}, stage {stage}]"
    )
}

fn shard_ranges(bytes: usize, width: usize, world: usize) -> Vec<Range<usize>> {
    let elements = bytes / width;
    let minimum = elements / world;
    let extra = elements % world;
    (0..world)
        .scan(0, |start, slot| {
            let end = *start + (minimum + usize::from(slot < extra)) * width;
            let range = *start..end;
            *start = end;
            Some(range)
        })
        .collect()
}

#[cfg(test)]
mod tests;
