//! Fixed-membership host transport. Every participant owns its listener.
//! Slot numbers determine tensor layout, reduction order, and wire order.

use std::collections::HashSet;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use plait_core::DeviceId;
use plait_core::collective::{PeerAllReducePlan, peer_all_reduce_plan, peer_shard_elements};
pub use plait_core::group::PeerEndpoint as Endpoint;
use plait_core::group::{GroupFault, GroupVersion, peer_digest};
use sha2::Digest;

use crate::profile::Timer;
use crate::{DataType, ReduceOp, reduce_bytes, reduce_fp8_to_f32};

mod frame;
mod rankless;
pub(crate) mod requests;
pub use rankless::{RanklessListener, RanklessMember};

const COLLECTIVE_LANE: u8 = 0;
const DIRECT_LANE: u8 = 1;
const REPLICATED_EXCHANGE: u8 = 2;

pub struct Listener {
    socket: TcpListener,
    endpoint: Endpoint,
    version: Option<GroupVersion>,
}

impl Listener {
    pub fn bind(address: SocketAddr, ticket: [u8; 16]) -> io::Result<Self> {
        Self::bind_scoped(address, ticket, None)
    }

    pub fn bind_in_epoch(
        address: SocketAddr,
        ticket: [u8; 16],
        version: GroupVersion,
    ) -> io::Result<Self> {
        if version.epoch == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "group epoch must be positive",
            ));
        }
        Self::bind_scoped(address, ticket, Some(version))
    }

    fn bind_scoped(
        address: SocketAddr,
        ticket: [u8; 16],
        version: Option<GroupVersion>,
    ) -> io::Result<Self> {
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
        Ok(Self {
            socket,
            endpoint,
            version,
        })
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
        self.connect_bound(slot, members, None, timeout, timeout)
    }

    fn connect_bound(
        self,
        slot: usize,
        members: Vec<Endpoint>,
        binding: Option<GrantBinding>,
        timeout: Duration,
        bootstrap_timeout: Duration,
    ) -> io::Result<Arc<PeerGroup>> {
        if timeout.is_zero()
            || bootstrap_timeout.is_zero()
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
            .checked_add(bootstrap_timeout)
            .ok_or_else(|| io::Error::other("invalid peer bootstrap timeout"))?;
        let bootstrap = Bootstrap {
            slot,
            timeout,
            deadline,
            manifest: binding.as_ref().map_or_else(
                || {
                    peer_digest(self.version, members.iter().copied())
                        .finalize()
                        .into()
                },
                |binding| binding.handshake_digest,
            ),
        };
        let mut connections = (0..members.len()).map(|_| [None, None]).collect::<Vec<_>>();
        let (incoming, local_and_outgoing) = connections.split_at_mut(slot);
        thread::scope(|scope| {
            let inbound = scope.spawn(|| bootstrap.accept_lower(&self.socket, incoming));
            let dialed = bootstrap.dial_higher(&members[slot + 1..], &mut local_and_outgoing[1..]);
            let accepted = inbound.join().expect("peer bootstrap listener panicked");
            dialed.and(accepted)
        })?;
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
            timeout,
            order: Mutex::new(CollectiveOrder::default()),
            turn: Condvar::new(),
            requests: requests::Dispatcher::default(),
            failed: AtomicBool::new(false),
            binding,
            #[cfg(feature = "control")]
            grant_watch: std::sync::OnceLock::new(),
        }))
    }
}

struct GrantBinding {
    devices: Vec<DeviceId>,
    digest: [u8; 32],
    handshake_digest: [u8; 32],
    #[cfg(feature = "python")]
    local_grant: plait_core::group::WorkerGrant,
}

type WireLanes = [Option<TcpStream>; 2];

struct Bootstrap {
    slot: usize,
    timeout: Duration,
    deadline: Instant,
    manifest: [u8; 32],
}

impl Bootstrap {
    fn accept_lower(&self, listener: &TcpListener, incoming: &mut [WireLanes]) -> io::Result<()> {
        for _ in 0..self.slot * 2 {
            let mut stream = accept(listener, self.deadline)?;
            configure(&stream, remaining(self.deadline)?)?;
            let mut hello = [0; 45];
            stream.read_exact(&mut hello)?;
            let source = u32::from_le_bytes(hello[4..8].try_into().unwrap()) as usize;
            let lane = hello[44];
            if source >= self.slot
                || lane > DIRECT_LANE
                || incoming[source][lane as usize].is_some()
                || hello != greeting(&self.manifest, source, self.slot, lane)
            {
                return Err(io::Error::other("wrong peer group greeting"));
            }
            stream.write_all(&greeting(&self.manifest, self.slot, source, lane))?;
            configure(&stream, self.timeout)?;
            incoming[source][lane as usize] = Some(stream);
        }
        Ok(())
    }

    fn dial_higher(&self, members: &[Endpoint], outgoing: &mut [WireLanes]) -> io::Result<()> {
        for (offset, (member, pair)) in members.iter().zip(outgoing).enumerate() {
            let target = self.slot + 1 + offset;
            for lane in [COLLECTIVE_LANE, DIRECT_LANE] {
                let mut stream =
                    TcpStream::connect_timeout(&member.address, remaining(self.deadline)?)?;
                configure(&stream, remaining(self.deadline)?)?;
                stream.write_all(&greeting(&self.manifest, self.slot, target, lane))?;
                let mut reply = [0; 45];
                stream.read_exact(&mut reply)?;
                if reply != greeting(&self.manifest, target, self.slot, lane) {
                    return Err(io::Error::other("wrong peer group acknowledgement"));
                }
                configure(&stream, self.timeout)?;
                pair[lane as usize] = Some(stream);
            }
        }
        Ok(())
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

fn greeting(manifest: &[u8; 32], source: usize, target: usize, lane: u8) -> [u8; 45] {
    let mut bytes = [0; 45];
    bytes[..4].copy_from_slice(b"PLP2");
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
    pub(crate) fn reads_input(self, slot: usize) -> bool {
        !matches!(self, Self::Broadcast(_, source) if source != slot)
    }

    pub(crate) fn wire(self) -> (u8, DataType, ReduceOp, usize) {
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

    fn reduce(self, inputs: &[Vec<u8>]) -> Vec<u8> {
        let (_, dtype, op, _) = self.wire();
        if matches!(self, Collective::AllReduceFp8ToF32) {
            reduce_fp8_to_f32(inputs)
        } else {
            reduce_bytes(op, dtype, inputs)
        }
    }
}

fn use_replicated_exchange(collective: Collective, bytes: usize, world: usize) -> bool {
    matches!(
        collective,
        Collective::AllReduce(..) | Collective::AllReduceFp8ToF32
    ) && peer_all_reduce_plan(bytes as u64, world) == PeerAllReducePlan::Replicated
}

pub(crate) struct PeerGroup {
    slot: usize,
    links: Vec<Option<Link>>,
    timeout: Duration,
    order: Mutex<CollectiveOrder>,
    turn: Condvar,
    pub(crate) requests: requests::Dispatcher,
    failed: AtomicBool,
    binding: Option<GrantBinding>,
    #[cfg(feature = "control")]
    grant_watch: std::sync::OnceLock<tokio::sync::mpsc::Sender<Option<GroupFault>>>,
}

#[derive(Default)]
struct CollectiveOrder {
    submitted: u64,
    completed: u64,
}

impl PeerGroup {
    pub(crate) fn live(&self) -> io::Result<()> {
        if self.is_failed() {
            return Err(io::Error::other("peer communicator has failed"));
        }
        Ok(())
    }

    pub(crate) fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub(crate) fn world(&self) -> usize {
        self.links.len()
    }

    pub(crate) fn device_slot(&self, device: DeviceId) -> Option<usize> {
        self.binding
            .as_ref()?
            .devices
            .iter()
            .position(|&id| id == device)
    }

    pub fn cancel(&self) {
        self.close(None);
    }

    pub(crate) fn fail(&self, fault: GroupFault) {
        self.close(Some(fault));
    }

    fn close(&self, _fault: Option<GroupFault>) {
        let first = {
            let _order = self.order.lock().unwrap();
            let first = !self.failed.swap(true, Ordering::AcqRel);
            self.turn.notify_all();
            first
        };
        if first {
            for link in self.links.iter().flatten() {
                let _ = link.collective.shutdown(Shutdown::Both);
                let _ = link.direct.shutdown(Shutdown::Both);
            }
            #[cfg(feature = "control")]
            if let Some(watch) = self.grant_watch.get() {
                let _ = watch.try_send(_fault);
            }
        }
    }

    #[cfg(feature = "control")]
    pub(crate) fn watch_grant(&self, watch: tokio::sync::mpsc::Sender<Option<GroupFault>>) {
        self.grant_watch
            .set(watch)
            .expect("one watcher per communicator");
    }

    pub(crate) fn manifest_digest(&self) -> Option<[u8; 32]> {
        let binding = self.binding.as_ref()?;
        (!self.is_failed()).then_some(binding.digest)
    }

    #[cfg(feature = "python")]
    pub(crate) fn local_grant(&self) -> Option<plait_core::group::WorkerGrant> {
        let binding = self.binding.as_ref()?;
        (!self.is_failed()).then_some(binding.local_grant)
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
            frame::write(stream, header, outgoing)?;
            frame::read(stream, peer, header, incoming)
        } else {
            frame::read(stream, peer, header, incoming)?;
            frame::write(stream, header, outgoing)
        }
    }

    #[cfg(test)]
    pub fn execute(&self, collective: Collective, input: &[u8]) -> io::Result<Vec<u8>> {
        self.execute_timed(
            collective,
            input,
            &mut Timer::new(false),
            &mut Timer::new(false),
        )
    }

    pub(crate) fn execute_timed(
        &self,
        collective: Collective,
        input: &[u8],
        wire: &mut Timer,
        fold: &mut Timer,
    ) -> io::Result<Vec<u8>> {
        let sequence = self.reserve(Ok)?;
        self.execute_reserved(sequence, collective, input, wire, fold)
    }

    fn reserve<T>(&self, admit: impl FnOnce(u64) -> io::Result<T>) -> io::Result<T> {
        let mut order = self.order.lock().unwrap();
        self.live()?;
        let next = order
            .submitted
            .checked_add(1)
            .ok_or_else(|| io::Error::other("peer collective sequence exhausted"))?;
        let value = admit(order.submitted)?;
        order.submitted = next;
        Ok(value)
    }

    fn execute_reserved(
        &self,
        sequence: u64,
        collective: Collective,
        input: &[u8],
        wire: &mut Timer,
        fold: &mut Timer,
    ) -> io::Result<Vec<u8>> {
        let order = self
            .turn
            .wait_while(self.order.lock().unwrap(), |order| {
                order.completed != sequence && !self.is_failed()
            })
            .unwrap();
        self.live()?;
        drop(order);
        let result = self.run(collective, sequence, input, wire, fold);
        match &result {
            Ok(_) => {
                self.order.lock().unwrap().completed += 1;
                self.turn.notify_all();
            }
            Err(error) if error.kind() == io::ErrorKind::Unsupported => self.cancel(),
            Err(_) => self.fail(GroupFault::Transport),
        }
        result
    }

    fn run(
        &self,
        collective: Collective,
        sequence: u64,
        input: &[u8],
        wire: &mut Timer,
        fold: &mut Timer,
    ) -> io::Result<Vec<u8>> {
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
        if use_replicated_exchange(collective, bytes, world) {
            let header = collective.header(sequence, bytes, REPLICATED_EXCHANGE);
            let mut inputs = vec![vec![0; bytes]; world];
            inputs[self.slot].copy_from_slice(input);
            let peers = self.links.iter().zip(&mut inputs).enumerate().filter_map(
                |(peer, (link, received))| {
                    Some((
                        peer,
                        &link.as_ref()?.collective,
                        input,
                        received.as_mut_slice(),
                    ))
                },
            );
            wire.measure(|| frame::exchange_all(peers, &header, self.timeout))?;
            return Ok(fold.measure(|| collective.reduce(&inputs)));
        }
        match collective {
            Collective::AllReduce(..)
            | Collective::ReduceScatter(..)
            | Collective::AllReduceFp8ToF32 => {
                let ranges = shard_ranges(bytes, dtype.bytes(), world);
                let inputs = self.contributions(&header, input, &ranges, wire)?;
                let fp8 = matches!(collective, Collective::AllReduceFp8ToF32);
                let reduced = fold.measure(|| collective.reduce(&inputs));
                if matches!(collective, Collective::ReduceScatter(..)) {
                    return Ok(reduced);
                }
                let output_width = if fp8 { 4 } else { 1 };
                let total = bytes
                    .checked_mul(output_width)
                    .ok_or_else(|| io::Error::other("reduction size overflow"))?;
                let mut output = vec![0; total];
                let header = collective.header(sequence, bytes, 1);
                self.gather(
                    &header,
                    &reduced,
                    &mut output,
                    ranges.iter().map(|range| range.len() * output_width),
                    wire,
                )?;
                Ok(output)
            }
            Collective::AllGather(_) => {
                let total = bytes
                    .checked_mul(world)
                    .ok_or_else(|| io::Error::other("gather size overflow"))?;
                let mut output = vec![0; total];
                self.gather(
                    &header,
                    input,
                    &mut output,
                    std::iter::repeat_n(bytes, world),
                    wire,
                )?;
                Ok(output)
            }
            Collective::Broadcast(..) => {
                let mut output = input.to_vec();
                if self.slot == endpoint {
                    for peer in self.partners() {
                        wire.measure(|| self.exchange(peer, &header, input, &mut []))?;
                    }
                } else {
                    wire.measure(|| self.exchange(endpoint, &header, &[], &mut output))?;
                }
                Ok(output)
            }
            Collective::Reduce(..) => {
                if self.slot == endpoint {
                    let mut contributions = vec![vec![0; bytes]; world];
                    contributions[self.slot].copy_from_slice(input);
                    for peer in self.partners() {
                        wire.measure(|| {
                            self.exchange(peer, &header, &[], &mut contributions[peer])
                        })?;
                    }
                    Ok(fold.measure(|| reduce_bytes(op, dtype, &contributions)))
                } else {
                    wire.measure(|| self.exchange(endpoint, &header, input, &mut []))?;
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
        wire: &mut Timer,
    ) -> io::Result<Vec<Vec<u8>>> {
        let own = ranges[self.slot].clone();
        let mut contributions = vec![vec![0; own.len()]; self.links.len()];
        contributions[self.slot].copy_from_slice(&input[own]);
        let peers = self
            .links
            .iter()
            .zip(&mut contributions)
            .enumerate()
            .filter_map(|(peer, (link, received))| {
                Some((
                    peer,
                    &link.as_ref()?.collective,
                    &input[ranges[peer].clone()],
                    received.as_mut_slice(),
                ))
            });
        wire.measure(|| frame::exchange_all(peers, header, self.timeout))?;
        Ok(contributions)
    }

    fn gather(
        &self,
        header: &[u8; 24],
        input: &[u8],
        mut output: &mut [u8],
        shard_lengths: impl Iterator<Item = usize>,
        wire: &mut Timer,
    ) -> io::Result<()> {
        let peers =
            self.links
                .iter()
                .zip(shard_lengths)
                .enumerate()
                .filter_map(|(peer, (link, bytes))| {
                    let (shard, rest) = std::mem::take(&mut output).split_at_mut(bytes);
                    output = rest;
                    if let Some(link) = link {
                        Some((peer, &link.collective, input, shard))
                    } else {
                        shard.copy_from_slice(input);
                        None
                    }
                });
        wire.measure(|| frame::exchange_all(peers, header, self.timeout))
    }

    pub fn send(self: &Arc<Self>, peer: usize, dtype: DataType, input: Vec<u8>) -> io::Result<()> {
        self.live()?;
        let link = self.link(peer);
        let mut queue = link.direct_write.lock().unwrap();
        if queue.is_none() {
            let stream = link.direct.try_clone()?;
            let group = Arc::downgrade(self);
            let (send, recv) = sync_channel::<DirectMessage>(8);
            // An NCCL caller can enqueue a send before posting its receive.
            // A per-link writer keeps the group thread free to post that receive.
            thread::Builder::new()
                .name("plait-peer-send".to_owned())
                .spawn(move || {
                    while let Ok(DirectMessage { dtype, payload }) = recv.recv() {
                        if frame::write(&stream, &direct_header(dtype, payload.len()), &payload)
                            .is_err()
                        {
                            if let Some(group) = group.upgrade() {
                                group.fail(GroupFault::Transport);
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
            self.fail(GroupFault::Transport);
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
            self.fail(GroupFault::Transport);
        }
        result
    }
}

impl Drop for PeerGroup {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn direct_header(dtype: DataType, bytes: usize) -> [u8; 9] {
    let mut header = [0; 9];
    header[0] = dtype as u8;
    header[1..].copy_from_slice(&(bytes as u64).to_le_bytes());
    header
}

fn shard_ranges(bytes: usize, width: usize, world: usize) -> Vec<Range<usize>> {
    let elements = (bytes / width) as u64;
    (0..world)
        .scan(0, |start, slot| {
            let end = *start + peer_shard_elements(elements, world, slot) as usize * width;
            let range = *start..end;
            *start = end;
            Some(range)
        })
        .collect()
}

#[cfg(test)]
mod tests;
