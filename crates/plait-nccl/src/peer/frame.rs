use std::io::{self, IoSlice, Read, Write};
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use super::PeerGroup;
use crate::profile::{self, Timer};

pub(super) fn write(mut stream: &TcpStream, header: &[u8], data: &[u8]) -> io::Result<()> {
    let mut slices = [IoSlice::new(header), IoSlice::new(data)];
    let mut remaining = slices.as_mut_slice();
    while !remaining.is_empty() {
        match stream.write_vectored(remaining) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(written) => IoSlice::advance_slices(&mut remaining, written),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn read(
    mut stream: &TcpStream,
    peer: usize,
    expected: &[u8; 24],
    data: &mut [u8],
) -> io::Result<()> {
    let mut incoming = [0; 24];
    stream.read_exact(&mut incoming)?;
    check_header(peer, expected, &incoming)?;
    stream.read_exact(data)
}

// The collective sequence lock owns permanently nonblocking lanes.
// Point-to-point traffic uses separate blocking sockets.
pub(super) fn exchange_all<'a>(
    peers: impl Iterator<Item = (usize, &'a TcpStream, &'a [u8], &'a mut [u8])>,
    header: &[u8; 24],
    timeout: Duration,
    wire: &mut Timer,
    trace: Option<&ShardTrace>,
) -> io::Result<()> {
    let transfers = wire.measure(|| {
        let deadline = phase_deadline(timeout)?;
        let mut transfers: Vec<_> = peers
            .map(|(peer, stream, outgoing, incoming)| Transfer {
                peer,
                stream,
                outgoing,
                incoming,
                progress: Progress::new(trace.is_some()),
            })
            .collect();
        transfer_all(&mut transfers, header, deadline)?;
        Ok::<_, io::Error>(transfers)
    })?;
    if let Some(trace) = trace {
        for transfer in &transfers {
            trace.record(header, transfer.peer, &transfer.progress);
        }
    }
    Ok(())
}

pub(super) fn exchange_pair(
    peer: usize,
    stream: &TcpStream,
    header: &[u8; 24],
    outgoing: &[u8],
    incoming: &mut [u8],
    timeout: Duration,
) -> io::Result<()> {
    let deadline = phase_deadline(timeout)?;
    let mut transfer = Transfer {
        peer,
        stream,
        outgoing,
        incoming,
        progress: Progress::default(),
    };
    loop {
        let events = transfer.progress(header)?;
        if events == 0 {
            return Ok(());
        }
        wait_ready(
            &mut [libc::pollfd {
                fd: stream.as_raw_fd(),
                events,
                revents: 0,
            }],
            deadline,
        )?;
    }
}

struct Transfer<'a> {
    peer: usize,
    stream: &'a TcpStream,
    outgoing: &'a [u8],
    incoming: &'a mut [u8],
    progress: Progress,
}

#[derive(Default)]
struct Progress {
    sent: usize,
    remote_header: [u8; 24],
    header_bytes: usize,
    body_bytes: usize,
    marks: Option<Box<Milestones>>,
}

#[derive(Default)]
struct Milestones {
    tx_first: Option<Instant>,
    tx_last: Option<Instant>,
    rx_header: Option<Instant>,
    rx_first: Option<Instant>,
    rx_last: Option<Instant>,
}

pub(super) struct ShardTrace {
    file: &'static std::fs::File,
    group: u128,
    slot: usize,
    sequence: u64,
    start: Instant,
}

impl ShardTrace {
    pub(super) fn new(group: &PeerGroup, sequence: u64) -> Option<Self> {
        Some(Self {
            file: profile::shard_log()?,
            group: group.profile_group(),
            slot: group.slot(),
            sequence,
            start: Instant::now(),
        })
    }

    fn record(&self, header: &[u8; 24], peer: usize, progress: &Progress) {
        let Some(marks) = &progress.marks else { return };
        let kind = header[20];
        let stage = header[23];
        let ns = |mark: Option<Instant>| {
            mark.map_or(0, |mark| mark.duration_since(self.start).as_nanos())
        };
        let line = format!(
            "{}\t{:032x}\t{}\t{}\t{peer}\t{kind}\t{stage}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            std::process::id(),
            self.group,
            self.slot,
            self.sequence,
            progress.sent.saturating_sub(progress.remote_header.len()),
            progress.body_bytes,
            ns(marks.tx_first),
            ns(marks.tx_last),
            ns(marks.rx_header),
            ns(marks.rx_first),
            ns(marks.rx_last),
        );
        let mut file = self.file;
        let _ = file.write_all(line.as_bytes());
    }
}

impl Transfer<'_> {
    fn progress(&mut self, header: &[u8; 24]) -> io::Result<libc::c_short> {
        self.progress.send(self.stream, header, self.outgoing)?;
        self.progress
            .receive(self.stream, self.peer, header, self.incoming)?;
        Ok(self
            .progress
            .events(self.outgoing.len(), self.incoming.len()))
    }
}

impl Progress {
    fn new(traced: bool) -> Self {
        Self {
            marks: traced.then(Box::default),
            ..Self::default()
        }
    }

    fn sending(&self, available: usize) -> bool {
        self.sent < self.remote_header.len() + available
    }

    fn receiving(&self, expected: usize) -> bool {
        self.header_bytes < self.remote_header.len() || self.body_bytes < expected
    }

    fn send(
        &mut self,
        mut stream: &TcpStream,
        header: &[u8; 24],
        payload: &[u8],
    ) -> io::Result<()> {
        if self.sending(payload.len()) {
            let remaining = [
                IoSlice::new(&header[self.sent.min(header.len())..]),
                IoSlice::new(&payload[self.sent.saturating_sub(header.len())..]),
            ];
            let written =
                socket_progress(stream.write_vectored(&remaining), io::ErrorKind::WriteZero)?;
            self.sent += written;
            if written != 0
                && let Some(marks) = &mut self.marks
            {
                let now = Instant::now();
                marks.tx_last = Some(now);
                if self.sent > header.len() {
                    marks.tx_first.get_or_insert(now);
                }
            }
        }
        Ok(())
    }

    fn receive(
        &mut self,
        mut stream: &TcpStream,
        peer: usize,
        header: &[u8; 24],
        payload: &mut [u8],
    ) -> io::Result<()> {
        if self.header_bytes < header.len() {
            self.header_bytes += socket_progress(
                stream.read(&mut self.remote_header[self.header_bytes..]),
                io::ErrorKind::UnexpectedEof,
            )?;
            if self.header_bytes == header.len() {
                check_header(peer, header, &self.remote_header)?;
                if let Some(marks) = &mut self.marks {
                    let now = Instant::now();
                    marks.rx_header = Some(now);
                    marks.rx_last = Some(now);
                }
            }
        }
        if self.header_bytes == header.len() && self.body_bytes < payload.len() {
            let received = socket_progress(
                stream.read(&mut payload[self.body_bytes..]),
                io::ErrorKind::UnexpectedEof,
            )?;
            self.body_bytes += received;
            if received != 0
                && let Some(marks) = &mut self.marks
            {
                let now = Instant::now();
                marks.rx_first.get_or_insert(now);
                marks.rx_last = Some(now);
            }
        }
        Ok(())
    }

    fn events(&self, outgoing: usize, incoming: usize) -> libc::c_short {
        let mut events = 0;
        if self.sending(outgoing) {
            events |= libc::POLLOUT;
        }
        if self.receiving(incoming) {
            events |= libc::POLLIN;
        }
        events
    }
}

pub(super) const PIPELINE_BLOCK_BYTES: usize = 64 * 1024;

pub(super) struct ScatterGatherLane<'a> {
    scatter: Transfer<'a>,
    gather: Progress,
    gather_recv: &'a mut [u8],
}

impl<'a> ScatterGatherLane<'a> {
    pub(super) fn new(
        peer: usize,
        stream: &'a TcpStream,
        scatter_send: &'a [u8],
        scatter_recv: &'a mut [u8],
        gather_recv: &'a mut [u8],
    ) -> Self {
        Self {
            scatter: Transfer {
                peer,
                stream,
                outgoing: scatter_send,
                incoming: scatter_recv,
                progress: Progress::default(),
            },
            gather: Progress::default(),
            gather_recv,
        }
    }

    pub(super) fn contribution(&self) -> &[u8] {
        self.scatter.incoming
    }

    fn advance_gather(&mut self, header: &[u8; 24], reduced: &[u8]) -> io::Result<libc::pollfd> {
        let scatter = &self.scatter;
        if !scatter.progress.sending(scatter.outgoing.len()) {
            self.gather.send(scatter.stream, header, reduced)?;
        }
        if !scatter.progress.receiving(scatter.incoming.len()) {
            self.gather
                .receive(scatter.stream, scatter.peer, header, self.gather_recv)?;
        }
        let events = scatter
            .progress
            .events(scatter.outgoing.len(), scatter.incoming.len())
            | self.gather.events(reduced.len(), self.gather_recv.len());
        Ok(libc::pollfd {
            fd: scatter.stream.as_raw_fd(),
            events,
            revents: 0,
        })
    }
}

pub(super) struct ScatterGather<'a> {
    pub scatter_header: [u8; 24],
    pub gather_header: [u8; 24],
    pub gather_send: &'a mut [u8],
}

impl ScatterGather<'_> {
    /// Reduce the prefix received from every peer. Each direction keeps the
    /// original frame order, so a peer can still read the full scatter first.
    pub(super) fn run(
        self,
        lanes: &mut [ScatterGatherLane<'_>],
        timeout: Duration,
        wire: &mut Timer,
        trace: Option<&ShardTrace>,
        mut reduce_block: impl FnMut(usize, &[ScatterGatherLane<'_>], &mut [u8]),
    ) -> io::Result<()> {
        let mut deadline = phase_deadline(timeout)?;
        let mut waiting = Vec::with_capacity(lanes.len());
        let mut reduced = 0;
        let mut scatter_finished = false;
        if trace.is_some() {
            for lane in &mut *lanes {
                lane.scatter.progress.marks = Some(Box::default());
                lane.gather.marks = Some(Box::default());
            }
        }
        loop {
            let (scatter_events, contiguous) = wire.measure(|| {
                let mut events = 0;
                let mut contiguous = self.gather_send.len();
                for lane in &mut *lanes {
                    events |= lane.scatter.progress(&self.scatter_header)?;
                    contiguous = contiguous.min(lane.scatter.progress.body_bytes);
                }
                Ok::<_, io::Error>((events, contiguous))
            })?;
            if scatter_events == 0 && !scatter_finished {
                deadline = phase_deadline(timeout)?;
                scatter_finished = true;
            }

            let ready = if contiguous == self.gather_send.len() {
                contiguous
            } else {
                contiguous / PIPELINE_BLOCK_BYTES * PIPELINE_BLOCK_BYTES
            };
            while reduced < ready {
                let end = ready.min(reduced + PIPELINE_BLOCK_BYTES);
                reduce_block(reduced, lanes, &mut self.gather_send[reduced..end]);
                reduced = end;
            }

            waiting.clear();
            wire.measure(|| {
                for lane in &mut *lanes {
                    let poll =
                        lane.advance_gather(&self.gather_header, &self.gather_send[..reduced])?;
                    if poll.events != 0 {
                        waiting.push(poll);
                    }
                }
                Ok::<_, io::Error>(())
            })?;
            if waiting.is_empty() {
                if let Some(trace) = trace {
                    for lane in &*lanes {
                        let peer = lane.scatter.peer;
                        trace.record(&self.scatter_header, peer, &lane.scatter.progress);
                        trace.record(&self.gather_header, peer, &lane.gather);
                    }
                }
                return Ok(());
            }
            wire.measure(|| wait_ready(&mut waiting, deadline))?;
        }
    }
}

fn phase_deadline(timeout: Duration) -> io::Result<Instant> {
    Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| io::Error::other("invalid peer exchange timeout"))
}

fn transfer_all(
    transfers: &mut [Transfer<'_>],
    header: &[u8; 24],
    deadline: Instant,
) -> io::Result<()> {
    let mut waiting = Vec::with_capacity(transfers.len());
    loop {
        waiting.clear();
        for transfer in transfers.iter_mut() {
            let events = transfer.progress(header)?;
            if events != 0 {
                waiting.push(libc::pollfd {
                    fd: transfer.stream.as_raw_fd(),
                    events,
                    revents: 0,
                });
            }
        }
        if waiting.is_empty() {
            return Ok(());
        }
        wait_ready(&mut waiting, deadline)?;
    }
}

fn socket_progress(result: io::Result<usize>, on_closed: io::ErrorKind) -> io::Result<usize> {
    match result {
        Ok(0) => Err(io::Error::from(on_closed)),
        Ok(bytes) => Ok(bytes),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) =>
        {
            Ok(0)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[test]
fn traced_frame_separates_a_valid_header_from_the_first_payload() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let local = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (mut remote, _) = listener.accept().unwrap();
    let timeout = Some(Duration::from_secs(3));
    local.set_read_timeout(timeout).unwrap();
    remote.set_read_timeout(timeout).unwrap();
    let header = [0; 24];
    let mut progress = Progress::new(true);

    progress.send(&local, &header, &[]).unwrap();
    let mut observed = [0; 24];
    remote.read_exact(&mut observed).unwrap();
    assert_eq!(observed, header);
    let marks = progress.marks.as_ref().unwrap();
    assert!(marks.tx_first.is_none());
    let header_write = marks.tx_last.unwrap();
    while progress.sending(3) {
        progress.send(&local, &header, &[1, 2, 3]).unwrap();
    }
    let mut sent = [0; 3];
    remote.read_exact(&mut sent).unwrap();
    assert_eq!(sent, [1, 2, 3]);

    remote.write_all(&header).unwrap();
    while progress.receiving(0) {
        progress.receive(&local, 1, &header, &mut []).unwrap();
    }
    let marks = progress.marks.as_ref().unwrap();
    assert!(marks.rx_first.is_none());
    assert_eq!(marks.rx_header, marks.rx_last);
    let payload_release = Instant::now();
    remote.write_all(&[4, 5, 6]).unwrap();
    let mut received = [0; 3];
    while progress.receiving(received.len()) {
        progress.receive(&local, 1, &header, &mut received).unwrap();
    }
    let marks = progress.marks.as_ref().unwrap();
    assert_eq!(received, [4, 5, 6]);
    assert!(header_write <= marks.tx_first.unwrap());
    assert!(marks.tx_first <= marks.tx_last);
    assert!(marks.rx_header.unwrap() <= payload_release);
    assert!(payload_release <= marks.rx_first.unwrap());
    assert!(marks.rx_first <= marks.rx_last);
}

fn wait_ready(waiting: &mut [libc::pollfd], deadline: Instant) -> io::Result<()> {
    loop {
        let time = deadline.saturating_duration_since(Instant::now());
        if time.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "peer collective exchange",
            ));
        }
        let millis = time.as_millis().clamp(1, i32::MAX as u128) as i32;
        // Each socket outlives the poll and waiting contains initialized pollfds.
        let ready =
            unsafe { libc::poll(waiting.as_mut_ptr(), waiting.len() as libc::nfds_t, millis) };
        if ready > 0 {
            return Ok(());
        }
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}

fn check_header(peer: usize, expected: &[u8; 24], incoming: &[u8; 24]) -> io::Result<()> {
    if incoming == expected {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "peer {peer} submitted a different collective descriptor, expected {}, received {}",
        describe(expected),
        describe(incoming)
    )))
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
