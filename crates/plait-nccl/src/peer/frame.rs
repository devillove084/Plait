use std::io::{self, IoSlice, Read, Write};
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use crate::profile::Timer;

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
) -> io::Result<()> {
    let deadline = phase_deadline(timeout)?;
    let mut transfers: Vec<_> = peers
        .map(|(peer, stream, outgoing, incoming)| Transfer {
            peer,
            stream,
            outgoing,
            incoming,
            progress: Progress::default(),
        })
        .collect();
    transfer_all(&mut transfers, header, deadline)
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
            self.sent +=
                socket_progress(stream.write_vectored(&remaining), io::ErrorKind::WriteZero)?;
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
            }
        }
        if self.header_bytes == header.len() && self.body_bytes < payload.len() {
            self.body_bytes += socket_progress(
                stream.read(&mut payload[self.body_bytes..]),
                io::ErrorKind::UnexpectedEof,
            )?;
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

pub(super) struct ScatterGather<'a> {
    pub peer: usize,
    pub stream: &'a TcpStream,
    pub scatter_header: [u8; 24],
    pub gather_header: [u8; 24],
    pub scatter_send: &'a [u8],
    pub scatter_recv: &'a mut [u8],
    pub gather_send: &'a mut [u8],
    pub gather_recv: &'a mut [u8],
}

impl ScatterGather<'_> {
    /// Keep each direction's frame order while letting the gather send start
    /// before the scatter receive finishes. Only reduced bytes are writable.
    pub(super) fn run(
        self,
        timeout: Duration,
        wire: &mut Timer,
        mut reduce_block: impl FnMut(usize, &[u8], &mut [u8]),
    ) -> io::Result<()> {
        let mut deadline = phase_deadline(timeout)?;
        let mut scatter = Progress::default();
        let mut gather = Progress::default();
        let mut reduced = 0;
        let mut scatter_finished = false;
        loop {
            wire.measure(|| {
                scatter.send(self.stream, &self.scatter_header, self.scatter_send)?;
                scatter.receive(
                    self.stream,
                    self.peer,
                    &self.scatter_header,
                    self.scatter_recv,
                )
            })?;
            let scatter_sending = scatter.sending(self.scatter_send.len());
            let scatter_receiving = scatter.receiving(self.scatter_recv.len());
            if !scatter_sending && !scatter_receiving && !scatter_finished {
                deadline = phase_deadline(timeout)?;
                scatter_finished = true;
            }

            let received = scatter.body_bytes;
            let ready = if scatter_receiving {
                received / PIPELINE_BLOCK_BYTES * PIPELINE_BLOCK_BYTES
            } else {
                received
            };
            while reduced < ready {
                let end = ready.min(reduced + PIPELINE_BLOCK_BYTES);
                reduce_block(
                    reduced,
                    &self.scatter_recv[reduced..end],
                    &mut self.gather_send[reduced..end],
                );
                reduced = end;
            }

            let events = wire.measure(|| {
                if !scatter_sending {
                    gather.send(
                        self.stream,
                        &self.gather_header,
                        &self.gather_send[..reduced],
                    )?;
                }
                if !scatter_receiving {
                    gather.receive(
                        self.stream,
                        self.peer,
                        &self.gather_header,
                        self.gather_recv,
                    )?;
                }
                Ok::<_, io::Error>(gather.events(reduced, self.gather_recv.len()))
            })?;
            let mut events = events;
            if scatter_sending {
                events |= libc::POLLOUT;
            }
            if scatter_receiving {
                events |= libc::POLLIN;
            }
            if events == 0 {
                return Ok(());
            }
            wire.measure(|| {
                wait_ready(
                    &mut [libc::pollfd {
                        fd: self.stream.as_raw_fd(),
                        events,
                        revents: 0,
                    }],
                    deadline,
                )
            })?;
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
