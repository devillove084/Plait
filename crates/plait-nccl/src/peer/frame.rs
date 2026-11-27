use std::io::{self, IoSlice, Read, Write};
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

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

// The collective sequence lock owns these lanes. Point-to-point traffic uses
// separate sockets with their own blocking mode.
pub(super) fn exchange_all<'a>(
    peers: impl Iterator<Item = (usize, &'a TcpStream, &'a [u8], &'a mut [u8])>,
    header: &[u8; 24],
    timeout: Duration,
) -> io::Result<()> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| io::Error::other("invalid peer exchange timeout"))?;
    let mut transfers: Vec<_> = peers
        .map(|(peer, stream, outgoing, incoming)| Transfer {
            peer,
            stream,
            outgoing,
            incoming,
            sent: 0,
            remote_header: [0; 24],
            header_bytes: 0,
            body_bytes: 0,
        })
        .collect();
    let mut enabled = 0;
    let mut result = (|| {
        for transfer in &transfers {
            transfer.stream.set_nonblocking(true)?;
            enabled += 1;
        }
        transfer_all(&mut transfers, header, deadline)
    })();
    for transfer in &transfers[..enabled] {
        result = result.and(transfer.stream.set_nonblocking(false));
    }
    result
}

struct Transfer<'a> {
    peer: usize,
    stream: &'a TcpStream,
    outgoing: &'a [u8],
    incoming: &'a mut [u8],
    sent: usize,
    remote_header: [u8; 24],
    header_bytes: usize,
    body_bytes: usize,
}

impl Transfer<'_> {
    fn progress(&mut self, header: &[u8; 24]) -> io::Result<libc::c_short> {
        let mut stream = self.stream;
        let total = header.len() + self.outgoing.len();
        if self.sent < total {
            let remaining = [
                IoSlice::new(&header[self.sent.min(header.len())..]),
                IoSlice::new(&self.outgoing[self.sent.saturating_sub(header.len())..]),
            ];
            self.sent +=
                socket_progress(stream.write_vectored(&remaining), io::ErrorKind::WriteZero)?;
        }
        if self.header_bytes < header.len() {
            self.header_bytes += socket_progress(
                stream.read(&mut self.remote_header[self.header_bytes..]),
                io::ErrorKind::UnexpectedEof,
            )?;
            if self.header_bytes == header.len() {
                check_header(self.peer, header, &self.remote_header)?;
            }
        }
        if self.header_bytes == header.len() && self.body_bytes < self.incoming.len() {
            self.body_bytes += socket_progress(
                stream.read(&mut self.incoming[self.body_bytes..]),
                io::ErrorKind::UnexpectedEof,
            )?;
        }

        let mut events = 0;
        if self.sent < total {
            events |= libc::POLLOUT;
        }
        if self.header_bytes < header.len() || self.body_bytes < self.incoming.len() {
            events |= libc::POLLIN;
        }
        Ok(events)
    }
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
