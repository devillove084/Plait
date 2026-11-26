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

// The collective sequence lock owns this lane. Point-to-point traffic uses
// a different socket with its own blocking mode.
pub(super) fn exchange_duplex(
    stream: &TcpStream,
    peer: usize,
    header: &[u8; 24],
    outgoing: &[u8],
    incoming: &mut [u8],
    timeout: Duration,
) -> io::Result<()> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| io::Error::other("invalid peer exchange timeout"))?;
    stream.set_nonblocking(true)?;
    let result = transfer(stream, peer, header, outgoing, incoming, deadline);
    let restored = stream.set_nonblocking(false);
    result.and(restored)
}

fn transfer(
    mut stream: &TcpStream,
    peer: usize,
    header: &[u8; 24],
    outgoing: &[u8],
    incoming: &mut [u8],
    deadline: Instant,
) -> io::Result<()> {
    let mut slices = [IoSlice::new(header), IoSlice::new(outgoing)];
    let mut pending_write = slices.as_mut_slice();
    let mut remote_header = [0; 24];
    let mut header_bytes = 0;
    let mut body_bytes = 0;
    loop {
        if !pending_write.is_empty() {
            let written = socket_progress(
                stream.write_vectored(pending_write),
                io::ErrorKind::WriteZero,
            )?;
            IoSlice::advance_slices(&mut pending_write, written);
        }
        if header_bytes < header.len() {
            header_bytes += socket_progress(
                stream.read(&mut remote_header[header_bytes..]),
                io::ErrorKind::UnexpectedEof,
            )?;
            if header_bytes == header.len() {
                check_header(peer, header, &remote_header)?;
            }
        }
        if header_bytes == header.len() && body_bytes < incoming.len() {
            body_bytes += socket_progress(
                stream.read(&mut incoming[body_bytes..]),
                io::ErrorKind::UnexpectedEof,
            )?;
        }

        let mut events = 0;
        if !pending_write.is_empty() {
            events |= libc::POLLOUT;
        }
        if header_bytes < header.len() || body_bytes < incoming.len() {
            events |= libc::POLLIN;
        }
        if events == 0 {
            return Ok(());
        }
        wait_ready(stream, events, deadline)?;
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

fn wait_ready(stream: &TcpStream, events: libc::c_short, deadline: Instant) -> io::Result<()> {
    let mut request = libc::pollfd {
        fd: stream.as_raw_fd(),
        events,
        revents: 0,
    };
    loop {
        let time = deadline.saturating_duration_since(Instant::now());
        if time.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "peer collective exchange",
            ));
        }
        let millis = time.as_millis().clamp(1, i32::MAX as u128) as i32;
        // The socket outlives the call and request contains one valid pollfd.
        let ready = unsafe { libc::poll(&mut request, 1, millis) };
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
