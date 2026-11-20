//! Host-staged two-level TCP allreduce.
//!
//! Four local ranks reduce into a host leader. The two host leaders exchange
//! eight full-duplex channels, then publish the aggregate through shared
//! memory to their local workers. This is a TCP fallback, not GPUDirect.

use std::env;
use std::ffi::CString;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use plait_collective::ipc::PeerRegion;

type AnyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[repr(C, align(64))]
struct Header {
    start: AtomicU32,
    output_ready: AtomicU32,
}

struct SharedResult {
    fd: i32,
    map: *mut u8,
    size: usize,
}

unsafe impl Send for SharedResult {}

impl SharedResult {
    fn create(path: &str, bytes: usize) -> AnyResult<Self> {
        let c_path = CString::new(path)?;
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let size = std::mem::size_of::<Header>() + bytes;
        if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self {
            fd,
            map: map.cast(),
            size,
        })
    }

    fn header(&self) -> &Header {
        unsafe { &*self.map.cast::<Header>() }
    }

    fn result_ptr(&self) -> *mut f32 {
        unsafe { self.map.add(std::mem::size_of::<Header>()).cast::<f32>() }
    }
}

impl Drop for SharedResult {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.map.cast(), self.size);
            libc::close(self.fd);
        }
    }
}

fn listen(port: u16) -> AnyResult<TcpListener> {
    Ok(TcpListener::bind(("0.0.0.0", port))?)
}

fn connect(ip: &str, port: u16) -> AnyResult<TcpStream> {
    let mut last = String::new();
    for _ in 0..240 {
        match TcpStream::connect((ip, port)) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                return Ok(stream);
            }
            Err(error) => {
                last = error.to_string();
                thread::sleep(std::time::Duration::from_millis(250));
            }
        }
    }
    Err(std::io::Error::other(format!("connect {ip}:{port}: {last}")).into())
}

fn accept(listener: &TcpListener) -> AnyResult<TcpStream> {
    let (stream, _) = listener.accept()?;
    stream.set_nodelay(true)?;
    Ok(stream)
}

fn send_values(mut stream: TcpStream, values: &[f32]) -> AnyResult<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    stream.write_all(&bytes)?;
    stream.flush()?;
    Ok(())
}

fn receive_into(stream: &mut TcpStream, values: &mut [f32]) -> AnyResult<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), values.len() * 4)
    };
    stream.read_exact(bytes)?;
    Ok(())
}

fn exchange_poll_into(stream: TcpStream, send: &[f32], recv: &mut [f32]) -> AnyResult<()> {
    let fd = stream.as_raw_fd();
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL, 0) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let send_bytes =
        unsafe { std::slice::from_raw_parts(send.as_ptr().cast::<u8>(), send.len() * 4) };
    let recv_bytes =
        unsafe { std::slice::from_raw_parts_mut(recv.as_mut_ptr().cast::<u8>(), recv.len() * 4) };
    let mut sent = 0usize;
    let mut received = 0usize;
    while sent < send_bytes.len() || received < recv_bytes.len() {
        let mut event = libc::pollfd {
            fd,
            events: 0,
            revents: 0,
        };
        if sent < send_bytes.len() {
            event.events |= libc::POLLOUT;
        }
        if received < recv_bytes.len() {
            event.events |= libc::POLLIN;
        }
        if unsafe { libc::poll(&mut event, 1, -1) } < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error.into());
        }
        if event.revents & libc::POLLOUT != 0 && sent < send_bytes.len() {
            let n = unsafe {
                libc::send(
                    fd,
                    send_bytes[sent..].as_ptr().cast(),
                    send_bytes.len() - sent,
                    libc::MSG_NOSIGNAL,
                )
            };
            if n > 0 {
                sent += n as usize;
            }
        }
        if event.revents & libc::POLLIN != 0 && received < recv_bytes.len() {
            let n = unsafe {
                libc::recv(
                    fd,
                    recv_bytes[received..].as_mut_ptr().cast(),
                    recv_bytes.len() - received,
                    0,
                )
            };
            if n > 0 {
                received += n as usize;
            } else if n == 0 {
                return Err(std::io::Error::other("cross peer closed").into());
            }
        }
    }
    Ok(())
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 9 {
        return Err(std::io::Error::other(
            "usage: tcp-twolevel rank world gpu bytes iters local_ip remote_ip base_port",
        )
        .into());
    }
    let rank: usize = args[1].parse()?;
    let world: usize = args[2].parse()?;
    let gpu: usize = args[3].parse()?;
    let bytes: usize = args[4].parse()?;
    let iters: usize = args[5].parse()?;
    let local_ip = &args[6];
    let remote_ip = &args[7];
    let base: u16 = args[8].parse()?;
    let count = bytes / std::mem::size_of::<f32>();
    let is_local = rank < 4;

    let region = PeerRegion::alloc(gpu, bytes)?;
    let input: Vec<f32> = (0..count).map(|_| (rank + 1) as f32).collect();
    region.write_host(&input)?;

    let shm_path = format!("/dev/shm/plait-rust-hybrid-{base}");
    let shared = SharedResult::create(&shm_path, bytes)?;

    let mut workers: Vec<TcpStream> = Vec::new();
    let mut crosses: Vec<TcpStream> = Vec::new();
    if rank == 0 {
        let listener = listen(base)?;
        for _ in 0..3 {
            workers.push(accept(&listener)?);
        }
        drop(listener);
        let listener = listen(base + 8)?;
        for _ in 0..4 {
            crosses.push(accept(&listener)?);
        }
    } else if rank == 4 {
        let listener = listen(base + 4)?;
        for _ in 0..7 {
            workers.push(accept(&listener)?);
        }
        drop(listener);
        for _ in 0..4 {
            crosses.push(connect(local_ip, base + 8)?);
        }
    } else if is_local {
        workers.push(connect(local_ip, base)?);
    } else {
        workers.push(connect(remote_ip, base + 4)?);
    }

    // All shared-memory and socket setup is outside the benchmark interval.
    if rank == 0 || rank == 4 {
        for worker in &mut workers {
            worker.write_all(&[1])?;
        }
        shared.header().start.store(1, Ordering::Release);
    } else {
        let mut token = [0u8; 1];
        workers[0].read_exact(&mut token)?;
        while shared.header().start.load(Ordering::Acquire) == 0 {
            thread::yield_now();
        }
    }

    let start = std::time::Instant::now();
    let mut received = vec![0.0f32; count];
    let mut sum = vec![0.0f32; count];
    let mut worker_buffers: Vec<Vec<f32>> = if rank == 0 || rank == 4 {
        (0..workers.len()).map(|_| vec![0.0; count]).collect()
    } else {
        Vec::new()
    };
    let mut cross_values: Vec<Vec<f32>> = if rank == 0 || rank == 4 {
        (0..4).map(|_| vec![0.0; count]).collect()
    } else {
        Vec::new()
    };
    let mut generation = 0u32;
    for _ in 0..iters {
        generation += 1;
        if rank == 0 || rank == 4 {
            sum.fill(0.0);
            for (target, value) in sum.iter_mut().zip(&input) {
                *target += *value;
            }
            thread::scope(|scope| {
                for (worker, buffer) in workers.iter_mut().zip(worker_buffers.iter_mut()) {
                    scope.spawn(|| receive_into(worker, buffer));
                }
            });
            for buffer in &worker_buffers {
                for (target, value) in sum.iter_mut().zip(buffer) {
                    *target += *value;
                }
            }

            let part = count / 4;
            let mut cross_owned: Vec<Option<Vec<f32>>> = cross_values.drain(..).map(Some).collect();
            let (sender, receiver) = std::sync::mpsc::channel();
            thread::scope(|scope| {
                for (channel, source) in crosses.iter().enumerate() {
                    let begin = channel * part;
                    let end = if channel == 3 { count } else { begin + part };
                    let mut buffer = cross_owned[channel].take().expect("cross buffer");
                    let stream = source.try_clone().expect("cross clone");
                    let payload = &sum[begin..end];
                    let sender = sender.clone();
                    scope.spawn(move || {
                        let result = exchange_poll_into(stream, payload, &mut buffer[begin..end])
                            .map(|_| (channel, buffer));
                        let _ = sender.send(result);
                    });
                }
                drop(sender);
                for _ in 0..4 {
                    let message = receiver.recv().expect("cross channel closed");
                    let (channel, buffer) = message.expect("cross exchange failed");
                    cross_owned[channel] = Some(buffer);
                }
            });
            cross_values.extend(cross_owned.into_iter().map(|v| v.expect("cross buffer")));
            for value in &cross_values {
                for (target, item) in sum.iter_mut().zip(value) {
                    *target += *item;
                }
            }

            unsafe {
                std::ptr::copy_nonoverlapping(sum.as_ptr(), shared.result_ptr(), count);
            }
            shared
                .header()
                .output_ready
                .store(generation, Ordering::Release);
            received.copy_from_slice(&sum);
        } else {
            send_values(workers[0].try_clone()?, &input)?;
            while shared.header().output_ready.load(Ordering::Acquire) != generation {
                thread::yield_now();
            }
            unsafe {
                let result = std::slice::from_raw_parts(shared.result_ptr(), count);
                received.copy_from_slice(result);
            }
        }
        region.write_host(&received)?;
    }
    let seconds = start.elapsed().as_secs_f64() / iters as f64;
    let output = region.read_host(count)?;
    let expected = (world * (world + 1)) as f32 / 2.0;
    let errors = output
        .iter()
        .filter(|value| (**value - expected).abs() > 1e-2)
        .count();
    let algbw = bytes as f64 / seconds / 1e9;
    println!(
        "rank={rank} world={world} gpu={gpu} bytes={bytes} ms={:.3} algbw={:.3} errors={errors}",
        seconds * 1e3,
        algbw
    );
    Ok(())
}
