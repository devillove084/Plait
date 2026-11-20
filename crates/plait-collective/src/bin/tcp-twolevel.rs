//! Host-staged, two-level TCP allreduce.
//!
//! Local ranks send their partials to a host leader. The two host leaders
//! exchange four full-duplex streams, then each leader broadcasts the result.
//! This is a TCP fallback for hosts without RDMA, not a GPUDirect path.

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use plait_collective::ipc::PeerRegion;

type AnyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn listen(port: u16) -> AnyResult<TcpListener> {
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    Ok(listener)
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

fn send_all(mut stream: TcpStream, bytes: Vec<u8>) -> AnyResult<()> {
    stream.write_all(&bytes)?;
    stream.flush()?;
    Ok(())
}

fn receive(stream: &mut TcpStream, count: usize) -> AnyResult<Vec<f32>> {
    let mut bytes = vec![0u8; count * 4];
    stream.read_exact(&mut bytes)?;
    let mut values = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn send_values(stream: TcpStream, values: &[f32]) -> AnyResult<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    send_all(stream, bytes)
}

fn exchange(stream: TcpStream, send: Vec<f32>, count: usize) -> AnyResult<Vec<f32>> {
    let mut reader = stream.try_clone()?;
    let writer = thread::spawn(move || send_values(stream, &send));
    let received = receive(&mut reader, count)?;
    writer
        .join()
        .map_err(|_| std::io::Error::other("cross writer panicked"))??;
    Ok(received)
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 9 {
        return Err(
            "usage: tcp-twolevel rank world gpu bytes iters local_ip remote_ip base_port".into(),
        );
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

    // Setup skew must not be counted as collective time.
    let ready = [1u8; 1];
    if rank == 0 || rank == 4 {
        for worker in &mut workers {
            worker.write_all(&ready)?;
        }
    } else {
        let mut token = [0u8; 1];
        workers[0].read_exact(&mut token)?;
    }

    let start = std::time::Instant::now();
    let mut received = vec![0.0f32; count];
    let mut sum = vec![0.0f32; count];
    for _ in 0..iters {
        if rank == 0 || rank == 4 {
            sum.fill(0.0);
            for (index, value) in input.iter().enumerate() {
                sum[index] += value;
            }
            let mut buffers: Vec<Vec<f32>> = (0..workers.len()).map(|_| vec![0.0; count]).collect();
            thread::scope(|scope| {
                for (worker, buffer) in workers.iter_mut().zip(buffers.iter_mut()) {
                    scope.spawn(|| -> AnyResult<()> {
                        buffer.copy_from_slice(&receive(worker, count)?);
                        Ok(())
                    });
                }
            });
            for buffer in &buffers {
                for (index, value) in buffer.iter().enumerate() {
                    sum[index] += value;
                }
            }
            let part = count / 4;
            let mut cross_values = vec![vec![0.0f32; count]; 4];
            for channel in 0..4 {
                let begin = channel * part;
                let end = if channel == 3 { count } else { begin + part };
                let stream = crosses[channel].try_clone()?;
                let remote = exchange(stream, sum[begin..end].to_vec(), end - begin)?;
                cross_values[channel][begin..end].copy_from_slice(&remote);
            }
            for value in cross_values {
                for (index, item) in value.iter().enumerate() {
                    sum[index] += item;
                }
            }
            thread::scope(|scope| {
                for worker in workers.iter_mut() {
                    let payload = &sum;
                    scope.spawn(move || send_values(worker.try_clone()?, payload));
                }
            });
            received.copy_from_slice(&sum);
        } else {
            send_values(workers[0].try_clone()?, &input)?;
            received = receive(&mut workers[0], count)?;
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
    let _ = received;
    Ok(())
}
