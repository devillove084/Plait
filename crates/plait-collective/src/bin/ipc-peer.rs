//! Two-process CUDA IPC allreduce.
//!
//! Rank 0 listens on TCP for descriptor exchange. Both ranks allocate a local
//! and receive region, export the receive handle, import the peer's handle,
//! copy into the peer's receive region with `cuMemcpyPeerAsync`, and run the
//! add kernel locally. Only the initial fill and final verification touch the
//! host.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use plait_collective::ipc::{AddKernel, IpcHandle, PeerRegion};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rank = 0usize;
    let mut device = 0usize;
    let mut bytes = 64usize;
    let mut addr = "127.0.0.1:18765".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rank" => rank = args.next().ok_or("--rank needs a value")?.parse()?,
            "--device" => device = args.next().ok_or("--device needs a value")?.parse()?,
            "--bytes" => bytes = args.next().ok_or("--bytes needs a value")?.parse()?,
            "--addr" => addr = args.next().ok_or("--addr needs a value")?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if rank > 1 || bytes < 4 || bytes % 4 != 0 {
        return Err("rank must be 0/1 and bytes must be a positive multiple of 4".into());
    }
    let count = bytes / std::mem::size_of::<f32>();

    let local = PeerRegion::alloc(device, bytes)?;
    let receive = PeerRegion::alloc(device, bytes)?;
    let local_values: Vec<f32> = (0..count)
        .map(|index| rank as f32 + 1.0 + (index % 7) as f32 * 0.5)
        .collect();
    local.write_host(&local_values)?;

    let mut stream = connect(rank, &addr)?;
    let receive_handle = receive.handle()?;
    stream.write_all(&receive_handle.0)?;
    let mut peer_bytes = [0u8; 64];
    stream.read_exact(&mut peer_bytes)?;
    let peer_handle = IpcHandle(peer_bytes);
    let peer_receive = PeerRegion::open(peer_handle, device, bytes)?;
    barrier(&mut stream)?;

    // Both processes write into the peer's receive region. No host staging
    // occurs on this data path.
    local.copy_to(&peer_receive, bytes)?;
    barrier(&mut stream)?;

    let kernel = AddKernel::current(device)?;
    kernel.add(&local, &receive, count)?;
    barrier(&mut stream)?;

    let result = local.read_host(count)?;
    let expected: Vec<f32> = (0..count)
        .map(|index| {
            let base = 1.0 + 2.0; // rank0 contributes 1, rank1 contributes 2
            base + (index % 7) as f32 * 0.5 * 2.0
        })
        .collect();
    let checksum = result.iter().sum::<f32>();
    if result
        .iter()
        .zip(&expected)
        .any(|(actual, expected)| (*actual - *expected).abs() > 1e-5)
    {
        return Err(format!("allreduce mismatch: {result:?} vs {expected:?}").into());
    }
    println!("ipc-peer: rank={rank} gpu={device} bytes={bytes} checksum={checksum:.3} ok");
    Ok(())
}

fn connect(rank: usize, addr: &str) -> Result<TcpStream, Box<dyn std::error::Error>> {
    if rank == 0 {
        let listener = TcpListener::bind(addr)?;
        let (stream, _) = listener.accept()?;
        Ok(stream)
    } else {
        let mut last = None;
        for _ in 0..50 {
            match TcpStream::connect(addr) {
                Ok(stream) => {
                    stream.set_nodelay(true)?;
                    return Ok(stream);
                }
                Err(error) => {
                    last = Some(error);
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
        Err(last.unwrap().into())
    }
}

fn barrier(stream: &mut TcpStream) -> Result<(), Box<dyn std::error::Error>> {
    stream.write_all(&[1])?;
    let mut byte = [0u8; 1];
    stream.read_exact(&mut byte)?;
    Ok(())
}
