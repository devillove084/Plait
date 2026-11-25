use std::io;

use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const FRAME_LIMIT: usize = 64 * 1024 * 1024;

pub(super) async fn read_frame<T: DeserializeOwned>(stream: &mut TcpStream) -> io::Result<T> {
    Ok(serde_json::from_slice(&read_payload(stream).await?)?)
}

pub(super) async fn read_payload(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let len = stream.read_u32().await? as usize;
    if len > FRAME_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tensor RPC frame exceeds limit",
        ));
    }
    let mut data = vec![0; len];
    stream.read_exact(&mut data).await?;
    Ok(data)
}

pub(super) async fn write_frame(stream: &mut TcpStream, value: &impl Serialize) -> io::Result<()> {
    write_payload(stream, &serde_json::to_vec(value)?).await
}

pub(super) async fn write_payload(stream: &mut TcpStream, data: &[u8]) -> io::Result<()> {
    if data.len() > FRAME_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "tensor RPC frame exceeds limit",
        ));
    }
    stream.write_u32(data.len() as u32).await?;
    stream.write_all(data).await
}
