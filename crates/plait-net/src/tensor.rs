//! Pull immutable tensor contributions on a socket separate from control traffic.

use std::io::{self, ErrorKind};
use std::net::SocketAddr;
use std::time::Duration;

use bincode::Options;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use plait_core::tensor::KPartial;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use super::{MAX_FRAME, codec, send_frame};

pub const MAX_PARTIAL_BYTES: usize = MAX_FRAME;

pub fn partial_size(partial: KPartial) -> io::Result<usize> {
    partial
        .payload_bytes()
        .filter(|size| *size <= MAX_PARTIAL_BYTES)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "invalid partial tensor size"))
}

pub struct PartialSource {
    listener: PartialListener,
    partial: KPartial,
    bytes: Bytes,
}

impl PartialSource {
    pub async fn bind(addr: SocketAddr, partial: KPartial, bytes: Vec<u8>) -> io::Result<Self> {
        if partial_size(partial)? != bytes.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "partial tensor size does not match its descriptor",
            ));
        }
        Ok(Self {
            listener: PartialListener::bind(addr).await?,
            partial,
            bytes: Bytes::from(bytes),
        })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Serve one pull request. The source remains available for subsequent
    /// pulls, including a retry or a request from a different consumer.
    pub async fn serve_next(&self, deadline: Duration) -> io::Result<()> {
        self.listener
            .serve_next(deadline, |request| {
                (request == self.partial).then(|| self.bytes.clone())
            })
            .await
    }
}

/// An endpoint that serves contributions supplied by a local producer.
pub struct PartialListener(TcpListener);
pub struct PartialConnection(TcpStream);

impl PartialListener {
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        Ok(Self(TcpListener::bind(addr).await?))
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.0.local_addr()
    }

    /// Accept separately from service so a slow connection need not hold up another.
    pub async fn accept(&self) -> io::Result<PartialConnection> {
        let (stream, _) = self.0.accept().await?;
        Ok(PartialConnection(stream))
    }

    pub async fn serve_next(
        &self,
        deadline: Duration,
        lookup: impl FnOnce(KPartial) -> Option<Bytes>,
    ) -> io::Result<()> {
        self.accept().await?.serve(deadline, lookup).await
    }
}

impl PartialConnection {
    /// Look up the full descriptor after reading it from the consumer.
    pub async fn serve(
        self,
        deadline: Duration,
        lookup: impl FnOnce(KPartial) -> Option<Bytes>,
    ) -> io::Result<()> {
        let mut framed = Framed::new(self.0, request_codec());
        let exchange = async {
            let request = framed.next().await.transpose()?.ok_or_else(closed)?;
            let payload = bincode::options()
                .with_fixint_encoding()
                .reject_trailing_bytes()
                .deserialize::<KPartial>(&request)
                .ok()
                .and_then(|request| {
                    let size = partial_size(request).ok()?;
                    lookup(request).filter(|bytes| bytes.len() == size)
                })
                .unwrap_or_default();
            // The small request limit should not also limit the reply.
            framed = Framed::new(framed.into_inner(), codec());
            framed.send(payload).await
        };
        tokio::time::timeout(deadline, exchange).await?
    }
}

pub async fn fetch_partial(
    addr: SocketAddr,
    partial: KPartial,
    deadline: Duration,
) -> io::Result<Bytes> {
    let size = partial_size(partial)?;
    let exchange = async {
        let stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        let mut framed = Framed::new(stream, request_codec());
        send_frame(&mut framed, &partial).await?;
        let reply_codec = LengthDelimitedCodec::builder()
            .max_frame_length(size)
            .new_codec();
        let mut framed = Framed::new(framed.into_inner(), reply_codec);
        let payload = framed.next().await.transpose()?.ok_or_else(closed)?;
        if payload.is_empty() {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                "source does not hold the requested contribution",
            ));
        }
        if payload.len() != size {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "partial tensor size does not match its descriptor",
            ));
        }
        Ok(payload.freeze())
    };
    tokio::time::timeout(deadline, exchange).await?
}

fn request_codec() -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .max_frame_length(256)
        .new_codec()
}

fn closed() -> io::Error {
    io::Error::new(
        ErrorKind::UnexpectedEof,
        "partial transfer ended before a frame",
    )
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{GroupId, GroupVersion, WorkerGrant};
    use plait_core::{DeviceId, NodeId, OpId, ShardId};

    use super::*;

    #[tokio::test]
    async fn pulls_are_bound_to_one_contribution_and_can_be_retried() {
        let owner = NodeId::derive("publisher");
        let partial = KPartial {
            grant: WorkerGrant {
                version: GroupVersion {
                    group: GroupId(owner),
                    epoch: 2,
                },
                device: DeviceId::new(owner, 1),
                incarnation: 3,
                fence: 4,
            },
            op: OpId::new(owner, 5),
            shard: ShardId(7),
            k_start: 0,
            k_end: 3,
            output_shape: [1, 1],
        };
        let output = 1.5f32.to_le_bytes();
        let source = PartialSource::bind("127.0.0.1:0".parse().unwrap(), partial, output.to_vec())
            .await
            .unwrap();
        let addr = source.local_addr().unwrap();
        let timeout = Duration::from_secs(2);
        let server = tokio::spawn(async move {
            for _ in 0..4 {
                source.serve_next(timeout).await.unwrap();
            }
        });
        let mut stale = partial;
        stale.grant.version.epoch -= 1;
        assert_eq!(
            fetch_partial(addr, stale, timeout)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::NotFound
        );
        let mut stale = partial;
        stale.grant.fence -= 1;
        assert_eq!(
            fetch_partial(addr, stale, timeout)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::NotFound
        );
        for _ in 0..2 {
            assert_eq!(
                fetch_partial(addr, partial, timeout)
                    .await
                    .unwrap()
                    .as_ref(),
                &output
            );
        }
        server.await.unwrap();
    }
}
