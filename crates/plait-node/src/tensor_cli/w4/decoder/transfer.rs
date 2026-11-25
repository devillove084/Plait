use std::collections::BTreeMap;
use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use plait_core::NodeId;
pub(super) use plait_core::serving::DecoderKvSource as Source;
use plait_ops::decoder::{DecoderStackKv, KvSnapshot};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

const MAGIC: &[u8; 8] = b"PLAITKV1";
const HEADER_BYTES: usize = MAGIC.len() + 8;
const LAYER_HEADER_BYTES: usize = 32 + 4;
const DEPOSIT: [u8; NodeId::LEN] = [0; NodeId::LEN];

fn transfer_token() -> NodeId {
    loop {
        let token = NodeId::random();
        if token.as_bytes() != &DEPOSIT {
            return token;
        }
    }
}

#[derive(Default)]
pub(super) struct Publication(Option<(NodeId, JoinHandle<()>)>);

impl Publication {
    pub(super) fn active(&self) -> bool {
        self.0.as_ref().is_some_and(|(_, task)| !task.is_finished())
    }

    pub(super) async fn publish(
        &mut self,
        listen: SocketAddr,
        lifetime_ms: u64,
        cache: &DecoderStackKv,
    ) -> io::Result<Source> {
        self.publish_with_capacity(listen, lifetime_ms, cache, 0)
            .await
    }

    pub(super) async fn publish_with_capacity(
        &mut self,
        listen: SocketAddr,
        lifetime_ms: u64,
        cache: &DecoderStackKv,
        checkpoint_bytes: usize,
    ) -> io::Result<Source> {
        if listen.ip().is_unspecified() || !(1..=120_000).contains(&lifetime_ms) {
            return Err(invalid(
                "KV publication needs a concrete IP and a bounded lifetime",
            ));
        }
        if self.active() {
            return Err(io::Error::other("previous KV publication is still active"));
        }
        let payload = encode_suffix(cache, 0)?;
        let listener = TcpListener::bind(listen).await?;
        let source = Source {
            addr: listener.local_addr()?,
            transfer: transfer_token(),
            bytes: payload.len() as u64,
            sha256: Sha256::digest(&payload).into(),
        };
        let task = tokio::spawn(async move {
            let mut checkpoints = BTreeMap::<[u8; NodeId::LEN], Vec<u8>>::new();
            let mut remaining = checkpoint_bytes;
            let _ = tokio::time::timeout(Duration::from_millis(lifetime_ms), async {
                while let Ok((mut stream, _)) = listener.accept().await {
                    let _ = tokio::time::timeout(Duration::from_secs(15), async {
                        let mut token = [0; NodeId::LEN];
                        stream.read_exact(&mut token).await?;
                        if token == DEPOSIT && checkpoint_bytes != 0 {
                            let mut parent = [0; NodeId::LEN];
                            stream.read_exact(&mut parent).await?;
                            let size = stream.read_u64().await?;
                            if &parent != source.transfer.as_bytes()
                                || size == 0
                                || size > remaining as u64
                            {
                                return stream.write_all(&DEPOSIT).await;
                            }
                            let size = size as usize;
                            let mut delta = Vec::new();
                            delta.try_reserve_exact(size).map_err(io::Error::other)?;
                            delta.resize(size, 0);
                            stream.read_exact(&mut delta).await?;
                            let mut transfer = *transfer_token().as_bytes();
                            while &transfer == source.transfer.as_bytes()
                                || checkpoints.contains_key(&transfer)
                            {
                                transfer = *transfer_token().as_bytes();
                            }
                            checkpoints.insert(transfer, delta);
                            remaining -= size;
                            return stream.write_all(&transfer).await;
                        }
                        let Some(payload) = (if &token == source.transfer.as_bytes() {
                            Some(payload.as_slice())
                        } else {
                            checkpoints.get(&token).map(Vec::as_slice)
                        }) else {
                            return stream.write_u64(0).await;
                        };
                        stream.write_u64(payload.len() as u64).await?;
                        stream.write_all(payload).await
                    })
                    .await;
                }
            })
            .await;
        });
        self.0 = Some((source.transfer, task));
        Ok(source)
    }

    pub(super) fn release(&mut self, transfer: NodeId) -> bool {
        if self.0.as_ref().is_some_and(|(id, _)| *id == transfer) {
            let (_, task) = self.0.take().expect("matched publication");
            task.abort();
            return true;
        }
        false
    }
}

impl Drop for Publication {
    fn drop(&mut self) {
        if let Some((_, task)) = self.0.take() {
            task.abort();
        }
    }
}

pub(super) struct Snapshot {
    pub(super) tokens: u32,
    pub(super) layers: Vec<KvSnapshot>,
}

impl Snapshot {
    pub(super) fn append(&mut self, suffix: Self) -> io::Result<()> {
        if self.tokens == 0 || suffix.tokens == 0 || self.layers.len() != suffix.layers.len() {
            return Err(invalid("invalid KV checkpoint suffix"));
        }
        let tokens = self
            .tokens
            .checked_add(suffix.tokens)
            .ok_or_else(|| invalid("KV checkpoint token count overflows"))?;
        for (base, delta) in self.layers.iter().zip(&suffix.layers) {
            let width = base.keys_f16.len() / self.tokens as usize;
            if base.layer != delta.layer
                || width.checked_mul(suffix.tokens as usize) != Some(delta.keys_f16.len())
                || delta.keys_f16.len() != delta.values_f16.len()
            {
                return Err(invalid(
                    "KV checkpoint layer or width differs from its base",
                ));
            }
        }
        for (base, delta) in self.layers.iter_mut().zip(suffix.layers) {
            base.keys_f16.extend(delta.keys_f16);
            base.values_f16.extend(delta.values_f16);
        }
        self.tokens = tokens;
        Ok(())
    }
}

pub(super) async fn deposit(base: Source, payload: &[u8]) -> io::Result<Source> {
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut stream = TcpStream::connect(base.addr).await?;
        stream.set_nodelay(true)?;
        stream.write_all(&DEPOSIT).await?;
        stream.write_all(base.transfer.as_bytes()).await?;
        stream.write_u64(payload.len() as u64).await?;
        stream.write_all(payload).await?;
        let mut transfer = DEPOSIT;
        stream.read_exact(&mut transfer).await?;
        if transfer == DEPOSIT {
            return Err(io::Error::other("KV checkpoint storage is unavailable"));
        }
        Ok(Source {
            addr: base.addr,
            transfer: NodeId::from_bytes(transfer),
            bytes: payload.len() as u64,
            sha256: Sha256::digest(payload).into(),
        })
    })
    .await?
}

pub(super) async fn pull(
    source: Source,
    limit: usize,
    layers: u32,
    deadline: Duration,
) -> io::Result<Snapshot> {
    let size = usize::try_from(source.bytes).map_err(io::Error::other)?;
    if size < HEADER_BYTES
        || size > limit
        || source.addr.ip().is_unspecified()
        || source.addr.port() == 0
    {
        return Err(invalid(
            "KV source exceeds decoder capacity or has an invalid endpoint",
        ));
    }
    let data = tokio::time::timeout(deadline, async {
        let mut stream = TcpStream::connect(source.addr).await?;
        stream.set_nodelay(true)?;
        stream.write_all(source.transfer.as_bytes()).await?;
        let len = stream.read_u64().await?;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "KV transfer token is unavailable",
            ));
        }
        if len != source.bytes {
            return Err(invalid("KV transfer length does not match its descriptor"));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(size).map_err(io::Error::other)?;
        data.resize(size, 0);
        stream.read_exact(&mut data).await?;
        Ok::<_, io::Error>(data)
    })
    .await??;
    let digest: [u8; 32] = Sha256::digest(&data).into();
    if digest != source.sha256 {
        return Err(invalid("KV transfer digest does not match its descriptor"));
    }
    decode(&data, layers)
}

pub(super) fn snapshot_size(layers: u32, tokens: u32, width: u32) -> io::Result<usize> {
    if layers == 0 || tokens == 0 {
        return Err(invalid(
            "decoder needs at least one layer and one token of capacity",
        ));
    }
    let elements = tokens
        .checked_mul(width)
        .ok_or_else(|| invalid("too many KV elements per layer"))?;
    (elements as usize)
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(LAYER_HEADER_BYTES))
        .and_then(|bytes| bytes.checked_mul(layers as usize))
        .and_then(|bytes| bytes.checked_add(HEADER_BYTES))
        .ok_or_else(|| invalid("KV snapshot is too large"))
}

pub(super) fn encode_suffix(cache: &DecoderStackKv, start: usize) -> io::Result<Vec<u8>> {
    if start > cache.tokens() {
        return Err(invalid("KV checkpoint starts after the cache"));
    }
    let skip = |len: usize| {
        if start == 0 {
            0
        } else {
            start * (len / cache.tokens())
        }
    };
    let layers = u32::try_from(cache.snapshots().len()).map_err(io::Error::other)?;
    let tokens = u32::try_from(cache.tokens() - start).map_err(io::Error::other)?;
    let size = cache.snapshots().try_fold(HEADER_BYTES, |total, kv| {
        (kv.keys_f16.len() - skip(kv.keys_f16.len()))
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(LAYER_HEADER_BYTES))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| invalid("KV snapshot is too large"))
    })?;
    let mut data = Vec::new();
    data.try_reserve_exact(size).map_err(io::Error::other)?;
    data.extend_from_slice(MAGIC);
    data.extend_from_slice(&layers.to_le_bytes());
    data.extend_from_slice(&tokens.to_le_bytes());
    for kv in cache.snapshots() {
        let skip = skip(kv.keys_f16.len());
        let len = u32::try_from(kv.keys_f16.len() - skip).map_err(io::Error::other)?;
        data.extend_from_slice(&kv.layer);
        data.extend_from_slice(&len.to_le_bytes());
        for word in kv.keys_f16[skip..].iter().chain(&kv.values_f16[skip..]) {
            data.extend_from_slice(&word.to_le_bytes());
        }
    }
    Ok(data)
}

fn decode(mut data: &[u8], expected_layers: u32) -> io::Result<Snapshot> {
    if take(&mut data, MAGIC.len())? != MAGIC || read_u32(&mut data)? != expected_layers {
        return Err(invalid("KV format or decoder layer count does not match"));
    }
    let tokens = read_u32(&mut data)?;
    let mut layers = Vec::new();
    for _ in 0..expected_layers {
        let layer = take(&mut data, 32)?
            .try_into()
            .expect("layer digest length");
        let count = read_u32(&mut data)? as usize;
        let bytes = count
            .checked_mul(2)
            .ok_or_else(|| invalid("KV layer is too large"))?;
        let read_words = |data: &mut &[u8]| -> io::Result<Vec<u16>> {
            Ok(take(data, bytes)?
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("binary16 length")))
                .collect())
        };
        layers.push(KvSnapshot {
            layer,
            keys_f16: read_words(&mut data)?,
            values_f16: read_words(&mut data)?,
        });
    }
    if !data.is_empty() {
        return Err(invalid("KV snapshot has trailing data"));
    }
    Ok(Snapshot { tokens, layers })
}

fn read_u32(data: &mut &[u8]) -> io::Result<u32> {
    Ok(u32::from_le_bytes(
        take(data, 4)?.try_into().expect("word length"),
    ))
}

fn take<'a>(data: &mut &'a [u8], len: usize) -> io::Result<&'a [u8]> {
    let (head, tail) = data
        .split_at_checked(len)
        .ok_or_else(|| invalid("truncated KV snapshot"))?;
    *data = tail;
    Ok(head)
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
