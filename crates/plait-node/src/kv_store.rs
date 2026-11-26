//! Bounded storage for complete, head-addressable KV blocks.

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

const MAGIC: &[u8; 8] = b"PLAITKV2";
const HEADER: usize = 73;
const MAX_BLOCK_BYTES: usize = 64 << 20;
const MAX_HEADS: usize = 1024;
const PARTIAL_LIFETIME: Duration = Duration::from_secs(60);
const DEADLINE: Duration = Duration::from_secs(15);
const MISS: u8 = 0;
const OK: u8 = 1;
const REJECTED: u8 = 2;

type Key = [u8; 32];
type Writer = [u8; 16];

struct Request {
    verb: u8,
    key: Key,
    writer: Writer,
    heads: usize,
    first: usize,
    count: usize,
    head_bytes: usize,
}

impl Request {
    fn parse(header: [u8; HEADER]) -> io::Result<Self> {
        if &header[..MAGIC.len()] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid KV header",
            ));
        }
        let field = |index: usize| {
            let start = 57 + 4 * index;
            u32::from_le_bytes(header[start..start + 4].try_into().unwrap()) as usize
        };
        let request = Self {
            verb: header[8],
            key: header[9..41].try_into().unwrap(),
            writer: header[41..57].try_into().unwrap(),
            heads: field(0),
            first: field(1),
            count: field(2),
            head_bytes: field(3),
        };
        let range_valid = match request.verb {
            b'H' => request.first == 0 && request.count == 0,
            b'G' | b'P' => {
                request.count != 0
                    && request
                        .first
                        .checked_add(request.count)
                        .is_some_and(|end| end <= request.heads)
            }
            _ => false,
        };
        if !range_valid
            || !(1..=MAX_HEADS).contains(&request.heads)
            || request.head_bytes == 0
            || request
                .head_bytes
                .checked_mul(request.heads)
                .is_none_or(|bytes| bytes > MAX_BLOCK_BYTES)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid KV block dimensions or head range",
            ));
        }
        Ok(request)
    }

    fn reserved(&self) -> usize {
        self.heads * (self.head_bytes + size_of::<Option<Bytes>>())
    }
}

struct Block {
    writer: Writer,
    expires: Instant,
    head_bytes: usize,
    heads: Vec<Option<Bytes>>,
    present: usize,
    reserved: usize,
}

impl Block {
    fn matches(&self, request: &Request) -> bool {
        self.head_bytes == request.head_bytes && self.heads.len() == request.heads
    }

    fn complete(&self) -> bool {
        self.present == self.heads.len()
    }
}

struct Store {
    blocks: HashMap<Key, Block>,
    capacity: usize,
    used: usize,
}

impl Store {
    fn new(capacity: usize) -> Self {
        Self {
            blocks: HashMap::new(),
            capacity,
            used: 0,
        }
    }

    fn get(&self, request: &Request) -> Option<Vec<Bytes>> {
        let block = self.blocks.get(&request.key)?;
        if !block.matches(request) || !block.complete() {
            return None;
        }
        Some(
            block.heads[request.first..request.first + request.count]
                .iter()
                .map(|head| head.as_ref().expect("complete block").clone())
                .collect(),
        )
    }

    fn put(&mut self, request: &Request, payload: Bytes, now: Instant) -> bool {
        if self
            .blocks
            .get(&request.key)
            .is_some_and(|block| !block.complete() && now >= block.expires)
        {
            self.used -= self
                .blocks
                .remove(&request.key)
                .expect("expired block")
                .reserved;
        }
        if !self.blocks.contains_key(&request.key) {
            let reserved = request.reserved();
            if reserved > self.capacity - self.used {
                self.blocks.retain(|_, block| {
                    let keep = block.complete() || now < block.expires;
                    if !keep {
                        self.used -= block.reserved;
                    }
                    keep
                });
            }
            if reserved > self.capacity - self.used {
                return false;
            }
            self.blocks.insert(
                request.key,
                Block {
                    writer: request.writer,
                    expires: now + PARTIAL_LIFETIME,
                    head_bytes: request.head_bytes,
                    heads: vec![None; request.heads],
                    present: 0,
                    reserved,
                },
            );
            self.used += reserved;
        }

        let block = self.blocks.get_mut(&request.key).expect("reserved block");
        if !block.matches(request) || (!block.complete() && block.writer != request.writer) {
            return false;
        }
        let heads = &mut block.heads[request.first..request.first + request.count];
        if heads
            .iter()
            .zip(payload.chunks_exact(request.head_bytes))
            .any(|(existing, incoming)| {
                existing
                    .as_ref()
                    .is_some_and(|head| head.as_ref() != incoming)
            })
        {
            return false;
        }
        for (index, head) in heads.iter_mut().enumerate() {
            if head.is_none() {
                let start = index * request.head_bytes;
                *head = Some(Bytes::copy_from_slice(
                    &payload[start..start + request.head_bytes],
                ));
                block.present += 1;
            }
        }
        true
    }
}

async fn exchange(mut stream: TcpStream, store: &Mutex<Store>, capacity: usize) -> io::Result<()> {
    let mut header = [0; HEADER];
    stream.read_exact(&mut header).await?;
    let request = Request::parse(header)?;
    if request.verb == b'P' {
        if request.reserved() > capacity {
            return stream.write_u8(REJECTED).await;
        }
        let mut payload = vec![0; request.count * request.head_bytes];
        stream.read_exact(&mut payload).await?;
        let stored =
            store
                .lock()
                .expect("KV store lock")
                .put(&request, payload.into(), Instant::now());
        return stream.write_u8(if stored { OK } else { REJECTED }).await;
    }
    let heads = store.lock().expect("KV store lock").get(&request);
    let Some(heads) = heads else {
        return stream.write_u8(MISS).await;
    };
    stream.write_u8(OK).await?;
    for head in heads {
        stream.write_all(&head).await?;
    }
    Ok(())
}

pub(super) async fn run(
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut listen = "127.0.0.1:7952".parse::<SocketAddr>()?;
    let mut capacity = 256 << 20;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen" => listen = args.next().ok_or("--listen needs IP:port")?.parse()?,
            "--capacity-bytes" => {
                capacity = args
                    .next()
                    .ok_or("--capacity-bytes needs a size")?
                    .parse::<usize>()?;
            }
            _ => return Err(format!("unknown KV store argument {arg}").into()),
        }
    }
    if capacity == 0 {
        return Err("KV store capacity must be positive".into());
    }
    let listener = TcpListener::bind(listen).await?;
    let store = Arc::new(Mutex::new(Store::new(capacity)));
    let connections = Arc::new(Semaphore::new(32));
    println!("Plait KV store listening on {}", listener.local_addr()?);
    loop {
        let permit = connections.clone().acquire_owned().await?;
        let (stream, _) = listener.accept().await?;
        stream.set_nodelay(true)?;
        let store = store.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let _ = tokio::time::timeout(DEADLINE, exchange(stream, &store, capacity)).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(first: usize, count: usize, writer: u8) -> Request {
        Request {
            verb: b'P',
            key: [1; 32],
            writer: [writer; 16],
            heads: 4,
            first,
            count,
            head_bytes: 1,
        }
    }

    #[test]
    fn exposes_only_complete_blocks_and_reads_in_logical_head_order() {
        let mut store = Store::new(1024);
        let now = Instant::now();
        assert!(store.put(&request(2, 2, 1), Bytes::from_static(&[30, 40]), now));
        assert!(store.get(&request(2, 1, 1)).is_none());
        assert!(!store.put(&request(0, 2, 2), Bytes::from_static(&[10, 20]), now));
        assert!(store.put(&request(0, 2, 1), Bytes::from_static(&[10, 20]), now));
        let heads = store.get(&request(1, 3, 2)).unwrap();
        assert_eq!(heads.concat(), [20, 30, 40]);
        assert!(!store.put(&request(2, 1, 2), Bytes::from_static(&[90]), now));
    }

    #[test]
    fn an_incomplete_write_expires_without_leaking_its_heads_into_its_successor() {
        let mut store = Store::new(request(0, 4, 1).reserved());
        let now = Instant::now();
        assert!(store.put(&request(0, 2, 1), Bytes::from_static(&[1, 2]), now));
        let later = now + PARTIAL_LIFETIME;
        assert!(store.put(&request(2, 2, 2), Bytes::from_static(&[7, 8]), later));
        assert!(store.get(&request(2, 1, 2)).is_none());
        assert!(store.put(&request(0, 2, 2), Bytes::from_static(&[5, 6]), later));
        let heads = store.get(&request(0, 4, 2)).unwrap();
        assert_eq!(heads.concat(), [5, 6, 7, 8]);
    }
}
