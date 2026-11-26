//! Bounded storage for complete KV blocks in logical head and layer order.

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

const MAGIC: &[u8; 8] = b"PLAITKV2";
const RANGED_MAGIC: &[u8; 8] = b"PLAITKV3";
const HEADER: usize = 73;
const MAX_BLOCK_BYTES: usize = 64 << 20;
const MAX_HEADS: usize = 1024;
const MAX_PARTS: usize = 65536;
const MAX_RANGES: usize = 4096;
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
    head_bytes: usize,
    part_bytes: usize,
    ranges: Vec<Range<usize>>,
}

impl Request {
    async fn read(stream: &mut TcpStream) -> io::Result<Self> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid KV request");
        let mut header = [0; HEADER];
        stream.read_exact(&mut header).await?;
        let ranged = match &header[..8] {
            bytes if bytes == MAGIC => false,
            bytes if bytes == RANGED_MAGIC => true,
            _ => return Err(invalid()),
        };
        let field = |index: usize| {
            let start = 57 + 4 * index;
            u32::from_le_bytes(header[start..start + 4].try_into().unwrap()) as usize
        };
        let heads = field(0);
        let head_bytes = field(3);
        let part_bytes = if ranged { field(2) } else { head_bytes };
        let Some(block_bytes) = heads.checked_mul(head_bytes) else {
            return Err(invalid());
        };
        if !(1..=MAX_HEADS).contains(&heads)
            || head_bytes == 0
            || part_bytes == 0
            || !head_bytes.is_multiple_of(part_bytes)
            || block_bytes > MAX_BLOCK_BYTES
            || block_bytes / part_bytes > MAX_PARTS
        {
            return Err(invalid());
        }
        let range = |first: usize, count: usize| {
            let end = first.checked_add(count)?;
            (count != 0 && end <= block_bytes / part_bytes)
                .then(|| first * part_bytes..end * part_bytes)
        };
        let ranges = match (header[8], ranged) {
            (b'H', true) if field(1) == 0 => Vec::new(),
            (b'H', false) if field(1) == 0 && field(2) == 0 => Vec::new(),
            (b'G' | b'P', false) => vec![range(field(1), field(2)).ok_or_else(invalid)?],
            (b'G' | b'P', true) if (1..=MAX_RANGES).contains(&field(1)) => {
                let mut table = vec![0; field(1) * 8];
                stream.read_exact(&mut table).await?;
                let mut end = 0;
                let mut ranges = Vec::with_capacity(field(1));
                for entry in table.chunks_exact(8) {
                    let first = u32::from_le_bytes(entry[..4].try_into().unwrap()) as usize;
                    let count = u32::from_le_bytes(entry[4..].try_into().unwrap()) as usize;
                    let next = range(first, count).ok_or_else(invalid)?;
                    if next.start < end {
                        return Err(invalid());
                    }
                    end = next.end;
                    ranges.push(next);
                }
                ranges
            }
            _ => return Err(invalid()),
        };
        Ok(Self {
            verb: header[8],
            key: header[9..41].try_into().unwrap(),
            writer: header[41..57].try_into().unwrap(),
            heads,
            head_bytes,
            part_bytes,
            ranges,
        })
    }

    fn parts(&self) -> usize {
        self.heads * (self.head_bytes / self.part_bytes)
    }

    fn reserved(&self) -> usize {
        self.heads * self.head_bytes + self.parts() * size_of::<bool>()
    }

    fn part_indices(&self) -> impl Iterator<Item = usize> + Clone + '_ {
        self.ranges
            .iter()
            .flat_map(|range| range.start / self.part_bytes..range.end / self.part_bytes)
    }
}

struct Block {
    writer: Writer,
    expires: Instant,
    heads: usize,
    head_bytes: usize,
    part_bytes: usize,
    data: BlockData,
    present: usize,
    reserved: usize,
}

enum BlockData {
    Staging { bytes: Vec<u8>, received: Vec<bool> },
    Published(Bytes),
}

impl Block {
    fn matches(&self, request: &Request) -> bool {
        self.heads == request.heads && self.head_bytes == request.head_bytes
    }

    fn complete(&self) -> bool {
        matches!(self.data, BlockData::Published(_))
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
        if !block.matches(request) {
            return None;
        }
        let BlockData::Published(bytes) = &block.data else {
            return None;
        };
        Some(
            request
                .ranges
                .iter()
                .map(|range| bytes.slice(range.clone()))
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
                    heads: request.heads,
                    head_bytes: request.head_bytes,
                    part_bytes: request.part_bytes,
                    data: BlockData::Staging {
                        bytes: vec![0; request.heads * request.head_bytes],
                        received: vec![false; request.parts()],
                    },
                    present: 0,
                    reserved,
                },
            );
            self.used += reserved;
        }

        let block = self.blocks.get_mut(&request.key).expect("reserved block");
        if !block.matches(request) {
            return false;
        }
        if let BlockData::Published(bytes) = &block.data {
            let mut remaining = payload.as_ref();
            for range in &request.ranges {
                let Some(rest) = remaining.strip_prefix(&bytes[range.clone()]) else {
                    return false;
                };
                remaining = rest;
            }
            return remaining.is_empty();
        }
        if block.writer != request.writer || block.part_bytes != request.part_bytes {
            return false;
        }
        let BlockData::Staging { bytes, received } = &mut block.data else {
            unreachable!("published blocks were checked already");
        };
        let target = |index| {
            let start = index * request.part_bytes;
            start..start + request.part_bytes
        };
        let parts = request.part_indices();
        if parts
            .clone()
            .zip(payload.chunks_exact(request.part_bytes))
            .any(|(index, incoming)| received[index] && bytes[target(index)] != *incoming)
        {
            return false;
        }
        for (index, incoming) in parts.zip(payload.chunks_exact(request.part_bytes)) {
            if !received[index] {
                bytes[target(index)].copy_from_slice(incoming);
                received[index] = true;
                block.present += 1;
            }
        }
        if block.present == received.len() {
            block.data = BlockData::Published(Bytes::from(std::mem::take(bytes)));
        }
        true
    }
}

async fn exchange(mut stream: TcpStream, store: &Mutex<Store>, capacity: usize) -> io::Result<()> {
    let request = Request::read(&mut stream).await?;
    if request.verb == b'P' {
        if request.reserved() > capacity {
            return stream.write_u8(REJECTED).await;
        }
        let mut payload = vec![0; request.ranges.iter().map(Range::len).sum()];
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
            head_bytes: 1,
            part_bytes: 1,
            ranges: vec![first..first + count],
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

    #[test]
    fn pipeline_ranges_share_one_writer_and_retain_legacy_head_order() {
        let mut store = Store::new(1024);
        let now = Instant::now();
        let ranges = |spans, part_bytes, writer| Request {
            verb: b'P',
            key: [2; 32],
            writer: [writer; 16],
            heads: 2,
            head_bytes: 4,
            part_bytes,
            ranges: spans,
        };
        let first_stage = ranges(vec![0..2, 4..6], 2, 1);
        let next_stage = ranges(vec![2..4, 6..8], 2, 1);
        let old_head = ranges(vec![4..8], 4, 1);
        let other_partition = ranges(vec![0..4, 6..8], 2, 1);
        assert!(store.put(&next_stage, Bytes::from_static(&[3, 4, 7, 8]), now));
        assert!(store.get(&first_stage).is_none());
        assert!(store.get(&old_head).is_none());
        assert!(!store.put(&old_head, Bytes::from_static(&[5, 6, 7, 8]), now));
        let rival = ranges(vec![0..2, 4..6], 2, 2);
        assert!(!store.put(&rival, Bytes::from_static(&[1, 2, 5, 6]), now));
        assert!(store.put(&first_stage, Bytes::from_static(&[1, 2, 5, 6]), now));
        assert_eq!(store.get(&old_head).unwrap().concat(), [5, 6, 7, 8]);
        assert_eq!(
            store.get(&other_partition).unwrap().concat(),
            [1, 2, 3, 4, 7, 8]
        );
        assert!(store.put(&old_head, Bytes::from_static(&[5, 6, 7, 8]), now));
        assert!(!store.put(&old_head, Bytes::from_static(&[9, 6, 7, 8]), now));

        let mut store = Store::new(1024);
        let all_heads = ranges(vec![0..8], 4, 3);
        assert!(store.put(
            &all_heads,
            Bytes::from_static(&[1, 2, 3, 4, 5, 6, 7, 8]),
            now
        ));
        assert_eq!(store.get(&next_stage).unwrap().concat(), [3, 4, 7, 8]);
    }
}
