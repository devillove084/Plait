//! Bounded TCP link sampling for hosted execution nodes.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use plait_core::NodeId;
use plait_core::serving::{HostedLinkSample, HostedProbeEndpoint};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpSocket, TcpStream};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinSet;

const MEASURE: &[u8; 8] = b"PLTPRB01";
const SNAPSHOT: &[u8; 8] = b"PLTSNP01";
const TIMEOUT: Duration = Duration::from_secs(3);
const SNAPSHOT_BYTES: usize = 64 << 10;
const PROBE_BYTES: usize = 1 << 20;
const CHUNK: usize = 32 << 10;

pub(crate) type Samples = Arc<RwLock<Vec<HostedLinkSample>>>;

pub(crate) async fn serve(
    listener: TcpListener,
    session: NodeId,
    samples: Samples,
) -> io::Result<()> {
    let capacity = Arc::new(Semaphore::new(8));
    loop {
        let (stream, _) = listener.accept().await?;
        let Ok(permit) = capacity.clone().try_acquire_owned() else {
            continue;
        };
        let samples = samples.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let _ = tokio::time::timeout(TIMEOUT, handle(stream, session, samples)).await;
        });
    }
}

async fn handle(mut stream: TcpStream, session: NodeId, samples: Samples) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let mut command = [0; 8];
    stream.read_exact(&mut command).await?;
    match &command {
        MEASURE => {
            stream.write_all(session.as_bytes()).await?;
            let mut buffer = [0; CHUNK];
            for _ in 0..PROBE_BYTES / CHUNK {
                stream.read_exact(&mut buffer).await?;
            }
            stream.write_u8(1).await?;
        }
        SNAPSHOT => {
            let report = serde_json::to_vec(&*samples.read().await)?;
            if report.len() > SNAPSHOT_BYTES {
                return Err(io::Error::other("hosted link snapshot is too large"));
            }
            stream.write_all(session.as_bytes()).await?;
            stream.write_u32(report.len() as u32).await?;
            stream.write_all(&report).await?;
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown link probe request",
            ));
        }
    }
    Ok(())
}

pub(crate) async fn measure(
    local_ip: IpAddr,
    target: HostedProbeEndpoint,
) -> io::Result<HostedLinkSample> {
    tokio::time::timeout(TIMEOUT, async {
        let socket = if local_ip.is_ipv4() {
            TcpSocket::new_v4()?
        } else {
            TcpSocket::new_v6()?
        };
        socket.bind(SocketAddr::new(local_ip, 0))?;
        let mut stream = socket.connect(target.address).await?;
        stream.set_nodelay(true)?;
        let start = Instant::now();
        stream.write_all(MEASURE).await?;
        check_session(&mut stream, target.session).await?;
        let rtt_ns = u64::try_from(start.elapsed().as_nanos())
            .map_err(io::Error::other)?
            .max(1);
        let start = Instant::now();
        let payload = [0; CHUNK];
        for _ in 0..PROBE_BYTES / CHUNK {
            stream.write_all(&payload).await?;
        }
        if stream.read_u8().await? != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid link probe acknowledgement",
            ));
        }
        let rate = PROBE_BYTES as u128 * 1_000_000_000 / start.elapsed().as_nanos().max(1);
        Ok(HostedLinkSample {
            target: target.host,
            target_session: target.session,
            sampled_at_ms: now_ms()?,
            bytes_per_second: u64::try_from(rate).unwrap_or(u64::MAX).max(1),
            rtt_ns,
        })
    })
    .await?
}

pub(crate) async fn collect(
    endpoints: Vec<HostedProbeEndpoint>,
    sources: Option<&BTreeSet<NodeId>>,
) -> BTreeMap<(NodeId, NodeId), HostedLinkSample> {
    let sessions: BTreeMap<_, _> = endpoints
        .iter()
        .map(|entry| (entry.host, entry.session))
        .collect();
    let mut requests = JoinSet::new();
    for endpoint in endpoints {
        if sources.is_some_and(|hosts| !hosts.contains(&endpoint.host)) {
            continue;
        }
        requests.spawn(async move {
            let report = tokio::time::timeout(TIMEOUT, snapshot(endpoint)).await??;
            Ok::<_, io::Error>((endpoint.host, report))
        });
    }
    let mut links = BTreeMap::new();
    while let Some(result) = requests.join_next().await {
        if let Ok(Ok((source, samples))) = result {
            for sample in samples {
                if sample.target != source
                    && sessions.get(&sample.target) == Some(&sample.target_session)
                {
                    links.insert((source, sample.target), sample);
                }
            }
        }
    }
    links
}

async fn snapshot(endpoint: HostedProbeEndpoint) -> io::Result<Vec<HostedLinkSample>> {
    let mut stream = TcpStream::connect(endpoint.address).await?;
    stream.write_all(SNAPSHOT).await?;
    check_session(&mut stream, endpoint.session).await?;
    let len = stream.read_u32().await? as usize;
    if len > SNAPSHOT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "hosted link snapshot is too large",
        ));
    }
    let mut data = vec![0; len];
    stream.read_exact(&mut data).await?;
    Ok(serde_json::from_slice(&data)?)
}

async fn check_session(stream: &mut TcpStream, expected: NodeId) -> io::Result<()> {
    let mut session = [0; NodeId::LEN];
    stream.read_exact(&mut session).await?;
    if &session != expected.as_bytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "hosted probe session changed",
        ));
    }
    Ok(())
}

pub(crate) fn now_ms() -> io::Result<u64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?;
    u64::try_from(elapsed.as_millis()).map_err(io::Error::other)
}
