//! Disjoint shares of a worker's network budget across KV stores.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::net::{IpAddr, SocketAddr};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task::JoinSet;

use super::checkpoints::{Limit, MAX_PEERS, PeerLimit};
use super::{DEADLINE, read_stats, unicast_peer};

const MAX_STORES: usize = 128;
const MAX_PLAN_BYTES: u64 = 8 << 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Peer {
    address: IpAddr,
    limit: PeerLimit,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Store {
    endpoint: SocketAddr,
    peers: Vec<Peer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    peers: Vec<Peer>,
    stores: Vec<Store>,
}

#[derive(Serialize)]
struct Allocation {
    peers: BTreeMap<IpAddr, PeerLimit>,
    stores: BTreeMap<SocketAddr, BTreeMap<IpAddr, PeerLimit>>,
}

pub(super) struct PeerPlan {
    allocation: Allocation,
    id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub(super) struct Membership {
    id: String,
    endpoint: SocketAddr,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn canonical_endpoint(endpoint: SocketAddr) -> SocketAddr {
    match endpoint {
        SocketAddr::V6(value) if value.ip().to_ipv4_mapped().is_some() => {
            SocketAddr::new(endpoint.ip().to_canonical(), endpoint.port())
        }
        _ => endpoint,
    }
}

fn peer_map(entries: Vec<Peer>) -> io::Result<BTreeMap<IpAddr, PeerLimit>> {
    if entries.len() > MAX_PEERS {
        return Err(invalid("invalid number of peers in checkpoint plan"));
    }
    let mut peers = BTreeMap::new();
    for Peer { address, limit } in entries {
        let ip = address.to_canonical();
        if !unicast_peer(ip) || !limit.read.valid_share() || !limit.write.valid_share() {
            return Err(invalid(format!("invalid checkpoint budget for peer {ip}")));
        }
        if peers.insert(ip, limit).is_some() {
            return Err(invalid(format!("duplicate checkpoint peer {ip}")));
        }
    }
    Ok(peers)
}

fn debit(capacity: &mut Limit, quota: Limit) -> Option<()> {
    let bytes_per_second = capacity
        .bytes_per_second
        .checked_sub(quota.bytes_per_second)?;
    let burst_bytes = capacity.burst_bytes.checked_sub(quota.burst_bytes)?;
    *capacity = Limit {
        bytes_per_second,
        burst_bytes,
    };
    Some(())
}

impl PeerPlan {
    pub(super) fn load(path: &str) -> io::Result<Self> {
        let mut body = Vec::new();
        File::open(path)?
            .take(MAX_PLAN_BYTES + 1)
            .read_to_end(&mut body)?;
        if body.len() as u64 > MAX_PLAN_BYTES {
            return Err(invalid("checkpoint budget plan is too large"));
        }
        Self::validate(serde_json::from_slice(&body)?)
    }

    fn validate(document: Document) -> io::Result<Self> {
        let peers = peer_map(document.peers)?;
        if peers.is_empty() {
            return Err(invalid("checkpoint plan needs at least one peer budget"));
        }
        if !(1..=MAX_STORES).contains(&document.stores.len()) {
            return Err(invalid("invalid number of stores in checkpoint plan"));
        }
        let mut remaining = peers.clone();
        let mut stores = BTreeMap::new();
        for Store { endpoint, peers } in document.stores {
            let endpoint = canonical_endpoint(endpoint);
            if endpoint.port() == 0 || !unicast_peer(endpoint.ip()) {
                return Err(invalid(format!(
                    "invalid checkpoint store endpoint {endpoint}"
                )));
            }
            let peers = peer_map(peers)?;
            if stores.contains_key(&endpoint) {
                return Err(invalid(format!("duplicate checkpoint endpoint {endpoint}")));
            }
            for (&ip, limit) in &peers {
                let capacity = remaining.get_mut(&ip).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint store {endpoint} names undeclared peer {ip}"
                    ))
                })?;
                debit(&mut capacity.read, limit.read).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint read shares exceed the rate or burst for {ip}"
                    ))
                })?;
                debit(&mut capacity.write, limit.write).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint write shares exceed the rate or burst for {ip}"
                    ))
                })?;
            }
            stores.insert(endpoint, peers);
        }
        let allocation = Allocation { peers, stores };
        let mut hash = Sha256::new();
        hash.update(b"plait-checkpoint-peer-plan-v1");
        hash.update(serde_json::to_vec(&allocation)?);
        let id = format!("{:x}", hash.finalize());
        Ok(Self { allocation, id })
    }

    pub(super) fn share_for(
        &self,
        endpoint: SocketAddr,
    ) -> io::Result<(Membership, BTreeMap<IpAddr, PeerLimit>)> {
        let endpoint = canonical_endpoint(endpoint);
        let peers = self.allocation.stores.get(&endpoint).ok_or_else(|| {
            invalid(format!(
                "checkpoint budget plan has no allocation for {endpoint}"
            ))
        })?;
        Ok((
            Membership {
                id: self.id.clone(),
                endpoint,
            },
            peers.clone(),
        ))
    }

    pub(super) async fn audit(&self) -> io::Result<String> {
        let mut jobs = JoinSet::new();
        for &endpoint in self.allocation.stores.keys() {
            jobs.spawn(async move {
                (
                    endpoint,
                    tokio::time::timeout(DEADLINE, read_stats(endpoint)).await,
                )
            });
        }
        while let Some(result) = jobs.join_next().await {
            let (endpoint, response) = result.map_err(io::Error::other)?;
            let body = response
                .map_err(|err| io::Error::other(format!("checkpoint store {endpoint}: {err}")))?
                .map_err(|err| io::Error::other(format!("checkpoint store {endpoint}: {err}")))?;
            let report: Report = serde_json::from_str(&body)
                .map_err(|err| invalid(format!("checkpoint store {endpoint}: {err}")))?;
            let (expected, peers) = self.share_for(endpoint)?;
            let actual: BTreeMap<_, _> = report
                .peers
                .into_iter()
                .map(|(ip, share)| {
                    (
                        ip,
                        PeerLimit {
                            read: share.read,
                            write: share.write,
                        },
                    )
                })
                .collect();
            if report.peer_budget_plan.as_ref() != Some(&expected) || peers != actual {
                return Err(invalid(format!(
                    "checkpoint store {endpoint} has a different budget plan or share"
                )));
            }
        }
        Ok(serde_json::json!({
            "id": self.id,
            "stores": self.allocation.stores.keys().collect::<Vec<_>>(),
        })
        .to_string())
    }
}

#[derive(Deserialize)]
struct Report {
    peer_budget_plan: Option<Membership>,
    peers: BTreeMap<IpAddr, ReportedShare>,
}

#[derive(Deserialize)]
struct ReportedShare {
    read: Limit,
    write: Limit,
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use serde_json::{Value, json};

    use super::*;
    use crate::kv_store::checkpoints::Checkpoints;

    fn fixture() -> Value {
        let limits = |rate, burst| {
            json!({
                "read": {"bytes_per_second": rate, "burst_bytes": burst},
                "write": {"bytes_per_second": rate, "burst_bytes": burst}
            })
        };
        let entries = |rate, burst| {
            json!([{
                "address": "192.0.2.3", "limit": limits(rate, burst)
            }])
        };
        json!({
            "peers": entries(100, 10),
            "stores": [
                {"endpoint": "192.0.2.5:7952", "peers": entries(40, 4)},
                {"endpoint": "192.0.2.6:7952", "peers": entries(60, 6)}
            ]
        })
    }

    fn parse(value: Value) -> io::Result<PeerPlan> {
        PeerPlan::validate(serde_json::from_value(value)?)
    }

    #[test]
    fn same_plan_has_a_stable_id_and_each_endpoint_receives_only_its_share() {
        let plan = parse(fixture()).unwrap();
        let mut reordered = fixture();
        reordered["stores"].as_array_mut().unwrap().reverse();
        reordered["stores"][1]["endpoint"] = json!("[::ffff:192.0.2.5]:7952");
        assert_eq!(plan.id, parse(reordered).unwrap().id);
        let endpoint = "192.0.2.5:7952".parse().unwrap();
        let peer = "192.0.2.3".parse::<IpAddr>().unwrap();
        let (membership, shares) = plan.share_for(endpoint).unwrap();
        assert_eq!(membership.endpoint, endpoint);
        assert_eq!(
            shares[&peer].read,
            Limit {
                bytes_per_second: 40,
                burst_bytes: 4
            }
        );
        assert!(plan.share_for("192.0.2.4:7952".parse().unwrap()).is_err());
    }

    #[test]
    fn rejects_overallocated_rates_bursts_unknown_peers_and_duplicate_endpoints() {
        for field in ["bytes_per_second", "burst_bytes"] {
            let mut invalid = fixture();
            let budget = &mut invalid["stores"][1]["peers"][0]["limit"]["write"][field];
            *budget = json!(budget.as_u64().unwrap() + 1);
            assert!(parse(invalid).is_err());
        }
        let mut unknown = fixture();
        unknown["stores"][1]["peers"][0]["address"] = json!("192.0.2.4");
        assert!(parse(unknown).is_err());

        let mut duplicate = fixture();
        duplicate["stores"][1]["endpoint"] = json!("[::ffff:192.0.2.5]:7952");
        assert!(parse(duplicate).is_err());
    }

    #[test]
    fn restart_accrues_fresh_credit_and_empty_or_zero_shares_stay_restricted() {
        let endpoint = "192.0.2.5:7952".parse().unwrap();
        let other = "192.0.2.6:7952".parse().unwrap();
        let peer = "192.0.2.3".parse().unwrap();
        let start = Instant::now();
        let plan = parse(fixture()).unwrap();
        let (member, shares) = plan.share_for(endpoint).unwrap();
        let first = Checkpoints::new(None, shares.clone(), start, Some(member.clone()));
        assert!(first.write(peer, 1, start).is_none());
        let restart = start + Duration::from_millis(100);
        first.write(peer, 4, restart).unwrap().complete();
        let second = Checkpoints::new(None, shares, restart, Some(member));
        assert!(second.write(peer, 1, restart).is_none());
        second
            .write(peer, 4, restart + Duration::from_millis(100))
            .unwrap()
            .complete();

        let mut disabled = fixture();
        disabled["stores"][0]["peers"][0]["limit"]["read"] =
            json!({"bytes_per_second": 0, "burst_bytes": 0});
        disabled["stores"][1]["peers"] = json!([]);
        let plan = parse(disabled).unwrap();
        let late = start + Duration::from_secs(30);
        let (member, shares) = plan.share_for(endpoint).unwrap();
        let store = Checkpoints::new(None, shares, start, Some(member));
        assert!(!store.reserve_read([1; 32], [2; 16], 1, Some(BTreeMap::from([(peer, 1)])), late));
        let (member, shares) = plan.share_for(other).unwrap();
        let store = Checkpoints::new(None, shares, start, Some(member));
        assert!(store.write(peer, 1, late).is_none());
        assert!(!store.reserve_read([1; 32], [2; 16], 1, None, late));
    }
}
