//! One byte budget for optional page transfers through a store.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

use super::{Key, Writer};

const NANOS_PER_SECOND: u128 = 1_000_000_000;
const MAX_LEASES: usize = 1024;
const LEASE_LIFETIME: Duration = Duration::from_secs(30);

#[derive(Clone, Copy)]
pub(super) struct Limit {
    pub(super) bytes_per_second: u64,
    pub(super) burst_bytes: u64,
}

pub(super) struct Checkpoints(Mutex<State>);

#[derive(Clone, Copy)]
enum Direction {
    Read,
    Write,
}

struct ReadLease {
    remaining: u64,
    expires: Instant,
}

#[derive(Default, Serialize)]
struct Counters {
    read_bytes: u64,
    write_bytes: u64,
    read_reservations_denied: u64,
    write_reservations_denied: u64,
    read_claims_denied: u64,
    expired_read_bytes: u64,
}

#[derive(Serialize)]
pub(super) struct Snapshot<'a> {
    uptime_ms: u64,
    bytes_per_second: Option<u64>,
    burst_bytes: Option<u64>,
    pending_bytes: u64,
    read_leases: usize,
    #[serde(flatten)]
    counters: &'a Counters,
}

struct State {
    limit: Option<Limit>,
    credits: u128,
    started: Instant,
    updated: Instant,
    pending: u64,
    reads: HashMap<(Key, Writer), ReadLease>,
    counters: Counters,
}

impl State {
    fn refresh(&mut self, now: Instant) {
        let now = now.max(self.updated);
        self.reads.retain(|_, lease| {
            if lease.expires > now {
                return true;
            }
            self.pending -= lease.remaining;
            self.counters.expired_read_bytes = self
                .counters
                .expired_read_bytes
                .saturating_add(lease.remaining);
            false
        });
        if let Some(limit) = self.limit {
            let elapsed = now.saturating_duration_since(self.updated).as_nanos();
            let available = u128::from(limit.burst_bytes - self.pending) * NANOS_PER_SECOND;
            let earned = elapsed.saturating_mul(u128::from(limit.bytes_per_second));
            self.credits = self.credits.saturating_add(earned).min(available);
        }
        self.updated = now;
    }

    fn charge(&mut self, bytes: u64) -> bool {
        if self.limit.is_some() {
            let cost = u128::from(bytes) * NANOS_PER_SECOND;
            let Some(credits) = self.credits.checked_sub(cost) else {
                return false;
            };
            self.credits = credits;
        }
        self.pending += bytes;
        true
    }
}

impl Checkpoints {
    pub(super) fn new(limit: Option<Limit>, now: Instant) -> Self {
        let burst = limit.map_or(0, |limit| limit.burst_bytes);
        Self(Mutex::new(State {
            limit,
            credits: u128::from(burst) * NANOS_PER_SECOND,
            started: now,
            updated: now,
            pending: 0,
            reads: HashMap::new(),
            counters: Counters::default(),
        }))
    }

    pub(super) fn reserve_read(&self, key: Key, writer: Writer, bytes: u64, now: Instant) -> bool {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        let id = (key, writer);
        if state.reads.len() >= MAX_LEASES || state.reads.contains_key(&id) || !state.charge(bytes)
        {
            state.counters.read_reservations_denied =
                state.counters.read_reservations_denied.saturating_add(1);
            return false;
        }
        let expires = state.updated + LEASE_LIFETIME;
        state.reads.insert(
            id,
            ReadLease {
                remaining: bytes,
                expires,
            },
        );
        true
    }

    pub(super) fn read(
        &self,
        key: Key,
        writer: Writer,
        bytes: u64,
        now: Instant,
    ) -> Option<Transfer<'_>> {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        let id = (key, writer);
        let Some(lease) = state
            .reads
            .get_mut(&id)
            .filter(|lease| lease.remaining >= bytes)
        else {
            state.counters.read_claims_denied = state.counters.read_claims_denied.saturating_add(1);
            return None;
        };
        lease.remaining -= bytes;
        if lease.remaining == 0 {
            state.reads.remove(&id);
        }
        Some(Transfer {
            budget: self,
            direction: Direction::Read,
            bytes,
            complete: false,
        })
    }

    pub(super) fn write(&self, bytes: u64, now: Instant) -> Option<Transfer<'_>> {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        if !state.charge(bytes) {
            state.counters.write_reservations_denied =
                state.counters.write_reservations_denied.saturating_add(1);
            return None;
        }
        Some(Transfer {
            budget: self,
            direction: Direction::Write,
            bytes,
            complete: false,
        })
    }

    pub(super) fn snapshot(&self, now: Instant) -> Vec<u8> {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        serde_json::to_vec(&Snapshot {
            uptime_ms: state.updated.duration_since(state.started).as_millis() as u64,
            bytes_per_second: state.limit.map(|limit| limit.bytes_per_second),
            burst_bytes: state.limit.map(|limit| limit.burst_bytes),
            pending_bytes: state.pending,
            read_leases: state.reads.len(),
            counters: &state.counters,
        })
        .expect("serializable checkpoint metrics")
    }
}

pub(super) struct Transfer<'a> {
    budget: &'a Checkpoints,
    direction: Direction,
    bytes: u64,
    complete: bool,
}

impl Transfer<'_> {
    pub(super) fn complete(mut self) {
        self.complete = true;
    }
}

impl Drop for Transfer<'_> {
    fn drop(&mut self) {
        let mut state = self.budget.0.lock().expect("checkpoint budget");
        state.refresh(Instant::now());
        state.pending -= self.bytes;
        if self.complete {
            let counter = match self.direction {
                Direction::Read => &mut state.counters.read_bytes,
                Direction::Write => &mut state.counters.write_bytes,
            };
            *counter = counter.saturating_add(self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_order_clock_samples_cannot_refill_the_same_interval_twice() {
        let start = Instant::now();
        let budget = Checkpoints::new(
            Some(Limit {
                bytes_per_second: 10,
                burst_bytes: 20,
            }),
            start,
        );
        let mut state = budget.0.lock().unwrap();
        assert!(state.charge(20));
        state.pending = 0;
        state.refresh(start + Duration::from_millis(500));
        state.refresh(start + Duration::from_millis(200));
        state.refresh(start + Duration::from_millis(800));
        assert_eq!(state.credits, 8 * NANOS_PER_SECOND);
        assert!(!state.charge(9));
    }

    #[test]
    fn outstanding_reads_and_writes_share_one_burst_and_refill() {
        let now = Instant::now();
        let budget = Checkpoints::new(
            Some(Limit {
                bytes_per_second: 100,
                burst_bytes: 100,
            }),
            now,
        );
        assert!(budget.reserve_read([1; 32], [1; 16], 80, now));
        let held = budget.write(20, now).unwrap();
        let later = now + Duration::from_secs(1);
        assert!(budget.write(1, later).is_none());
        assert!(budget.read([1; 32], [2; 16], 1, later).is_none());
        let read = budget.read([1; 32], [1; 16], 30, later).unwrap();
        assert!(budget.write(1, later).is_none());
        read.complete();
        drop(held);
        assert!(budget.write(1, later).is_none());
        let later = later + Duration::from_millis(500);
        budget.write(50, later).unwrap().complete();
        assert!(budget.read([1; 32], [1; 16], 51, later).is_none());
        assert!(
            budget
                .read([1; 32], [1; 16], 50, now + LEASE_LIFETIME)
                .is_none()
        );
        let snapshot: serde_json::Value =
            serde_json::from_slice(&budget.snapshot(now + LEASE_LIFETIME)).unwrap();
        assert_eq!(snapshot["pending_bytes"], 0);
        assert_eq!(snapshot["read_bytes"], 30);
        assert_eq!(snapshot["write_bytes"], 50);
        assert_eq!(snapshot["expired_read_bytes"], 50);
    }
}
