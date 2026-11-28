use std::fs::File;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, Weak};
use std::thread;
use std::time::Instant;

use super::{Collective, PeerGroup};
use crate::profile::{self, Phases};

pub(super) const QUEUE_DEPTH: usize = 8;
const DEFAULT_STAGING_BYTES: usize = 1024 * 1024 * 1024;

struct Job {
    sequence: u64,
    operation: Collective,
    input: Vec<u8>,
    result: SyncSender<io::Result<StagedOutput>>,
    admission: Admission,
    timing: Option<(&'static File, Instant)>,
}

impl Job {
    fn run(self, group: &Weak<PeerGroup>) {
        let Admission { slot, staging } = self.admission;
        drop(slot);
        let timing = self.timing.map(|(log, enqueued)| (log, enqueued.elapsed()));
        let outcome = match group.upgrade() {
            Some(group) => {
                let mut phases = Phases::new(timing.is_some());
                let outcome = phases.collective.measure(|| {
                    group.execute_reserved(
                        self.sequence,
                        self.operation,
                        &self.input,
                        &mut phases.wire,
                        &mut phases.fold,
                    )
                });
                if let (Ok(_), Some((log, queued))) = (&outcome, timing) {
                    phases.record_async(
                        log,
                        &group,
                        self.sequence,
                        self.operation,
                        self.input.len(),
                        queued,
                    );
                }
                outcome
            }
            None => Err(disconnected()),
        };
        drop(self.input);
        let result = outcome.map(|bytes| StagedOutput {
            bytes,
            _staging: staging,
        });
        let _ = self.result.send(result);
    }
}

pub(crate) struct Dispatcher {
    sender: Mutex<Option<SyncSender<Job>>>,
    pending: Arc<AtomicUsize>,
    staged_bytes: Arc<AtomicUsize>,
    staging_limit: usize,
}

impl Dispatcher {
    pub(super) fn from_env(staging_ceiling: usize) -> io::Result<Self> {
        let limit = match std::env::var("PLAIT_NCCL_PEER_ASYNC_STAGING_BYTES") {
            Ok(value) => value.parse().map_err(|_| invalid_limit())?,
            Err(std::env::VarError::NotPresent) => DEFAULT_STAGING_BYTES,
            Err(_) => return Err(invalid_limit()),
        };
        Ok(Self::new(limit.min(staging_ceiling)))
    }

    pub(super) fn new(staging_limit: usize) -> Self {
        Self {
            sender: Mutex::new(None),
            pending: Arc::new(AtomicUsize::new(0)),
            staged_bytes: Arc::new(AtomicUsize::new(0)),
            staging_limit,
        }
    }

    pub(crate) fn admit(&self, bytes: usize) -> Option<Admission> {
        let slot = Reservation::try_new(&self.pending, 1, QUEUE_DEPTH)?;
        let staging = Reservation::try_new(&self.staged_bytes, bytes, self.staging_limit)?;
        Some(Admission { slot, staging })
    }

    pub(crate) fn submit(
        group: &Arc<PeerGroup>,
        operation: Collective,
        input: Vec<u8>,
        admission: Admission,
    ) -> io::Result<Completion> {
        group.live()?;
        let log = profile::async_log();
        let mut sender = group.requests.sender.lock().unwrap();
        if sender.is_none() {
            let (tx, rx) = sync_channel::<Job>(QUEUE_DEPTH);
            let weak = Arc::downgrade(group);
            thread::Builder::new()
                .name("plait-peer-collectives".into())
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        job.run(&weak);
                    }
                })?;
            *sender = Some(tx);
        }
        let (result, receiver) = sync_channel(1);
        let sender = sender.as_ref().expect("collective queue started");
        group.reserve(|sequence| {
            sender
                .try_send(Job {
                    sequence,
                    operation,
                    input,
                    result,
                    admission,
                    timing: log.map(|log| (log, Instant::now())),
                })
                .map_err(|error| match error {
                    TrySendError::Full(_) => {
                        io::Error::new(io::ErrorKind::WouldBlock, "peer collective queue is full")
                    }
                    TrySendError::Disconnected(_) => disconnected(),
                })
        })?;
        Ok(Completion {
            receiver,
            result: None,
        })
    }
}

pub(crate) struct Admission {
    slot: Reservation,
    staging: Reservation,
}

struct Reservation {
    used: Arc<AtomicUsize>,
    count: usize,
}

impl Reservation {
    fn try_new(used: &Arc<AtomicUsize>, count: usize, limit: usize) -> Option<Self> {
        used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
            used.checked_add(count).filter(|&next| next <= limit)
        })
        .ok()?;
        Some(Self {
            used: Arc::clone(used),
            count,
        })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.used.fetch_sub(self.count, Ordering::Relaxed);
    }
}

fn invalid_limit() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "PLAIT_NCCL_PEER_ASYNC_STAGING_BYTES must be a nonnegative byte count",
    )
}

pub(crate) struct Completion {
    receiver: Receiver<io::Result<StagedOutput>>,
    result: Option<io::Result<StagedOutput>>,
}

pub(crate) struct StagedOutput {
    pub(crate) bytes: Vec<u8>,
    // Held by the channel, then by whoever copies or drops these bytes.
    _staging: Reservation,
}

impl Completion {
    pub(crate) fn ready(&mut self) -> bool {
        if self.result.is_none() {
            self.result = match self.receiver.try_recv() {
                Ok(result) => Some(result),
                Err(TryRecvError::Disconnected) => Some(Err(disconnected())),
                Err(TryRecvError::Empty) => None,
            };
        }
        self.result.is_some()
    }

    pub(crate) fn wait(self) -> io::Result<StagedOutput> {
        self.result
            .unwrap_or_else(|| self.receiver.recv().unwrap_or_else(|_| Err(disconnected())))
    }
}

fn disconnected() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "peer collective worker closed")
}
