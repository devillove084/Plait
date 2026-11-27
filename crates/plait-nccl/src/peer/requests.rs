use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

use super::{Collective, PeerGroup, Timer};

pub(super) const QUEUE_DEPTH: usize = 8;
const DEFAULT_STAGING_BYTES: usize = 1024 * 1024 * 1024;

struct Job {
    sequence: u64,
    operation: Collective,
    input: Vec<u8>,
    result: SyncSender<io::Result<StagedOutput>>,
    staging: StagingReservation,
}

pub(crate) struct Dispatcher {
    sender: Mutex<Option<SyncSender<Job>>>,
    staged_bytes: Arc<AtomicUsize>,
    staging_limit: usize,
}

impl Dispatcher {
    pub(super) fn from_env() -> io::Result<Self> {
        let limit = match std::env::var("PLAIT_NCCL_PEER_ASYNC_STAGING_BYTES") {
            Ok(value) => value.parse().map_err(|_| invalid_limit())?,
            Err(std::env::VarError::NotPresent) => DEFAULT_STAGING_BYTES,
            Err(_) => return Err(invalid_limit()),
        };
        Ok(Self::new(limit))
    }

    pub(super) fn new(staging_limit: usize) -> Self {
        Self {
            sender: Mutex::new(None),
            staged_bytes: Arc::new(AtomicUsize::new(0)),
            staging_limit,
        }
    }

    pub(crate) fn reserve_staging(&self, bytes: usize) -> Option<StagingReservation> {
        self.staged_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes)
                    .filter(|&next| next <= self.staging_limit)
            })
            .ok()?;
        Some(StagingReservation {
            used: Arc::clone(&self.staged_bytes),
            bytes,
        })
    }

    pub(crate) fn submit(
        group: &Arc<PeerGroup>,
        operation: Collective,
        input: Vec<u8>,
        staging: StagingReservation,
    ) -> io::Result<Completion> {
        group.live()?;
        let mut sender = group.requests.sender.lock().unwrap();
        if sender.is_none() {
            let (tx, rx) = sync_channel::<Job>(QUEUE_DEPTH);
            let weak = Arc::downgrade(group);
            thread::Builder::new()
                .name("plait-peer-collectives".into())
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        let outcome = match weak.upgrade() {
                            Some(group) => group.execute_reserved(
                                job.sequence,
                                job.operation,
                                &job.input,
                                &mut Timer::new(false),
                                &mut Timer::new(false),
                            ),
                            None => Err(disconnected()),
                        };
                        drop(job.input);
                        let result = outcome.map(|bytes| StagedOutput {
                            bytes,
                            _staging: job.staging,
                        });
                        let _ = job.result.send(result);
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
                    staging,
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

pub(crate) struct StagingReservation {
    used: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for StagingReservation {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::Relaxed);
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
    _staging: StagingReservation,
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
