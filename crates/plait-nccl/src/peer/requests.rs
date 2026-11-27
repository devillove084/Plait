use std::io;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

use super::{Collective, PeerGroup, Timer};

pub(super) const QUEUE_DEPTH: usize = 8;

struct Job {
    sequence: u64,
    operation: Collective,
    input: Vec<u8>,
    result: SyncSender<io::Result<Vec<u8>>>,
}

#[derive(Default)]
pub(crate) struct Dispatcher(Mutex<Option<SyncSender<Job>>>);

impl Dispatcher {
    pub(crate) fn submit(
        group: &Arc<PeerGroup>,
        operation: Collective,
        input: Vec<u8>,
    ) -> io::Result<Completion> {
        group.live()?;
        let mut sender = group.requests.0.lock().unwrap();
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
                        let _ = job.result.send(outcome);
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

pub(crate) struct Completion {
    receiver: Receiver<io::Result<Vec<u8>>>,
    result: Option<io::Result<Vec<u8>>>,
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

    pub(crate) fn wait(self) -> io::Result<Vec<u8>> {
        self.result
            .unwrap_or_else(|| self.receiver.recv().unwrap_or_else(|_| Err(disconnected())))
    }
}

fn disconnected() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "peer collective worker closed")
}
