//! Opt-in phase timings for host-staged peer collectives.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::peer::{Collective, PeerGroup};

static LOG: OnceLock<Option<File>> = OnceLock::new();
static ASYNC_LOG: OnceLock<Option<File>> = OnceLock::new();
static SHARD_LOG: OnceLock<Option<File>> = OnceLock::new();

pub(crate) fn log() -> Option<&'static File> {
    LOG.get_or_init(|| open_log("PLAIT_NCCL_PHASE_LOG"))
        .as_ref()
}

pub(crate) fn async_log() -> Option<&'static File> {
    ASYNC_LOG
        .get_or_init(|| open_log("PLAIT_NCCL_ASYNC_PHASE_LOG"))
        .as_ref()
}

pub(crate) fn shard_log() -> Option<&'static File> {
    SHARD_LOG
        .get_or_init(|| open_log("PLAIT_NCCL_SHARD_PHASE_LOG"))
        .as_ref()
}

fn open_log(variable: &str) -> Option<File> {
    let path = std::env::var_os(variable).filter(|path| !path.is_empty())?;
    match OpenOptions::new().create(true).append(true).open(path) {
        Ok(file) => Some(file),
        Err(error) => {
            eprintln!("Plait collective phase log ({variable}): {error}");
            None
        }
    }
}

pub(crate) struct Timer {
    enabled: bool,
    elapsed: Duration,
}

impl Timer {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            elapsed: Duration::ZERO,
        }
    }

    pub(crate) fn measure<T>(&mut self, work: impl FnOnce() -> T) -> T {
        let started = self.enabled.then(Instant::now);
        let result = work();
        if let Some(started) = started {
            self.elapsed += started.elapsed();
        }
        result
    }
}

pub(crate) struct Phases {
    pub(crate) sync: Timer,
    pub(crate) download: Timer,
    pub(crate) collective: Timer,
    pub(crate) wire: Timer,
    pub(crate) fold: Timer,
    pub(crate) upload: Timer,
}

impl Phases {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            sync: Timer::new(enabled),
            download: Timer::new(enabled),
            collective: Timer::new(enabled),
            wire: Timer::new(enabled),
            fold: Timer::new(enabled),
            upload: Timer::new(enabled),
        }
    }

    pub(crate) fn record(
        &self,
        mut output: &File,
        slot: usize,
        world: usize,
        operation: Collective,
        bytes: usize,
    ) {
        let (kind, dtype, _, _) = operation.wire();
        let line = format!(
            "{}\t{slot}\t{world}\t{kind}\t{}\t{bytes}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            std::process::id(),
            dtype as u8,
            self.sync.elapsed.as_nanos(),
            self.download.elapsed.as_nanos(),
            self.collective.elapsed.as_nanos(),
            self.wire.elapsed.as_nanos(),
            self.fold.elapsed.as_nanos(),
            self.upload.elapsed.as_nanos(),
        );
        // One append per row, including when multiple worker processes share a log.
        let _ = output.write_all(line.as_bytes());
    }

    pub(crate) fn record_async(
        &self,
        mut output: &File,
        group: &PeerGroup,
        sequence: u64,
        operation: Collective,
        bytes: usize,
        queued: Duration,
    ) {
        let (kind, dtype, _, _) = operation.wire();
        let line = format!(
            "{}\t{:032x}\t{}\t{}\t{sequence}\t{kind}\t{}\t{bytes}\t{}\t{}\t{}\t{}\n",
            std::process::id(),
            group.profile_group(),
            group.slot(),
            group.world(),
            dtype as u8,
            queued.as_nanos(),
            self.collective.elapsed.as_nanos(),
            self.wire.elapsed.as_nanos(),
            self.fold.elapsed.as_nanos(),
        );
        let _ = output.write_all(line.as_bytes());
    }
}
