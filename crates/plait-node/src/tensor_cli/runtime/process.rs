use std::fs::{File, OpenOptions};
use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use plait_core::serving::WorkerGrant;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::{Artifact, ControlPeers, Request};

mod lease;
use lease::WorkerLease;
pub(in crate::tensor_cli) use lease::enter_worker;

pub(super) struct SlotLock {
    pub(super) path: PathBuf,
    pub(super) identity: [u64; 2],
}

impl SlotLock {
    pub(super) fn open(path: PathBuf) -> io::Result<Self> {
        let identity = lock_identity(&open_lock(&path)?)?;
        Ok(Self { path, identity })
    }

    pub(super) fn acquire(&self) -> io::Result<File> {
        let file = lock_device(&self.path)?;
        if lock_identity(&file)? != self.identity {
            return Err(io::Error::other("resident device lock was replaced"));
        }
        Ok(file)
    }
}

pub(super) struct WorkerProcess {
    child: Child,
    input: ChildStdin,
    output: Lines<BufReader<ChildStdout>>,
    usable: bool,
    lease: WorkerLease,
}

impl WorkerProcess {
    pub(super) async fn start(
        grant: WorkerGrant,
        artifact: &Artifact,
        listen: SocketAddr,
        lock: &SlotLock,
        control: &ControlPeers,
    ) -> io::Result<Self> {
        let config = json!({
            "library": artifact.library, "ordinal": grant.device.ordinal,
            "group_size": artifact.group_size, "grant": grant, "k_range": artifact.k_range,
            "output_columns": artifact.output_columns, "listen": listen,
            "device_lock": lock.path, "control": control, "kernel_profiles": artifact.kernel_profiles
        });
        let weights = tokio::fs::read_to_string(&artifact.weights).await?;
        let weights = weights.trim();
        if weights.contains('\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "worker weights must be a single JSON line",
            ));
        }
        let mut process = Self::spawn(config, lock)?;
        tokio::time::timeout(Duration::from_secs(90), async {
            process.send(weights.as_bytes()).await?;
            let ready = process.read().await?;
            if ready
                .get("ready")
                .and_then(Value::as_str)
                .and_then(|addr| addr.parse::<SocketAddr>().ok())
                .is_none()
            {
                return Err(io::Error::other("worker did not report a data endpoint"));
            }
            Ok::<_, io::Error>(())
        })
        .await??;
        Ok(process)
    }

    fn spawn(config: Value, lock: &SlotLock) -> io::Result<Self> {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .args(["tensor", "worker-w4"])
            .arg(config.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        let (mut child, lease) = WorkerLease::spawn(&mut command, lock)?;
        let input = child.stdin.take().expect("piped worker stdin");
        let output = BufReader::new(child.stdout.take().expect("piped worker stdout")).lines();
        Ok(Self {
            child,
            input,
            output,
            usable: true,
            lease,
        })
    }

    pub(super) fn check_running(&mut self) -> io::Result<()> {
        if let Some(status) = self.try_wait()? {
            return Err(io::Error::other(format!(
                "resident worker exited with {status}"
            )));
        }
        Ok(())
    }

    pub(super) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    pub(super) fn has_usable_pipe(&self) -> bool {
        self.usable
    }

    pub(super) async fn call(&mut self, request: &Request) -> io::Result<Value> {
        self.check_running()?;
        if !self.usable {
            return Err(io::Error::other(
                "resident worker response stream is unavailable",
            ));
        }
        let data = serde_json::to_vec(request)?;
        let result = tokio::time::timeout(Duration::from_secs(35), async {
            self.send(&data).await?;
            self.read().await
        })
        .await
        .map_err(io::Error::from)
        .and_then(|result| result);
        if result.is_err() {
            self.usable = false;
        }
        result
    }

    pub(super) async fn stop(self) -> io::Result<()> {
        let Self {
            mut child,
            input,
            lease: _lease,
            ..
        } = self;
        drop(input);
        let status = tokio::time::timeout(Duration::from_secs(40), child.wait()).await??;
        if status.success() {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "resident worker did not exit cleanly: {status}"
            )))
        }
    }

    async fn send(&mut self, payload: &[u8]) -> io::Result<()> {
        self.input.write_all(payload).await?;
        self.input.write_all(b"\n").await
    }

    async fn read(&mut self) -> io::Result<Value> {
        let line = self.output.next_line().await?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "resident worker closed its output",
            )
        })?;
        Ok(serde_json::from_str(&line)?)
    }
}

#[cfg(unix)]
pub(in crate::tensor_cli) fn lock_device(path: &Path) -> io::Result<File> {
    use std::os::fd::AsRawFd;
    let file = open_lock(path)?;
    // The worker owns this descriptor until its device session has been dropped.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
}

fn open_lock(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

#[cfg(unix)]
pub(super) fn lock_identity(file: &File) -> io::Result<[u64; 2]> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata()?;
    Ok([metadata.dev(), metadata.ino()])
}

#[cfg(not(unix))]
pub(super) fn lock_identity(_file: &File) -> io::Result<[u64; 2]> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resident device locks require Unix",
    ))
}

#[cfg(not(unix))]
pub(in crate::tensor_cli) fn lock_device(_path: &Path) -> io::Result<File> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "resident device locks require Unix",
    ))
}
