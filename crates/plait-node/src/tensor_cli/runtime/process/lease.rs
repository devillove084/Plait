//! Tie a worker to its supervisor while retaining the device lock across exec.

use std::fs::File;
use std::io;
use std::path::Path;

use tokio::process::{Child, Command};

use super::SlotLock;

#[cfg(unix)]
mod unix {
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::net::UnixStream;

    use super::*;
    use crate::tensor_cli::runtime::lock_device;

    const LOCK_FD: &str = "PLAIT_RESIDENT_LOCK_FD";
    const WATCH_FD: &str = "PLAIT_RESIDENT_WATCH_FD";

    pub(in crate::tensor_cli::runtime) struct WorkerLease {
        _supervisor: UnixStream,
    }

    impl WorkerLease {
        pub(in crate::tensor_cli::runtime) fn spawn(
            command: &mut Command,
            lock: &SlotLock,
        ) -> io::Result<(Child, Self)> {
            let lease = lock.acquire()?;
            let (supervisor, watch) = UnixStream::pair()?;
            let lock_fd = lease.as_raw_fd();
            let watch_fd = watch.as_raw_fd();
            command
                .env(LOCK_FD, lock_fd.to_string())
                .env(WATCH_FD, watch_fd.to_string());
            // SAFETY: Both descriptors stay open until spawn returns. The child
            // clears close-on-exec on just these descriptors using async-signal-safe fcntl.
            unsafe {
                command.pre_exec(move || {
                    for fd in [lock_fd, watch_fd] {
                        if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                            return Err(io::Error::last_os_error());
                        }
                    }
                    Ok(())
                });
            }
            let child = command.spawn()?;
            Ok((
                child,
                Self {
                    _supervisor: supervisor,
                },
            ))
        }
    }

    pub(in crate::tensor_cli) fn enter_worker(path: &Path) -> io::Result<File> {
        if std::env::var_os(LOCK_FD).is_none() && std::env::var_os(WATCH_FD).is_none() {
            return lock_device(path);
        }
        let lock_fd = descriptor_number(LOCK_FD)?;
        let watch_fd = descriptor_number(WATCH_FD)?;
        if lock_fd == watch_fd {
            return Err(invalid_lease());
        }
        let lease = File::from(take_inherited(lock_fd)?);
        let mut watch = UnixStream::from(take_inherited(watch_fd)?);
        watch.peer_addr()?;
        let expected = std::fs::metadata(path)?;
        let actual = lease.metadata()?;
        if (actual.dev(), actual.ino()) != (expected.dev(), expected.ino()) {
            return Err(invalid_lease());
        }
        if unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::last_os_error());
        }

        watch.set_nonblocking(true)?;
        let mut byte = [0];
        if !matches!(watch.read(&mut byte), Err(error) if error.kind() == io::ErrorKind::WouldBlock)
        {
            return Err(io::Error::other("resident supervisor is gone"));
        }
        watch.set_nonblocking(false)?;
        std::thread::Builder::new()
            .name("plait-supervisor-watch".into())
            .spawn(move || {
                loop {
                    if !matches!(watch.read(&mut byte), Err(error) if error.kind() == io::ErrorKind::Interrupted) {
                        // Parent death cannot wait for a blocked accelerator call.
                        // SAFETY: _exit terminates the whole worker without running
                        // library destructors that may wait for the dead supervisor.
                        unsafe { libc::_exit(1) };
                    }
                }
            })?;
        Ok(lease)
    }

    fn descriptor_number(name: &str) -> io::Result<RawFd> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|fd| *fd >= 3)
            .ok_or_else(invalid_lease)
    }

    fn take_inherited(fd: RawFd) -> io::Result<OwnedFd> {
        // A directly launched worker's descriptors created by Rust have CLOEXEC.
        // Check that this is explicitly inherited and restore CLOEXEC before taking it.
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } != 0
            || unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } != 0
        {
            return Err(invalid_lease());
        }
        // SAFETY: fcntl confirmed the descriptor is open. Each descriptor is taken once.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    fn invalid_lease() -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput, "invalid resident worker lease")
    }
}

#[cfg(unix)]
pub(in crate::tensor_cli::runtime) use unix::WorkerLease;
#[cfg(unix)]
pub(in crate::tensor_cli) use unix::enter_worker;

#[cfg(not(unix))]
pub(in crate::tensor_cli::runtime) struct WorkerLease;

#[cfg(not(unix))]
impl WorkerLease {
    pub(in crate::tensor_cli::runtime) fn spawn(
        _command: &mut Command,
        lock: &SlotLock,
    ) -> io::Result<(Child, Self)> {
        lock.acquire()?;
        unreachable!("resident worker locks require Unix")
    }
}

#[cfg(not(unix))]
pub(in crate::tensor_cli) fn enter_worker(path: &Path) -> io::Result<File> {
    super::lock_device(path)
}
