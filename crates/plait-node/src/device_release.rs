//! Driver process checks for dedicated devices used by managed TCP P/D groups.

use std::collections::BTreeSet;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::time::Duration;

use plait_core::NodeId;
use plait_core::serving::{GroupVersion, WorkerGrant};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::time::{Instant, sleep, timeout_at};

mod amd_kfd;

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Driver {
    Nvidia,
    Amd,
    AmdKfd,
    Ppu,
    Ascend,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReleaseDriver {
    driver: Driver,
    program: PathBuf,
}

impl ReleaseDriver {
    pub(crate) fn backend(&self) -> &'static str {
        match self.driver {
            Driver::Nvidia => "cuda",
            Driver::Amd | Driver::AmdKfd => "rocm",
            Driver::Ppu => "ppu",
            Driver::Ascend => "huawei",
        }
    }

    pub(crate) fn available(&self) -> bool {
        cfg!(target_os = "linux")
            && self.program.is_absolute()
            && is_executable(&self.program)
            && (!matches!(self.driver, Driver::AmdKfd) || amd_kfd::available())
    }

    pub(crate) fn supports_mask(&self, device_env: &str) -> bool {
        !matches!(self.driver, Driver::AmdKfd)
            || matches!(device_env, "CUDA_VISIBLE_DEVICES" | "HIP_VISIBLE_DEVICES")
    }

    async fn idle(&self, ordinal: u32, deadline: Instant) -> io::Result<bool> {
        let index = ordinal.to_string();
        let args = match self.driver {
            Driver::Nvidia | Driver::Ppu => &[
                "-i",
                &index,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ][..],
            Driver::Amd => &["process", "--gpu", &index, "--json"],
            Driver::Ascend => &["info", "-t", "proc-mem", "-i", &index, "-p", "0"],
            Driver::AmdKfd => unreachable!("AMD KFD release checks the entire local grant set"),
        };
        self.driver
            .idle_output(ordinal, &self.query(args, deadline).await?)
    }

    async fn query(&self, args: &[&str], deadline: Instant) -> io::Result<Vec<u8>> {
        let mut query = Command::new(&self.program);
        query.kill_on_drop(true).env("LC_ALL", "C").args(args);
        let output = timeout_at(deadline, query.output()).await??;
        if !output.status.success() || !output.stderr.iter().all(u8::is_ascii_whitespace) {
            return Err(io::Error::other("accelerator driver query failed"));
        }
        Ok(output.stdout)
    }
}

#[cfg(unix)]
pub(crate) fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
pub(crate) fn is_executable(_: &Path) -> bool {
    false
}

impl Driver {
    fn idle_output(self, ordinal: u32, output: &[u8]) -> io::Result<bool> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "unrecognized driver process report",
            )
        };
        let text = std::str::from_utf8(output).map_err(|_| invalid())?;
        let mut rows = text.lines().map(str::trim).filter(|line| !line.is_empty());
        match self {
            Self::Nvidia | Self::Ppu => {
                let mut idle = true;
                for row in rows {
                    if !row.parse::<u64>().is_ok_and(|pid| pid > 0) {
                        return Err(invalid());
                    }
                    idle = false;
                }
                Ok(idle)
            }
            Self::Amd => {
                #[derive(Deserialize)]
                struct Report {
                    gpu: u32,
                    process_list: Vec<serde_json::Value>,
                }
                let reports: Vec<Report> = serde_json::from_str(text).map_err(|_| invalid())?;
                let [report] = reports.as_slice() else {
                    return Err(invalid());
                };
                if report.gpu != ordinal {
                    return Err(invalid());
                }
                Ok(report.process_list.is_empty())
            }
            Self::Ascend => {
                let matches =
                    rows.next()
                        .and_then(|row| row.split_once(':'))
                        .is_some_and(|(name, id)| {
                            name.trim() == "NPU ID" && id.trim().parse::<u32>() == Ok(ordinal)
                        });
                if !matches {
                    return Err(invalid());
                }
                Ok(rows.next() == Some("No process in device.") && rows.next().is_none())
            }
            Self::AmdKfd => unreachable!("AMD KFD release uses the kernel process table"),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleaseRequest {
    monitor: ReleaseDriver,
    host: NodeId,
    version: GroupVersion,
    grants: Vec<WorkerGrant>,
    #[serde(default)]
    identities: Vec<amd_kfd::Identity>,
}

impl ReleaseRequest {
    fn devices(&self) -> io::Result<BTreeSet<u32>> {
        let mut devices = BTreeSet::new();
        if !self.monitor.available()
            || self.grants.is_empty()
            || self.grants.iter().any(|grant| {
                grant.device.node != self.host
                    || grant.version != self.version
                    || !devices.insert(grant.device.ordinal)
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid local release request",
            ));
        }
        Ok(devices)
    }
}

fn request(mut args: impl Iterator<Item = String>) -> io::Result<ReleaseRequest> {
    if args.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected a local grant record on stdin",
        ));
    }
    let mut input = Vec::new();
    io::Read::take(io::stdin().lock(), 65_537).read_until(b'\n', &mut input)?;
    if input.len() > 65_536 || !input.ends_with(b"\n") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid release request length",
        ));
    }
    Ok(serde_json::from_slice(&input)?)
}

pub(crate) async fn capture(
    args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = request(args)?;
    let devices = request.devices()?;
    if !matches!(request.monitor.driver, Driver::AmdKfd) || !request.identities.is_empty() {
        return Err("expected an AMD KFD device capture request".into());
    }
    let identities = amd_kfd::capture(
        &request.monitor,
        &devices,
        Instant::now() + Duration::from_secs(20),
    )
    .await?;
    serde_json::to_writer(io::stdout().lock(), &identities)?;
    Ok(())
}

pub(crate) async fn run(
    args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = request(args)?;
    let devices = request.devices()?;
    let deadline = Instant::now() + Duration::from_secs(20);
    let kfd = matches!(request.monitor.driver, Driver::AmdKfd);
    if !kfd && !request.identities.is_empty() {
        return Err("unexpected release identity record".into());
    }
    loop {
        let idle = if kfd {
            amd_kfd::idle(&request.monitor, &devices, &request.identities, deadline).await?
        } else {
            let mut idle = true;
            for &ordinal in &devices {
                idle &= request.monitor.idle(ordinal, deadline).await?;
            }
            idle
        };
        if idle {
            return Ok(());
        }
        timeout_at(deadline, sleep(Duration::from_millis(500))).await?;
    }
}

#[cfg(test)]
mod tests {
    use super::Driver;

    #[test]
    fn process_queries_require_a_complete_unambiguous_idle_report() {
        for driver in [Driver::Nvidia, Driver::Ppu] {
            assert!(driver.idle_output(3, b"\n").unwrap());
            assert!(!driver.idle_output(3, b"27\n41\n").unwrap());
            assert!(driver.idle_output(3, b"Not supported\n").is_err());
        }
        assert!(
            Driver::Amd
                .idle_output(3, br#"[{"gpu":3,"process_list":[]}]"#)
                .unwrap()
        );
        assert!(
            !Driver::Amd
                .idle_output(
                    3,
                    br#"[{"gpu":3,"process_list":[{"process_info":{"pid":27}}]}]"#
                )
                .unwrap()
        );
        assert!(
            Driver::Amd
                .idle_output(3, br#"[{"gpu":2,"process_list":[]}]"#)
                .is_err()
        );
        assert!(
            Driver::Amd
                .idle_output(3, br#"[{"gpu":3,"process_list":"N/A"}]"#)
                .is_err()
        );
        assert!(Driver::Amd.idle_output(3, b"[]").is_err());
        assert!(
            Driver::Ascend
                .idle_output(3, b" NPU ID : 3\n No process in device.\n")
                .unwrap()
        );
        assert!(
            !Driver::Ascend
                .idle_output(3, b" NPU ID : 3\n Process ID : 27\n")
                .unwrap()
        );
        assert!(
            !Driver::Ascend
                .idle_output(3, b" NPU ID : 3\n No process in device.\n incomplete\n")
                .unwrap()
        );
        assert!(
            Driver::Ascend
                .idle_output(3, b" NPU ID : 2\n No process in device.\n")
                .is_err()
        );
    }
}
