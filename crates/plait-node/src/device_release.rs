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

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Driver {
    Nvidia,
    Amd,
    Ppu,
    Ascend,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReleaseDriver {
    driver: Driver,
    program: PathBuf,
}

impl ReleaseDriver {
    pub(crate) fn backend(&self) -> &'static str {
        match self.driver {
            Driver::Nvidia => "cuda",
            Driver::Amd => "rocm",
            Driver::Ppu => "ppu",
            Driver::Ascend => "huawei",
        }
    }

    pub(crate) fn available(&self) -> bool {
        cfg!(target_os = "linux") && self.program.is_absolute() && is_executable(&self.program)
    }

    async fn idle(&self, ordinal: u32, deadline: Instant) -> io::Result<bool> {
        let mut query = Command::new(&self.program);
        query.kill_on_drop(true).env("LC_ALL", "C");
        let index = ordinal.to_string();
        match self.driver {
            Driver::Nvidia | Driver::Ppu => query.args([
                "-i",
                &index,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]),
            Driver::Amd => query.args(["process", "--gpu", &index, "--json"]),
            Driver::Ascend => query.args(["info", "-t", "proc-mem", "-i", &index, "-p", "0"]),
        };
        let output = timeout_at(deadline, query.output()).await??;
        if !output.status.success() || !output.stderr.iter().all(u8::is_ascii_whitespace) {
            return Err(io::Error::other(format!(
                "driver query failed for device {ordinal}"
            )));
        }
        self.driver.idle_output(ordinal, &output.stdout)
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

pub(crate) async fn run(
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.next().is_some() {
        return Err("verify-host-release expects a local grant record on stdin".into());
    }
    let mut input = Vec::new();
    io::Read::take(io::stdin().lock(), 65_537).read_until(b'\n', &mut input)?;
    if input.len() > 65_536 || !input.ends_with(b"\n") {
        return Err("invalid release request length".into());
    }
    let request: ReleaseRequest = serde_json::from_slice(&input)?;
    let devices = request.devices()?;
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let mut idle = true;
        for &ordinal in &devices {
            idle &= request.monitor.idle(ordinal, deadline).await?;
        }
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
