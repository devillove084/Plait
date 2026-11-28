//! Match ROCm ordinals to KFD devices and check their allocations and queues.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tokio::time::{Instant, timeout_at};

use super::ReleaseDriver;

const KFD: &str = "/sys/class/kfd/kfd";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Identity {
    hip: u32,
    management: u32,
    kfd: u32,
    node: u32,
    render: u32,
    bdf: String,
}

#[derive(Deserialize)]
struct Enumeration {
    gpu: u32,
    hip_id: u32,
    kfd_id: u32,
    node_id: u32,
    partition_id: u32,
    render: String,
    bdf: String,
}

pub(super) fn available() -> bool {
    let root = Path::new(KFD);
    root.join("topology/nodes").is_dir() && root.join("proc").is_dir()
}

pub(super) async fn capture(
    monitor: &ReleaseDriver,
    ordinals: &BTreeSet<u32>,
    deadline: Instant,
) -> io::Result<Vec<Identity>> {
    let output = monitor.query(&["list", "-e", "--json"], deadline).await?;
    let identities = identities(&output, ordinals)?;
    validate_topology(Path::new(KFD), &identities)?;
    Ok(identities)
}

pub(super) async fn idle(
    monitor: &ReleaseDriver,
    ordinals: &BTreeSet<u32>,
    expected: &[Identity],
    deadline: Instant,
) -> io::Result<bool> {
    if expected
        .iter()
        .map(|identity| identity.hip)
        .ne(ordinals.iter().copied())
    {
        return Err(invalid(
            "release request lacks the launch-time AMD device identities",
        ));
    }
    if capture(monitor, ordinals, deadline).await? != expected {
        return Err(invalid("AMD device identity changed since launch"));
    }
    let report = monitor.query(&["process", "--json"], deadline).await?;
    let pids = reported_processes(&report, expected)?;
    let expected = expected.to_vec();
    let snapshot =
        tokio::task::spawn_blocking(move || idle_snapshot(Path::new(KFD), &expected, &pids));
    match timeout_at(deadline, snapshot)
        .await?
        .map_err(io::Error::other)?
    {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        result => result,
    }
}

fn identities(output: &[u8], ordinals: &BTreeSet<u32>) -> io::Result<Vec<Identity>> {
    let rows: Vec<Enumeration> = serde_json::from_slice(output)?;
    let mut identities = Vec::new();
    for &hip in ordinals {
        let mut matches = rows.iter().filter(|row| row.hip_id == hip);
        let row = matches
            .next()
            .ok_or_else(|| invalid("unmapped HIP device"))?;
        if matches.next().is_some()
            || row.kfd_id == 0
            || row.kfd_id == u32::MAX
            || row.partition_id != 0
            || row.bdf.is_empty()
            || rows
                .iter()
                .filter(|other| {
                    other.bdf == row.bdf
                        || other.kfd_id == row.kfd_id
                        || other.node_id == row.node_id
                        || other.gpu == row.gpu
                        || other.render == row.render
                })
                .count()
                != 1
        {
            return Err(invalid(
                "AMD release requires a uniquely mapped whole device",
            ));
        }
        let render = row
            .render
            .strip_prefix("renderD")
            .and_then(|render| render.parse().ok())
            .ok_or_else(|| invalid("unmapped AMD render device"))?;
        identities.push(Identity {
            hip,
            management: row.gpu,
            kfd: row.kfd_id,
            node: row.node_id,
            render,
            bdf: row.bdf.clone(),
        });
    }
    Ok(identities)
}

fn validate_topology(root: &Path, devices: &[Identity]) -> io::Result<()> {
    for device in devices {
        let node = root.join("topology/nodes").join(device.node.to_string());
        let properties = fs::read_to_string(node.join("properties"))?;
        let mut renders = properties
            .lines()
            .filter_map(|line| line.split_once(' '))
            .filter_map(|(name, value)| (name == "drm_render_minor").then_some(value.trim()));
        if number(&node.join("gpu_id"))? != u64::from(device.kfd)
            || renders.next().and_then(|value| value.parse::<u32>().ok()) != Some(device.render)
            || renders.next().is_some()
        {
            return Err(invalid(
                "AMD SMI and KFD disagree about the device identity",
            ));
        }
    }
    Ok(())
}

fn reported_processes(output: &[u8], devices: &[Identity]) -> io::Result<BTreeSet<u32>> {
    #[derive(Deserialize)]
    struct Report {
        gpu: u32,
        process_list: Vec<Process>,
    }
    #[derive(Deserialize)]
    struct Process {
        process_info: ProcessInfo,
    }
    #[derive(Deserialize)]
    struct ProcessInfo {
        pid: u32,
    }
    let reports: Vec<Report> = serde_json::from_slice(output)?;
    let mut seen = BTreeSet::new();
    let mut pids = BTreeSet::new();
    for report in reports {
        if !seen.insert(report.gpu) {
            return Err(invalid("duplicate GPU process report"));
        }
        for process in report.process_list {
            if process.process_info.pid == 0 {
                return Err(invalid("invalid AMD process identifier"));
            }
            pids.insert(process.process_info.pid);
        }
    }
    if devices
        .iter()
        .any(|device| !seen.contains(&device.management))
    {
        return Err(invalid("missing AMD GPU process report"));
    }
    Ok(pids)
}

fn idle_snapshot(root: &Path, devices: &[Identity], reported: &BTreeSet<u32>) -> io::Result<bool> {
    validate_topology(root, devices)?;
    let dir = root.join("proc");
    let processes = numeric_dirs(&dir)?;
    if reported.iter().any(|pid| !processes.contains_key(pid)) {
        return Ok(false);
    }
    let gpu_ids: BTreeSet<_> = devices.iter().map(|device| u64::from(device.kfd)).collect();
    for &pid in processes.keys() {
        let process = dir.join(pid.to_string());
        let queues = process.join("queues");
        let before = numeric_dirs(&queues)?;
        for &queue in before.keys() {
            if gpu_ids.contains(&number(&queues.join(queue.to_string()).join("gpuid"))?) {
                return Ok(false);
            }
        }
        for &gpu in &gpu_ids {
            if number(&process.join(format!("vram_{gpu}")))? != 0 {
                return Ok(false);
            }
        }
        if before != numeric_dirs(&queues)? {
            return Ok(false);
        }
    }
    validate_topology(root, devices)?;
    Ok(processes == numeric_dirs(&dir)?)
}

#[cfg(unix)]
fn numeric_dirs(path: &Path) -> io::Result<BTreeMap<u32, u64>> {
    use std::os::unix::fs::DirEntryExt;
    let mut entries = BTreeMap::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let id = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
            .filter(|id| id.to_string() == entry.file_name().to_string_lossy())
            .ok_or_else(|| invalid("invalid KFD directory entry"))?;
        if !entry.file_type()?.is_dir() || entries.insert(id, entry.ino()).is_some() {
            return Err(invalid("invalid KFD directory layout"));
        }
    }
    Ok(entries)
}

#[cfg(not(unix))]
fn numeric_dirs(_: &Path) -> io::Result<BTreeMap<u32, u64>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "KFD requires Linux",
    ))
}

fn number(path: &Path) -> io::Result<u64> {
    fs::read_to_string(path)?
        .trim()
        .parse()
        .map_err(|_| invalid("invalid KFD counter"))
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(all(test, unix))]
mod tests {
    use std::path::PathBuf;

    use plait_core::NodeId;

    use super::*;

    const ENUMERATION: &str = r#"[
        {"gpu":5,"hip_id":0,"kfd_id":81,"node_id":2,"partition_id":0,"render":"renderD128","bdf":"0000:01:00.0"},
        {"gpu":7,"hip_id":1,"kfd_id":90,"node_id":3,"partition_id":0,"render":"renderD129","bdf":"0000:02:00.0"}
    ]"#;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("plait-kfd-test-{}", NodeId::random()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn write(&self, relative: &str, contents: &str) {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn other_gpu_activity_does_not_withhold_a_complete_idle_report() {
        let fixture = Fixture::new();
        fixture.write("topology/nodes/2/gpu_id", "81\n");
        fixture.write("topology/nodes/2/properties", "drm_render_minor 128\n");
        fixture.write("topology/nodes/3/gpu_id", "90\n");
        fixture.write("topology/nodes/3/properties", "drm_render_minor 129\n");
        fixture.write("proc/20/queues/4/gpuid", "81\n");
        fixture.write("proc/20/vram_81", "4096\n");
        fixture.write("proc/20/vram_90", "0\n");
        let devices = identities(ENUMERATION.as_bytes(), &BTreeSet::from([0, 1])).unwrap();
        let report = br#"[
            {"gpu":5,"process_list":[{"process_info":{"pid":20}}]},
            {"gpu":7,"process_list":[{"process_info":{"pid":20}}]}
        ]"#;
        let pids = reported_processes(report, &devices).unwrap();
        assert!(!idle_snapshot(&fixture.0, &devices, &pids).unwrap());
        assert!(idle_snapshot(&fixture.0, &devices[1..], &pids).unwrap());

        fixture.write("proc/20/queues/4/gpuid", "90\n");
        assert!(!idle_snapshot(&fixture.0, &devices[1..], &pids).unwrap());
        fixture.write("proc/20/queues/4/gpuid", "81\n");
        fixture.write("proc/20/vram_90", "32\n");
        assert!(!idle_snapshot(&fixture.0, &devices[1..], &pids).unwrap());

        fs::remove_file(fixture.0.join("proc/20/vram_90")).unwrap();
        assert!(idle_snapshot(&fixture.0, &devices[1..], &pids).is_err());
        assert!(!idle_snapshot(&fixture.0, &devices[1..], &BTreeSet::from([99])).unwrap());
        fixture.write("topology/nodes/3/gpu_id", "91\n");
        assert!(idle_snapshot(&fixture.0, &devices[1..], &pids).is_err());
    }

    #[test]
    fn capture_rejects_partitioned_or_ambiguous_device_mappings() {
        let device = BTreeSet::from([1]);
        let same_board = ENUMERATION.replace("0000:02:00.0", "0000:01:00.0");
        assert!(identities(same_board.as_bytes(), &device).is_err());
        let partitioned = ENUMERATION.replace("\"partition_id\":0", "\"partition_id\":1");
        assert!(identities(partitioned.as_bytes(), &device).is_err());
    }
}
