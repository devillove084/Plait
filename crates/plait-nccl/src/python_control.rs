use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::time::Duration;

use plait_core::group::{GroupId, GroupVersion};
use plait_core::{DeviceId, NodeId};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::{CommHandle, JoinGroup, Transport, backend_from_env, join_committed_group};

fn control_addresses(address: &str, seeds: Vec<String>) -> PyResult<(SocketAddr, Vec<SocketAddr>)> {
    let listen = address
        .parse::<SocketAddr>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let seeds = seeds
        .iter()
        .map(|seed| seed.parse())
        .collect::<Result<Vec<SocketAddr>, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((listen, seeds))
}

pub(super) fn compute_rate(
    handle: &CommHandle,
    py: Python<'_>,
    address: &str,
    seeds: Vec<String>,
    calibration: &str,
    update: Option<(u64, u64)>,
    timeout_ms: u32,
) -> PyResult<u64> {
    let (listen, seeds) = control_addresses(address, seeds)?;
    let calibration =
        NodeId::from_hex(calibration).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if timeout_ms == 0 || listen.ip().is_unspecified() || seeds.is_empty() {
        return Err(PyValueError::new_err(
            "compute rate requires a reachable listen address, seeds, and a timeout",
        ));
    }
    let update = update
        .map(|(revision, rate)| {
            NonZeroU64::new(rate)
                .map(|rate| (revision, rate))
                .ok_or_else(|| PyValueError::new_err("compute rate must be positive"))
        })
        .transpose()?;
    let Transport::Peers(group) = &unsafe { &*handle.pointer() }.transport else {
        return Err(PyValueError::new_err(
            "compute rate requires a committed group",
        ));
    };
    let grant = group
        .local_grant()
        .ok_or_else(|| PyValueError::new_err("compute rate requires a live group grant"))?;
    py.allow_threads(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            tokio::time::timeout(
                Duration::from_millis(u64::from(timeout_ms)),
                crate::control::compute_rate_revision(listen, seeds, grant, calibration, update),
            )
            .await?
        })
    })
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub(super) fn parse_version(version: &str) -> PyResult<GroupVersion> {
    let invalid = || PyValueError::new_err("expected a group version as <hex id>@<epoch>");
    let (group, epoch) = version.split_once('@').ok_or_else(invalid)?;
    let group = NodeId::from_hex(group).map_err(|_| invalid())?;
    let epoch = epoch.parse::<u64>().map_err(|_| invalid())?;
    Ok(GroupVersion {
        group: GroupId(group),
        epoch,
    })
}

#[pyfunction]
pub(super) fn join_versioned_group(
    py: Python<'_>,
    address: &str,
    seeds: Vec<String>,
    version: &str,
    devices: Vec<(String, u32)>,
    slot: usize,
    device: usize,
    timeout_ms: u32,
) -> PyResult<CommHandle> {
    let (listen, seeds) = control_addresses(address, seeds)?;
    let layout = devices
        .into_iter()
        .map(|(host, ordinal)| NodeId::from_hex(&host).map(|node| DeviceId::new(node, ordinal)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let member = *layout
        .get(slot)
        .ok_or_else(|| PyValueError::new_err("slot is outside the group layout"))?;
    let version = parse_version(version)?;
    let backend = backend_from_env().map_err(PyValueError::new_err)?;
    let request = JoinGroup {
        version,
        member,
        layout,
        listen,
        seeds,
    };
    let timeout = Duration::from_millis(u64::from(timeout_ms));
    py.allow_threads(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime
            .block_on(join_committed_group(request, backend, device, timeout))
            .map(|comm| CommHandle {
                ptr: comm.into_raw(),
            })
    })
    .map_err(|error| PyRuntimeError::new_err(format!("Plait group join for {version:?}: {error}")))
}
