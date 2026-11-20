//! Real `libibverbs` transport built on the community `rdma-io` bindings.
//!
//! `rdma-io` owns the rdma-core ABI, device/PD/CQ/MR wrappers, work-request
//! builders, and completion types. This module adapts its low-level synchronous
//! API to Plait's descriptor model and adds the two-QP self-test.
//!
//! A device is required at runtime. `cargo test -p plait-rdma` reports a skip
//! when no device is present; it does not fake hardware. For Soft-RoCE,
//! configure an `rxe` link on the host (see `docs/rdma.md`).

use std::io;
use std::sync::Arc;

use plait_core::NodeId;
use plait_mem::{AlignedBuffer, MemoryKind, RegionDescriptor, RegionId};
use rdma_io::cq::CompletionQueue;
use rdma_io::device::Context;
use rdma_io::error as rdma_error;
use rdma_io::mr::{AccessFlags, MemoryRegion};
use rdma_io::qp::{QpInitAttr, QueuePair};
use rdma_io::wc::WorkCompletion;
use rdma_io::wr::{QpType, SendFlags, SendWr, Sge, WrOpcode};
use rdma_io_sys::ibverbs as ffi;
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    pub qpn: u32,
    pub lid: u16,
    pub gid: [u8; 16],
    pub gid_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteRegion {
    pub addr: u64,
    pub rkey: u32,
    pub len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    pub wr_id: u64,
    pub opcode: u32,
    pub status: u32,
}

impl Completion {
    pub fn success(&self) -> bool {
        self.status == 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelfTest {
    Skipped(String),
    Passed { device: String, bytes: usize },
}

pub struct RdmaDevice {
    ctx: Arc<Context>,
    pd: Arc<rdma_io::pd::ProtectionDomain>,
    name: String,
    port: u8,
    gid_index: i32,
    link_layer: u8,
}

impl RdmaDevice {
    /// Open the first active verbs device. Returns `Ok(None)` when the host has
    /// no RDMA device (including an unconfigured Soft-RoCE host).
    pub fn open_first() -> io::Result<Option<Self>> {
        let devices = rdma_io::device::devices().map_err(io_error)?;
        let Some(device) = devices.first() else {
            return Ok(None);
        };
        let name = device.name().to_owned();
        let ctx = Arc::new(device.open().map_err(io_error)?);
        let device_attr = ctx.query_device().map_err(io_error)?;
        let mut port = 0u8;
        let mut port_attr = None;
        for candidate in 1..=device_attr.phys_port_cnt {
            let attr = ctx.query_port(candidate).map_err(io_error)?;
            if attr.state == ffi::IBV_PORT_ACTIVE {
                port = candidate;
                port_attr = Some(attr);
                break;
            }
        }
        let Some(port_attr) = port_attr else {
            return Ok(None);
        };
        let gid_index = choose_gid(
            &ctx,
            port,
            port_attr.link_layer as u32,
            port_attr.gid_tbl_len,
        )?;
        let pd = ctx.alloc_pd().map_err(io_error)?;
        Ok(Some(Self {
            ctx,
            pd,
            name,
            port,
            gid_index,
            link_layer: port_attr.link_layer,
        }))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn gid_index(&self) -> i32 {
        self.gid_index
    }

    pub fn create_qp(&self) -> io::Result<RdmaQp> {
        let cq = CompletionQueue::new(self.ctx.clone(), 128).map_err(io_error)?;
        let init = QpInitAttr {
            qp_type: QpType::Rc,
            max_send_wr: 128,
            max_recv_wr: 128,
            max_send_sge: 1,
            max_recv_sge: 1,
            max_inline_data: 0,
            sq_sig_all: false,
        };
        let qp = self.pd.create_qp(&cq, &cq, &init).map_err(io_error)?;
        let access =
            (AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE | AccessFlags::REMOTE_READ)
                .bits();
        qp.to_init(self.port, 0, access).map_err(io_error)?;
        let port_attr = self.ctx.query_port(self.port).map_err(io_error)?;
        let gid = self
            .ctx
            .query_gid(self.port, self.gid_index)
            .map_err(io_error)?;
        let gid = unsafe { gid.raw };
        let endpoint = Endpoint {
            qpn: qp.qp_num(),
            lid: port_attr.lid,
            gid,
            gid_index: self.gid_index as u32,
        };
        Ok(RdmaQp {
            qp,
            cq,
            endpoint,
            port: self.port,
            gid_index: self.gid_index,
            link_layer: self.link_layer,
        })
    }

    /// Register an aligned staging buffer. The returned registration borrows the
    /// buffer; keep both alive for the lifetime of the transfer.
    /// Probe whether the verbs provider can register a device pointer as an
    /// MR. This is the GPUDirect gate: a successful probe means the kernel has
    /// a peer-memory/dmabuf path for this GPU, not that a full RDMA collective
    /// has been validated.
    /// # Safety
    /// `ptr` must point to a live device allocation of at least `len` bytes.
    pub unsafe fn probe_gpu_region(&self, ptr: *mut u8, len: usize) -> io::Result<()> {
        if ptr.is_null() || len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GPU region pointer/length is invalid",
            ));
        }
        // SAFETY: the caller owns a live device allocation for `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        let access =
            AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE | AccessFlags::REMOTE_READ;
        let mr = self.pd.reg_mr(slice, access).map_err(io_error)?;
        drop(mr);
        Ok(())
    }

    pub fn register<'a>(&self, buffer: &'a mut AlignedBuffer) -> io::Result<Registration<'a>> {
        if buffer.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot register an empty buffer",
            ));
        }
        let access =
            AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE | AccessFlags::REMOTE_READ;
        let mr = self.pd.reg_mr(&mut buffer[..], access).map_err(io_error)?;
        Ok(Registration { mr })
    }
}

impl std::fmt::Debug for RdmaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdmaDevice")
            .field("name", &self.name)
            .field("port", &self.port)
            .field("gid_index", &self.gid_index)
            .finish()
    }
}

pub struct RdmaQp {
    qp: QueuePair,
    cq: Arc<CompletionQueue>,
    endpoint: Endpoint,
    port: u8,
    gid_index: i32,
    link_layer: u8,
}

impl RdmaQp {
    pub fn endpoint(&self) -> Endpoint {
        self.endpoint
    }

    pub fn connect(&self, remote: Endpoint) -> io::Result<()> {
        let ah_attr = if self.link_layer as u32 == ffi::IBV_LINK_LAYER_ETHERNET {
            let mut gid = ffi::ibv_gid::default();
            gid.raw = remote.gid;
            ffi::ibv_ah_attr {
                grh: ffi::ibv_global_route {
                    dgid: gid,
                    sgid_index: self.gid_index as u8,
                    hop_limit: 255,
                    traffic_class: 0,
                    flow_label: 0,
                },
                port_num: self.port,
                is_global: 1,
                ..Default::default()
            }
        } else {
            ffi::ibv_ah_attr {
                dlid: remote.lid,
                port_num: self.port,
                is_global: 0,
                ..Default::default()
            }
        };
        let mut attr = ffi::ibv_qp_attr {
            qp_state: ffi::IBV_QPS_RTR,
            path_mtu: ffi::IBV_MTU_1024,
            dest_qp_num: remote.qpn,
            rq_psn: 0,
            max_dest_rd_atomic: 1,
            min_rnr_timer: 12,
            ah_attr,
            ..Default::default()
        };
        let rtr_mask = ffi::IBV_QP_STATE
            | ffi::IBV_QP_AV
            | ffi::IBV_QP_PATH_MTU
            | ffi::IBV_QP_DEST_QPN
            | ffi::IBV_QP_RQ_PSN
            | ffi::IBV_QP_MAX_DEST_RD_ATOMIC
            | ffi::IBV_QP_MIN_RNR_TIMER;
        self.qp.modify(&mut attr, rtr_mask).map_err(io_error)?;

        let mut attr = ffi::ibv_qp_attr {
            qp_state: ffi::IBV_QPS_RTS,
            sq_psn: 0,
            timeout: 14,
            retry_cnt: 7,
            rnr_retry: 7,
            max_rd_atomic: 1,
            ..Default::default()
        };
        let rts_mask = ffi::IBV_QP_STATE
            | ffi::IBV_QP_TIMEOUT
            | ffi::IBV_QP_RETRY_CNT
            | ffi::IBV_QP_RNR_RETRY
            | ffi::IBV_QP_SQ_PSN
            | ffi::IBV_QP_MAX_QP_RD_ATOMIC;
        self.qp.modify(&mut attr, rts_mask).map_err(io_error)?;
        Ok(())
    }

    pub fn post_write(
        &self,
        local: &Registration<'_>,
        offset: usize,
        len: usize,
        remote: &RemoteRegion,
        wr_id: u64,
    ) -> io::Result<()> {
        validate_remote(remote, len)?;
        let sge = local.sge(offset, len)?;
        let mut wr = SendWr::new(wr_id, WrOpcode::RdmaWrite)
            .flags(SendFlags::SIGNALED)
            .sg(sge)
            .rdma(remote.addr, remote.rkey);
        self.qp.post_send(&mut wr).map_err(io_error)
    }

    pub fn post_read(
        &self,
        local: &Registration<'_>,
        offset: usize,
        len: usize,
        remote: &RemoteRegion,
        wr_id: u64,
    ) -> io::Result<()> {
        validate_remote(remote, len)?;
        let sge = local.sge(offset, len)?;
        let mut wr = SendWr::new(wr_id, WrOpcode::RdmaRead)
            .flags(SendFlags::SIGNALED)
            .sg(sge)
            .rdma(remote.addr, remote.rkey);
        self.qp.post_send(&mut wr).map_err(io_error)
    }

    pub fn post_write_descriptor(
        &self,
        local: &Registration<'_>,
        offset: usize,
        len: usize,
        remote: &RegionDescriptor,
        wr_id: u64,
    ) -> io::Result<()> {
        let remote = RemoteRegion {
            addr: remote.addr + remote.offset,
            rkey: remote.rkey as u32,
            len: remote.len,
        };
        self.post_write(local, offset, len, &remote, wr_id)
    }

    pub fn post_read_descriptor(
        &self,
        local: &Registration<'_>,
        offset: usize,
        len: usize,
        remote: &RegionDescriptor,
        wr_id: u64,
    ) -> io::Result<()> {
        let remote = RemoteRegion {
            addr: remote.addr + remote.offset,
            rkey: remote.rkey as u32,
            len: remote.len,
        };
        self.post_read(local, offset, len, &remote, wr_id)
    }

    pub fn poll(&self, timeout_ms: u32) -> io::Result<Option<Completion>> {
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms as u64);
        loop {
            let mut completions = [WorkCompletion::default(); 1];
            let count = self.cq.poll(&mut completions).map_err(io_error)?;
            if count > 0 {
                let wc = completions[0];
                return Ok(Some(Completion {
                    wr_id: wc.wr_id(),
                    opcode: wc.as_raw().opcode,
                    status: wc.status_raw(),
                }));
            }
            if std::time::Instant::now() >= deadline {
                return Ok(None);
            }
            std::hint::spin_loop();
        }
    }
}

pub struct Registration<'a> {
    mr: MemoryRegion<'a>,
}

impl Registration<'_> {
    pub fn lkey(&self) -> u32 {
        self.mr.lkey()
    }

    pub fn rkey(&self) -> u32 {
        self.mr.rkey()
    }

    pub fn addr(&self) -> u64 {
        self.mr.addr() as u64
    }

    pub fn len(&self) -> u64 {
        self.mr.length() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.mr.length() == 0
    }

    pub fn remote(&self, offset: u64, len: u64) -> io::Result<RemoteRegion> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "remote range overflow"))?;
        if end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "remote range exceeds the registered region",
            ));
        }
        Ok(RemoteRegion {
            addr: self.addr() + offset,
            rkey: self.rkey(),
            len,
        })
    }

    pub fn descriptor(
        &self,
        node: NodeId,
        offset: u64,
        len: u64,
        kind: MemoryKind,
    ) -> io::Result<RegionDescriptor> {
        let remote = self.remote(offset, len)?;
        Ok(RegionDescriptor {
            node,
            region: RegionId(self.addr()),
            addr: remote.addr,
            offset: 0,
            len,
            rkey: remote.rkey as u64,
            kind,
        })
    }

    pub fn read(&self, offset: usize, out: &mut [u8]) -> io::Result<()> {
        let end = offset
            .checked_add(out.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > self.len() as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read exceeds the registered region",
            ));
        }
        // SAFETY: the registration owns a valid mapping of `len` bytes; the
        // bounds check above keeps the read inside it.
        let slice = unsafe { std::slice::from_raw_parts(self.mr.addr().add(offset), out.len()) };
        out.copy_from_slice(slice);
        Ok(())
    }

    fn sge(&self, offset: usize, len: usize) -> io::Result<Sge> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
        if end > self.len() as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "transfer exceeds the registered region",
            ));
        }
        Ok(Sge::new(
            self.addr() + offset as u64,
            len as u32,
            self.lkey(),
        ))
    }
}

fn choose_gid(ctx: &Context, port: u8, link_layer: u32, gid_tbl_len: i32) -> io::Result<i32> {
    if link_layer != ffi::IBV_LINK_LAYER_ETHERNET || gid_tbl_len <= 0 {
        return Ok(0);
    }
    for index in 0..gid_tbl_len {
        let Ok(gid) = ctx.query_gid(port, index) else {
            continue;
        };
        // SAFETY: GID is a 16-byte union.
        let raw = unsafe { gid.raw };
        if raw.iter().any(|byte| *byte != 0) {
            return Ok(index);
        }
    }
    Ok(0)
}

fn validate_remote(remote: &RemoteRegion, len: usize) -> io::Result<()> {
    if len as u64 > remote.len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "operation length exceeds the remote region",
        ));
    }
    Ok(())
}

fn io_error(error: rdma_error::Error) -> io::Error {
    match error {
        rdma_error::Error::Verbs(error) => error,
        other => io::Error::other(other.to_string()),
    }
}

/// Exercise real two-QP loopback RDMA when a device is present.
///
/// The caller can run this on a Soft-RoCE host with:
/// `sudo rdma link add rxe0 type rxe netdev <iface>`.
pub fn selftest() -> io::Result<SelfTest> {
    let Some(device) = RdmaDevice::open_first()? else {
        return Ok(SelfTest::Skipped(
            "no verbs device; configure Soft-RoCE or an RNIC".to_owned(),
        ));
    };
    let qp_a = device.create_qp()?;
    let qp_b = device.create_qp()?;
    qp_a.connect(qp_b.endpoint())?;
    qp_b.connect(qp_a.endpoint())?;

    let mut buffer_a = AlignedBuffer::new(4096, 4096)?;
    let mut buffer_b = AlignedBuffer::new(4096, 4096)?;
    for (index, byte) in buffer_a.iter_mut().take(64).enumerate() {
        *byte = (index as u8).wrapping_mul(7).wrapping_add(3);
    }
    let region_a = device.register(&mut buffer_a)?;
    let region_b = device.register(&mut buffer_b)?;

    let node = NodeId::derive("rdma-selftest");
    let remote_b = region_b.descriptor(node, 0, 64, MemoryKind::Host)?;
    qp_a.post_write_descriptor(&region_a, 0, 64, &remote_b, 1)?;
    let completion = wait(&qp_a, 1)?;
    if !completion.success() {
        return Err(io::Error::other(format!(
            "RDMA write failed with status {}",
            completion.status
        )));
    }
    let expected: Vec<u8> = (0..64u8)
        .map(|index| index.wrapping_mul(7).wrapping_add(3))
        .collect();
    let mut written = [0u8; 64];
    region_b.read(0, &mut written)?;
    if written != expected[..] {
        return Err(io::Error::other("RDMA write data mismatch"));
    }

    qp_a.post_read_descriptor(&region_a, 128, 64, &remote_b, 2)?;
    let completion = wait(&qp_a, 2)?;
    if !completion.success() {
        return Err(io::Error::other(format!(
            "RDMA read failed with status {}",
            completion.status
        )));
    }
    let mut read_back = [0u8; 64];
    region_a.read(128, &mut read_back)?;
    if read_back != expected[..] {
        return Err(io::Error::other("RDMA read data mismatch"));
    }

    Ok(SelfTest::Passed {
        device: device.name().to_owned(),
        bytes: 128,
    })
}

fn wait(qp: &RdmaQp, wr_id: u64) -> io::Result<Completion> {
    for _ in 0..5_000 {
        if let Some(completion) = qp.poll(1)? {
            if completion.wr_id == wr_id {
                return Ok(completion);
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        format!("RDMA completion {wr_id} did not arrive"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbs_selftest_or_skip() {
        match selftest() {
            Ok(SelfTest::Skipped(reason)) => eprintln!("skipping verbs test: {reason}"),
            Ok(SelfTest::Passed { device, bytes }) => {
                eprintln!("verbs loopback passed on {device} ({bytes} bytes)")
            }
            Err(error) => panic!("verbs self-test failed: {error}"),
        }
    }
}
