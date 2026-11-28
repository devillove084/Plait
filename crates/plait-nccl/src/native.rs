//! C entry points for the device-identity bootstrap.

use std::ffi::{CStr, c_char, c_void};
use std::io;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Weak};
use std::time::Duration;

use plait_core::group::{GroupFault, GroupId, GroupVersion, WorkerGrant};
use plait_core::{DeviceId, NodeId};

use crate::peer::requests::{Completion, Dispatcher};
use crate::profile::Phases;
use crate::{
    BackendKind, Collective, Comm, DataType, NCCL_INVALID_ARGUMENT, NCCL_SUCCESS,
    NCCL_UNHANDLED_CUDA_ERROR, NCCL_UNSUPPORTED, NcclResult, PeerEndpoint, PeerGroup,
    RanklessListener, RanklessMember, ReduceOp, Transport, set_error, stage_peer_input,
    synchronize_stream,
};

const ADDRESS_BYTES: usize = 80;
pub const PLAIT_PEER_BUSY: NcclResult = 5;

#[repr(C)]
pub struct PlaitPeerCollective {
    pub(crate) kind: i32,
    pub(crate) datatype: i32,
    pub(crate) reduction: i32,
    pub(crate) root: i32,
    pub(crate) count: usize,
}

impl PlaitPeerCollective {
    fn plan(&self, comm: &Comm) -> Option<(Collective, usize, usize)> {
        let dtype = DataType::from_raw(self.datatype)?;
        let width = self.count.checked_mul(dtype.bytes())?;
        let root = || {
            usize::try_from(self.root)
                .ok()
                .filter(|&root| root < comm.world)
        };
        let reduction = || ReduceOp::from_raw(self.reduction);
        Some(match self.kind {
            1 => (Collective::AllReduce(dtype, reduction()?), width, width),
            2 => (
                Collective::AllGather(dtype),
                width,
                width.checked_mul(comm.world)?,
            ),
            3 => (
                Collective::ReduceScatter(dtype, reduction()?),
                width.checked_mul(comm.world)?,
                width,
            ),
            4 => (Collective::Broadcast(dtype, root()?), width, width),
            5 => {
                let dest = root()?;
                (
                    Collective::Reduce(dtype, reduction()?, dest),
                    width,
                    if comm.rank == dest { width } else { 0 },
                )
            }
            _ => return None,
        })
    }
}

pub struct PlaitPeerRequest {
    owner: Weak<PeerGroup>,
    completion: Completion,
    output_bytes: usize,
}

impl PlaitPeerRequest {
    pub(crate) fn submit(
        comm: &Comm,
        operation: &PlaitPeerCollective,
        send: u64,
        stream: u64,
    ) -> Result<Self, NcclResult> {
        let Transport::Peers(group) = &comm.transport else {
            return Err(NCCL_UNSUPPORTED);
        };
        let Some((op, input_bytes, output_bytes)) = operation.plan(comm) else {
            return Err(NCCL_INVALID_ARGUMENT);
        };
        if op.reads_input(comm.rank) && input_bytes != 0 && send == 0 {
            return Err(NCCL_INVALID_ARGUMENT);
        }
        group.live().map_err(|error| set_error(error.to_string()))?;
        let staging_bytes = input_bytes
            .checked_add(output_bytes)
            .ok_or(NCCL_INVALID_ARGUMENT)?;
        let admission = group.requests.admit(staging_bytes).ok_or(PLAIT_PEER_BUSY)?;
        let input = stage_peer_input(
            comm,
            group,
            op,
            send as *const c_void,
            input_bytes,
            stream as *mut c_void,
            &mut Phases::new(false),
        )?;
        let completion = Dispatcher::submit(group, op, input, admission).map_err(|error| {
            if error.kind() == io::ErrorKind::WouldBlock {
                PLAIT_PEER_BUSY
            } else {
                set_error(error.to_string())
            }
        })?;
        Ok(Self {
            owner: Arc::downgrade(group),
            completion,
            output_bytes,
        })
    }

    pub(crate) fn ready(&mut self) -> bool {
        self.completion.ready()
    }

    pub(crate) fn destination<'a>(
        &self,
        comm: &'a Comm,
        recv: u64,
        recv_bytes: usize,
    ) -> Result<&'a PeerGroup, NcclResult> {
        let Transport::Peers(group) = &comm.transport else {
            return Err(NCCL_UNSUPPORTED);
        };
        if !self.owner.ptr_eq(&Arc::downgrade(group))
            || self.output_bytes != recv_bytes
            || (recv_bytes != 0 && recv == 0)
        {
            return Err(NCCL_INVALID_ARGUMENT);
        }
        Ok(group)
    }

    pub(crate) fn deliver(
        self,
        comm: &Comm,
        group: &PeerGroup,
        recv: u64,
        stream: u64,
    ) -> Result<(), NcclResult> {
        let output = self
            .completion
            .wait()
            .map_err(|error| set_error(error.to_string()))?;
        if output.bytes.len() != self.output_bytes {
            group.fail(GroupFault::Transport);
            return Err(set_error(
                "peer collective returned an unexpected output size",
            ));
        }
        group.live().map_err(|error| set_error(error.to_string()))?;
        if output.bytes.is_empty() {
            return Ok(());
        }
        synchronize_stream(comm, stream as *mut c_void)?;
        group.live().map_err(|error| set_error(error.to_string()))?;
        comm.runtime
            .copy_host_to_device(recv, &output.bytes)
            .map_err(|error| comm.device_error(error, NCCL_UNHANDLED_CUDA_ERROR))
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PlaitPeerGroupVersion {
    group: [u8; 16],
    epoch: u64,
}

impl From<PlaitPeerGroupVersion> for GroupVersion {
    fn from(value: PlaitPeerGroupVersion) -> Self {
        Self {
            group: GroupId(NodeId::from_bytes(value.group)),
            epoch: value.epoch,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PlaitPeerDevice {
    node: [u8; 16],
    ordinal: u32,
}

impl From<PlaitPeerDevice> for DeviceId {
    fn from(value: PlaitPeerDevice) -> Self {
        Self::new(NodeId::from_bytes(value.node), value.ordinal)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PlaitPeerGrant {
    device: PlaitPeerDevice,
    incarnation: u64,
    fence: u64,
}

impl PlaitPeerGrant {
    fn scoped(self, version: GroupVersion) -> WorkerGrant {
        WorkerGrant {
            version,
            device: self.device.into(),
            incarnation: self.incarnation,
            fence: self.fence,
        }
    }
}

#[repr(C)]
pub struct PlaitPeerMember {
    version: PlaitPeerGroupVersion,
    grant: PlaitPeerGrant,
    address: [u8; ADDRESS_BYTES],
    ticket: [u8; 16],
}

impl PlaitPeerMember {
    fn decode(&self) -> Option<RanklessMember> {
        let address = CStr::from_bytes_until_nul(&self.address)
            .ok()?
            .to_str()
            .ok()?
            .parse()
            .ok()?;
        Some(RanklessMember {
            grant: self.grant.scoped(self.version.into()),
            endpoint: PeerEndpoint {
                address,
                ticket: self.ticket,
            },
        })
    }
}

/// # Safety
/// Input pointers refer to initialized values. The address is NUL terminated.
/// Outputs refer to separate writable values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerListen(
    address: *const c_char,
    version: *const PlaitPeerGroupVersion,
    grant: *const PlaitPeerGrant,
    ticket: *const [u8; 16],
    listener: *mut *mut RanklessListener,
    member: *mut PlaitPeerMember,
) -> NcclResult {
    if address.is_null()
        || version.is_null()
        || grant.is_null()
        || ticket.is_null()
        || listener.is_null()
        || member.is_null()
    {
        return NCCL_INVALID_ARGUMENT;
    }
    let Ok(address) = unsafe { CStr::from_ptr(address) }.to_str() else {
        return NCCL_INVALID_ARGUMENT;
    };
    let Ok(address) = address.parse() else {
        return NCCL_INVALID_ARGUMENT;
    };
    let (version, grant, ticket) = unsafe { (*version, *grant, *ticket) };
    let bound = match RanklessListener::bind(address, ticket, grant.scoped(version.into())) {
        Ok(bound) => bound,
        Err(error) => return set_error(error.to_string()),
    };
    let mut advertised = [0; ADDRESS_BYTES];
    let endpoint = bound.member().endpoint.address.to_string();
    let Some(target) = advertised
        .get_mut(..endpoint.len())
        .filter(|target| target.len() < ADDRESS_BYTES)
    else {
        return NCCL_INVALID_ARGUMENT;
    };
    target.copy_from_slice(endpoint.as_bytes());
    unsafe {
        member.write(PlaitPeerMember {
            version,
            grant,
            address: advertised,
            ticket,
        });
        listener.write(Box::into_raw(Box::new(bound)));
    }
    NCCL_SUCCESS
}

/// # Safety
/// A non-null listener must be live and exclusively owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerSetReplicatedEgressBytes(
    listener: *mut RanklessListener,
    bytes: u64,
) -> NcclResult {
    let Some(listener) = (unsafe { listener.as_mut() }) else {
        return NCCL_INVALID_ARGUMENT;
    };
    listener.set_replicated_egress_bytes(bytes);
    NCCL_SUCCESS
}

/// # Safety
/// The listener is exclusively owned. The manifest refers to `count` initialized
/// members. The output does not alias the listener or manifest.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerConnect(
    listener: *mut *mut RanklessListener,
    members: *const PlaitPeerMember,
    count: usize,
    backend: i32,
    device: u32,
    timeout_ms: u32,
    comm: *mut *mut Comm,
) -> NcclResult {
    if listener.is_null()
        || members.is_null()
        || comm.is_null()
        || timeout_ms == 0
        || count == 0
        || count > (i32::MAX as usize).min(isize::MAX as usize / size_of::<PlaitPeerMember>())
    {
        return NCCL_INVALID_ARGUMENT;
    }
    let backend = match backend {
        0 => BackendKind::Cuda,
        1 => BackendKind::Rocm,
        2 => BackendKind::Huawei,
        3 => BackendKind::Ppu,
        _ => return NCCL_INVALID_ARGUMENT,
    };
    let pointer = unsafe { *listener };
    if pointer.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(manifest) = (unsafe { std::slice::from_raw_parts(members, count) })
        .iter()
        .map(PlaitPeerMember::decode)
        .collect::<Option<Vec<_>>>()
    else {
        return NCCL_INVALID_ARGUMENT;
    };
    let bound = unsafe {
        listener.write(ptr::null_mut());
        *Box::from_raw(pointer)
    };
    match Comm::connect_rankless(
        bound,
        manifest,
        backend,
        device as usize,
        Duration::from_millis(timeout_ms.into()),
    ) {
        Ok(handle) => {
            unsafe {
                comm.write(handle.into_raw());
            }
            NCCL_SUCCESS
        }
        Err(error) => set_error(error),
    }
}

/// # Safety
/// The non-null listener is live and exclusively owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerListenerDestroy(listener: *mut RanklessListener) -> NcclResult {
    if !listener.is_null() {
        unsafe {
            drop(Box::from_raw(listener));
        }
    }
    NCCL_SUCCESS
}

/// # Safety
/// The communicator remains live and the device and slot pointers are valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerDeviceSlot(
    comm: *const Comm,
    device: *const PlaitPeerDevice,
    slot: *mut i32,
) -> NcclResult {
    if comm.is_null() || device.is_null() || slot.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(value) = unsafe { &*comm }.peer_slot(unsafe { *device }.into()) else {
        return NCCL_INVALID_ARGUMENT;
    };
    unsafe {
        slot.write(value as i32);
    }
    NCCL_SUCCESS
}

/// # Safety
/// The caller keeps the communicator alive through cancellation and until
/// every concurrent operation on it has returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerCancel(comm: *const Comm) -> NcclResult {
    let Some(comm) = (unsafe { comm.as_ref() }) else {
        return NCCL_INVALID_ARGUMENT;
    };
    if comm.cancel_peers() {
        NCCL_SUCCESS
    } else {
        NCCL_UNSUPPORTED
    }
}

/// # Safety
/// The communicator remains live and the output points to writable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerFailed(comm: *const Comm, failed: *mut i32) -> NcclResult {
    if comm.is_null() || failed.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(value) = unsafe { &*comm }.peers_failed() else {
        return NCCL_UNSUPPORTED;
    };
    unsafe {
        failed.write(i32::from(value));
    }
    NCCL_SUCCESS
}

/// # Safety
/// The communicator remains live and `digest` points to 32 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerManifestDigest(
    comm: *const Comm,
    digest: *mut [u8; 32],
) -> NcclResult {
    if comm.is_null() || digest.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let Some(value) = unsafe { &*comm }.peer_manifest_digest() else {
        return NCCL_UNSUPPORTED;
    };
    unsafe {
        digest.write(value);
    }
    NCCL_SUCCESS
}

/// # Safety
/// The communicator and descriptor are live, and the input device allocation
/// and producer stream remain valid for this call. The output is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerCollectiveSubmit(
    comm: *const Comm,
    operation: *const PlaitPeerCollective,
    send: *const c_void,
    stream: *mut c_void,
    request: *mut *mut PlaitPeerRequest,
) -> NcclResult {
    let (Some(comm), Some(operation)) = (unsafe { (comm.as_ref(), operation.as_ref()) }) else {
        return NCCL_INVALID_ARGUMENT;
    };
    if request.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    let request_value = match PlaitPeerRequest::submit(comm, operation, send as u64, stream as u64)
    {
        Ok(request) => request,
        Err(error) => return error,
    };
    unsafe { request.write(Box::into_raw(Box::new(request_value))) };
    NCCL_SUCCESS
}

/// # Safety
/// The request is live and exclusively accessed. The output is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerRequestTest(
    request: *mut PlaitPeerRequest,
    ready: *mut i32,
) -> NcclResult {
    let Some(request) = (unsafe { request.as_mut() }) else {
        return NCCL_INVALID_ARGUMENT;
    };
    if ready.is_null() {
        return NCCL_INVALID_ARGUMENT;
    }
    unsafe { ready.write(i32::from(request.ready())) };
    NCCL_SUCCESS
}

/// # Safety
/// The request and communicator remain live until consumed. The request is
/// exclusively owned. The output allocation and consumer stream are valid
/// throughout the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerRequestWait(
    comm: *const Comm,
    request: *mut *mut PlaitPeerRequest,
    recv: *mut c_void,
    recv_bytes: usize,
    stream: *mut c_void,
) -> NcclResult {
    let (Some(comm), Some(request)) = (unsafe { (comm.as_ref(), request.as_mut()) }) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let Some(pending) = (unsafe { request.as_ref() }) else {
        return NCCL_INVALID_ARGUMENT;
    };
    let group = match pending.destination(comm, recv as u64, recv_bytes) {
        Ok(group) => group,
        Err(error) => return error,
    };
    let request_value = unsafe { *Box::from_raw(std::mem::replace(request, ptr::null_mut())) };
    request_value
        .deliver(comm, group, recv as u64, stream as u64)
        .map_or_else(|error| error, |_| NCCL_SUCCESS)
}

/// # Safety
/// A non-null request is live and exclusively owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn plaitPeerRequestDestroy(request: *mut PlaitPeerRequest) -> NcclResult {
    if !request.is_null() {
        unsafe { drop(Box::from_raw(request)) };
    }
    NCCL_SUCCESS
}
