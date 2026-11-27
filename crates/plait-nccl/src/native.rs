//! C entry points for the device-identity bootstrap.

use std::ffi::{CStr, c_char};
use std::mem::size_of;
use std::ptr;
use std::time::Duration;

use plait_core::group::{GroupId, GroupVersion, WorkerGrant};
use plait_core::{DeviceId, NodeId};

use crate::{
    BackendKind, Comm, NCCL_INVALID_ARGUMENT, NCCL_SUCCESS, NCCL_UNSUPPORTED, NcclResult,
    PeerEndpoint, RanklessListener, RanklessMember, set_error,
};

const ADDRESS_BYTES: usize = 80;

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
