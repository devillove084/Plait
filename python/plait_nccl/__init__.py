"""Python API for the Plait NCCL-compatible collective backend.

This module is the Python-facing half of the Rust backend. It deliberately
mirrors the small surface vLLM's ``pynccl_wrapper.NCCLLibrary`` uses, but the
data path runs through PyO3 objects rather than ctypes/dlopen.

Typical use:

    import plait_nccl
    plait_nccl.patch_vllm()          # before importing vLLM communicators

The Rust extension is named ``_plait_nccl`` by the maturin/Cargo build.
"""

from __future__ import annotations

import atexit
import os
import secrets
import time
from typing import Any

from . import _plait_nccl
from .router import CapabilityRouter, Tier, default_tiers

SUCCESS = 0
VERSION = 23004


class ncclUniqueId:
    def __init__(self) -> None:
        self.internal = bytearray(128)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ncclUniqueId":
        if len(data) != 128:
            raise ValueError(f"expected 128 bytes, got {len(data)}")
        value = cls()
        value.internal[:] = data
        return value


class CommHandle:
    """Thin Python view over the Rust communicator."""

    def __init__(self, handle: "_plait_nccl.CommHandle") -> None:
        self._handle = handle
        self._rank = handle.rank
        self._world = handle.world

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world(self) -> int:
        return self._world

    @property
    def peer_failed(self) -> bool | None:
        """Report local peer communicator failure, including grant revocation."""
        return self._handle.peer_failed

    def compute_rate_revision(self, host: str, seeds: list[str], calibration: str) -> int:
        """Read the measurement revision for this communicator's device."""
        address = f"[{host}]:0" if ":" in host else f"{host}:0"
        return self._handle.compute_rate(address, seeds, calibration)

    def publish_compute_rate(
        self, host: str, seeds: list[str], calibration: str,
        expected_revision: int, operations_per_second: int,
    ) -> int:
        """Publish a measurement under this communicator's device grant."""
        address = f"[{host}]:0" if ":" in host else f"{host}:0"
        return self._handle.compute_rate(
            address, seeds, calibration, (expected_revision, operations_per_second),
        )

    def all_reduce(self, send: int, recv: int, count: int, dtype: int, op: int) -> int:
        return self._handle.all_reduce(send, recv, count, dtype, op)

    def all_reduce_stream(
        self, send: int, recv: int, count: int, dtype: int, op: int, stream: int
    ) -> int:
        return self._handle.all_reduce_stream(send, recv, count, dtype, op, stream)

    def all_reduce_fp8_to_f32(
        self, send: int, recv: int, count: int
    ) -> int:
        return self._handle.all_reduce_fp8_to_f32(send, recv, count)

    def all_gather(self, send: int, recv: int, count: int, dtype: int, stream: int = 0) -> int:
        return self._handle.all_gather(send, recv, count, dtype, stream)

    def broadcast(self, send: int, recv: int, count: int, dtype: int, root: int, stream: int = 0) -> int:
        return self._handle.broadcast(send, recv, count, dtype, root, stream)

    def reduce_scatter(self, send: int, recv: int, count: int, dtype: int, op: int, stream: int = 0) -> int:
        return self._handle.reduce_scatter(send, recv, count, dtype, op, stream)

    def reduce(self, send: int, recv: int, count: int, dtype: int, op: int, root: int, stream: int = 0) -> int:
        return self._handle.reduce(send, recv, count, dtype, op, root, stream)

    def send(self, ptr: int, count: int, dtype: int, dest: int, stream: int = 0) -> int:
        return self._handle.send(ptr, count, dtype, dest, stream)

    def recv(self, ptr: int, count: int, dtype: int, src: int, stream: int = 0) -> int:
        return self._handle.recv(ptr, count, dtype, src, stream)


def _port() -> int:
    return int(os.environ.get("PLAIT_NCCL_PORT", 25000 + os.getpid() % 2000))


def _check(result: int) -> None:
    if result != 0:
        take_error = getattr(_plait_nccl, "take_last_error", None)
        detail = take_error() if take_error is not None else ""
        raise RuntimeError(f"Plait NCCL error {result}" + (f": {detail}" if detail else ""))


def _ptr(value: Any) -> int:
    """Convert ctypes pointer-like values to an integer address."""
    if value is None:
        return 0
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, "little")
    return int(value or 0)


class NCCLLibrary:
    """Drop-in replacement for ``vllm...pynccl_wrapper.NCCLLibrary``."""

    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, Any] = {}

    def __init__(self, so_file: str | None = None, *, device: int | None = None) -> None:
        self.so_file = so_file
        self.device = device
        self._comms: set[CommHandle] = set()

    def NCCL_CHECK(self, result: int) -> None:
        _check(result)

    def has_symbol(self, name: str) -> bool:
        return callable(getattr(self, name, None))

    def ncclGetRawVersion(self) -> int:
        return VERSION

    def ncclGetVersion(self) -> str:
        return "2.30.4"

    def ncclGetErrorString(self, result: int) -> str:
        return {
            0: "success",
            1: "unhandled CUDA error",
            2: "invalid argument",
            3: "invalid usage",
            4: "unsupported by Plait backend",
        }.get(result, "unknown NCCL error")

    def ncclGetUniqueId(self) -> ncclUniqueId:
        return ncclUniqueId.from_bytes(_plait_nccl.unique_id(_port()))

    def unique_id_from_bytes(self, data: bytes) -> ncclUniqueId:
        return ncclUniqueId.from_bytes(data)

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId, rank: int) -> CommHandle:
        raw = bytes(unique_id.internal)
        port = int.from_bytes(raw[4:6], "little") if raw[:4] == b"PLNC" else _port()
        # A vLLM worker can belong to a subgroup whose rank is not its device index.
        device = self.device
        if device is None:
            device = int(os.environ.get("PLAIT_NCCL_DEVICE", os.environ.get("LOCAL_RANK", rank)))
        handle = _plait_nccl.CommHandle(rank, world_size, device, port)
        comm = CommHandle(handle)
        self._comms.add(comm)
        return comm

    def connect_peers(self, host: str, slot: int, exchange_endpoints) -> CommHandle:
        """Exchange per-device endpoints, then connect the peer communicator."""
        address = f"[{host}]:0" if ":" in host else f"{host}:0"
        ticket = secrets.token_bytes(16)
        listener = None
        try:
            listener = _plait_nccl.PeerListener(
                address, ticket, os.environ.get("PLAIT_TP_GROUP_VERSION")
            )
            local = (listener.endpoint(), ticket)
        except (AttributeError, OSError, ValueError) as exc:
            local = str(exc)
        members = exchange_endpoints(local)
        for peer, member in enumerate(members):
            if isinstance(member, str):
                raise RuntimeError(f"Plait peer {peer} failed to bind: {member}")
        if listener is None:
            raise RuntimeError("local Plait peer listener failed to bind")
        timeout_ms = int(os.environ.get("PLAIT_NCCL_TIMEOUT_MS", "120000"))
        device = self.device
        if device is None:
            device = int(os.environ.get("PLAIT_NCCL_DEVICE", os.environ.get("LOCAL_RANK", slot)))
        comm = CommHandle(listener.connect(members, slot, device, timeout_ms))
        self._comms.add(comm)
        return comm

    def join_versioned_group(
        self,
        host: str,
        seeds: list[str],
        version: str,
        devices: list[tuple[str, int]],
        slot: int,
    ) -> CommHandle:
        """Join committed devices in the caller's logical collective order."""
        address = f"[{host}]:0" if ":" in host else f"{host}:0"
        timeout_ms = int(os.environ.get("PLAIT_NCCL_TIMEOUT_MS", "120000"))
        if self.device is None:
            raise ValueError("a versioned group requires the local accelerator index")
        comm = CommHandle(_plait_nccl.join_versioned_group(
            address, seeds, version, devices, slot, self.device, timeout_ms,
        ))
        self._comms.add(comm)
        return comm

    def ncclAllReduce(self, sendbuff, recvbuff, count: int, datatype: int, op: int, comm: CommHandle, stream) -> None:
        send = _ptr(sendbuff)
        recv = _ptr(recvbuff)
        cstream = _ptr(stream)
        handle = comm._handle
        if _TIMING_ENABLED:
            start_wall = time.time_ns()
            start = time.perf_counter_ns()
            try:
                result = handle.all_reduce_stream(
                    send, recv, count, datatype, op, cstream
                )
            finally:
                _record_allreduce(
                    comm._rank, start_wall, time.perf_counter_ns() - start, count, datatype
                )
        else:
            result = handle.all_reduce_stream(send, recv, count, datatype, op, cstream)
        _check(result)

    def ncclAllGather(self, sendbuff, recvbuff, count: int, datatype: int, comm: CommHandle, stream) -> None:
        _check(comm.all_gather(_ptr(sendbuff), _ptr(recvbuff), count, datatype, _ptr(stream)))

    def ncclReduceScatter(self, sendbuff, recvbuff, count: int, datatype: int, op: int, comm: CommHandle, stream) -> None:
        _check(comm.reduce_scatter(_ptr(sendbuff), _ptr(recvbuff), count, datatype, op, _ptr(stream)))

    def ncclReduce(self, sendbuff, recvbuff, count: int, datatype: int, op: int, root: int, comm: CommHandle, stream) -> None:
        _check(comm.reduce(_ptr(sendbuff), _ptr(recvbuff), count, datatype, op, root, _ptr(stream)))

    def ncclBroadcast(self, sendbuff, recvbuff, count: int, datatype: int, root: int, comm: CommHandle, stream) -> None:
        _check(comm.broadcast(_ptr(sendbuff), _ptr(recvbuff), count, datatype, root, _ptr(stream)))

    def ncclSend(self, sendbuff, count: int, datatype: int, dest: int, comm: CommHandle, stream) -> None:
        _check(comm.send(_ptr(sendbuff), count, datatype, dest, _ptr(stream)))

    def ncclRecv(self, recvbuff, count: int, datatype: int, src: int, comm: CommHandle, stream) -> None:
        _check(comm.recv(_ptr(recvbuff), count, datatype, src, _ptr(stream)))

    def ncclCommDestroy(self, comm: CommHandle) -> None:
        abort = getattr(comm._handle, "abort", None)
        if abort is not None:
            abort()
        self._comms.discard(comm)

    def ncclCommAbort(self, comm: CommHandle) -> None:
        self.ncclCommDestroy(comm)

    def ncclGroupStart(self) -> None:
        return None

    def ncclGroupEnd(self) -> None:
        return None

    def ncclCommWindowRegister(self, comm: CommHandle, buff, size: int, win_flags: int):
        return buff

    def ncclCommWindowDeregister(self, comm: CommHandle, window) -> None:
        return None

    def ncclCommQueryProperties(self, comm: CommHandle, properties) -> None:
        return None


def patch_vllm(disable_custom_allreduce: bool = True) -> None:
    """Install this backend into vLLM's pynccl modules.

    Call before ``PyNcclCommunicator`` is constructed. Both the wrapper module
    and the communicator module bind ``NCCLLibrary`` by name, so both attributes
    are replaced. vLLM's CUDA custom all-reduce and NCCL symmetric-memory
    dispatch are disabled by default: they bypass the replaced library, so
    leaving them enabled would silently keep NVIDIA-only data paths.
    """
    os.environ.setdefault("VLLM_USE_NCCL_SYMM_MEM", "0")
    os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
    from vllm.distributed.device_communicators import pynccl, pynccl_wrapper

    pynccl_wrapper.NCCLLibrary = NCCLLibrary
    pynccl.NCCLLibrary = NCCLLibrary
    if disable_custom_allreduce:
        from vllm.distributed import parallel_state

        parallel_state.set_custom_all_reduce(False)




def preferred_linear_backend(device: int | None = None, quantization: str | None = None) -> str:
    """Pick a vLLM quantized-linear backend for this GPU and checkpoint.

    SM120 (NVIDIA Blackwell) gets Humming, including NVFP4. Ampere keeps
    Marlin for W4A16 checkpoints and Cutlass for FP8, so a heterogeneous
    deployment can use the same checkpoint on both host groups.
    """
    import torch

    if not torch.cuda.is_available():
        return "auto"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) >= (12, 0):
        return "humming"
    if quantization in {"awq", "gptq", "awq_marlin", "gptq_marlin"}:
        return "marlin"
    if (major, minor) >= (8, 9):
        return "cutlass"
    return "cutlass"


def kernel_config(device: int | None = None, quantization: str | None = None):
    """Return a vLLM ``KernelConfig`` tuned for this GPU.

    Pass it as ``LLM(..., kernel_config=kernel_config(...))``. The allreduce
    path is unchanged: quantized weights still produce bf16/fp16 activations,
    which Plait's NCCL kernel already handles.
    """
    from vllm.config import KernelConfig

    backend = preferred_linear_backend(device, quantization)
    return KernelConfig(linear_backend=backend, moe_backend=backend)

__all__ = [
    "NCCLLibrary",
    "CommHandle",
    "Group",
    "join_group",
    "ncclUniqueId",
    "patch_vllm",
    "patch_vllm_control",
    "preferred_linear_backend",
    "kernel_config",
    "CapabilityRouter",
    "Tier",
    "default_tiers",
]


class Group:
    """Rankless group handle.

    The root creates the group and shares `unique_id` out of band (for vLLM,
    through its existing store). Workers join with `join()`; rank assignment
    happens at the rendezvous, so callers do not construct a rank table.
    """

    def __init__(self, world: int, port: int | None = None) -> None:
        self.world = world
        self.port = _port() if port is None else port
        self.unique_id = _plait_nccl.new_group(self.world, self.port)

    @classmethod
    def from_id(cls, unique_id: bytes) -> "Group":
        value = cls.__new__(cls)
        value.world = int.from_bytes(unique_id[6:10], "little")
        value.port = int.from_bytes(unique_id[4:6], "little")
        value.unique_id = unique_id
        return value

    def join_root(self, device: int) -> CommHandle:
        return CommHandle(_plait_nccl.join_root(self.unique_id, device))

    def join(self, device: int) -> CommHandle:
        return CommHandle(_plait_nccl.join(self.unique_id, device))


def join_group(unique_id: bytes, device: int, *, root: bool = False) -> CommHandle:
    group = Group.from_id(unique_id)
    return group.join_root(device) if root else group.join(device)


_TIMING: dict[int, list[float]] = {}
_TIMING_LOG = os.environ.get("PLAIT_NCCL_TIMING_LOG", "")
_TIMING_ENABLED = os.environ.get("PLAIT_NCCL_TIMING", "0") == "1"
_TIMING_FD = (
    os.open(_TIMING_LOG, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    if _TIMING_ENABLED and _TIMING_LOG
    else -1
)


def _timing_enabled() -> bool:
    return _TIMING_ENABLED


def _record_allreduce(
    rank: int, started_ns: int, elapsed_ns: int, count: int, datatype: int
) -> None:
    if _TIMING_ENABLED and _TIMING_FD >= 0:
        line = f"{rank}\t{started_ns}\t{elapsed_ns}\t{count}\t{datatype}\n".encode()
        os.write(_TIMING_FD, line)
    entry = _TIMING.setdefault(rank, [0.0, 0.0, 1e30, 0.0])
    entry[0] += 1
    entry[1] += elapsed_ns / 1e6
    entry[2] = min(entry[2], elapsed_ns / 1e3)
    entry[3] = max(entry[3], elapsed_ns / 1e3)


def _flush_timing() -> None:
    for rank, (calls, total_ms, minimum_us, maximum_us) in sorted(_TIMING.items()):
        print(
            f"PLAIT_NCCL_ALLREDUCE rank={rank} calls={int(calls)} "
            f"total_ms={total_ms:.3f} avg_us={total_ms * 1000 / calls:.1f} "
            f"min_us={minimum_us:.1f} max_us={maximum_us:.1f}",
            flush=True,
        )


if _timing_enabled():
    atexit.register(_flush_timing)


_control_patched = False


def patch_vllm_control() -> None:
    """Route vLLM's control process groups through Gloo, not NCCL.

    vLLM still needs a ``torch.distributed`` store for rank discovery and
    object broadcast. That control plane does not carry tensor data; the
    device communicator is replaced by :class:`NCCLLibrary`. Mapping the
    control backend to Gloo removes the remaining NCCL dependency while the
    Plait TCP/device path owns all tensor collectives.
    """
    global _control_patched
    if _control_patched:
        return
    import torch.distributed as dist

    original_init = dist.init_process_group
    original_new_group = dist.new_group

    def _rewrite(backend):
        if backend is None:
            return "gloo"
        text = str(backend)
        return "gloo" if "nccl" in text else backend

    def init_process_group(*args, **kwargs):
        if "backend" in kwargs:
            kwargs["backend"] = _rewrite(kwargs["backend"])
        elif args:
            args = (_rewrite(args[0]), *args[1:])
        kwargs.setdefault("device_id", None)
        return original_init(*args, **kwargs)

    def new_group(*args, **kwargs):
        if "backend" in kwargs:
            kwargs["backend"] = _rewrite(kwargs["backend"])
        return original_new_group(*args, **kwargs)

    dist.init_process_group = init_process_group
    dist.new_group = new_group
    _control_patched = True
