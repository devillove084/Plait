# NCCL-compatible Python backend

`crates/plait-nccl` exposes a PyO3 extension (`plait_nccl._plait_nccl`) and a
Python package (`plait_nccl`). The vLLM fork uses this package for its collective
and point-to-point device communication.

## Build and install

```sh
maturin build --release
python -m pip install --force-reinstall target/wheels/plait_nccl-*.whl
```

The Rust extension also exports the NCCL C symbols for non-Python callers.

## vLLM peer transport

Install the wheel in each worker's environment and set `VLLM_USE_PLAIT=1`.
Choose the device adapter locally on each worker. The workers exchange endpoints
over the vLLM Gloo group and connect directly over TCP. Each adapter stages its
device buffers through host memory.

| `PLAIT_NCCL_BACKEND` | Device runtime |
|---|---|
| `cuda` (default) | CUDA driver |
| `rocm` | HIP |
| `huawei` | CANN |
| `ppu` | HGGC |

On ROCm, the default loader opens `libamdhip64.so`. If the launch environment
requires an explicit `PLAIT_ROCM_HIP_LIB`, point it to the HIP library used by
PyTorch in that worker. Loading a second HIP runtime into the same worker can
cause a crash at shutdown.

To use a group published by the Plait control plane, set
`VLLM_PLAIT_TP_GROUP_CONFIG`. A worker then joins using its device grant and
validates the ordered group manifest.
[Join a committed TP group](rankless-nccl.md#join-a-committed-tp-group-from-vllm)

### Pass a CANN stream from PyTorch

When calling Plait with raw `torch_npu` device pointers, synchronize the
PyTorch stream before passing its `npu_stream` handle. This flushes the
framework's host-side queue. Plait synchronizes the ACL stream, which covers
operations that have already reached ACL. The caller must make pending writes
to both the input and output buffers visible before the call.

```python
stream = torch.npu.current_stream()
stream.synchronize()
status = comm.reduce_scatter(
    input_tensor.data_ptr(), output_tensor.data_ptr(), output_tensor.numel(),
    6, 0, stream.npu_stream,  # FP16, sum
)
if status != 0:
    raise RuntimeError(f"Plait ReduceScatter failed with status {status}")
```

Keep the buffers alive until the operation finishes. For a queued collective,
apply the stream ordering requirement both when submitting an input and when
delivering the result to an output buffer. The vLLM fork handles this when
converting a `torch_npu` stream to a native handle.

### Profile asynchronous collectives

Set `PLAIT_NCCL_ASYNC_PHASE_LOG` to a writable file path before submitting
requests. The worker appends one tab-separated row for each successful
background collective.

```text
pid  group_fingerprint  slot  world  sequence  operation  dtype  input_bytes  queue_ns  collective_ns  wire_ns  fold_ns
```

Match rows across workers by group fingerprint and sequence. The fingerprint
comes from the manifest agreed at peer bootstrap. The sequence is shared with
blocking collectives within that group.

`queue_ns` measures from enqueue to worker pickup. `collective_ns` includes
waiting for the collective's turn and executing it. `wire_ns` includes socket
I/O and readiness waits, and `fold_ns` measures host reduction. Both are parts
of `collective_ns`. The worker records the row after input staging and before
delivery to the caller. Blocking requests use `PLAIT_NCCL_PHASE_LOG`.

Set `PLAIT_NCCL_SHARD_PHASE_LOG` to trace scatter and gather frames on each
peer connection. This trace works for both blocking and asynchronous calls.
Match it to the asynchronous trace by group fingerprint, slot, and sequence.

```text
pid  group_fingerprint  slot  sequence  peer  operation  stage  tx_bytes  rx_bytes  tx_first_payload_ns  tx_last_ns  rx_header_ns  rx_first_payload_ns  rx_last_ns
```

AllReduce uses stage 0 for scatter and stage 1 for gather. Times are relative
to entry into the sharded operation on that worker. Transmit times record
acceptance by the local socket. Receive times record when the worker validates
the header, reads the first payload, and finishes reading the frame. A zero
first-payload time indicates an empty payload. Compare intervals on each worker
using its local monotonic clock.

## Legacy NCCL bootstrap

Call `patch_vllm()` to replace the Python NCCL wrapper before communicator
construction in integrations that use the legacy bootstrap.

```python
import plait_nccl
plait_nccl.patch_vllm()   # replace pynccl.NCCLLibrary / pynccl_wrapper.NCCLLibrary

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
comm = PyNcclCommunicator(group, device=local_rank)
```

The legacy `ncclCommInitRank` bootstrap uses a coordinator at slot 0.

| Variable | Meaning |
|---|---|
| `PLAIT_NCCL_MASTER_ADDR` | rank-0 host; falls back to `MASTER_ADDR` |
| `PLAIT_NCCL_PORT` | TCP rendezvous port; rank 0 listens, peers retry for 60s |
| `PLAIT_NCCL_DEVICE` | Process-local device ordinal override. Falls back to `LOCAL_RANK` |
| `PLAIT_NCCL_SHM` | `1` enables the single-host shared-memory all-reduce fast path |
| `PLAIT_NCCL_SHM_MAX` | largest message using shared memory (default 64 KiB) |
| `PLAIT_NCCL_HOSTS` | per-host rank ranges, e.g. `0-3,4-11`; enables two-level allreduce |
| `PLAIT_NCCL_CROSS_CHANNELS` | parallel TCP channels between host leaders (default 8) |
| `PLAIT_NCCL_TIMING` | `1` prints per-rank all-reduce call/latency summary at exit |
| `PLAIT_NCCL_TIMING_LOG` | append `rank,start_ns,elapsed_ns` rows to this file |
| `PLAIT_NCCL_SHM_BYTES` | per-rank shm slot size (default 64 MiB with host layout) |

`ncclGetUniqueId` encodes the port in bytes 4..6 after the `PLNC` magic, so
vLLM's existing `dist.broadcast` of the 128-byte unique id carries the
rendezvous port to every rank. The legacy point-to-point path uses a separate
relay port.

## Operations

Implemented on GPU pointers: `all_reduce`, `all_gather`, `reduce_scatter`,
`reduce`, `broadcast`, `send`/`recv`, group start/end (no-op), comm
destroy/abort, and symmetric window register/deregister (no-op handles).
