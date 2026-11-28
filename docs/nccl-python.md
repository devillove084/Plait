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
