# NCCL-compatible Python backend

`crates/plait-nccl` exposes a PyO3 extension (`plait_nccl._plait_nccl`) and a
Python package (`plait_nccl`). vLLM's `PyNcclCommunicator` can use it without a
ctypes `dlopen` of `libnccl.so`: install the wheel, call `patch_vllm()` before
constructing the communicator, and keep the existing vLLM call sites.

## Build and install

```sh
maturin build --release
pip install --force-reinstall target/wheels/plait_nccl-*.whl
```

The Rust extension is a normal PyO3 module; the `cdylib` output also keeps the
NCCL C symbols for non-Python embedders, but the Python path never uses ctypes.

## vLLM

```python
import plait_nccl
plait_nccl.patch_vllm()   # replace pynccl.NCCLLibrary / pynccl_wrapper.NCCLLibrary

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
comm = PyNcclCommunicator(group, device=local_rank)
```

Environment:

| Variable | Meaning |
|---|---|
| `PLAIT_NCCL_MASTER_ADDR` | rank-0 host; falls back to `MASTER_ADDR` |
| `PLAIT_NCCL_PORT` | TCP rendezvous port; rank 0 listens, peers retry for 60s |
| `PLAIT_NCCL_DEVICE` | CUDA device override; falls back to `LOCAL_RANK` |
| `PLAIT_NCCL_BACKEND` | `cuda` (default), `huawei`, or `ppu`; vendor adapters load dynamically |
| `PLAIT_NCCL_SHM` | `1` enables the single-host shared-memory all-reduce fast path |
| `PLAIT_NCCL_SHM_MAX` | largest message using shared memory (default 64 KiB) |
| `PLAIT_NCCL_HOSTS` | per-host rank ranges, e.g. `0-3,4-11`; enables two-level allreduce |
| `PLAIT_NCCL_CROSS_CHANNELS` | parallel TCP channels between host leaders (default 8) |
| `PLAIT_NCCL_TIMING` | `1` prints per-rank all-reduce call/latency summary at exit |
| `PLAIT_NCCL_TIMING_LOG` | append `rank,start_ns,elapsed_ns` rows to this file |
| `PLAIT_NCCL_SHM_BYTES` | per-rank shm slot size (default 64 MiB with host layout) |

`ncclGetUniqueId` encodes the port in bytes 4..6 after the `PLNC` magic, so
vLLM's existing `dist.broadcast` of the 128-byte unique id carries the
rendezvous port to every rank.

## Operations

Implemented on GPU pointers: `all_reduce`, `all_gather`, `reduce_scatter`,
`reduce`, `broadcast`, `send`/`recv`, group start/end (no-op), comm
destroy/abort, and symmetric window register/deregister (no-op handles).
`send`/`recv` use a rank-0 relay on a separate port so a sender does not block
waiting for its peer to post the matching receive.

The current point-to-point and collective transports are host-staged TCP:
every rank copies its device buffer to host, rank 0 reduces/gathers or routes,
then copies back. It is correct and portable, but not the tuned two-level
leader/channel/shm path used by `tcp-twolevel`; replacing the transport does
not change this Python ABI.
