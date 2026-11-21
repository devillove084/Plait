# Rankless NCCL compatibility

The NCCL C ABI takes `(nranks, uniqueId, rank)` at init. Plait keeps that ABI
for existing applications, but the runtime itself does not need a rank table:
the unique id is the only group metadata.

## Three layers

1. **Wire group** — `Group(world, port)` writes `PLNC`, the TCP port, and the
   world size into one 128-byte `ncclUniqueId`. The root listens; workers join
   the same id and receive their rank at rendezvous. No separate metadata
   service, JSON file, or rank list is read.
2. **Rankless API** — `Group.join_root(device)` / `Group.join(device)` return a
   `CommHandle` without caller-supplied rank or world. Arrival order assigns
   ranks, so callers only need the id and their device.
3. **NCCL ABI** — `ncclGetUniqueId` / `ncclCommInitRank` are thin adapters over
   the same wire group. `rank` is accepted because NCCL applications pass it;
   it is not needed by the collective algorithms.

```python
import plait_nccl
root = plait_nccl.Group(world=2, port=35000)
id = root.unique_id                       # send through the existing store
comm = root.join_root(device=0)           # rank 0
# worker:
comm = plait_nccl.Group.from_id(id).join(device=1)  # rank assigned at join
```

## vLLM

`plait_nccl.patch_vllm()` replaces `NCCLLibrary` and disables the NVIDIA-only
custom all-reduce/symmetric-memory fast paths so tensor collectives reach the
Plait communicator. `patch_vllm_control()` maps the remaining
`torch.distributed` control groups to Gloo; vLLM still needs that group for its
store and object broadcast, but no tensor data crosses NCCL.

```python
import plait_nccl
plait_nccl.patch_vllm()
plait_nccl.patch_vllm_control()
```

## Device backends

| backend | selection | implementation |
|---|---|---|
| CUDA | default | `cudarc` context + `cuMemcpyHtoD/DtoH` |
| Huawei | `PLAIT_NCCL_BACKEND=huawei` | dynamically loads `libascendcl.so` and resolves `aclInit`, `aclrtSetDevice`, `aclrtMemcpy`, `aclrtSynchronizeDevice` |
| PPU | `PLAIT_NCCL_BACKEND=ppu` | dynamically loads `libplait_ppu.so` with the adapter ABI below |

The vendor runtimes are loaded with `libloading`, so the same wheel builds and
runs on an NVIDIA-only machine. If the vendor library is absent the init fails
with the library path rather than falling back to CUDA.

PPU adapter ABI (`libplait_ppu.so`):

```c
int  plait_ppu_set_device(int device);
int  plait_ppu_memcpy_h2d(uint64_t dst, const void *src, size_t bytes);
int  plait_ppu_memcpy_d2h(uint64_t src, void *dst, size_t bytes);
int  plait_ppu_sync(void);
```

## Current limits

- The all-reduce path is two-level when `PLAIT_NCCL_HOSTS` is set and
  `PLAIT_NCCL_SHM=1`: one shared-memory mailbox per host group, one cross-host
  exchange between group leaders, then a second shared-memory publish. Example:
  `PLAIT_NCCL_HOSTS=0-3,4-11`.
- The two-level path covers all-reduce, all-gather, broadcast from rank 0, and
  reduce-scatter. It uses `PLAIT_NCCL_SHM_BYTES` (64 MiB default with a host
  layout) per rank; larger messages, `Avg`, and non-zero-root broadcasts fall
  back to the TCP star. GPU peer memory is not wired into this ABI yet.
- `patch_vllm_control()` is for control groups only. A full vLLM engine on a
  vendor platform also needs that platform's `torch` device backend; Plait
  supplies the collective side, not the tensor runtime.
- `ncclCommWindowRegister` returns a handle-shaped value but does not create
  symmetric memory; symmetric-memory dispatch is disabled by `patch_vllm()`.
