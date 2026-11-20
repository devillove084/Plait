# Peer-memory collectives and the NCCL gap

`plait-collective` is the first GPU data-plane layer in Plait. It is real CUDA
peer memory, not a CPU simulation, but it is **not** a drop-in NCCL replacement
yet.

## What is implemented

- One primary CUDA context and stream per visible GPU in one process.
- Peer-to-peer copies through `cuMemcpyPeerAsync` (via cudarc's cross-context
  `memcpy_dtod`, which records the source stream event before the copy).
- A real CUDA add kernel, compiled to a cubin for the detected architecture
  (`sm_86` locally, `sm_120` on `pro5000`) so older production drivers do not
  reject newer NVRTC PTX.
- Collectives:
  - all-reduce: ring reduce-scatter + all-gather, recursive doubling, and
    reduce-to-root + broadcast;
  - broadcast;
  - reduce;
  - all-gather (ring);
  - reduce-scatter (ring);
  - all-to-all (pairwise peer copies).
- CPU-reference tests for every operation, including all visible ranks on the
  host.

The API lives in `crates/plait-collective/src/lib.rs`. `just collectives` runs
the one-megabyte all-reduce benchmark and writes a Chrome trace.

## Rank model and Chrome timeline

The public API uses `RankId` and `PeerGroup::world_size()` / `rank()`. This
initial implementation assigns one rank per visible GPU. The bench prints
`ranks=...`, not `gpus=...`.

`PeerGroup::enable_trace(capacity)` records one `X` event per rank for every
collective, with `tid = rank`. `PeerGroup::write_trace(path)` emits the Chrome
Trace Event JSON format. Load it in `chrome://tracing` or Perfetto:

```sh
just collectives
# produces target/collectives-trace.json
```

The trace is opt-in: when disabled there is no mutex on the collective hot
path. The buffer is preallocated, event names are static strings, and no
per-event `String` is allocated.

## Channelized ring

`PeerGroup::new_with_channels(world_size, channels)` creates one default stream
plus `channels - 1` nonblocking streams per rank. Ring allreduce splits the
message into disjoint channel segments; each channel runs on its own stream and
the trace lane is `tid = rank * channels + channel`.

```sh
cargo run -q --locked -p plait-node -- collectives --gpus 8 --channels 2 --bytes 1048576 --iters 10
```

1 MiB, 8 ranks on `pro5000`:

| Channels | Time (ms) | Bus BW (GB/s) |
|---:|---:|---:|
| 1 | 13.353 | 0.137 |
| 2 | 30.422 | 0.060 |
| 4 | 52.162 | 0.035 |

More channels are currently **slower**. The stream pool and trace lanes are
implemented, but the copy/kernel launch and cross-stream event overhead dominate
at this size. This is recorded as a failed performance gate, not hidden.

## Topology and tree

`plait-collective/src/topology.rs` parses `nvidia-smi topo -m` into
`LinkClass::{NvLink, Pix, Pxb, Phb, Node, Sys, Unknown}`. `--algorithm auto`
uses the conservative `GpuTopology::recommended_allreduce`; an unavailable
topology falls back to ring.

The binary tree implementations build parent `rank & (rank - 1)`:
`broadcast_tree`, `reduce_tree`, and tree allreduce. Root-relative indexing is
used for non-zero roots. `--algorithm auto` selects reduce+broadcast below
32 KiB on PCIe, tree at 64 KiB or more, and ring otherwise. 1 MiB, 8 ranks on
`pro5000`:

| Algorithm | Time (ms) | Bus BW (GB/s) |
|---|---:|---:|
| Ring | 6.276 | 0.292 |
| Tree | 3.996 | 0.459 |
| NCCL | 0.218 | 8.436 |

Tree is faster than the current ring at this size but still far below NCCL.
CollNet/NVLS/PAT are not implemented and are never reported as available.

## Cross-process CUDA IPC

`crates/plait-collective/src/ipc.rs` exports a real `cuMemAlloc` region with
`cuIpcGetMemHandle`, imports it with `cuIpcOpenMemHandle`, and copies with
`cuMemcpyPeerAsync`. `crates/plait-collective/src/bin/ipc-peer.rs` runs two OS
processes, exchanges the receive handle over TCP, and runs a two-rank allreduce
with the same cubin add kernel.

```sh
just ipc
PLAIT_IPC_DEV0=0 PLAIT_IPC_DEV1=1 just ipc
```

Local result (same-GPU cross-process path; this host refuses cross-GPU IPC):

```text
ipc-peer: rank=1 gpu=0 bytes=256 checksum=381.000 ok
ipc-peer: rank=0 gpu=0 bytes=256 checksum=381.000 ok
```

`pro5000` result (cross-GPU IPC):

```text
ipc-peer: rank=1 gpu=1 bytes=256 checksum=381.000 ok
ipc-peer: rank=0 gpu=0 bytes=256 checksum=381.000 ok
```

The descriptor exchange in this harness is a TCP control test, not the EPaxos
replicated log. Host copies occur only for the initial input and final
verification.

### VMM path

`crates/plait-collective/src/ipc.rs` also implements `VmmRegion`:

- `cuMemGetAllocationGranularity`;
- `cuMemCreate` with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`;
- `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess`;
- `cuMemExportToShareableHandle` and `cuMemImportFromShareableHandle`.

The TLA/I PC harness does not yet pass the fd over a Unix socket; the VMM test
covers export/import in one process. Cross-process fd transport is the remaining
M2 item. Both local and `pro5000` VMM tests pass.

## GPUDirect registration probe

`just gpu-rdma` allocates a VMM region and asks the first verbs device to
register it as an MR:

```text
plait gpu-rdma: skipped (no active ibverbs device)
```

Both the local host and the ordinary `pro5000` container currently skip. This
is an explicit hardware absence, not a successful GPUDirect test. The next M5
step requires an RNIC/rxe host and a peer-memory/dmabuf provider, then a
two-host GPU-buffer collective.

## All-collective benchmark matrix

The Plait side now accepts `--op` for all core collectives:

```sh
for op in allreduce broadcast reduce allgather reducescatter alltoall; do
  cargo run -q --locked -p plait-node -- collectives --gpus 8 --bytes 1048576 --op "$op"
done
```

The benchmark validates the CPU reference for every operation and emits the
same `algbw_GB/s` / `busbw_GB/s` fields. On `pro5000` (8 ranks, 1 MiB):

| Op | Time (ms) | Bus BW (GB/s) |
|---|---:|---:|
| allreduce | 28.218 | 0.065 |
| broadcast | 17.882 | 0.103 |
| reduce | 7.280 | 0.252 |
| allgather | 25.592 | 0.072 |
| reducescatter | 6.186 | 0.297 |
| alltoall | 22.908 | 0.080 |

`just nccl-ops` runs the NCCL and Plait sides for all six core collectives.
1 MiB, 8 ranks, one run:

| Op | NCCL Bus BW (GB/s) | Plait Bus BW (GB/s) |
|---|---:|---:|
| allreduce | 8.446 | 0.087 |
| broadcast | 6.193 | 0.106 |
| reduce | 3.219 | 0.704 |
| allgather | 14.644 | 0.084 |
| reducescatter | 12.775 | 0.295 |
| alltoall | 14.213 | 0.099 |

All correctness checks pass. The performance gap is still large. `just nccl-ops`
is the M7 artifact generator; JSON trace files are written by the Plait side when
`--trace` is supplied. The full size sweep and a CI performance threshold remain
open. NCCL alltoall is implemented with grouped `ncclSend`/`ncclRecv`.

## Measured performance

Eight RTX PRO 5000, 1 MiB, 8 ranks, 5 warmups / 10 timed iterations:

| Implementation | Time (ms) | Alg BW (GB/s) | Bus BW (GB/s) |
|---|---:|---:|---:|
| Plait ring all-reduce | 5.813 | 0.180 | 0.316 |
| Plait recursive doubling | 3.005 | 0.349 | 0.611 |
| Plait reduce+broadcast | 3.728 | 0.281 | 0.492 |
| NCCL 2.30.4 all-reduce | 0.218 | 4.821 | 8.436 |

This is **not faster than NCCL**. The ring path now enqueues all steps and
synchronizes once per collective instead of synchronizing the host after every
step, but it still uses one stream per rank and one chunk copy/reduction at a
time. NCCL uses channels, multiple in-flight chunks, LL/LL128 protocols, and
topology-aware paths. A performance claim would require fixing those before
measuring.

The ring result is also correct: `just collectives` verifies every rank against
the CPU sum; the collectives test suite covers 2 through 8 visible GPUs.

## Gap matrix

| Area | NCCL | Plait now | Gap / next step |
|---|---|---|---|
| Same-process peer memory | CUDA IPC / VMM / P2P | CUDA P2P between primary contexts | Add cross-process CUDA IPC and VMM handle exchange |
| Cross-process peer memory | IPC + NVLink/IB | none | Export/import allocation handles through the replicated log |
| GPUDirect RDMA | ibverbs + NVLink | host `rdma-io` only | Register GPU memory and build GPU-direct descriptors |
| All-reduce algorithms | ring, tree, CollNet, NVLS, PAT | ring, recursive doubling, reduce+broadcast | Tree/CollNet/NVLS scheduling; topology-aware selection |
| All-gather / reduce-scatter | ring/tree, multiple channels | ring, host-synchronized | Channelized kernels and nonblocking pipeline |
| Broadcast / reduce | tree and chain | chain/root fan-out | Topology-aware trees, in-network reduction |
| All-to-all | pairwise/topology-aware | pairwise | Chunked pipelines and buffer reuse |
| Protocols | LL, LL128, Simple | f32 peer copy + scalar add | Vectorized/local kernels, flags, protocol selection |
| Channels | multiple per GPU, topology-derived | one stream per GPU | Channel scheduler and stream pools; per-step host barrier removed but not channelized |
| Bootstrap | TCP store (control only) | EPaxos TCP control plane | Not a data-plane substitute; keep bootstrap separate from transport |
| Memory allocation | caching allocator, registered regions | cudarc primary-context allocations | Reusable peer pool, pinned host buffers, registered regions |
| Performance harness | nccl-tests matrix | one-size benchmark + correctness tests | Full size/protocol sweep and regression thresholds |

## Why the TCP-store comparison is not the right benchmark

NCCL's TCP store is a **bootstrap/rendezvous** mechanism: it exchanges addresses,
sizes, and communicator metadata. Tensor bytes move over NVLink, PCIe P2P, or
RDMA. Plait's consensus TCP control plane is also control metadata; it does not
make the data path faster. Replacing NCCL's store with EPaxos would not change
NCCL's collective bandwidth. Plait must win on the data plane: peer memory,
channels, protocols, and topology.

## Reproduce

```sh
just collectives
cargo run -q --locked -p plait-node -- collectives --gpus 8 --bytes 1048576 --iters 20 --algorithm ring
cargo run -q --locked -p plait-node -- collectives --gpus 8 --bytes 1048576 --iters 20 --algorithm recursive
```

NCCL comparison data and methodology are in [nccl.md](nccl.md).