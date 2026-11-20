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

## Measured performance

Eight RTX PRO 5000, 1 MiB, 8 ranks, 5 warmups / 10 timed iterations:

| Implementation | Time (ms) | Alg BW (GB/s) | Bus BW (GB/s) |
|---|---:|---:|---:|
| Plait ring all-reduce | 5.813 | 0.180 | 0.316 |
| Plait recursive doubling | 3.005 | 0.349 | 0.611 |
| Plait reduce+broadcast | 3.728 | 0.281 | 0.492 |
| NCCL 2.30.4 all-reduce | 0.218 | 4.821 | 8.436 |

This is **not faster than NCCL**. The current implementation synchronizes the
host after every ring step and launches one chunk copy/reduction at a time. NCCL
uses channels, multiple in-flight chunks, LL/LL128 protocols, and topology-aware
paths. A performance claim would require fixing those before measuring.

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
| Protocols | LL, LL128, Simple | f32 peer copy + scalar add | Vectorized/local kernels, flags, no host barrier per step |
| Channels | multiple per GPU, topology-derived | one stream per GPU | Channel scheduler and stream pools |
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