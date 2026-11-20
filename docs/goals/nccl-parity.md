# Goal: NCCL data-plane parity

Status: **in progress, not achieved**. The current implementation is a
correctness baseline; it must not be described as NCCL parity or as faster than
NCCL.

## Objective

Make `plait-collective` a credible same-host and multi-host alternative to NCCL
for the collectives Plait uses, with measured correctness and performance:

- same collective semantics and rank model;
- peer memory without host staging;
- one host synchronization per collective, not per ring step;
- topology-aware algorithm selection;
- explicit, measured comparison against NCCL with the same buffers, ranks, and
  iteration counts.

Performance is a gate, not a claim made in advance. A collective is done only
when the harness records correctness and a repeated benchmark result.

## Milestones

### M1 — Rank model and trace (done)

- [x] `RankId`, `world_size`, `rank()` and rank-indexed inputs.
- [x] Chrome Trace Event JSON with one lane per rank.
- [x] Preallocated trace buffer; no tracing mutex on the hot path.
- [x] Core collectives: all-reduce ring/recursive/reduce+broadcast, broadcast,
  reduce, all-gather, reduce-scatter, all-to-all.
- [ ] Replace `Vec<Vec<f32>>` public inputs with a rank-indexed buffer view after
  the memory API is stable.

### M2 — Cross-process peer memory

- [ ] CUDA IPC handle export/import for a peer allocation.
- [ ] VMM (`cuMemCreate` + POSIX fd) path for allocations that cannot use IPC.
- [ ] Descriptor exchange through the replicated log / TCP bootstrap.
- [ ] Two-process, two-GPU allreduce with no host staging.
- [ ] Timeline includes process/rank lanes.

Acceptance: a two-process test imports a peer region, performs allreduce, and
matches the CPU reference; no `cudaMemcpy` through the host after setup.

### M3 — Channelized GPU execution

- [ ] `PeerGroup::with_channels(C)` and per-rank stream pools.
- [ ] Split each collective into C disjoint channel segments.
- [ ] Ring reduce-scatter/all-gather enqueued across channels without a host
  barrier per step.
- [ ] CUDA events/flags for cross-channel readiness; one final synchronization.
- [ ] Trace shows overlapping channel work; `--channels C` in the CLI.
- [ ] Remove per-step `self.sync()` from every ring path (currently the explicit
  barrier is gone, but the engine still has not been channelized).

Acceptance: 8-rank, 1 MiB/16 MiB allreduce is correct with C=1/2/4 and the trace
shows channel overlap.

### M4 — Topology-aware algorithms

- [ ] Read PCIe/NVLink topology (`nvidia-smi topo -m` or CUDA attributes).
- [ ] Detect NVLink, PIX, PXB, NODE, SYS and build a rank tree.
- [ ] Select ring vs tree vs reduce+broadcast by message size and topology.
- [ ] Tree allreduce and tree broadcast/reduce.
- [ ] CollNet/NVLS/PAT: implement only when the hardware exposes the required
  capability; otherwise explicit fallback with a recorded reason.
- [ ] Timeline records the selected algorithm and topology class.

Acceptance: on the 8× Blackwell host (two NUMA groups, no NVLink) the selector
prefers the measured faster algorithm for each size; on NVLink hardware it
detects `NV#`.

### M5 — GPUDirect RDMA

- [ ] Register GPU memory with `ibv_reg_mr` (or dmabuf where required).
- [ ] Probe peer-memory support and fail explicitly when absent.
- [ ] Two-host RDMA allreduce/all-gather with GPU buffers, no host bounce.
- [ ] Topology/manifest exchange replaces implicit rank assumptions.
- [ ] Cross-host correctness and bandwidth test.

Acceptance: two hosts with RNIC/rxe exchange GPU-backed regions and produce the
same checksum as NCCL; absent hardware is a clear skip, not a fake pass.

### M6 — LL / LL128 protocols

- [ ] Flag/ack protocol for small messages.
- [ ] 128-byte line protocol for large messages.
- [ ] Per-protocol correctness tests.
- [ ] Select protocol by size/topology; expose in trace args.
- [ ] Compare latency/bandwidth against NCCL at 4 KiB–64 MiB.

### M7 — Validation gate

- [ ] `nccl-tests`-style matrix: allreduce, broadcast, reduce, all-gather,
  reduce-scatter, all-to-all.
- [ ] Same rank counts, byte sizes, warmup, and iterations.
- [ ] Correctness check with finite tolerance and per-rank errors.
- [ ] JSON timeline and benchmark artifact per run.
- [ ] CI gate on a GPU host; explicit skip only when hardware is absent.

## Current measured baseline

8 ranks, 1 MiB, one host (RTX PRO 5000, PCIe/NODE/SYS, no NVLink):

| Implementation | Bus BW (GB/s) | Status |
|---|---:|---|
| Plait ring | 0.31 | correct, slow |
| Plait recursive doubling | 0.61 | correct, slow |
| NCCL 2.30.4 | 8.44 | reference |

The baseline is retained as a regression floor. A performance change is not
accepted without a new measured table and the matching timeline artifact.

## Non-goals until M2–M5 exist

- Claiming NCCL parity from API coverage alone.
- Claiming TCP-store replacement improves collective bandwidth.
- Treating CPU or host-staged copies as GPU peer memory.
- Fabricating CollNet/NVLS/PAT support on hardware that does not expose it.