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

- [x] CUDA IPC handle export/import for a peer allocation.
- [x] VMM (`cuMemCreate` + POSIX fd) region allocation, export, and import.
- [ ] Cross-process VMM fd transport (SCM_RIGHTS/Unix socket).
- [x] Experimental TCP descriptor exchange for the IPC peer test.
- [ ] Integrate peer descriptors into the replicated log.
- [x] Two-process, two-GPU allreduce with no host staging.
- [ ] Timeline includes process/rank lanes (the two-process harness is not yet
  part of the node trace).

Acceptance: a two-process test imports a peer region, performs allreduce, and
matches the CPU reference; no `cudaMemcpy` through the host after setup.
The IPC path passes; VMM and replicated-log descriptor exchange remain open.

### M3 — Channelized GPU execution

- [x] `PeerGroup::new_with_channels(C)` and per-rank stream pools.
- [x] Split ring allreduce into C disjoint channel segments.
- [x] Enqueue ring steps without a host barrier per step; one final sync.
- [x] Trace channel lanes (`tid = rank * channels + channel`); `--channels C`.
- [ ] CUDA events/flags for cross-channel readiness beyond stream ordering.
- [ ] Performance gate: C=1/2/4 is correct, but the current copy/kernel
  scheduling is slower with more channels.

Acceptance: correctness passes for C=1/2/4; **performance does not yet pass**.
The measured 1 MiB result is 0.137 GB/s at C=1, 0.060 at C=2, 0.035 at C=4.
The channel engine is a structural step, not a performance win.

### M4 — Topology-aware algorithms

- [x] Read PCIe/NVLink topology (`nvidia-smi topo -m`); parse NVLink/PIX/PXB/
  PHB/NODE/SYS, including ANSI-stripped headers.
- [x] Conservative selection API; `--algorithm auto` falls back to ring when
  topology discovery fails.
- [x] Binary tree allreduce, broadcast, and reduce on the rank tree.
- [x] Size heuristic: small PCIe messages use reduce+broadcast, larger use tree;
  ring remains the fallback. The heuristic is recorded, not yet tuned across a
  full size sweep.
- [x] Explicit fallback: CollNet/NVLS/PAT are never advertised as available;
  the selector falls back to ring/reduce+broadcast/tree.
- [ ] Timeline records the selected topology class (algorithm is recorded).

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

## Related formal workstream

An EPaxos variant is tracked separately in
[epaxos-formal.md](epaxos-formal.md). Modifying EPaxos can improve control-plane
metadata and ordering, but it cannot substitute for GPU peer memory, channels,
or GPUDirect. TLA+/Lean acceptance gates are there; the first scheduler model is
`specs/PlaitExecutionGraph.tla`.

## Non-goals until M2–M5 exist

- Claiming NCCL parity from API coverage alone.
- Claiming TCP-store replacement improves collective bandwidth.
- Treating CPU or host-staged copies as GPU peer memory.
- Fabricating CollNet/NVLS/PAT support on hardware that does not expose it.