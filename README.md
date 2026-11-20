<p align="center">
  <img src="docs/assets/plait-logo.png" width="220" alt="Plait logo">
</p>

<h1 align="center">Plait</h1>

<p align="center">
  <strong>A rankless, ordering-driven tensor fabric.</strong>
</p>

<p align="center">
  Rust 2024 · leaderless ordering · no rank 0 · TP/PP/EP/DP/CP/SP · heterogeneous accelerators · GPU peer-memory collectives
</p>

Plait is a research runtime for distributed tensor work. It replaces the fixed
process group and rank-zero driver of a traditional collective with a
leaderless ordered log. Every node has a stable `NodeId`, every contribution is
an ordered command, and every replica applies the same state transition.

This is not a production training system. The control plane, fused runtime,
GPU peer-memory collectives, heterogeneous planner, and formal models are
implemented and tested. Cross-host GPUDirect RDMA, durable state, authentication,
and production-scale validation are not.

## Why this design

A conventional collective assumes a fixed communicator. Ranks are array
positions; membership changes require rebuilding the group and the communication
plan. Plait assumes churn. Nodes join and leave, GPUs differ in compute and link
class, and a single operator may have a different contributor set from the one
before it.

```mermaid
flowchart LR
  A[Node A] -->|propose| L[(Plait ordering log)]
  B[Node B] -->|propose| L
  C[Node C] -->|propose| L
  L -->|ordered commands| R[Fused runtime]
  R -->|materialize shard| T[(Tensor state)]
  T -. bulk bytes .-> M[Registered memory / peer data plane]
```

The log carries control metadata and small partials. Large tensors stay in
registered memory and move through peer pointers or one-sided descriptors.

## What is implemented

- **Leaderless ordering.** Any voter may propose. Fast-path dependency
  agreement, slow-path acceptance, ballot-based recovery, snapshots, and
  membership changes are implemented in `plait-consensus`.
- **Fused runtime.** `DeclareOp`, `Contribute`, and `Fuse` materialize tensor
  shards after all declared contributors have been applied. `Sum`, `Max`,
  `Gather`, `TopK`, and `OnlineSoftmax` are implemented.
- **SCC execution.** `ExecutionGraph` maintains an online SCC condensation.
  Independent components are ready in parallel; conflicting components stay
  together.
- **GPU peer-memory collectives.** Ring allreduce, broadcast, reduce,
  all-gather, reduce-scatter, and all-to-all use CUDA peer memory and
  device-resident flags in `plait-collective`. The device-ring allreduce is the
  default path.
- **Heterogeneous planning.** `HeterogeneousLayout` solves a max-min linear
  program over compute rate, effective bandwidth, and latency, then maps the
  result to contiguous row ranges.
- **Rejoin policy.** `RejoinCost` chooses between strict recovery, snapshot
  replay, and deterministic recomputation.
- **Formal models.** Finite TLA+ models under `specs/` cover the scheduler,
  ordering safety, dependency stability, recovery liveness, and the unified
  compute/fusion/commit/execute design.

## Measured baseline

On one host with eight RTX PRO 5000 GPUs, PCIe topology, no NVLink, Plait's
device-ring allreduce measured:

| Size | Plait | NCCL 2.30.4 |
|---:|---:|---:|
| 1 MiB | 11.0 GB/s | 4.82 GB/s |
| 16 MiB | 16.3 GB/s | 12.72 GB/s |
| 64 MiB | 15.7 GB/s | 13.60 GB/s |

The matrix path also matches NCCL's float allreduce checksum at `8x8x6` and
`64x64x64`. These numbers cover same-host allreduce only. They are not an
end-to-end training claim.

LL and LL128 protocol kernels exist but remain opt-in. On the PCIe test host,
their per-line synchronization is slower than the ring; PCIe defaults to ring
plus `Simple`.

## Heterogeneous 4+8 example

The planner models four RTX 3090 ranks and eight RTX PRO 5000 ranks:

```text
4 × RTX 3090     PCIe, compute 1.0, 25 Gbps
8 × RTX PRO 5000 UB,   compute 4.0, 200 Gbps
```

Run:

```sh
cargo run -q -p plait-node -- hetero-plan
```

It reports fractional shares, integer row ranges, and the equal-split
comparison:

```text
local_share=0.111 remote_share=0.889
rows=64 local_rows=7 remote_rows=57
rows=64 equal_split_local=21 equal_split_remote=43
```

The eight faster ranks receive most of the work. Quorum membership remains one
vote per node; weights only change work assignment.

## Quick start

Install `just` and a Rust toolchain, then:

```sh
just test        # workspace tests
just demo        # three TCP nodes, fused sum
just dsa         # TP/CP sparse attention reference
just parallel    # EP MoE, DP optimizer, PP stage handoff
just devices     # discovered device profiles
just hetero      # capability-aware compute placement
just hetero-plan # 4+8 heterogeneous LP demo
cargo run -q -p plait-node -- hetero-plan --local-gflops 1394 --remote-gflops 4317
just matrix      # 12-replica matrix collective
just ipc         # two-process CUDA IPC allreduce
just tcp-allreduce # two-host TCP GPU-buffer star fallback
just tcp-ring      # two-host full-duplex TCP ring
just tcp-twolevel  # host-local reduce + cross-host exchange
just tcp-twolevel-shm # shared-memory local group + cross-host exchange
just tcp-twolevel-hybrid # shared-memory result + 8 cross-host channels
just tcp-twolevel-rust # Rust host-staged two-level prototype
just hetero-gather # two-host variable-size allgather
just nccl-gather   # NCCL allgather baseline over TCP
just gpu         # CUDA AXPY and CUTLASS SGEMM
just gpu-bench   # per-GPU CUTLASS SGEMM throughput
just rdma        # real ibverbs or an explicit skip
just nccl-tcp    # experimental NCCL-over-TCP baseline
just ci          # fmt, lint, tests, demos
```

`just setup` clones CUTLASS into `target/cutlass` if needed. The CUDA build
selects the local architecture; no generated CUTLASS tree is committed.

## Commands and hardware

The GPU data plane requires peer-capable CUDA devices. Local RTX 3090 machines
may refuse cross-GPU peer access; tests report the skip instead of hiding it.
The `pro5000` host passes cross-GPU IPC.

The ibverbs backend compiles and links. The ordinary containers expose no
`/dev/infiniband`, no active verbs device, and no `CAP_NET_ADMIN`, so
`just gpu-rdma` and `just rdma` report explicit skips.

### TCP fallback

`scripts/tcp-allreduce-two-host.sh` runs a 12-rank, host-staged allreduce across
four local GPUs and eight GPUs on the remote host. It exercises GPU H2D/D2H,
TCP, and a star reducer; it is not a GPUDirect path. On the current 4+8 host
pair it completed with `errors=0` at 1 MiB and 16 MiB:

```text
1 MiB   0.009 GB/s
16 MiB  0.032 GB/s
```

A nonblocking ring variant is in `scripts/tcp-ring-pipeline.cu`. It uses
`epoll` and nonblocking sockets, with send and receive in the same event loop.
It completes with `errors=0`. With local and remote ranks interleaved around
the ring, 1 MiB ranges from 0.019 to 0.024 GB/s and 16 MiB from 0.106 to
0.216 GB/s. The spread is smaller than the unpinned ring, but NCCL's 0.050 and
0.273 GB/s are still better. The TCP host-staged path remains a correctness
fallback, not a winner.

A two-level host-leader allreduce (`scripts/tcp-twolevel.cu`) reduces within
the four local and eight remote ranks, exchanges one host aggregate, and
broadcasts locally. With a ready barrier before timing it reports `errors=0`
at 0.091-0.103 GB/s for 1 MiB, roughly 2x NCCL's 0.050 GB/s. At 16 MiB the
host leaders reach 0.301-0.330 GB/s and the workers 0.258 GB/s; NCCL's
0.273 GB/s is still slightly ahead for the slowest workers.

The star path is a correctness fallback, not a performance result. NCCL 2.28.9
over the same TCP path reports `errors=0` at:

```text
1 MiB   0.050 GB/s
16 MiB  0.273 GB/s
```

NCCL is faster because it uses a ring and avoids a serialized star root. The
Plait TCP path still needs a ring implementation.

`crates/plait-collective/src/bin/tcp-twolevel.rs` is a Rust implementation of
the same host-leader shape. It is correct but currently slower than the C++
hybrid because the Rust prototype serializes the four cross-host channels and
copies the result back to the GPU every iteration.

The hybrid path (`scripts/tcp-twolevel-hybrid.cu`) sends local inputs to the
host leader, exchanges the aggregate over eight cross-host channels, and
publishes the result through shared memory. It reaches 0.070-0.071 GB/s at
1 MiB and 0.293-0.319 GB/s at 16 MiB, above NCCL's 0.050 and 0.273 GB/s for
every rank.

`scripts/tcp-ring-weighted.cu` exercises the heterogeneous path: four local
ranks own 2 rows each and eight remote ranks own 7 rows each, so send and
receive lengths differ and results are placed by offset. At 12 MiB total it
reaches about 0.25 GB/s. The matching `ncclAllGather` baseline reaches about
1.15 GB/s, so the variable-size Plait path is still roughly 4.5x slower. `scripts/nccl-tcp-two-host.sh`
expects the local CUDA 12.9 NCCL wheel at `PLAIT_LOCAL_NCCL` and the remote
conda NCCL build at the same version.

TLA+ tools are not vendored. Run:

```sh
./scripts/check-tla.sh
```

The script expects `tlc` on `PATH`.

## Repository layout

| Crate | Responsibility |
| --- | --- |
| `plait-core` | identities, tensors, commands, conflict keys, membership, quorum policy |
| `plait-consensus` | leaderless ordering, ballots, recovery, SCC execution, snapshots |
| `plait-fabric` | fused runtime and replicated state transitions |
| `plait-collective` | CUDA peer-memory collectives, IPC/VMM, topology, protocols |
| `plait-parallel` | TP/PP/EP/DP/CP/SP planning, DSA/MoE/DP/PP runners, heterogeneous LP, rejoin |
| `plait-mem` | aligned buffers, pooling, registered regions, RDMA contract |
| `plait-rdma` | real libibverbs RC QP, one-sided read/write, completion polling |
| `plait-net` | framed TCP control transport |
| `plait-device` | device discovery and capability scheduling |
| `plait-gpu` | CUDA AXPY and CUTLASS SGEMM execution |
| `plait-node` | cluster process, client, benchmarks, demos |

Design documents:

- [docs/design/plait.md](docs/design/plait.md) — design of record
- [docs/design/plait-requirements.md](docs/design/plait-requirements.md) — requirements
- [docs/design/asymmetric.md](docs/design/asymmetric.md) — heterogeneous row scheduling
- [docs/architecture.md](docs/architecture.md) — protocol and data flow
- [docs/parallelism.md](docs/parallelism.md) — parallelism-axis mapping
- [docs/rdma.md](docs/rdma.md) — verbs backend and setup
- [docs/goals/tcp-parity.md](docs/goals/tcp-parity.md) — TCP/NCCL acceptance goal
- [specs/](specs/) — TLA+ models and TLC configurations

## Limitations

- No authentication or encryption. The transport trusts the network.
- No durable state. Restart with the same identity is not handled.
- No verified cross-host GPU-buffer collective.
- Only NVIDIA has a real kernel adapter. AMD, Huawei, and PPU profiles are
  discovered and scheduled, but their kernel adapters are not implemented.
- LL/LL128 remain opt-in and are slower on the tested PCIe host.
- TLA+ results are finite-state checks, not a refinement proof of the Rust
  implementation.

## License

See [LICENSE](LICENSE).
