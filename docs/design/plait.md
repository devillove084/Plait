# Design: Plait — a rankless, consensus-fused tensor fabric

Status: research prototype. This document is the design of record; the code is
the reference implementation, not a production training system. Requirements and
assumptions are in [plait-requirements.md](plait-requirements.md); the rendered
architecture is [plait-architecture.html](plait-architecture.html). Remote
validation evidence is in [../validation/pro5000.md](../validation/pro5000.md).

## 0. Executive summary

Plait replaces a rank-ordered collective with **a replicated command log that
every equal process can propose to**. A tensor contribution is a command. The
same ordered command that coordinates a collective drives the fused state-machine
transition on every replica, so communication and computation are one mechanism.
Bulk tensor bytes stay in registered memory and move through one-sided RDMA;
the log carries only order, keys, small partials, and region descriptors.

There is no rank 0. Identity is a 128-bit `NodeId`; membership is replicated
state; any voter can propose, recover, and serve a snapshot.

The central design tension is **coordination granularity**. A per-query,
per-head consensus instance would make the log the bottleneck. The scale plan
therefore uses **shard-level batched commands** and **small consensus domains**,
while the reference runners use finer-grained commands to prove exactness.

## 1. Problem & scope

### Functional

- Leaderless proposal and recovery for tensor/operator work.
- Fused contributions (`Sum`, `Max`, `Gather`, `TopK`, `OnlineSoftmax`) that
  materialize a tensor shard without a separate collective.
- TP, PP, EP, DP, CP, SP as participant groups and conflict domains.
- Executable reference operators: TP/CP DSA, EP MoE, DP optimizer, PP stages.
- Identity-based join/leave with snapshot catch-up.
- Descriptor-based bulk movement with a real ibverbs backend.
- Async Tokio control plane with an Asio-style actor and no application mutex.
- Real NVIDIA execution: embedded-cubin AXPY and CUTLASS SGEMM FFI.
- Heterogeneous compute: replicated device profiles, capability matching, a
  compute access mode, and deterministic placement across CPU/NVIDIA/AMD/Huawei/
  PPU.

### Non-functional targets

| Quality | Target |
|---|---|
| Consensus domain size | 3–8 voters (bounded fanout) |
| Command rate | 1k–20k/s per domain |
| Commit p99 | ≤2 ms same-rack; ≤10 ms cross-node |
| Failure tolerance | Majority: 3 voters survive 1, 5 survive 2 |
| Bulk data | 100–400 Gb/s per node; one-sided RDMA |
| State freshness | A query sees only executed state; per-domain linearizable log |
| Durability | Prototype in-memory; production WAL/snapshot target, RPO 0 for committed |
| Security | Production mTLS + authorized membership; prototype trusted network |
| Heterogeneous devices | CPU + NVIDIA + AMD + Huawei + PPU profiles in one registry | A rank cannot imply a vendor; placement is capability-driven |
| Explicit dynamic execution graph | `petgraph` SCCs over committed EPaxos instances; cached between commits; edges classified as data/device/config/control | Ready components are visible to the scheduler; no per-tick O(V+E) rebuild; graph is the ordering authority | Fork/merge copies state per SCC; large state needs persistent/sharded maps |
| SCC batch parallel dispatch | `Runtime` forks independent ready components and merges in deterministic batch order via Rayon | Uses available cores without breaking convergent state; conflict keys remain the parallelism contract | Membership batches are serialized; independent components must have disjoint write effects |

### Out of scope

Model frontend/autograd, kernels, production scheduler, multi-region consensus,
GPUDirect RDMA, multi-NIC striping, authentication, and durable restart. The
reference runners are forward-only where stated.

### Heterogeneous compute requirement

A compute task is not just a read or write. It reserves a physical device and
must be schedulable on a CPU or an accelerator family with sufficient memory,
dtype support, and advertised capability:

```text
DeviceProfile { id, name, arch, memory, flops hint, compute_units,
                 fp16/bf16/fp64, rdma_capable, numa }
ComputeRequirement { arch?, min_memory, min_flops, dtype, prefer_rdma }
Command::Compute { task, kernel, device, requirement, inputs, output, values }
```

`Command::Compute` adds reads of its inputs, a write of its output, and a
`Mode::Compute` reservation on `Key::Device(device)`. Two kernels on one device
conflict and are ordered; kernels on different devices do not. The registered
profiles are replicated data, so every replica evaluates the same eligibility
rule and records the same placement.

### Assumptions that drive the design

- **A1 — batch at shard level.** Production DSA commands cover a CP shard's query
  block and local heads; the reference runner uses per-query ops for exactness.
- **A2 — domain-scoped consensus.** The target is 3–8 voters per tensor partition.
  The prototype currently has one global `ClusterConfig`; per-domain groups are
  the next scale step.
- **A3 — workload.** 61 DSA layers, 128 heads, head_dim 128, 64k context, top-k
  2048; training ~0.25–1 step/s, serving ~5 forwards/s. Illustrative, not a
  benchmark.
- **A4 — hardware.** No RDMA device in the local or remote container; the backend
  compiles and links, and its two-QP path runs only on a host with an RNIC or
  Soft-RoCE. The remote host does expose 8 NVIDIA RTX PRO 5000 72GB GPUs.
- **A5 — vendor kernels are adapters.** Discovery, capability matching, compute
  conflicts, scheduling, and placement are implemented; CUDA/ROCm/CANN/PPU
  kernel execution is the next adapter layer.

## 2. Scale estimates

These are order-of-magnitude numbers to choose an architecture, not measured
throughput. Units and assumptions are explicit.

| Quantity | Estimate | Derivation |
|---|---|---|
| Voters per domain | 3–8 | Small group owning a tensor partition |
| Majority / fast quorum | 2–5 / 2–6 | `floor(N/2)+1`; fast `max(majority, 3N/4)` |
| Commands per DSA forward | ~1.5k | 61 layers × 8 CP shards × 3 boundary phases, batched |
| Command rate at 5 forwards/s | ~7.5k/s | 1.5k × 5 |
| Control sends/s | ~52k | 7.5k commands × 7 non-self voters |
| Log metadata/day | ~6.5 GB | 7.5k/s × 256 B × 86,400 s |
| Descriptor log/day | ~26 GB | 1 kB descriptor per command |
| Q bulk per node per layer | ~14 MB | 64k seq × 16 heads × 128 dim × 2 B / TP-CP shard |
| KV cache per 64k sequence | ~262 GB | 2 × 61 × 64k × 128 × 128 × 2 B, sharded by CP/TP |
| Same-rack commit budget | ≤2 ms | One fast-path RTT + local apply |
| Hetero demo | 9 devices / 32 tasks | 8 NVIDIA + CPU; placements 29/3 |

What the numbers force:

1. **Do not use a 512-voter global quorum.** A fast quorum of 384 makes every
   command O(N). Use 3–8-voter domains; global membership is discovery and
   authorization, not the data-plane quorum.
2. **Do not create one instance per query/head.** At 64k × 128 heads a per-query
   log is millions of instances per layer. Batch a shard's query block and heads
   into one contribution.
3. **Keep bulk bytes out of the log.** A 14 MB Q shard per node/layer at 5
   forwards/s is ~70 MB/s per node of Q traffic; inline serialization would
   multiply that through consensus.
4. **Plan for log compaction.** Even 256 B metadata reaches GB/day; snapshot and
   truncate committed history.
5. **RDMA must tolerate loss.** A QP/NIC failure cannot stall the control plane;
   the log must make progress and the data plane must retry or fall back.

## 3. API

### Control-plane entry points

```text
Submit {
  request: u64,
  command: Command,
  reply_to: "host:port"
} -> Ack { request: u64, ok: bool, message: string }

Query {
  request: u64,
  op: OpId,
  reply_to: "host:port"
} -> QueryReply {
  request, op, declared: bool,
  values: Option<Vec<f64>>,
  executed: usize,
  voters: Vec<NodeId>
}
```

Idempotency: an instance is `(proposer NodeId, local counter)`. Retries reuse the
instance; the runtime deduplicates by `InstanceId`. A client retry after an
unknown `Ack` is safe; a duplicate `Submit` may create a second command only if
the client mints a new request, which is explicit.

### Command contract

```text
Command =
  | Nop
  | DeclareOp(OpSpec)                         // key: Op(op), write
  | Contribute { op, contributor, values }    // key: Op(op), write
  | Publish { key, version, values }          // key: Tensor(key), write
  | Stage { input?, output, version, values } // read input, write output
  | Membership(StageAdd|CommitAdd|StageRemove|CommitRemove)
```

`OpSpec { id, kind, contributors, tensor, epoch }` names the contributor set and
combination rule explicitly. There is no implicit rank list.

```text
RegionDescriptor {
  node: NodeId, region: u64, addr: u64, offset: u64,
  len: u64, rkey: u64, kind: Host|Pinned|Device
}
```

Small values remain inline. Bulk values use a descriptor and a one-sided
`RDMA_WRITE`/`RDMA_READ`; the log never carries the bytes.

### Compute and device entry points

```text
RegisterDevice(DeviceProfile) -> replicated device capability record
Compute {
  task: OpId,
  kernel: string,
  device: DeviceId,
  requirement: ComputeRequirement,
  inputs: [TensorKey],
  output: TensorKey,
  version: u64,
  values: [f64]
} -> placement recorded on every replica
```

The scheduler picks `device` before proposal. The command is accepted only when
the registered profile satisfies the requirement; the device key serializes
kernels on that device. A production adapter would carry a descriptor instead of
`values` and execute the kernel on the owning node.

### Real RDMA endpoint

```text
Endpoint { qpn: u32, lid: u16, gid: [u8;16], gid_index: u32 }
QP::connect(remote Endpoint)
QP::post_write(local Registration, offset, len, RemoteRegion)
QP::post_read(local Registration, offset, len, RemoteRegion)
QP::poll(timeout) -> Completion { wr_id, opcode, status }
```

The endpoint exchange is out of band today (caller-supplied; the two-QP self-test
uses both local endpoints). A production deployment would exchange it through
the authenticated control channel.

## 4. High-level design

[Open the rendered architecture diagram](plait-architecture.html)

The diagram is generated from the system-design `architecture-diagram` skill's
dark-theme template. Components and the requirement/number each earns:

| Component | Why it exists | Requirement / number |
|---|---|---|
| Peer process / `NodeId` | Equal proposer, recoverer, snapshot server | Leaderless steady state |
| EPaxos log | Orders conflicting commands, survives proposer failure | Majority/fast quorum, p99 commit |
| Fused runtime | Applies commands and materializes tensors | Communication = computation |
| Membership config | Join/leave without rank 0 | Identity-based membership |
| Registered memory + RDMA | Moves bulk bytes outside log | 14 MB Q shard/layer, 100–400 Gb/s |
| DSA / MoE / DP / PP runners | Executable axis mappings | TP/PP/EP/DP/CP/SP |
| Async TCP actor | One manager owns sockets; bounded channels; no locks | 3-node async demo, p99 budget |
| CUTLASS GPU executor | Real GEMM/attention capability on NVIDIA | 8 Blackwell / 4 Ampere GPUs verified |
| Device registry + scheduler | Maps compute to CPU/NVIDIA/AMD/Huawei/PPU capability | Heterogeneous replicas, device conflicts |

Flow:

1. A client submits a command to **any** voter.
2. The proposer sends `PreAccept` to its domain; fast quorum agreement commits
   without an extra accept round.
3. `Commit` drives the fused runtime; the last declared contributor materializes
   the shard.
4. Bulk Q/K/V movement uses registered regions and one-sided RDMA; small
   scores/partials stay in the command.
5. A joining node receives a snapshot, replays the executed prefix, acknowledges
   unanimous promotion, then proposes normally.

## 5. Data model

### Consensus record

```text
InstanceId { proposer: NodeId, counter: u64 }
Attrs      { seq: u64, deps: set<InstanceId> }
Value      { command: Command, attrs: Attrs, ballot: Ballot }
Record     { value: Value, phase: PreAccepted|Accepted|Committed|Executed,
             quorum: QuorumPolicy, config_version: u64, updated_ms: u64 }
```

Indexes: `records: InstanceId -> Record`, `record_index: Key -> set<InstanceId>`,
`executed: set<InstanceId>`, per-proposer `counters`, and `active` proposals.
The conflict index is the only derived structure needed for EPaxos dependencies.

### Device and placement state

```text
devices:   DeviceId -> DeviceProfile
placement: OpId -> DeviceId
```

Device profiles are replicated through `RegisterDevice`; placement is a
deterministic result of applying `Compute`. No scheduler state is hidden in a
rank-local table, so any replica can answer where a task ran.

### Runtime state

```text
ops:     OpId -> OpSpec
pending: OpId -> (contributor -> Vec<f64>)
results: OpId -> Vec<f64>
shards:  TensorKey -> TensorView { version, values }
members: NodeId -> address
applied: set<InstanceId>          // exactly-once state effect
```

A contribution is applied at most once; a fused result is keyed by `OpId`; a
tensor shard carries its epoch/version so queries can detect stale materialized
state.

### Partitioning

The future unit of consensus is a **domain**: a tensor partition plus a 3–8 voter
group. Keys are scoped to the domain and operator:

```text
dp:<domain>:<step>:gradient:<shard>
tp:<cp>:output
pp:<stage>:<microbatch>
ep:<expert>:dispatch|combine
ctx:<tp>:<head>:<query-block>:candidates|attention
seq:<tp>:<head>:q-index
```

The access mode is part of the key evaluation: a `Stage` command reads its input
and writes its output, so EPaxos orders it after the predecessor without a
global barrier.

## 6. Key decisions & trade-offs

| Decision | Solves | Worsens | Change it when |
|---|---|---|---|
| Leaderless EPaxos-style log | No rank-0 SPOF; any proposer/recoverer; parallel non-conflicting work | Recovery, ballots, and conflict handling are more complex than Raft; a fast-path conflict falls back to an accept round | Conflict rate makes slow-path p99 exceed 10 ms, or the workload becomes single-writer per key → Raft/primary |
| One global `ClusterConfig` (prototype) | Simple membership and snapshot story; all nodes see the same voter set | O(N) fanout per command; 512 voters is untenable | Any domain exceeds ~8 voters or command sends exceed ~50k/s → per-domain consensus groups |
| Shard-level batched commands | Keeps instance rate ~1.5k/forward; large payloads amortize RTT | A slow shard straggles the whole layer; payload size grows and needs RDMA | Median shard command payload > ~1 MB or one shard dominates tail → split the domain / pipeline the batch |
| Descriptor + RDMA for bulk, inline for small control | Removes MB-scale bytes from consensus; uses one RTT per bulk transfer | NIC/QP failure becomes a data-plane dependency; descriptor lifecycle is complex | Payload <1 KB (inline is cheaper) or no device (TCP copy fallback) |
| Community `rdma-io` instead of a project-local C shim | Avoids maintaining verbs struct layouts; safe RAII; one-sided READ/WRITE API; builds against system rdma-core | Young 0.0.2 crate; API may change; no GPUDirect | `rdma-io` breaks/unmaintained, or `ibverbs` gains one-sided ops, or GPUDirect is required → switch/fork with a pinned commit |
| Unanimous membership change | Removes the split-brain window between old/new majorities | Any voter down stalls join/leave; not online-reconfigurable | Membership changes happen during failures or >1 voter is down → joint consensus / Raft-style joint config |
| Snapshot catch-up | New nodes join without replaying all history | Snapshot transfer can be GBs; stale existing voters are not auto-repaired | Snapshot > ~1 GB or join latency > 30 s → incremental log catch-up + chunked snapshot |
| Reference runners use loopback/tcp | CI runs everywhere; exactness tests are deterministic | No performance evidence for the real data plane; per-query demo granularity is not production | Hardware available or a performance SLO is claimed → wire runners to verbs, batch commands, benchmark |
| `Mode::Compute` + `Key::Device` alongside read/write | Two kernels cannot reserve one device; unrelated devices run concurrently | Adds a scarce-resource dimension and more conflict keys; a device becomes a serialization point | Device queue depth/latency dominates; shard the device into virtual slots or partition by SM/MIG |
| Replicated device profiles + scheduler | Vendor-neutral placement; every replica agrees on eligibility | Profile advertisement is another replicated record; stale profile can mis-schedule | A device is hot-swapped, or profile churn exceeds ~1 command/min → lease/versioned profiles |
| Async actor instead of shared locked transport | No application mutex; one owner of sockets; natural backpressure | Single manager task serializes connection setup; async runtime dependency | Manager saturates or per-peer parallelism is needed → shard managers by peer group |
| Embedded cubin + CUTLASS FFI | Real execution on Ampere and Blackwell; no NVRTC/driver PTX mismatch | Two architecture artifacts; host copies in reference FFI | Add a new GPU family or device-pointer path → compile new cubin / descriptor path |
| GPU/CPU GEMM in `hetero` per placement | Placement and compute result are one replicated transition; no CPU placeholder for NVIDIA | CUTLASS host copies in the reference path; unsupported architectures fail | Device-pointer/stream path or AMD/Huawei/PPU adapters land |

### Why not a community RDMA library from the start?

This was asked explicitly. The Rust RDMA landscape is fragmented:

- **`ibverbs` (jonhoo)** is maintained but exposes SEND/RECV, not one-sided
  `RDMA_WRITE`/`RDMA_READ`; Plait needs one-sided bulk movement.
- **`rdma`** and **`async-rdma`** have the right verbs but currently fail
  dependency resolution because `hex-simd 0.5` pulls yanked `simd-abstraction`.
- **`rdma-core-sys`** expects a musl/make build wrapper not used here.
- A local C shim was implemented first, then **removed** in favor of
  **`rdma-io 0.0.2`** once it was verified to build and expose one-sided
  operations. `rdma-io-sys` is used directly only for QP/GID attribute types.

The remaining risk is that `rdma-io` is young. Mitigation: the descriptor model
and `RdmaDevice`/`RdmaQp` wrapper isolate it; switching to `ibverbs` + raw
one-sided WRs or a future stable crate is a contained change.

### Why not Raft?

Raft is the simpler correct choice when a single leader is acceptable. Plait's
premise is equal processes and no permanent coordinator; EPaxos lets any voter
propose conflicting work concurrently and preserves per-key order through
dependencies. If a workload is naturally single-writer per parameter shard, Raft
would be easier, but it would reintroduce the coordinator/leader bottleneck the
design is trying to avoid.

## 7. Failure modes & degradation

| Failure | What happens | User-visible behavior | Mitigation / trigger |
|---|---|---|---|
| Proposer crash mid-preaccept | Another voter's watchdog raises a ballot, recovers the instance | Commit delayed ~1–2 recovery intervals; no lost committed state | Jittered watchdog; ballot fencing; alert on recovery rate |
| Minority partition | No quorum → no commit; existing executed state remains queryable | New commands reject/timeout; reads return last executed state | Fail fast with `Ack ok=false`; do not retry into the wall |
| Quorum loss / rolling restart | Domain stops accepting writes | Degraded: stale state served; no silent divergence | Bulkhead domains; restart one voter at a time; monitor quorum health |
| Recovery stampede | Many replicas recover the same instance with rising ballots | CPU/ballot churn; commit latency spike | Deterministic jitter; cap recovery rounds; backoff |
| Membership change with a voter down | Stage/commit requires all relevant voters | Join/leave stalls; steady state unaffected | Documented tradeoff; future joint consensus |
| Stale existing voter | Its `config_version` is rejected; it cannot propose | That node is effectively a learner until restart/rejoin | Automatic snapshot catch-up is future work; alert version skew |
| RDMA QP/NIC failure | Completion timeout; one-sided transfer does not complete | Tensor stage stalls; control plane still commits unrelated work | Timeout + QP teardown/reconnect; fall back to TCP copy for the payload; do not resend a non-idempotent bulk write blindly |
| RDMA completion lost | Local CQ never sees `wr_id` | Caller waits | Bounded poll; fence/teardown; retry with a new idempotent descriptor |
| Log growth | Memory/disk pressure; snapshot transfer grows | Longer recovery/join; eventual OOM | Compaction/WAL; snapshot trigger at ~1 GB or 7 days |
| Device dies / profile stale | Compute on that device stalls or is ineligible; queued tasks wait | A shard stops; other devices continue | Heartbeat/lease on device profiles; requeue the task with a new device reservation |
| One device saturated | All tasks on that device serialize behind the device key | Other devices idle; tail latency rises | Scheduler load/queue metric; split by virtual slots or add devices |
| Control manager task stalls | All outbound sends queue behind one manager | Node appears alive but cannot send; inbound still readable | Bound the queue, close on backpressure, shard managers per peer group; alert queue depth |
| CUDA/CUTLASS error | Kernel returns an error or invalid result | That compute task fails; consensus state remains intact | Fail the task explicitly; never publish unverified GPU values |
| Unauthenticated peer | Any process can propose/query | Integrity/confidentiality failure | Production mTLS + authorized membership; out of scope now |

**Degradation stance:** the log is CP — under a partition the minority rejects
new work rather than diverging. Reads may serve the last **executed** state
(stale but explicit). The fabric never returns a partially fused result as if it
were complete; `Query` returns `values: None` until all contributors execute.

**Monitor:** commit p99 and fast/slow-path ratio, recovery count and ballot
rounds, quorum health (reachable voters), active instances, execution lag,
rejected stale-config commands, log bytes and snapshot time, RDMA completion
errors/retries, and per-shard straggler time.

## 8. Scale evolution

| Rung | Scale | Move | Breaking point / signal |
|---|---|---|---|
| 0 | 3 nodes | Single global EPaxos; per-query demo runners | Works for correctness, not rate |
| 1 | 8–64 nodes | Shard-level batching; descriptor-only bulk; snapshot compaction | Command rate >20k/s or p99 commit >10 ms |
| 2 | 64–512 nodes | Per-domain 3–8 voter groups; hierarchical membership; cross-domain dependency only at layer boundaries | A domain's fanout >8 or global config changes weekly |
| 3 | 512+ / multi-region | Two-level consensus, domain leaders only for cross-domain ordering, multi-QP RDMA | Cross-region RTT >100 ms or a region loss must not stop progress |

**Current bottleneck:** control-plane fanout and command granularity, not tensor
math. The prototype's single global voter set makes every command O(N); the
reference DSA runner's per-query ops would be millions of instances per layer.
The next rung is per-domain groups plus batched shard commands, with the same
`Command`/`Runtime` contracts.

**What I would not change yet:** the fused runtime and descriptor model. They are
the stable seams; scaling changes the quorum topology and batching, not the
meaning of a contribution.

## 9. Open questions

1. What is the measured fast-path commit p99 on 3/5/8 voters with a real RDMA
   control path? The current numbers are estimates.
2. Can `TopK` candidate merging stay exact when a query block spans domains with
   different top-k contributions, without a second score exchange?
3. What is the safest membership transition for online reconfiguration — joint
   consensus, or a two-phase unanimous stage with a quorum lease?
4. Should the consensus log persist commands or only descriptors + checksums?
   Persisting descriptors requires pinned memory lifetime guarantees.
5. How should a stale voter rejoin: full snapshot, incremental log, or a
   read-repair protocol?
6. Does GPUDirect RDMA change the contribution boundary (device memory
   descriptors) enough to split the command type?
7. What is the right device lease protocol when a GPU is reset or a profile is
   replaced — a new `DeviceProfile` command, a lease epoch, or both?
8. Should compute placement be fixed before proposal (current) or negotiated by
   a plan-level scheduler that reserves several devices atomically?

## Self-assessment (skill quality bar)

| Dimension | Score | Evidence / gap |
|---|---:|---|
| Distributed fundamentals | 4/5 | EPaxos fast/slow, recovery, SCC tests; formal safety proof absent |
| Building blocks opened | 4/5 | Consensus, RDMA, membership, failure behavior documented; no measured NIC behavior |
| Requirements scoped | 4/5 | Functional/non-functional/out-of-scope explicit; workload numbers illustrative |
| Trade-offs articulated | 5/5 | Eight decision rows with concrete change triggers |
| Numbers / estimates | 3/5 | Unit-correct estimates; no hardware benchmark |
| Failure modes | 4/5 | Degradation and monitoring named; security/durability only planned |
| Scale evolution | 3/5 | Ladder and triggers named; per-domain groups not implemented |
| Interfaces / data model | 4/5 | Concrete command, descriptor, endpoint, record schemas |

**Total: 34/40. Weakest dimension: multi-host/vendor performance evidence.** The
local 4× RTX 3090 and remote 8× RTX PRO 5000 hosts both executed real AXPY and
CUTLASS SGEMM; the async TCP control plane and a real Soft-RoCE two-QP transfer
were validated. Still unmeasured: multi-host RDMA traffic, AMD/Huawei/PPU
kernels, and 3/5/8-voter commit p99 under load.
