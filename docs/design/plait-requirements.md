# Requirements — Plait leaderless tensor fabric

## One-sentence restatement

Plait is a rankless execution fabric where equal processes use a replicated
consensus log to order tensor work, fuse communication with computation, and
move bulk bytes through registered memory instead of through the log.

## Functional requirements (core)

1. **Leaderless proposal and recovery.** Any voter can propose a tensor/operator
   command and any voter can recover a stalled instance. There is no permanent
   rank 0.
2. **Fused operator execution.** `Sum`, `Max`, `Gather`, `TopK`, and
   `OnlineSoftmax` contributions are replicated commands; the runtime
   materializes the result once all declared contributors execute.
3. **Parallelism axes.** Express TP, PP, EP, DP, CP, and SP as participant groups
   and conflict domains, with executable reference runners for DSA, MoE, DP
   optimizer, and PP stages.
4. **Identity-based membership.** Nodes have stable `NodeId`s; join/leave is
   replicated state, not an array index. A joining node receives a snapshot.
5. **Bulk data plane.** Large tensors stay in registered memory; consensus
   commands carry descriptors. TCP is the portable control fallback; real
   `libibverbs` RC queue pairs are available for one-sided read/write.
6. **Observable status.** A client can query an op's declaration/result,
   executed-instance count, and current voter set from any node.
7. **Heterogeneous compute placement.** Advertise CPU, NVIDIA, AMD, Huawei, PPU,
   and other devices as replicated capability profiles; schedule a compute task
   onto an eligible device; serialize compute per device through EPaxos.

## Non-functional requirements

| Quality | Target / assumption | Why it shapes the design |
|---|---|---|
| Scale | 3–64 voters per consensus domain; 8–512 accelerator processes per job | Global 512-voter quorums would make every command O(N); domains bound fanout |
| Command rate | 1k–20k commands/s per domain | Determines quorum fanout, batching, and log compaction |
| Commit latency | p99 ≤ 2 ms same-rack, ≤ 10 ms cross-node | Fast path needs one RTT; slow path adds a Paxos accept round |
| Read freshness | A query sees only executed state; committed state is linearizable per domain | Tensor results must not depend on replica-local ordering |
| Failure tolerance | Tolerate `floor((N-1)/2)` voter failures in steady state | Majority quorum; 3 voters survive 1, 5 survive 2 |
| Membership availability | Membership change requires all relevant voters | Deliberate safety-over-availability choice; see decision table |
| Durability | Prototype: in-memory; production target: WAL + snapshot, RPO 0 for committed entries | Current code has snapshots but no disk persistence |
| Security | Production: authenticated `NodeId`, mTLS control, authorized membership | Prototype transport is trusted-network only |
| Bulk bandwidth | 100–400 Gb/s per node; RDMA one-sided for tensors | Control must not serialize megabytes into consensus |
| Payload policy | Inline ≤ ~1 KB control; descriptor + RDMA above that | Keeps log bandwidth and CPU bounded |
| Device heterogeneity | CPU + NVIDIA + AMD + Huawei + PPU profiles in one plan | Scheduling cannot assume one vendor or one backend |
| Compute conflict | Same device serialized; different devices concurrent | A device is a scarce execution resource, not just data |

## Out of scope (explicit)

- A model frontend/autograd, CUDA/Triton kernels, and a production training
  scheduler.
- Multi-region consensus, cross-datacenter reconfiguration, and disaster recovery.
- GPUDirect RDMA, multi-NIC striping, and completion-queue batching.
- Production authentication/authorization, key management, and audit.
- Durable recovery after process restart with the same identity (future WAL).
- Full MoE training (backward pass); the EP runner is a forward reference.

## Assumptions (load-bearing)

- **A1.** Commands are shard-level, not per query/head, in production. A DSA
  command covers a CP shard's query block and all local heads. The reference
  runner uses smaller per-query ops for exactness; that granularity is not the
  scale plan.
- **A2.** Consensus is scoped to a small domain (3–8 voters) that owns a tensor
  partition. The current prototype still has one global `ClusterConfig`; per-domain
  groups are the next scale step.
- **A3.** The training step target is ~0.25–1 step/s, not 10 steps/s. Serving
  targets are ~5 forwards/s with 64k context.
- **A4.** DSA is the representative sparse-attention workload: 61 layers, 128
  heads, head_dim 128, seq 64k, top-k 2048. Numbers are illustrative, not a
  benchmark claim.
- **A5.** RDMA hardware is not present in the development container or the
  provided remote container; the real backend compiles and links, but the two-QP
  path is validated only on a host with an RNIC or configured Soft-RoCE.
- **A7.** Vendor kernel execution is an adapter layer. Device discovery,
  capability matching, compute conflicts, and placement are in scope now; CUDA,
  ROCm, CANN, and PPU kernels are future adapters.
- **A6.** Membership changes are rare and can tolerate a full-quorum pause.

## Key numbers (→ back-of-the-envelope)

| Quantity | Estimate | Assumption |
|---|---|---|
| Voters per domain | 3–8 | Small groups; majority 2–5, fast quorum 2–6 |
| Commands per DSA forward | ~1.5k | 61 layers × 8 CP shards × 3 boundary phases, batched |
| Command rate at 5 forwards/s | ~7.5k/s | Batched shard-level commands |
| Control fanout per command | 7 sends | 8-voter domain, excluding self |
| Control messages/s | ~52k | 7.5k × 7; small descriptors |
| Log metadata/day | ~6.5 GB | 7.5k/s × 256 B × 86.4k s |
| Descriptor log/day | ~26 GB | 1 KB descriptor per command |
| RDMA bulk Q per node/layer | ~14 MB | 64k seq × 16 heads × 128 dim × 2 B / shard |
| KV cache per 64k sequence | ~262 GB | 2 × 61 × 64k × 128 × 128 × 2 B; sharded |
| Same-rack commit budget | ≤2 ms p99 | One fast-path RTT + local apply |
| Remote GPU discovery | 8× RTX PRO 5000 72GB + CPU | `pro5000` test run |
| Hetero demo tasks | 32 planned ops | placements `{Nvidia: 29, Cpu: 3}` |

## Validation gate

- [x] At least one explicit out-of-scope list exists.
- [x] Each non-functional item has a number or an explicit target/assumption.
- [x] Assumptions are written down and referenced by the design.
- [x] Core features are six, not a full product; the rest is deferred.
