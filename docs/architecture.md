# Architecture

![Plait architecture](assets/plait-architecture.svg)

Plait has four layers with deliberately small contracts:

1. **Core vocabulary** (`plait-core`): node identities, tensors, commands,
   access keys, membership configurations, and quorum policies.
2. **Leaderless replication** (`plait-consensus`): leaderless ordering instances,
   ballots, recovery, deterministic execution, and snapshots.
3. **Fused runtime** (`plait-fabric`): the replicated state machine that
   combines contributions into tensor shards.
4. **Data plane** (`plait-mem`, `plait-net`): registered regions and one-sided
   transfers for bulk bytes; framed TCP for control and portable deployments.

## 1. Identity and membership

`NodeId` is 128 stable bits. A caller can derive it from a persisted label or
generate it from the OS RNG. It is never an array index. Addresses are separate
data and may change without changing identity.

`ClusterConfig` is replicated state:

```text
voters:       set of voting nodes
staged_add:   learners that received/are receiving a snapshot
staged_remove: voters in the middle of leaving
version:      incremented when the voter set changes
```

Ordinary consensus instances use a majority quorum and a larger Plait ordering fast
quorum. Membership instances use a **unanimous** policy:

- `StageAdd` / `StageRemove` require every current voter.
- `CommitAdd` requires every old voter **and** the joining node.
- `CommitRemove` requires every old voter.

This gives up availability during reconfiguration in exchange for removing the
window in which two disjoint majorities could both claim to be the cluster.

A joining process starts with only itself in its local configuration, contacts
one member, and receives a `Snapshot` containing the executed prefix plus the
known peer addresses. It replays the prefix into a fresh runtime and then
acknowledges the promotion. It does not need a rank-zero endpoint.

## 2. Commands and conflict keys

A command declares `(Key, Mode)` accesses. `Mode` is `Read`, `Write`, or
`Compute`. Two commands conflict when they share a key and at least one access is
not a read. A compute task reads its inputs, writes its output, and reserves its
device; two kernels on one device conflict, while kernels on different devices
are independent.

| Command | Keys |
| --- | --- |
| `Nop` | read `Config` |
| `DeclareOp` | read `Config`, write `Op(op)` |
| `Contribute` | read `Config`, write `Op(op)` |
| `Publish` | read `Config`, write `Tensor(tensor, shard)` |
| `Stage` | read `Config` + input tensor, write output tensor |
| `RegisterDevice` | read `Config`, write `Device(device)` |
| `Compute` | read `Config` + inputs, write output, **compute** `Device(device)` |
| `Membership` | write `Config` |

Because every command reads `Config` and membership writes it, reconfiguration
is serialized with all commands. Contributions to the same operator conflict
and are ordered; different operators have disjoint keys and execute
concurrently.

## 3. leaderless ordering replication

Every proposal has a global identity `InstanceId { proposer, counter }`. The
counter is local to the proposer, so there is no global instance allocator.

### Attributes

```text
Attrs { seq: u64, deps: set<InstanceId> }
```

`deps` contains the latest conflicting instance per proposer that the proposing
replica knows about. `seq` is `1 + max(seq of conflicts)`. The proposer includes
its own local records; receivers merge their local conflicts into the incoming
attributes before replying. Both paths use the same conflict index.

### Fast path

1. The proposer stores the instance as `PreAccepted` and sends `PreAccept` to
   every voter in the instance quorum.
2. Each receiver merges local dependencies, stores the instance, and returns
   the merged attributes plus any current accepted/committed state.
3. If the proposer has replies from the **fast quorum** and every reply's
   attributes equal the proposed attributes, it broadcasts `Commit`. This is
   the path used by uncontended work. For 3 voters the fast quorum is 2, for 5
   it is 3, and for 7 it is 5.
4. If the fast-quorum replies differ, the proposer unions their dependencies,
   takes the maximum sequence number, and runs Paxos `Accept` with a higher
   ballot. A majority of acceptances commits.

A reply with a higher ballot aborts the fast path. This matters when a recovery
raises the ballot while the original proposer is still collecting fast replies.

### Recovery

Every replica runs a watchdog over `PreAccepted` and `Accepted` records. After
the recovery delay (plus deterministic jitter), a replica that has no active
proposal for the instance sends `Recover` to the instance quorum.

- A committed response wins immediately.
- Otherwise the coordinator chooses the highest-ballot accepted value; if none
  exists, it chooses the preaccepted value with the highest `(seq, ballot)`.
- An accepted value is carried forward with a strictly higher ballot through
  `Accept`.
- A merely preaccepted value goes through `TryPreAccept`: responders merge their
  local dependencies, and if a fast quorum agrees after the merge the value
  commits; otherwise it falls back to `Accept` with the merged attributes.

Recovery is initiated by whichever replica notices the stall. There is no
recovery leader.

### Deterministic execution

`ExecutionGraph` is the explicit dynamic scheduler:

- nodes are committed, unexecuted instances;
- edges are Plait ordering dependencies, classified from the conflicting access keys as
  `Data` (tensor/operator), `Device` (compute reservation), `Config`
  (membership), or `Control` (an explicit dependency with no shared key);
- the SCC condensation is maintained online. A commit creates a singleton;
  an edge either joins the condensation DAG or closes a cycle. In the latter
  case the intersection of forward reachability from the dependency and reverse
  reachability from the dependent is merged into one component;
- removing an executed component updates its predecessors immediately; a
  predecessor becomes ready when its last outgoing dependency disappears;
- unresolved dependencies (an instance not committed yet) keep their component
  blocked; when the dependency commits, the edge is inserted and the block is
  cleared;
- `ready_batches` reads the maintained ready frontier and sorts batches by
  `(minimum seq, minimum instance id)`. It does not run a full SCC pass.

Dependency cycles are the normal consequence of concurrent conflicting
proposals; SCC order prevents a topological-sort deadlock. Independent
components commute, so replicas that learn them in different orders still reach
the same tensor state.

This is a complete online SCC/condensation algorithm over the pending frontier,
not a cached Tarjan rebuild. A cycle merge pays for a reachability search over
the current condensation; it does not scan the executed log. `petgraph` remains
in the test suite as an independent Tarjan oracle.

`Runtime` opts into SCC-level parallel execution: each ready component is
applied to a forked `Runtime`, then forks are merged in deterministic batch
order. Batches containing membership commands stay serialized. A state machine
that does not implement `fork_batch`/`merge_batch` keeps the sequential path.
Forking currently clones the runtime state; the next step is a persistent or
sharded state representation so parallel dispatch does not pay O(state) per SCC.

### Snapshots

A snapshot contains the membership configuration, per-proposer instance
counters, all records (including uncommitted candidates), and the executed
order. A joining replica replays the executed prefix into a fresh
`StateMachine`; committed-but-unexecuted records remain available for normal
execution and recovery.

## 4. Fused runtime

`Runtime` is the state machine:

- `DeclareOp(OpSpec)` registers an operator, its contributor set, combination
  rule, target tensor key, and epoch.
- `Contribute { op, contributor, values }` stores one local partial. When every
  declared contributor has been applied, the runtime combines the partials and
  installs a `TensorView`.
- `Publish` installs an externally produced shard.
- `Stage` installs a pipeline-stage activation and declares a read dependency on
  its predecessor tensor. This is how a pipeline stage is ordered after its
  input without a global barrier.
- `RegisterDevice` records a `DeviceProfile`; `Compute` records the chosen
  device and result. Device profiles are replicated data, so every replica
  evaluates the same capability requirement and sees the same placement.
- Membership commands update the runtime's member/address map and are also
  applied to `ClusterConfig` by the consensus layer.

Combination rules:

| Kind | Semantics |
| --- | --- |
| `Sum` | element-wise sum |
| `Max` | element-wise maximum |
| `Gather` | concatenate in contributor-id order |
| `TopK { k }` | merge `[score, index]` candidate lists, sort by score and index, dedupe, take `k` |
| `OnlineSoftmax` | combine `(max, sum, accumulator)` states associatively |

A contribution can be applied before its declaration if proposals race; the
runtime holds it and fuses when the declaration arrives. Invalid contributors
or ragged fixed-width partials produce deterministic error events instead of
panicking.

## 5. Data plane

Small control values belong in the log. Bulk tensors do not.

`RegionDescriptor` contains the owning node, region id, offset, length, remote
key, and memory kind. `RdmaPlane` exposes one-sided `read`/`write`. The shipped
`LoopbackRdma` validates the same descriptor and bounds rules in process; a
real `ibverbs` backend can replace it without changing consensus or planning.

`plait-net` is an Asio-style Tokio actor. A single connection-manager task owns
the outbound socket map and peer table; the node owns one inbound receiver and
is driven by `tokio::select!`. Callers use bounded channels and oneshot replies,
so the control plane has no application mutex and no blocking receive. Frames
are length-delimited bincode.

`plait-gpu` executes the real CUDA path: an embedded cubin for AXPY and a
CUTLASS 4.8.1 C++ FFI shim for SGEMM. The build detects the local compute
capability and chooses an ABI-compatible compiler (CUDA 12.9 for the Ampere
NVIDIA Ampere, 13.2 for Blackwell). `just setup` clones CUTLASS into `target/cutlass` on
demand.

`plait-device` discovers CPU/NVIDIA/AMD/Huawei/PPU profiles and provides a
deterministic capability scheduler. `ParallelOp::requirement()` supplies the
dtype/architecture constraints; the node registers profiles and emits `Compute`
commands with the selected device. The reference kernels are CPU placeholders;
vendor kernels plug in behind the executor.

`plait-rdma` is the real hardware path. It uses the community `rdma-io` crate
for the verbs ABI, device/PD/CQ/MR wrappers, work-request builders, and
completion types; Plait's wrapper adds endpoint exchange, registered aligned
buffers, the `RegionDescriptor` bridge, and one-sided `RDMA_WRITE`/`RDMA_READ`.
`RegionDescriptor` now carries the absolute registered address as well as the
remote key, so the same descriptor is usable by loopback and verbs transports.

## 6. Failure and reconfiguration behavior

- Steady-state majority failure: no new decisions, but existing committed state
  remains readable.
- Proposer failure: another voter recovers after the watchdog delay.
- Minority partition: no commit; ballots may rise, but no state is applied.
- Membership change: requires all relevant voters; a failure during the change
  stalls it. This is a deliberate safety/availability tradeoff.
- Durable control replicas restore their identity, accepted state, and
  execution order from the replica journal.

## 7. Asynchronous versioned reduction

With NCCL, each rank issues the matching operation on its communicator.
Execution is ordered on CUDA streams, and NCCL provides asynchronous enqueue
and nonblocking group calls. [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) · [NCCL group calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)

In the native W4 path, any execution host can submit an operation. The control
plane admits it against one group epoch. A plan lists the complete K partition
and exact worker grants. The submitter dispatches all shards concurrently and
pulls each immutable result as soon as its worker reports the descriptor.
The reducer accumulates the contiguous K prefix and holds later shards until
their predecessors arrive. This preserves FP32 summation order across network
schedules.

Result delivery requires the whole partition and a final ordered grant check.
The current group can admit new work while an earlier epoch drains. A
straggler in the earlier epoch holds up its own result. A revocation ordered
before the delivery check rejects output even when shard bytes have arrived. The TLA+ model covers both
epochs, independent publication and pull, out-of-order arrival, and stale
descriptors. [Async reduction model](../specs/PlaitAsyncReduction.tla)
