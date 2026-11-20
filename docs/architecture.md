# Architecture

![Plait architecture](assets/plait-architecture.svg)

Plait has four layers with deliberately small contracts:

1. **Core vocabulary** (`plait-core`): node identities, tensors, commands,
   access keys, membership configurations, and quorum policies.
2. **Leaderless replication** (`plait-consensus`): EPaxos-style instances,
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

Ordinary consensus instances use a majority quorum and a larger EPaxos fast
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

## 3. EPaxos-style replication

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

A committed instance is executable when every dependency is either executed or
inside the same strongly connected component. Plait runs Tarjan's algorithm over
committed-but-unexecuted instances, picks a ready component by
`(minimum seq, minimum instance id)`, and executes its instances in
`(seq, instance id)` order.

Dependency cycles are the normal consequence of concurrent conflicting
proposals; the SCC order is what prevents a topological-sort deadlock.
Non-conflicting components commute, so replicas that learn them in different
orders still reach the same tensor state.

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

`plait-net` uses length-prefixed bincode over TCP for control and for the
portable deployment path. It reuses send and receive buffers. The DSA reference
runner uses the registered-memory path for bulk Q and consensus commands for
small index/partial values.

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
- Restart: not supported yet. Identities and counters must be persisted before
  this prototype is used for durable deployments.

## 7. Why this avoids the old collective bottleneck

The old model couples three things: a fixed process group, a rank-ordered
collective, and a central control point. Plait decouples them:

- the process group is membership state, changeable without a special node;
- the collective is a normal command with conflict keys, so unrelated shards
  make progress independently;
- large payloads move through the data plane while the log carries only order
  and descriptors.

That does not make coordination free. It makes coordination **sparse**: only
operations that actually conflict pay for ordering, and only the shards that
participate enter the quorum.