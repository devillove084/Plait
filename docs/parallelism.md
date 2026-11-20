# Parallelism mapping

Plait treats parallelism as a coordinate system, not as a rank hierarchy.
`Topology` assigns each node a coordinate on six axes:

```text
Data, Pipeline, Tensor, Expert, Context, Sequence
```

A communication group is "all nodes that differ only along one axis". The
planner (`plait-parallel/src/plan.rs`) turns every axis into a fused operation
with an explicit participant group, conflict key, transfer policy, and operator
id.

## The contract

```rust
pub struct ParallelOp {
    pub name: String,
    pub op_id: OpId,
    pub axis: Axis,
    pub sync: SyncKind,
    pub participants: Vec<NodeId>,
    pub conflict_key: String,
    pub transfer: TransferPolicy,
}
```

The runtime observes only `OpSpec`:

```rust
pub struct OpSpec {
    pub id: OpId,
    pub kind: FuseKind,
    pub contributors: BTreeSet<NodeId>,
    pub tensor: TensorKey,
    pub epoch: u64,
}
```

The planner's job is to preserve the model's semantics while keeping the
conflict domain as small as possible. A collective that all-gathers a 64 MiB
activation becomes a `Gather` command with a registered-memory descriptor; a
collective that every rank enters becomes a set of per-shard commands that can
finish independently.

## Axis mappings

### TP — tensor parallel

- Attention heads and projection columns are split across TP coordinates.
- Q/K/V head slices are local; the Q-index gather and output projection use the
  TP/CP groups.
- The output projection is a fused `Sum`: each TP shard contributes a full-width
  partial with zeros outside its columns. The sum is the projected result.
- Conflict keys: `tp:<cp>:output`, `seq:<tp>:<head>:*`.
- Bulk Q/K/V transfers use registered regions; small partials are inline.

The reference DSA runner executes a TP topology with three shards and matches
the single-node reference.

### PP — pipeline parallel

- A stage boundary is an activation handoff for one microbatch.
- The command is a `PipelineHandoff`; the bulk activation stays in registered
  memory and the log carries the descriptor.
- Dependencies are per microbatch, so different microbatches occupy different
  pipeline slots without a global step.
- Conflict key: `pp:<stage>:<microbatch>`.
- A stage that has not received its dependency does not execute; unrelated
  microbatches continue.

The planner emits these operations, and `PpRunner` executes them: a wavefront
scheduler proposes every ready `(stage, microbatch)` task, and the `Stage`
command's read dependency orders it after its predecessor. The reference test
checks that multiple microbatches are in flight and that the final activations
match a single-node run.

### EP — expert parallel

- Dispatch routes tokens to expert shards; the bulk token activation moves
  through registered memory.
- Each expert shard computes locally; combine is a per-token fused sum of expert
  outputs.
- Conflict keys are per expert (`ep:<expert>:dispatch`, `ep:<expert>:combine`),
  so experts do not wait for a global all-to-all.
- The planner emits dispatch/combine for every expert coordinate. `MoeRunner`
  executes EP MoE: routing is replicated, each expert owns one EP coordinate,
  expert deltas are contributed with zeros outside the routed tokens, and the
  `Combine` operation fuses them. Both top-1 and top-2 routing match the
  single-node reference.

### DP — data parallel

- Forward replicas do not synchronize; they see different batches.
- Gradient exchange is per parameter shard: a fused `Sum` over the DP group with
  a registered region for the gradient bytes.
- There is no global optimizer barrier: a parameter shard can materialize as
  soon as all of its DP contributions are ordered. The optimizer still needs a
  per-shard readiness condition, but it is not a rank-ordered collective.
- `DpRunner` computes local gradients from each replica's batch, runs one fused
  sum per `(step, shard)`, and updates weights after the shard result; the
  result matches a single global-gradient run.
- Conflict key: `dp:<step>:gradient:<shard>`.

### CP — context parallel

- Sequence positions are split into KV shards.
- Each CP shard computes local indexer scores for every query, selects its local
  top-k candidates, and contributes `[score, index]` pairs.
- `TopK` merges the candidate lists. This is exact: the union of per-shard
  top-k lists contains the global top-k.
- Each shard computes an online-softmax partial `(max, sum, accumulator)` over
  its selected local keys; `OnlineSoftmax` combines the partials associatively.
- Bulk Q is fetched through the registered-memory plane; K/V stay sharded.
- Conflict keys: `ctx:<tp>:<head>:<query>:candidates|attention`.

The reference DSA runner executes a three-shard CP topology and matches the
single-node reference. `cargo run -p plait-node -- parallel` executes the EP,
DP, and PP reference runners; `-- dsa` executes TP/CP DSA.

### SP — sequence parallel

- SP is the sequence-sharded view of the same physical group as CP in the
  reference runner. The topology model keeps `Sequence` as an axis so planners
  can label the group independently; the DSA executor requires
  `sequence = 1` and aliases SP to CP.
- Q-index data is gathered through a fused `Gather`; larger sequence
  activations use a registered gather.
- Conflict key: `seq:<tp>:<head>:*`.

## DSA execution sequence

For every head and query, the runner performs:

1. **Q-index gather** — CP shards contribute their local Q-index rows to a
   `Gather` op. The result is reordered by CP shard coordinate.
2. **Q bulk read** — each replica reads peer Q head slices through
   `RegionDescriptor`s. The bytes never enter the log.
3. **Candidates** — every CP shard computes local indexer scores against its
   local keys, keeps its local top-k, and contributes `[score, index]` pairs to
   a `TopK` op.
4. **Attention partial** — every CP shard computes an online-softmax partial over
   the selected keys it owns and contributes `(max, sum, accumulator)` to an
   `OnlineSoftmax` op.
5. **Output projection** — each TP shard contributes a full-width partial to a
   fused `Sum`.

Every phase declares its operator through consensus, then proposes one
contribution per participant. No phase has a rank-zero driver.

## Adding an axis or operator

1. Add the coordinate to `Axis`/`AxisSizes` if it is a new dimension.
2. Add a `ParallelOp` with a conflict key that is disjoint from unrelated work.
3. Choose a `SyncKind`; extend `fuse_kind` and the runtime if the combination
   rule is new.
4. Choose `TransferPolicy::Inline` for small ordered control data and
   `Registered` for bulk tensors.
5. Add a planner test for participant counts and an execution test against a
   single-node reference when the operator is runnable.

The invariant to preserve: commands that do not model a real conflict must not
share a key, or the fabric will serialize work that could run concurrently.
## What is actually executed

| Runner | Topology exercised | Reference check |
| --- | --- | --- |
| `DsaRunner` | TP and CP/SP | single-node DSA, `1.1e-16` max error |
| `MoeRunner` | EP, top-1 and top-2 | single-node MoE |
| `DpRunner` | DP, 2–4 parameter shards | single global-gradient optimizer |
| `PpRunner` | PP, 2–3 stages, multiple microbatches | single-node sequential forward |

All four use `LocalCluster`, which is the same `Replica<Runtime>` and EPaxos
command path as the TCP node. The difference is only the transport.
