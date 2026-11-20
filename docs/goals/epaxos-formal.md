# Goal: formal gate for an EPaxos variant

Status: **proposed, not proven**. This workstream is separate from NCCL data-plane
parity and must not be used to claim a performance win.

## Position

Modifying EPaxos is allowed. A TLA+ or Lean model is necessary but not
sufficient:

- TLA+ can establish safety/liveness of a protocol model over finite or symbolic
  executions.
- Lean can machine-check protocol invariants and refinement lemmas.
- Neither proves the Rust implementation, allocator behavior, network failure
  handling, or performance.
- A refined implementation needs conformance tests and, ideally, an
  implementation-level refinement argument.

The first useful artifact is a small, checkable execution model, not a claim
that the current code is verified.

## Candidate EPaxos changes

Evaluate each as a separate proposal, with a safety argument and benchmark:

1. **Rank-aware dependency compression** — replace large dependency sets with
   version intervals or interval tree clocks. Invariant: dependency closure is
   preserved under merge and recovery.
2. **Batched EPaxos** — amortize preaccept/accept messages across multiple
   commands. Invariant: per-command dependency closure and deterministic order
   remain intact.
3. **Execution watermark** — a committed prefix/watermark allows earlier
   execution without waiting for an SCC of unrelated instances. Invariant:
   no conflicting command crosses the watermark out of order.
4. **Deterministic cycle ordering** — replace a dynamic SCC merge with a stable
   cycle signature and a fixed internal order. Invariant: all replicas choose
   the same order and execute the same state.
5. **Conflict-domain partitioning** — per-tensor/per-device EPaxos groups with a
   cross-domain boundary protocol. Invariant: cross-domain dependencies are
   never dropped.

Do not start with a protocol change. First model the **current** protocol and
execution scheduler, then make one change at a time and compare models.

## Formal acceptance gates

### F1 — Execution scheduler (started)

`specs/PlaitExecutionGraph.tla` models component readiness, execution safety,
and eventual progress under weak fairness. This proves the scheduler layer
conditional on a correct SCC partition.

- [x] Finite TLA+ model of component execution.
- [x] TLC model-check `Safety` and `ExecutedOnce` for 3 and 5 nodes.
- [ ] Add a reachability model that derives SCCs instead of taking the partition
  as a constant.
- [ ] Prove no deadlock and eventual completion for every bounded graph.

### F2 — EPaxos safety (finite model)

- [x] Finite TLC model of ballots, preaccept/accept/commit, fast/slow commit,
  and recovery (`specs/PlaitEPaxos.tla`).
- [x] Check agreement and committed stability for 3 replicas, 2 instances,
  2 commands, ballots bounded by 2.
- [ ] Model dependency merge explicitly; current records carry empty deps.
- [ ] Model message loss, duplication, and reordering.
- [ ] Check liveness for recovery and execution.

### F3 — Modified protocol

- [ ] State the exact delta from F2.
- [ ] Add/modify invariants and a refinement mapping.
- [ ] Model-check counterexamples and liveness.
- [ ] Lean proof for the core invariant if the TLA+ state space is too large.
- [ ] Map every model action to a Rust handler and add conformance traces.

## System-level model

`specs/PlaitUnified.tla` models the layer above EPaxos: rank assignment under
heterogeneous capacity, local compute, collective fusion groups, dependency-
ordered group commit, and execution. TLC checks capacity bounds, disjoint
fusion, commit/execute ordering, and eventual execution for a 3-rank/2-group
instance. This is still an abstraction: it does not claim the Rust scheduler or
GPU kernels are refined by TLC.

## Relationship to NCCL work

The EPaxos variant affects **control-plane** ordering and metadata. It does not
replace GPU peer memory, channels, GPUDirect, or NCCL's collective kernels. A
control-plane improvement cannot by itself make collectives faster than NCCL.

## Tooling

TLA+ tools are not vendored. Download `tla2tools.jar` and run TLC with
`specs/PlaitExecutionGraph.cfg`; Lean tooling is not installed in this container.
The model is committed as source, not represented as already checked.