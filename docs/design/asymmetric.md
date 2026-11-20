# Asymmetric 4+8 scheduling

The asymmetric path separates **quorum membership** from **work weight**.

## Work partition

`plait-parallel::AsymmetricLayout` assigns contiguous row ranges from positive
per-rank weights. `HeterogeneousLayout` derives those weights with a max-min
linear program:

```text
minimize T
subject to  a_i * x_i + b_i <= T
            sum x_i = 1, x_i >= 0

a_i = 1 / compute_i + message_bytes / bandwidth_i
b_i = latency_i
```

The one-coupling-constraint LP is solved by parametric search on `T`; no
external solver is required. The resulting shares are quantized to integer
rank weights, then `row_ranges` covers `0..M` exactly once.

For the 4+8 demo:

```text
4 local ranks:  RTX 3090, PCIe, compute 1.0, 25 Gbps
8 remote ranks: RTX PRO 5000, UB, compute 4.0, 200 Gbps
```

the LP gives `local_share=0.111`, `remote_share=0.889`, so the eight Pro 5000
ranks own the large majority of rows. With only 8 matrix rows the local group
can quantize to zero rows; at 64 rows it receives about 7. The planner also
exposes the fractional shares so a scheduler need not rely on row counts for
small matrices.

## ordering integration

`ExecutionGraph::ready_batches_for(local)` keeps the online SCC frontier but
orders independent ready components by:

```text
(number of remote device accesses, minimum seq, minimum instance id)
```

Only independent SCCs are reordered; conflict edges and SCC contents are
unchanged. Each replica still uses one vote per node. Weighting quorum votes
would be a safety change and is not done here.

## Demo

```sh
./scripts/matrix-asymmetric-local.sh
```

This starts four replicas in one process and eight in another on one host,
uses the weighted row split, and verifies the fused matrix checksum. For a real
two-host run, use the same manifest with four local files and eight remote
entries; `run_matrix_node` takes `--asymmetric` only on the driver.

## Limits

- Weights currently feed the row dimension only; TP/EP dimensions are still
  uniform.
- Integer row quantization can round a low-share group to zero when
  `rows < ranks`; use fractional shares or a larger row count.
- Matrix rows are the weighted dimension; TP/EP dimensions are not yet
  independently weighted.
- The demo runs 12 processes on one host for repeatability; the network
  topology of a 4+8 cross-host run is not encoded in the matrix checksum.

## Relaxed rejoin

`plait-parallel::RejoinCost` chooses the cheapest of strict ordering recovery,
snapshot replay, and deterministic recomputation. Recompute is selected when
it is cheaper than communication plus replay; it does not bypass consensus for
already committed commands.
