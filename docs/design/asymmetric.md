# Asymmetric 4+8 scheduling

The asymmetric path separates **quorum membership** from **work weight**.

## Work partition

`plait-parallel::AsymmetricLayout` assigns contiguous row ranges from positive
per-rank weights. For 4 local ranks and 8 remote ranks with equal per-rank
weight:

```text
local group   = 4/12 = 1/3 of rows
remote group  = 8/12 = 2/3 of rows
```

The ranges are exact and cover `0..M` once. A device-capability weight can be
added later (for example a Blackwell rank with a higher `weight` than a 3090)
without changing EPaxos membership.

## EPaxos integration

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

- Current weights are equal per rank; only group cardinality is asymmetric.
- Matrix rows are the weighted dimension; TP/EP dimensions are not yet
  independently weighted.
- The demo runs 12 processes on one host for repeatability; the network
  topology of a 4+8 cross-host run is not encoded in the matrix checksum.
