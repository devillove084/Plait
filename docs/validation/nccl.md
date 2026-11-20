# NCCL comparison

This validation checks the **result** of Plait's 12-way fused matrix collective
against NVIDIA NCCL all-reduce. It is a correctness comparison, not an
apples-to-apples throughput benchmark.

## Test logic

`scripts/nccl-allreduce.cu` mirrors the relevant parts of `nccl-tests`:

- deterministic input buffers, optional CPU reference check;
- warmup iterations before timed iterations;
- `algbw = bytes / time` and `busbw = algbw * 2*(n-1)/n`;
- a size sweep for plain all-reduce;
- `ncclCommInitAll` with one communicator per visible GPU (no MPI dependency).
- matrix mode generates the same DP2 x TP3 x EP2 partials as
  `plait-node`'s matrix runner and maps the 12 logical contributions onto the
  available NCCL ranks. Each rank locally sums its assigned partials, then NCCL
  all-reduces the rank buffers.

`scripts/nccl-compare.sh` runs the NCCL matrix check, runs
`scripts/matrix-local.sh` for Plait's 12 replicated contributors, and compares
the checksums.

## Results (pro5000, 8x RTX PRO 5000, NCCL 2.30.4, CUDA 13.2)

```text
matrix=8x8x6 logical=12 gpus=8 checksum_expected=103.000000000 checksum_nccl=103.000000000 max_errors=0
plait matrix: 12 replicas DP2×TP3×EP2, 8x8x6, 12 contributions, checksum 103, collective_ms 38.459
checksum match: plait=103 nccl=103.000000000
```

For a larger matrix:

```text
matrix=64x64x64 logical=12 gpus=8 checksum_expected=197.000000000 checksum_nccl=197.000000000 max_errors=0
plait matrix: 12 replicas DP2×TP3×EP2, 64x64x64, 12 contributions, checksum 197
checksum match: plait=197 nccl=197.000000000
```

NCCL all-reduce bandwidth sweep (`just nccl-sweep`):

| Bytes | Time (ms) | Alg BW (GB/s) | Bus BW (GB/s) | Errors |
|---:|---:|---:|---:|---:|
| 4,096 | 0.108 | 0.038 | 0.067 | 0 |
| 16,384 | 0.110 | 0.149 | 0.261 | 0 |
| 65,536 | 0.120 | 0.544 | 0.953 | 0 |
| 262,144 | 0.126 | 2.077 | 3.636 | 0 |
| 1,048,576 | 0.218 | 4.821 | 8.436 | 0 |
| 4,194,304 | 0.444 | 9.455 | 16.547 | 0 |
| 16,777,216 | 1.319 | 12.722 | 22.263 | 0 |
| 67,108,864 | 4.934 | 13.602 | 23.804 | 0 |

## What this does and does not prove

- It proves the fused sum semantics: NCCL's float all-reduce of the same 12
  partials and Plait's replicated `FuseKind::Sum` produce the same checksum.
- It does not compare throughput. Plait's matrix run includes EPaxos agreement
  over TCP, 12 proposal submissions, and a CPU f64 reference path. The NCCL path
  is GPU collective only.
- The NCCL matrix maps 12 logical contributors onto 8 physical ranks. This is
  deliberate: the result must be independent of rank packing. `--gpus` can
  change the packing without changing the reference.
- The Plait matrix currently carries values in consensus commands. Bulk
  registered-memory/RDMA transfer and GPU-resident all-reduce are separate work;
  this comparison must not be used as a bandwidth claim.

## Reproduce

```sh
just nccl
just nccl-sweep
PLAIT_MATRIX_M=64 PLAIT_MATRIX_N=64 PLAIT_MATRIX_K=64 just nccl
```

`NCCL_HOME` and `NVCC` select the toolkit; `PLAIT_NCCL_GPUS` overrides the GPU
count. On a host without NCCL the script fails at compile time rather than
silently skipping the comparison.