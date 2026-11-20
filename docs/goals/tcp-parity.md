# Goal: two-host TCP collective parity with NCCL

Status: **pass for 1 MiB and 16 MiB**. This is a TCP fallback goal; it does not
replace the GPUDirect/RDMA gate.

## Acceptance

For 1 MiB, 16 MiB, and 64 MiB allreduce over the 4×3090 + 8×Pro5000 TCP path:

1. every rank reports `errors=0`;
2. every rank's `algbw_GB/s` is at least the matching NCCL 2.28.9 TCP result;
3. setup and connection time are excluded by a ready barrier;
4. the same total bytes, rank count, warmup, and iteration count are used.

Averages are not enough. The slowest rank must pass.

## Current result

Rust `tcp-twolevel`:

| Size | Plait slowest rank | NCCL TCP | Status |
|---:|---:|---:|---|
| 1 MiB | 0.081 GB/s | 0.050 GB/s | pass |
| 16 MiB | 0.281 GB/s | 0.273 GB/s | pass |
| 64 MiB | not measured | not measured | open |

The current path uses one host leader per host, four full-duplex cross-host
TCP channels, shared-memory result publication, and a ready barrier.
