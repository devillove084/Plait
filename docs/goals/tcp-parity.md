# Goal: two-host TCP collective parity with NCCL

Status: **in progress**. This is a TCP fallback goal; it does not replace the
GPUDirect/RDMA gate.

## Acceptance

For 1 MiB, 16 MiB, and 64 MiB allreduce over the 4×3090 + 8×Pro5000 TCP path:

1. every rank reports `errors=0`;
2. every rank's `algbw_GB/s` is at least the matching NCCL 2.28.9 TCP result;
3. setup and connection time are excluded by a ready barrier;
4. the same total bytes, rank count, warmup, and iteration count are used.

Averages are not enough. The slowest rank must pass.

64 MiB is not yet measured.

## Current baseline

| Size | Plait two-level min rank | NCCL TCP | Status |
|---:|---:|---:|---|
| 1 MiB | 0.070 GB/s | 0.050 GB/s | pass |
| 16 MiB | 0.293 GB/s | 0.273 GB/s | pass |
| 64 MiB | not measured | not measured | open |

The 1 MiB result uses:

- host-local reduce on each side;
- four full-duplex cross-host TCP channels;
- parallel worker receive and leader broadcast;
- ready barrier before timing.

## Next work

- close the 16 MiB worker tail;
- measure 64 MiB;
- keep TCP results separate from GPUDirect claims.
