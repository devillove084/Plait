# LL/LL128 protocol status

The protocol layer is implemented, but neither LL nor LL128 is selected by
default: both are currently slower than the device-ring path because the
line/flag kernel is not yet resident or multi-block.

## Implemented

- `Protocol::{Simple, Ll, Ll128}`.
- `PeerGroup::all_reduce_protocol`.
- LL device flag/ack path (`all_reduce_device_ring`).
- LL128 line kernel (`plait_ll128_allreduce`) with 32-float/128-byte lines and
  per-line flags; 2-rank correctness passes.
- `PeerGroup::recommended_protocol(bytes)`:
  - ≤32 KiB → LL
  - ≥64 KiB → LL128
  - otherwise Simple.

## Measured

8 ranks, naive per-call protocol harness:

| Protocol | 1 MiB | 16 MiB |
|---|---:|---:|
| LL | 0.228 GB/s | 0.438 GB/s |
| LL128 | 1.099 GB/s | 2.254 GB/s |

These are worse than the resident device ring (20+ GB/s / 27+ GB/s). They are
not wired into `--device-ring` defaults. The LL128 kernel is one block per rank
with line-level host launch/allocation; it needs a persistent workspace and
multi-block pipeline before it can be compared fairly.

## Why not default

Selecting these today would regress the passing 1/16 MiB results. The protocol
selector exists for experimentation and is not used by the default collective
path.

## Next kernel work

- persistent LL/LL128 workspace;
- multi-block per line group;
- protocol-specific chunk scheduler;
- re-run `just nccl-ops` and only then change the default.
