# Where the Plait collective time goes

Profilers are present on `pro5000` (`/usr/local/cuda/bin/nsys`). The 8-rank,
1 MiB allreduce benchmark is captured with:

```sh
nsys profile -o /tmp/plait-prof --force-overwrite true --trace=cuda \
  --stats=true target/debug/plait-node collectives \
  --gpus 8 --bytes 1048576 --iters 2 --op allreduce
```

## Observed profile

The 20.8 ms run (5 warmups + 2 timed iterations) consisted of seven allreduce
calls. Nsight counted 57 stream synchronizations, 785 peer copies, 392 add
kernels, and 114 device allocations.

Top host/API costs:

| API | Calls | Total time | Avg |
|---|---:|---:|---:|
| `cuStreamSynchronize` | 57 | 9.19 ms | 161 µs |
| `cuMemcpyPeerAsync` | 785 | 6.74 ms | 8.6 µs |
| `cuMemAllocAsync` | 114 | 4.78 ms | 42 µs |
| `cuMemcpyDtoHAsync` | 57 | 3.18 ms | 56 µs |
| `cuMemcpyHtoDAsync` | 57 | 1.75 ms | 31 µs |
| `cuStreamWaitEvent` | 785 | 1.35 ms | 1.7 µs |
| `cuMemcpyDtoDAsync` | 392 | 1.23 ms | 3.1 µs |
| `cuLaunchKernel` | 392 | 0.99 ms | 2.5 µs |

GPU work:

| GPU operation | Count | Time |
|---|---:|---:|
| `plait_add_inplace` kernels | 392 | 0.318 ms |
| D2D copies | 392 | 0.252 ms |

## Interpretation

- **Host orchestration dominates.** 112 peer copies per allreduce (7 ring steps
  × 8 ranks × 2 phases) are issued from the host. At ~8.6 µs per
  `cuMemcpyPeerAsync`, the copies alone cost about 1 ms per collective before
  GPU work is considered.
- **The add kernel is not the bottleneck.** All 392 kernels together take
  0.318 ms; the tensors are small and the kernel is launch-bound.
- **Allocation per call is visible.** The current API allocates data and receive
  buffers for every rank on every call: 16 allocations per collective.
- **The benchmark includes H2D/D2H that NCCL's loop excludes.** Plait's public
  API takes `Vec<Vec<f32>>` and returns host vectors, so each iteration performs
  input H2D and output D2H; the NCCL harness allocates and fills once, outside
  the timed region. The 0.087 GB/s number is therefore a host-inclusive number,
  not an apples-to-apples device-resident collective.
- **Channelization increases the problem.** C=2/4 creates more streams and more
  host-issued copies without reducing the number of API calls enough; measured
  bus bandwidth fell from 0.137 to 0.060/0.035 GB/s.

## What would change the result

1. Device-resident input/output API: allocate once, time only the collective.
2. Persistent per-rank workspaces; remove `cuMemAllocAsync` from the hot path.
3. Enqueue larger segments and use CUDA graphs or device-side flags instead of
   one host copy per ring hop.
4. Reduce host-issued peer copies with channel-level batching; use one stream
   per channel only after buffers are reused.
5. Then revisit tree/ring selection and LL/LL128.

The profile is the evidence for the next optimization order; it is not a claim
that the current numbers are device-resident parity.