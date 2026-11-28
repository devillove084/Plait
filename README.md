<p align="center">
  <img src="docs/assets/plait-logo.png" width="136" alt="Plait">
</p>

<h1 align="center">Plait</h1>

<p align="center"><strong>Rankless collective communication for heterogeneous inference.</strong></p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#measured-results">Measurements</a> ·
  <a href="specs/README.md">Formal verification</a> ·
  <a href="docs/rankless-nccl.md">Deployment guide</a>
</p>

Plait is a Rust runtime with an NCCL-compatible interface for PyTorch and vLLM.
It groups accelerators by capability and topology, moves tensors between workers,
and uses replicated metadata to replace an execution group when devices join or fail.

<p align="center">
  <img src="docs/assets/plait-architecture.svg" width="1000" alt="Plait architecture. Callers submit tensor work to peers. Any voter can propose placement and group changes. Quorum-ordered grants fence the data plane. Host loss revokes affected epochs and can promote a ready standby.">
</p>

## How it works

- **Versioned groups.** Stable host and device IDs identify resources. Logical ranks
  assign tensor or pipeline slots inside one group version. Any voter can propose a
  change, and a quorum orders conflicting metadata updates.
- **Heterogeneous placement.** The planner prices device compute, device and host
  memory, and directed network links. A committed recipe binds the TP and PP shape
  that the vLLM launcher starts.
- **Asynchronous collectives.** Workers exchange tensors over peer connections.
  Native submit and wait calls provide bounded staging so callers can overlap a
  collective with independent compute. The C ABI retains the NCCL-style entry points.
- **Host failover.** A confirmed device fault or an expired host heartbeat closes
  affected epochs. The control plane can promote an armed, ready standby on healthy
  hosts. The engine owns model state, request continuation, and KV.

[Editable Excalidraw scene](docs/assets/plait-architecture.excalidraw) ·
[Group lifecycle and APIs](docs/rankless-nccl.md)

## Measured results

| Experiment | Baseline → Plait |
| --- | --- |
| Same-host PCIe AllReduce, 16 MiB | 12.72 → 16.3 GB/s against NCCL 2.30.4 |
| Two-host vLLM TP2, 64-token input, concurrency 4 | 9.87 → 15.94 generated tokens/s against NCCL socket transport. Median time to first token fell from 1,014 to 568 ms. |
| Four-host asynchronous FP16 AllReduce, 128 KiB plus approximately 40 ms of independent GEMM | 80.97 → 41.63 ms against blocking Plait. Async completed sooner in 24 of 24 paired trials. |

The two-host comparison used the same FP16 checkpoint and crossed two input
lengths with two concurrency levels. With 512-token inputs at concurrency 4,
Plait raised generation throughput from 9.80 to 14.23 tokens/s and increased
p95 inter-token latency from 350 to 434 ms. [Benchmark commands](Justfile) ·
[Deployment and evaluation notes](docs/rankless-nccl.md)

**Four-backend serving.** A single vLLM group ran `TP2 × PP2` across CUDA, ROCm,
CANN, and HGGC. In a separate failure trial with an armed single-device standby,
stopping one execution host's heartbeat led to promotion in 8.43 seconds. The
same HTTP endpoint returned a complete response in 8.71 seconds with matching
output tokens. The timings describe one controlled process-stop trial.

## Quick start

Run the three-node ordering and reduction demo with a Rust toolchain:

```sh
cargo run -p plait-node --no-default-features -- demo
```

Use the [NCCL-compatible Python integration](docs/nccl-python.md) and the
[Plait vLLM fork](https://github.com/devillove084/vllm/tree/plait/v0.31-runtime)
for heterogeneous serving. The [deployment guide](docs/rankless-nccl.md) covers
device enrollment, connectivity, mixed TP × PP, and standby activation.

## Verification

TLC checks the configured ordering, grant, collective, and host-failure models.
TLAPS checks the guarded packed-read, device-publication, zero-point, and mixed
axis geometry theorems. See the [results and reproduction commands](specs/README.md).

## Scope

The cross-vendor data path uses host-staged TCP. Vendor runtimes supply device
execution. The journal recovers control metadata. Engines reconstruct device
buffers and handle request or KV recovery. Deployment assumes a trusted network.

[Architecture](docs/architecture.md) ·
[Parallelism](docs/parallelism.md) ·
[License](LICENSE)
