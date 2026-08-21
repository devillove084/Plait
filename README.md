<p align="center">
  <img src="docs/assets/plait-logo.png" width="220" alt="Plait logo">
</p>

<h1 align="center">Plait</h1>

<p align="center">
  <strong>A rankless, asynchronous tensor fabric for heterogeneous machines.</strong>
</p>

<p align="center">
  C++26 · CMake · CUDA · HIP · Metal · OpenCL · distributed by design
</p>

<p align="center">
  <a href="https://github.com/devillove084/Plait/actions/workflows/ci.yml"><img src="https://github.com/devillove084/Plait/actions/workflows/ci.yml/badge.svg?branch=main" alt="C++ gate"></a>
</p>

Plait is an experimental systems runtime for executing one tensor program
across unlike accelerators and unlike hosts. It is intended to become the
substrate beneath projects such as **Ferrule**: Plait owns tensors, placement,
dependencies, movement, and execution; an inference or training system owns
models, operators, policies, and user-facing APIs.

The name is the architecture. A plait has no privileged strand: independent
lanes cross, separate, and rejoin while remaining part of one fabric.

> **Status:** pre-alpha foundation. The repository currently implements the
> first core vocabulary and its build/quality system. It does not yet execute a
> model or distribute tensor storage.

## Why Plait?

Most distributed tensor systems begin with a homogeneous world: a fixed set of
processes, integer ranks, globally ordered collectives, and rank 0 as a special
control point. That is a productive abstraction for a uniform GPU cluster, but
it becomes restrictive when the same computation spans NVIDIA generations,
AMD devices, vendor NPUs/TPUs, and Apple silicon.

Plait starts from a different contract:

- **Identity, not rank.** Nodes and devices have stable identities and
  advertised capabilities; their meaning does not depend on array position.
- **Dependencies, not global steps.** Operations become runnable when their
  tensor versions and placement requirements are satisfied, without forcing
  unrelated work through one collective sequence.
- **Placement is data.** Local, replicated, sharded, and partial values are
  explicit, inspectable states rather than hidden backend convention.
- **Movement is execution.** Copies, conversions, reductions, and
  materialization are scheduled work with costs, events, and failures.
- **Backends are adapters.** CUDA, HIP, Metal, OpenCL, PJRT, and transports plug
  into a backend-neutral core instead of defining the public tensor model.

“Rankless” does **not** mean coordination-free. Membership changes, conflicting
writes, recovery, and ownership still require precise protocols. Plait is
designed so that no permanent rank 0 is required for steady-state progress;
leaderless replication ideas such as EPaxos are a research direction, not a
prematurely frozen wire protocol.

## Scope

![Plait architecture](docs/assets/plait-architecture.svg)

Plait aims to provide:

1. a backend-neutral tensor metadata and storage model;
2. an event-driven execution DAG with explicit tensor versions;
3. capability-aware placement and cost-aware migration;
4. peer-to-peer discovery, transport, and coordination;
5. fault-aware primitives from which inference and training runtimes can build.

Plait is not a Torch-compatible frontend, a model format, an inference engine,
or a replacement operator library. It also does not assume that OpenCL alone
can expose every accelerator feature. OpenCL is the first portable adapter
foundation; native backends remain first-class.

## Current foundation

- C++26 is the single language dialect—there is no C++23 compatibility mode.
- Stable node and device identities.
- Tensor identity, version, dtype, shape, and layout vocabulary.
- Placement algebra for local, replicated, sharded, and partial tensors.
- Core validation and backend-boundary smoke tests.
- Strict Google formatting, warnings-as-errors, clang-tidy, ASan/UBSan/LSan,
  and a separate TSan preset.

## Build

Toolchain: LLVM/Clang 22.1.8 with LLD and libc++, CMake 3.28 or newer
(validated with 4.4.2), and Ninja.

```sh
make
make release
make tsan
make format
```

## Quality contract

The default `dev` gate is intentionally demanding:

- Google-based `.clang-format` with a zero-diff check;
- clang-tidy analyzer, bug-prone, concurrency, Core Guidelines, modernization,
  performance, portability, and readability families, promoted to errors;
- strict compiler warnings promoted to errors;
- AddressSanitizer, UndefinedBehaviorSanitizer, and leak detection;
- a separate ThreadSanitizer build because TSan and ASan cannot be combined.

GitHub Actions repeats the gate on a public Ubuntu runner with LLVM's public
Clang/LLD/libc++ 22 packages. Pull requests and pushes to `main` run the `dev`,
`release`, and `tsan` presets; their results converge on the stable `C++ gate`
check so branch protection does not depend on matrix job names.

## Roadmap

- [x] Core identity, tensor specification, and placement vocabulary
- [x] Reproducible C++26 build and dependency foundation
- [ ] Device discovery and capability negotiation
- [ ] Local event loop, storage handles, and asynchronous execution DAG
- [ ] OpenCL adapter and conformance harness
- [ ] Native CUDA, HIP, and Metal adapters
- [ ] Cross-host object transfer and topology-aware placement
- [ ] Rankless membership and replicated metadata experiments
- [ ] Recovery, backpressure, observability, and deterministic replay
- [ ] Higher-level inference and training primitives

The sequencing is deliberate: tensor state and failure semantics must be sound
before choosing a leaderless protocol or optimizing collective operations.

## Relationship to Ferrule

Ferrule can become a consumer of Plait, not a component hidden inside it.
Ferrule decides how a model should run; Plait decides where tensor state lives,
when dependent work is ready, and how that state crosses devices and hosts.
Keeping this boundary makes the tensor fabric useful to training, inference,
simulation, and non-ML array workloads alike.
