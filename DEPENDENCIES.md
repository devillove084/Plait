# Dependency policy

Plait is a Rust 2024 workspace. The authoritative lock is the committed
`Cargo.lock`; CI runs with `--locked`.

| Dependency | Version | Role |
| --- | --- | --- |
| `serde` | 1 | Wire, command, snapshot, and region-descriptor derives |
| `bincode` | 1.3 | Framed TCP payload codec |
| `sha2` | 0.10 | Stable `NodeId` derivation from human labels |
| `rand` | 0.8 | Fresh node and client identities when no id is supplied |
| `tokio` | 1 | Async TCP control plane, timers, cancellation |
| `tokio-util` | 0.7 | Length-delimited framing |
| `futures-util` | 0.3 | `Framed` stream/sink adapters |
| `bytes` | 1 | Zero-copy framed payloads |
| `petgraph` | 0.8 | Test-oracle Tarjan SCC reference for the online execution graph |
| `rayon` | 1 | Parallel ready-SCC dispatch and fork/merge execution |
| `NCCL` | 2.30.4 (system, optional) | Validation harness comparison; not a Plait runtime dependency |
| `cudarc` | 0.19 | CUDA driver, NVRTC, kernel launch, peer-context copies |
| `CUTLASS` | v4.8.1 (cloned by `just setup`) | Official NVIDIA C++ GEMM templates, FFI shim |
| `rdma-io` | 0.0.2, no async features | Safe RAII wrappers over rdma-core: device, PD, CQ, MR, QP, WR builders |
| `rdma-io-sys` | 0.0.1 | Generated rdma-core FFI used for QP attributes |
| `pkg-config` | 0.3 (build) | Exposes the system `libibverbs`/`librdmacm` link paths |
| `libibverbs` | rdma-core 1.14+ (system) | Real RC queue pairs, MRs, one-sided RDMA |
| `librdmacm` | rdma-core 1.3+ (system) | Link dependency of rdma-core; future CM integration |

Tokio is the only async runtime; Rayon provides the bounded CPU worker pool for
parallel SCC dispatch. No C++ submodule is used: CUTLASS is cloned by `just
setup` into `target/cutlass`. The RDMA path uses the community `rdma-io` crate
rather than a project-local ABI shim.

## Updating a dependency

1. Change the workspace version in `Cargo.toml`.
2. Run `cargo update -p <crate>` and commit `Cargo.lock`.
3. Run the full gate from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
just test
just demo
just dsa
just parallel
just gpu
just rdma
```

4. Update this table in the same change.

## Removed foundation

The previous C++26/CMake project, its dependency snapshots (Abseil,
GoogleTest, YalantingLibs, OpenCL), build presets, clang-format/tidy gates, and
the C++ CI workflow were removed as part of the Rust rewrite. They remain in
Git history.