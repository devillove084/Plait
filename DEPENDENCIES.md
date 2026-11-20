# Dependency policy

Plait is a Rust 2024 workspace. The authoritative lock is the committed
`Cargo.lock`; CI runs with `--locked`.

| Dependency | Version | Role |
| --- | --- | --- |
| `serde` | 1 | Wire, command, snapshot, and region-descriptor derives |
| `bincode` | 1.3 | Framed TCP payload codec |
| `sha2` | 0.10 | Stable `NodeId` derivation from human labels |
| `rand` | 0.8 | Fresh node and client identities when no id is supplied |
| `rdma-io` | 0.0.2, no async features | Safe RAII wrappers over rdma-core: device, PD, CQ, MR, QP, WR builders |
| `rdma-io-sys` | 0.0.1 | Generated rdma-core FFI used for QP attributes |
| `pkg-config` | 0.3 (build) | Exposes the system `libibverbs`/`librdmacm` link paths |
| `libibverbs` | rdma-core 1.14+ (system) | Real RC queue pairs, MRs, one-sided RDMA |
| `librdmacm` | rdma-core 1.3+ (system) | Link dependency of rdma-core; future CM integration |

Standard library only is used for threads, sockets, channels, allocation, and
time. There is no async runtime, no C++ and no Git submodule. The RDMA path uses
the community `rdma-io` crate rather than a project-local ABI shim.

## Updating a dependency

1. Change the workspace version in `Cargo.toml`.
2. Run `cargo update -p <crate>` and commit `Cargo.lock`.
3. Run the full gate from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo run --locked -p plait-node -- demo
cargo run --locked -p plait-node -- dsa
cargo run --locked -p plait-node -- parallel
cargo run --locked -p plait-node -- rdma-selftest
```

4. Update this table in the same change.

## Removed foundation

The previous C++26/CMake project, its dependency snapshots (Abseil,
GoogleTest, YalantingLibs, OpenCL), build presets, clang-format/tidy gates, and
the C++ CI workflow were removed as part of the Rust rewrite. They remain in
Git history.