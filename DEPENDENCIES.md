# Dependency policy

Plait is a Rust 2024 workspace. The authoritative lock is the committed
`Cargo.lock`; CI runs with `--locked`.

| Dependency | Version | Role |
| --- | --- | --- |
| `serde` | 1 | Wire, command, snapshot, and region-descriptor derives |
| `bincode` | 1.3 | Framed TCP payload codec |
| `sha2` | 0.10 | Stable `NodeId` derivation from human labels |
| `rand` | 0.8 | Fresh node and client identities when no id is supplied |

Standard library only is used for threads, sockets, channels, allocation, and
time. There is no async runtime, C/C++ bridge, build script, or Git submodule.

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
```

4. Update this table in the same change.

## Removed foundation

The previous C++26/CMake project, its dependency snapshots (Abseil,
GoogleTest, YalantingLibs, OpenCL), build presets, clang-format/tidy gates, and
the C++ CI workflow were removed as part of the Rust rewrite. They remain in
Git history.