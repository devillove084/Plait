# Plait task runner. CUTLASS is cloned on demand into target/cutlass.
cutlass_dir := "target/cutlass"

default: test

setup:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f "{{cutlass_dir}}/include/cutlass/gemm/device/gemm.h" ]; then
      rm -rf "{{cutlass_dir}}"
      ref="${PLAIT_CUTLASS_REF:-v4.8.1}"
      if ! git clone --depth 1 --branch "$ref" \
        "${PLAIT_CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}" \
        "{{cutlass_dir}}"; then
        rm -rf "{{cutlass_dir}}"
        git clone --depth 1 --branch "$ref" \
          "git@github.com:NVIDIA/cutlass.git" "{{cutlass_dir}}"
      fi
    fi

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

lint: setup
    cargo clippy --workspace --all-targets --locked -- -D warnings

test: setup
    cargo test --workspace --locked

demo: setup
    cargo run -q --locked -p plait-node -- demo

dsa: setup
    cargo run -q --locked -p plait-node -- dsa

parallel: setup
    cargo run -q --locked -p plait-node -- parallel

devices: setup
    cargo run -q --locked -p plait-node -- devices

hetero: setup
    cargo run -q --locked -p plait-node -- hetero

collectives: setup
    cargo run -q --locked -p plait-node -- collectives --channels ${PLAIT_CHANNELS:-1} --trace target/collectives-trace.json

ipc: setup
    ./scripts/ipc-peer.sh

matrix: setup
    ./scripts/matrix-local.sh

nccl: setup
    ./scripts/nccl-compare.sh

nccl-sweep: setup
    ./scripts/nccl-compare.sh --sweep

nccl-ops: setup
    ./scripts/nccl-ops.sh

gpu: setup
    cargo run -q --locked -p plait-node -- gpu-demo

gpu-rdma: setup
    cargo run -q --locked -p plait-node -- gpu-rdma

rdma: setup
    cargo run -q --locked -p plait-node -- rdma-selftest

run: setup
    cargo run -q --locked -p plait-node -- run

ci: fmt-check lint test demo dsa parallel gpu rdma
    @echo "plait ci: every gate passed"
