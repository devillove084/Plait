#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cargo build -q -p plait-collective --bin ipc-peer
bin=target/debug/ipc-peer
bytes=${PLAIT_IPC_BYTES:-256}
device0=${PLAIT_IPC_DEV0:-0}
device1=${PLAIT_IPC_DEV1:-0}
port=${PLAIT_IPC_PORT:-18765}
log=$(mktemp)
trap 'kill "$pid" 2>/dev/null || true; rm -f "$log"' EXIT

"$bin" --rank 0 --device "$device0" --bytes "$bytes" --addr "127.0.0.1:$port" >"$log" 2>&1 &
pid=$!
sleep 0.5
"$bin" --rank 1 --device "$device1" --bytes "$bytes" --addr "127.0.0.1:$port"
wait "$pid"
cat "$log"
