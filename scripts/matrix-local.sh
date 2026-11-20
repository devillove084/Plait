#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cargo build -q -p plait-node
bin=target/debug/plait-node
manifest=$(mktemp)
trap 'kill "$b_pid" 2>/dev/null || true; rm -f "$manifest"' EXIT

for i in $(seq 1 12); do
  id=$("$bin" id)
  echo "$id@127.0.0.1:$((7100 + i))" >>"$manifest"
done

"$bin" matrix --manifest "$manifest" --start 6 --count 6 --hold 40 >/tmp/plait-matrix-b.log 2>&1 &
b_pid=$!
sleep 2
"$bin" matrix --manifest "$manifest" --start 0 --count 6 --driver --hold 20
