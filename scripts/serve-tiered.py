#!/usr/bin/env python3
"""Launch independent native vLLM replicas for the Plait tiered layout.

Each replica is a normal ``vllm serve`` process. Native tensor parallelism
uses vLLM's own multiprocessing; do not use ``external_launcher`` for
Qwen3.5 TP, because that path produces degenerate output.

Examples:
  # local 4x3090: four TP1 W4A16 replicas
  python scripts/serve-tiered.py --model /mnt/nas1/hf/Qwen3.5-9B-W4A16 \
      --replicas 4 --tp 1 --gpu-start 0 --port-base 8000

  # remote 8xPRO: four independent TP2 122B-NVFP4 replicas
  python scripts/serve-tiered.py --model /mnt/nas1/hf/Qwen3.5-122B-A10B-NVFP4 \
      --replicas 4 --tp 2 --gpu-start 0 --port-base 8100 \
      --max-model-len 1024 --gpu-memory-utilization 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-start", type=int, default=0)
    parser.add_argument("--port-base", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--vllm", default=None)
    parser.add_argument("--no-enforce-eager", action="store_true")
    return parser.parse_args()


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> int:
    args = parse_args()
    vllm = args.vllm or shutil.which("vllm")
    if not vllm:
        print("vllm executable not found; pass --vllm", file=sys.stderr)
        return 2
    total_gpus = args.replicas * args.tp
    processes: list[subprocess.Popen] = []
    base_env = os.environ.copy()
    base_env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    try:
        for replica in range(args.replicas):
            first = args.gpu_start + replica * args.tp
            gpus = ",".join(str(first + offset) for offset in range(args.tp))
            port = args.port_base + replica
            name = args.served_model_name or os.path.basename(args.model.rstrip("/"))
            command = [
                vllm,
                "serve",
                args.model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--served-model-name",
                name,
                "--tensor-parallel-size",
                str(args.tp),
                "--max-model-len",
                str(args.max_model_len),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--trust-remote-code",
            ]
            if not args.no_enforce_eager:
                command.append("--enforce-eager")
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus
            print(
                f"start replica={replica} gpus={gpus} port={port} tp={args.tp}",
                flush=True,
            )
            processes.append(
                subprocess.Popen(
                    command,
                    env=env,
                    start_new_session=True,
                )
            )

        deadline = time.monotonic() + args.startup_timeout
        pending = set(range(args.replicas))
        while pending:
            for replica in list(pending):
                if health(args.port_base + replica):
                    print(f"ready replica={replica} port={args.port_base + replica}", flush=True)
                    pending.remove(replica)
                elif processes[replica].poll() is not None:
                    print(
                        f"replica={replica} exited with {processes[replica].returncode}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
            if time.monotonic() > deadline:
                print("startup timeout", file=sys.stderr, flush=True)
                return 1
            if pending:
                time.sleep(2)

        print(
            json.dumps(
                {
                    "replicas": args.replicas,
                    "tp": args.tp,
                    "gpus": total_gpus,
                    "endpoints": [
                        f"http://127.0.0.1:{args.port_base + i}"
                        for i in range(args.replicas)
                    ],
                }
            ),
            flush=True,
        )
        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"replica exited with {process.returncode}", file=sys.stderr)
                    return 1
            time.sleep(5)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
