#!/usr/bin/env python3
"""Launch independent native vLLM replicas for the Plait tiered layout.

Each replica is a normal ``vllm serve`` process. Native tensor parallelism
uses vLLM's own multiprocessing; do not use ``external_launcher`` for
Qwen3.5 TP, because that path produces degenerate output.
The launcher reports the current ready endpoints whenever the set changes.

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
import hashlib
import ipaddress
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import serving_group


SHUTDOWN_GRACE = 20.0


@dataclass
class Replica:
    index: int
    port: int
    process: subprocess.Popen
    startup_deadline: float
    ready: bool = False


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
    parser.add_argument("--no-prefix-caching", action="store_true", help="disable engine prefix reuse for calibration")
    parser.add_argument("--kernel-config", help="JSON object passed to the native vLLM engine")
    parser.add_argument("--plait-w4a16-library", type=Path, help="use the Plait FP16 compressed-tensors kernel")
    parser.add_argument("--attention-backend", help="native vLLM attention backend for matching KV layouts")
    parser.add_argument("--role", choices=("aggregated", "prefill", "decode"), default="aggregated")
    parser.add_argument("--kv-host", help="IPv4 address advertised by the NIXL side channel")
    parser.add_argument("--kv-port", type=int, help="NIXL side channel port")
    parser.add_argument("--kv-buffer-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--kv-lease-seconds", type=int, default=30)
    parser.add_argument("--group-id", help="hex ID of one committed group")
    parser.add_argument("--group-epoch", type=int)
    parser.add_argument("--host-id", help="hex ID of the host recorded in the plan")
    parser.add_argument("--control-bin", default="plait-node")
    parser.add_argument("--control-peer", action="append", default=[])
    parser.add_argument("--control-listen", default="127.0.0.1:0")
    parser.add_argument("--backend-port", type=int, help="private native vLLM port")
    parser.add_argument("--proxy-host", default="127.0.0.1", help="bind address of the managed HTTP proxy")
    parser.add_argument("--advertise-http", help="managed P/D proxy address and port visible to the router")
    parser.add_argument("--drain-timeout", type=float, default=30.0)
    parser.add_argument("--verify-release", help="local executable that verifies the worker grants are released")
    parser.add_argument("--print-launch-digest", action="store_true")
    args = parser.parse_args()
    if args.replicas < 1 or args.tp < 1:
        parser.error("--replicas and --tp must be positive")
    if args.gpu_start < 0 or not (0 < args.port_base <= 65536 - args.replicas):
        parser.error("invalid GPU start or port range")
    if not (0 < args.startup_timeout < float("inf")):
        parser.error("--startup-timeout must be finite and positive")
    if args.kernel_config is not None:
        try:
            config = json.loads(args.kernel_config)
            if not isinstance(config, dict):
                raise ValueError("expected a JSON object")
            args.kernel_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
        except ValueError as error:
            parser.error(f"invalid --kernel-config: {error}")
    if args.plait_w4a16_library is not None:
        if args.no_enforce_eager:
            parser.error("--plait-w4a16-library requires eager execution")
        try:
            args.plait_w4a16_library = args.plait_w4a16_library.resolve(strict=True)
            args.plait_w4a16_digest = hashlib.sha256(args.plait_w4a16_library.read_bytes()).hexdigest()
        except OSError as error:
            parser.error(f"cannot read --plait-w4a16-library: {error}")
    if args.group_id or args.print_launch_digest:
        if args.replicas != 1 or not (
            args.backend_port and 0 < args.backend_port < 65536 and args.backend_port != args.port_base
        ):
            parser.error("group launches require one replica and a distinct --backend-port")
    if args.role != "aggregated":
        try:
            host = ipaddress.IPv4Address(args.kv_host)
            if host.is_unspecified or host.is_multicast:
                raise ValueError("not a unicast address")
            args.kv_host = str(host)
        except ValueError:
            parser.error("prefill and decode require an explicit --kv-host IPv4 address")
        if args.replicas != 1 or args.kv_port is None or not (0 < args.kv_port < 65536):
            parser.error("prefill and decode need one replica and a valid --kv-port")
        if args.kv_port in (args.port_base, args.backend_port) or args.kv_lease_seconds < 1:
            parser.error("the NIXL port must be distinct and the lease must be positive")
    elif args.kv_host is not None or args.kv_port is not None:
        parser.error("--kv-host and --kv-port require a prefill or decode role")
    if args.advertise_http is not None:
        if args.role == "aggregated" or not (args.group_id or args.print_launch_digest):
            parser.error("--advertise-http requires a managed prefill or decode group")
        try:
            origin = urllib.parse.urlsplit("http://" + args.advertise_http)
            host = ipaddress.ip_address(origin.hostname or "")
            if (
                not origin.port or origin.username is not None or origin.path
                or origin.query or origin.fragment or host.is_unspecified or host.is_multicast
            ):
                raise ValueError("not a usable endpoint")
            args.advertise_http = f"[{host}]:{origin.port}" if host.version == 6 else f"{host}:{origin.port}"
        except ValueError:
            parser.error("--advertise-http must be an IP address and port")
    if args.group_id:
        if not (args.group_epoch and args.group_epoch > 0 and args.control_peer and args.host_id):
            parser.error("--group-id requires --group-epoch, --host-id, and --control-peer")
        try:
            for identity in (args.group_id, args.host_id):
                if len(identity) != 32 or len(bytes.fromhex(identity)) != 16:
                    raise ValueError("wrong identity size")
        except ValueError:
            parser.error("group and host IDs must contain 32 hexadecimal digits")
        if not (0 <= args.drain_timeout < float("inf")):
            parser.error("--drain-timeout must be finite and nonnegative")
    args.served_model_name = args.served_model_name or os.path.basename(args.model.rstrip("/"))
    return args


def native_command(args: argparse.Namespace, vllm: str, port: int) -> list[str]:
    command = [
        vllm, "serve", args.model, "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", args.served_model_name, "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--trust-remote-code",
    ]
    if not args.no_enforce_eager:
        command.append("--enforce-eager")
    if args.no_prefix_caching:
        command.append("--no-enable-prefix-caching")
    if args.kernel_config is not None:
        command.extend(("--kernel-config", args.kernel_config))
    if args.plait_w4a16_library is not None:
        command.extend(("--quantization", "plait-w4a16", "--dtype", "half"))
    if args.attention_backend is not None:
        command.extend(("--attention-backend", args.attention_backend))
    if args.role != "aggregated":
        config = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer" if args.role == "prefill" else "kv_consumer",
            "kv_buffer_device": args.kv_buffer_device,
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"kv_lease_duration": args.kv_lease_seconds},
        }
        command.extend(("--kv-transfer-config", json.dumps(config)))
    return command


def engine_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = {}
    if args.role != "aggregated":
        environment["VLLM_NIXL_SIDE_CHANNEL_HOST"] = args.kv_host
        environment["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(args.kv_port)
    if args.plait_w4a16_library is not None:
        environment["PLAIT_W4A16_LIBRARY"] = str(args.plait_w4a16_library)
        environment["PLAIT_W4A16_SHA256"] = args.plait_w4a16_digest
    return environment


def launch_digest(command: list[str], engine_env: dict[str, str], endpoint: dict | None) -> bytes:
    spec = [command, engine_env] if engine_env else command
    if endpoint is not None:
        spec = [spec, endpoint]
    return hashlib.sha256(json.dumps(spec).encode()).digest()


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def signal_group(replica: Replica, sig: int) -> bool:
    try:
        os.killpg(replica.process.pid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        if sig == 0:
            # A permission error from a liveness probe does not prove exit.
            return True
        raise
    return True


def begin_stop(
    replica: Replica, stopping: dict[int, tuple[Replica, float]]
) -> None:
    if signal_group(replica, signal.SIGTERM):
        stopping[replica.index] = (replica, time.monotonic() + SHUTDOWN_GRACE)


def reap_stopping(stopping: dict[int, tuple[Replica, float]]) -> None:
    now = time.monotonic()
    for index, (replica, deadline) in list(stopping.items()):
        replica.process.poll()
        if not signal_group(replica, 0):
            del stopping[index]
        elif now >= deadline:
            signal_group(replica, signal.SIGKILL)
            del stopping[index]


def supervise(
    replicas: dict[int, Replica],
    stopping: dict[int, tuple[Replica, float]],
    tensor_parallel: int,
    should_stop: Callable[[], bool],
) -> int:
    last_ports: tuple[int, ...] | None = None
    while replicas and not should_stop():
        for index, replica in list(replicas.items()):
            exit_code = replica.process.poll()
            if exit_code is not None:
                reason = f"exited with {exit_code}"
            elif replica.ready:
                continue
            elif health(replica.port):
                if replica.process.poll() is not None:
                    reason = f"exited with {replica.process.returncode}"
                else:
                    replica.ready = True
                    print(f"ready replica={index} port={replica.port}", flush=True)
                    continue
            elif time.monotonic() >= replica.startup_deadline:
                reason = "startup timeout"
            else:
                continue
            print(f"replica={index} {reason}", file=sys.stderr, flush=True)
            del replicas[index]
            begin_stop(replica, stopping)

        ports = tuple(replica.port for replica in replicas.values() if replica.ready)
        if ports != last_ports and (ports or last_ports is not None or not replicas):
            print(
                json.dumps(
                    {
                        "replicas": len(ports),
                        "tp": tensor_parallel,
                        "gpus": len(ports) * tensor_parallel,
                        "endpoints": [f"http://127.0.0.1:{port}" for port in ports],
                    }
                ),
                flush=True,
            )
            last_ports = ports
        reap_stopping(stopping)
        if replicas:
            time.sleep(0.5)
    return 0 if should_stop() else 1


def main() -> int:
    args = parse_args()
    vllm = args.vllm or shutil.which("vllm")
    if not vllm:
        print("vllm executable not found; pass --vllm", file=sys.stderr)
        return 2
    engine_env = engine_environment(args)
    endpoint = None
    if args.advertise_http is not None:
        endpoint = {
            "api_model": args.served_model_name,
            "http": args.advertise_http,
            "prefill_kv": f"{args.kv_host}:{args.kv_port}" if args.role == "prefill" else None,
        }
    if args.print_launch_digest:
        command = native_command(args, vllm, args.backend_port)
        print(launch_digest(command, engine_env, endpoint).hex())
        return 0
    replicas: dict[int, Replica] = {}
    stopping: dict[int, tuple[Replica, float]] = {}
    base_env = os.environ.copy()
    base_env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    base_env.update(engine_env)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGTERM, request_stop)

    try:
        if args.group_id:
            command = native_command(args, vllm, args.backend_port)
            digest = list(launch_digest(command, engine_env, endpoint))
            return serving_group.run(
                args, command, digest, base_env, health, lambda: stop_requested, endpoint=endpoint,
            )
        for replica in range(args.replicas):
            if stop_requested:
                break
            first = args.gpu_start + replica * args.tp
            gpus = ",".join(str(first + offset) for offset in range(args.tp))
            port = args.port_base + replica
            command = native_command(args, vllm, port)
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus
            print(
                f"start replica={replica} gpus={gpus} port={port} tp={args.tp}",
                flush=True,
            )
            process = subprocess.Popen(command, env=env, start_new_session=True)
            replicas[replica] = Replica(
                replica,
                port,
                process,
                time.monotonic() + args.startup_timeout,
            )

        return supervise(replicas, stopping, args.tp, lambda: stop_requested)
    except KeyboardInterrupt:
        return 0
    except serving_group.ControlError as error:
        print(f"serving group: {error}", file=sys.stderr)
        return 1
    finally:
        for replica in replicas.values():
            begin_stop(replica, stopping)
        while stopping:
            reap_stopping(stopping)
            if stopping:
                time.sleep(0.1)
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    raise SystemExit(main())
