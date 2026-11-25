"""Run one committed, host-local native vLLM group behind an admission proxy."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import re
import select
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ADMISSION_SECONDS = 4.0
POLL_SECONDS = 0.5
VERSION_HEADER = "X-Plait-Group-Version"
ROLE_HEADER = "X-Plait-Group-Role"
HANDOFF_HEADER = "X-Plait-KV-Handoff"
ADMISSION_HEADER = "X-Plait-Admission"
RELEASE_KV_PATH = "/plait/release-kv"
INFERENCE_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})


class ControlError(RuntimeError):
    pass


class Rejected(ControlError):
    pass


class Control:
    def __init__(self, binary: str, peers: list[str], listen: str) -> None:
        self.argv = [binary, "serving", "--listen", listen]
        for peer in peers:
            self.argv.extend(("--peer", peer))
        self.timeout = 3.0 + 2.0 * len(peers)
        self.process: subprocess.Popen | None = None
        self.holder = list(uuid.uuid4().bytes)
        self.counter = 0

    def execute(self, action: dict, *, retry: bool = True) -> dict:
        self.counter += 1
        operation = {"id": {"origin": self.holder, "counter": self.counter}, "action": action}
        request = {"request": "execute", "operation": operation}
        try:
            return self.request(request)["result"]
        except ControlError:
            if not retry:
                raise
            return self.request(request)["result"]

    def request(self, request: dict) -> dict:
        try:
            if self.process is None:
                self.process = subprocess.Popen(
                    self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
                )
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            readable, _, _ = select.select([self.process.stdout], [], [], self.timeout)
            if not readable:
                raise ControlError("serving control timed out")
            line = self.process.stdout.readline()
            if not line:
                raise ControlError("serving control exited")
            reply = json.loads(line)
        except (OSError, ValueError, ControlError) as error:
            self.close()
            raise ControlError(str(error)) from error
        if "error" in reply:
            raise ControlError(reply["error"])
        return reply

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()


class Gate:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.until = 0.0
        self.running: set[socket.socket] = set()
        self.revoked = False
        self.transfers: dict[str, float] = {}

    def allow_until(self, deadline: float) -> None:
        with self.lock:
            if not self.revoked:
                self.until = deadline

    def close(self, *, revoke: bool = False) -> None:
        with self.lock:
            self.until = 0.0
            self.revoked = self.revoked or revoke
            connections = tuple(self.running) if revoke else ()
            if revoke:
                self.transfers.clear()
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)

    def admit(self, connection: socket.socket) -> bool:
        with self.lock:
            if self.revoked or time.monotonic() >= self.until:
                return False
            self.running.add(connection)
            return True

    def release(self, connection: socket.socket) -> None:
        with self.lock:
            self.running.remove(connection)

    def drained(self) -> bool:
        with self.lock:
            return not self.running and not self.transfers

    def hold_transfer(self, token: str, lease: float) -> None:
        with self.lock:
            if not self.revoked:
                self.transfers[token] = time.monotonic() + lease

    def release_transfer(self, token: str) -> None:
        with self.lock:
            self.transfers.pop(token, None)

    def expire_transfers(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.transfers = {token: end for token, end in self.transfers.items() if end > now}

    def may_stream(self) -> bool:
        with self.lock:
            return not self.revoked


HOP_HEADERS = frozenset({
    "connection", "content-length", "host", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
})
PRIVATE_HEADERS = frozenset(header.lower() for header in (
    VERSION_HEADER, ROLE_HEADER, HANDOFF_HEADER, ADMISSION_HEADER,
))


def handoff_token(value: str | None) -> str | None:
    if value is not None and re.fullmatch(r"[a-f0-9]{32}", value):
        return value
    return None


def group_proxy(
    port: int, backend_port: int, gate: Gate, version: str, *,
    bind: str = "127.0.0.1", role: str = "aggregated", kv_lease: float = 0,
) -> ThreadingHTTPServer:
    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.relay()

        def do_POST(self) -> None:
            self.relay()

        def relay(self) -> None:
            release = self.path == RELEASE_KV_PATH and role == "prefill" and self.command == "POST"
            native = self.path.startswith("/v1/") if role == "aggregated" else (
                (self.command == "POST" and self.path in INFERENCE_PATHS) or self.path == "/v1/models"
            )
            if not release and self.path != "/health" and not native:
                self.send_error(404)
                return
            requested = self.headers.get(VERSION_HEADER)
            required = release or (role != "aggregated" and self.path in INFERENCE_PATHS)
            if requested != version and (requested is not None or required):
                self.reject(409, "stale or missing execution group version")
                return
            requested_role = self.headers.get(ROLE_HEADER)
            if requested_role is not None and requested_role != role:
                self.reject(409, "execution group role changed")
                return
            token = handoff_token(self.headers.get(HANDOFF_HEADER))
            if release:
                if token is None:
                    self.send_error(400, "invalid KV handoff token")
                    return
                gate.release_transfer(token)
                self.send_response(204)
                self.end_headers()
                return
            if role == "prefill" and self.path in INFERENCE_PATHS and token is None:
                self.send_error(400, "missing KV handoff token")
                return
            if not gate.admit(self.connection):
                self.reject(503, "execution group is not admitting requests")
                return
            try:
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header(VERSION_HEADER, version)
                    self.send_header(ROLE_HEADER, role)
                    self.end_headers()
                    return
                self.forward(token if role == "prefill" and self.path in INFERENCE_PATHS else None)
            finally:
                gate.release(self.connection)

        def reject(self, status: int, reason: str) -> None:
            self.send_response(status, reason)
            self.send_header(ADMISSION_HEADER, "rejected")
            self.end_headers()

        def forward(self, transfer: str | None) -> None:
            upstream = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=60)
            sent_headers = False
            try:
                if self.headers.get("Transfer-Encoding"):
                    self.send_error(411)
                    return
                self.connection.settimeout(60)
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    self.send_error(400)
                    return
                body = self.rfile.read(length) if length else None
                headers = {
                    k: v for k, v in self.headers.items()
                    if k.lower() not in HOP_HEADERS and k.lower() not in PRIVATE_HEADERS
                }
                upstream.request(self.command, self.path, body=body, headers=headers)
                response = upstream.getresponse()
                if not gate.may_stream():
                    self.send_error(503, "execution group was revoked")
                    return
                if transfer is not None and response.status == 200:
                    # The native P response exports blocks that D has yet to read.
                    gate.hold_transfer(transfer, kv_lease)
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in HOP_HEADERS and key.lower() not in PRIVATE_HEADERS:
                        self.send_header(key, value)
                self.end_headers()
                sent_headers = True
                # HTTP/1.0 uses connection close to delimit the dechunked response.
                while gate.may_stream():
                    chunk = response.read1(65536)
                    if not chunk or not gate.may_stream():
                        break
                    self.wfile.write(chunk)
            except (OSError, ValueError, http.client.HTTPException):
                if not sent_headers:
                    with contextlib.suppress(OSError):
                        self.send_error(502)
            finally:
                upstream.close()

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    return ThreadingHTTPServer((bind, port), Proxy)


def expect_phase(result: dict) -> str:
    if "Err" in result:
        raise Rejected(f"serving operation rejected: {result['Err']}")
    return result["Ok"]["Group"]["phase"]


def stop_process_group(process: subprocess.Popen, grace: float) -> bool:
    def send(sig: int) -> bool:
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            if sig == 0:
                return True
            raise

    stopped_at = time.monotonic() + grace
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not send(sig):
            process.poll()
            return True
        while time.monotonic() < stopped_at:
            process.poll()
            if not send(0):
                return True
            time.sleep(0.05)
        stopped_at = time.monotonic() + 1.0
    return False


def run(
    args: argparse.Namespace,
    command: list[str],
    launch_digest: list[int],
    env: dict[str, str],
    probe: Callable[[int], bool],
    should_stop: Callable[[], bool],
    *,
    endpoint: dict | None = None,
) -> int:
    control = Control(args.control_bin, args.control_peer, args.control_listen)
    version = {"group": list(bytes.fromhex(args.group_id)), "epoch": args.group_epoch}
    host = list(bytes.fromhex(args.host_id))
    identity = {"version": version, "holder": control.holder}
    gate = Gate()
    process = None
    claimed = False
    server = None
    proxy_thread = None
    phase = None
    drain_until = None

    def change(name: str, **fields: object) -> str:
        return expect_phase(control.execute({name: {**identity, **fields}}, retry=name != "ObserveLaunch"))

    try:
        view = control.request({"request": "inspect", "version": version})["group"]
        if not view or view["phase"] != "Preparing":
            raise ControlError("expected a prepared execution group")
        if (
            view["version"] != version
            or view["role"] != args.role.capitalize()
            or view["model"] != args.model
            or view["tensor_parallel"] != args.tp
            or view["pipeline_parallel"] != 1
            or view["launch_digest"] != launch_digest
        ):
            raise ControlError("the native launch configuration differs from the committed plan")
        grants = view["grants"]
        if len(grants) != args.tp or any(grant["device"]["node"] != host for grant in grants):
            raise ControlError("all grants must belong to the local host")
        if len({grant["device"]["ordinal"] for grant in grants}) != args.tp:
            raise ControlError("duplicate device ordinal in group grants")

        server = group_proxy(
            args.port_base, args.backend_port, gate, f"{args.group_id.lower()}@{args.group_epoch}",
            bind=args.proxy_host, role=args.role, kv_lease=args.kv_lease_seconds,
        )
        with socket.socket() as port_check:
            port_check.bind(("127.0.0.1", args.backend_port))
        phase = expect_phase(control.execute({
            "ClaimLaunch": {**identity, "host": host, "launch_digest": launch_digest},
        }))
        claimed = True
        if should_stop():
            return 0
        gpus = ",".join(str(grant["device"]["ordinal"]) for grant in grants)
        env["CUDA_VISIBLE_DEVICES"] = gpus
        process = subprocess.Popen(command, env=env, start_new_session=True)
        proxy_thread = threading.Thread(target=server.serve_forever, daemon=True)
        proxy_thread.start()
        deadline = time.monotonic() + args.startup_timeout
        announced = False
        while not should_stop() and process.poll() is None:
            gate.expire_transfers()
            if drain_until is not None and (gate.drained() or time.monotonic() >= drain_until):
                return 0
            if not announced and probe(args.backend_port) and process.poll() is None:
                phase = change("LaunchReady") if endpoint is None else change("LaunchReadyAt", endpoint=endpoint)
                announced = True

            sent = time.monotonic()
            try:
                phase = change("ObserveLaunch")
            except Rejected:
                raise
            except ControlError:
                # Already admitted streams can finish while the last admission window expires.
                if not announced and time.monotonic() >= deadline:
                    raise
                time.sleep(POLL_SECONDS)
                continue

            if phase == "Active":
                if announced and process.poll() is None and probe(args.backend_port):
                    gate.allow_until(sent + ADMISSION_SECONDS)
                else:
                    gate.close()
            elif phase in ("Revoked", "Retired"):
                return 1
            elif phase == "Draining":
                gate.close()
                if drain_until is None:
                    drain_until = time.monotonic() + args.drain_timeout

            if not announced and time.monotonic() >= deadline:
                raise ControlError("vLLM startup timed out")
            time.sleep(POLL_SECONDS)
        return 0 if should_stop() else 1
    finally:
        try:
            gate.close(revoke=True)
            if server is not None:
                if proxy_thread is not None:
                    server.shutdown()
                    proxy_thread.join()
                server.server_close()
            exited = process is None or stop_process_group(process, 20.0)
            if claimed and phase not in ("Draining", "Revoked", "Retired"):
                with contextlib.suppress(ControlError):
                    control.execute({"Revoke": version}, retry=False)
            deadline = time.monotonic() + 1.0
            while not gate.drained() and time.monotonic() < deadline:
                time.sleep(0.05)
            if claimed and exited and gate.drained():
                if process is None:
                    released = True
                elif args.verify_release:
                    # Process-group exit alone cannot attest device and transport cleanup.
                    proof = subprocess.run(
                        [args.verify_release],
                        input=json.dumps({"version": version, "grants": grants}) + "\n",
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    released = proof.returncode == 0
                else:
                    released = False
                if released:
                    change("LaunchStopped")
                    workers = [grant["device"] for grant in grants]
                    expect_phase(control.execute({"Retire": {"version": version, "workers": workers}}))
        finally:
            control.close()
