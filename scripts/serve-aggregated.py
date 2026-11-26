#!/usr/bin/env python3
"""Serve one OpenAI endpoint across versions of a committed Plait group."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serving_group import (
    DIRECTORY_POLL_SECONDS, DIRECTORY_READ_TTL, HOP_HEADERS, HTTP_ERRORS, INFERENCE_PATHS,
    MAX_JSON_BYTES, PRIVATE_HEADERS, Control, ControlError,
    Endpoint, admission_rejected, endpoint_ready,
)


class ActiveGroupDirectory:
    def __init__(self, model: str, group: bytes, control: Control) -> None:
        self.model = model
        self.group = list(group)
        self.control = control
        self.lock = threading.Lock()
        self.epoch = 0
        self.current: Endpoint | None = None
        self.until = 0.0

    def refresh(self) -> None:
        sent = time.monotonic()
        result = self.control.request({
            "request": "observe_group", "group": self.group, "id": self.control.next_operation_id(),
        })["result"]
        if result.get("Err") == {"NoActiveGroup": self.group}:
            with self.lock:
                self.current = None
            return

        view = result["Ok"]
        version = view["version"]
        epoch = version["epoch"]
        if version["group"] != self.group or type(epoch) is not int or epoch < 1:
            raise ValueError("control returned a different group")
        advertised = view["endpoint"]
        endpoint = None
        if (
            view["role"] == "Aggregated" and view["phase"] == "Active"
            and advertised and advertised["api_model"] == self.model
        ):
            endpoint = Endpoint.parse({
                "url": "http://" + advertised["http"],
                "version": f"{bytes(self.group).hex()}@{epoch}",
            }, "aggregated")

        with self.lock:
            if epoch < self.epoch or (epoch == self.epoch and self.current is None):
                return
            self.epoch = epoch
            self.current = endpoint
            self.until = sent + DIRECTORY_READ_TTL

    def endpoint(self) -> Endpoint | None:
        with self.lock:
            return self.current if time.monotonic() < self.until else None

    def watch(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                with contextlib.suppress(ControlError, ValueError, KeyError, TypeError, AttributeError):
                    self.refresh()
                stop.wait(DIRECTORY_POLL_SECONDS)
        finally:
            self.control.close()


def server(bind: str, port: int, directory: ActiveGroupDirectory, timeout: float) -> ThreadingHTTPServer:
    class Entry(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self.write_json(200, {"object": "list", "data": [{"id": directory.model, "object": "model"}]})
            elif self.path == "/health":
                endpoint = directory.endpoint()
                if endpoint is None or not endpoint_ready(endpoint, timeout):
                    self.fail(503, "active execution group is not ready")
                    return
                self.write_json(200, {}, endpoint.headers())
            else:
                self.fail(404, "unknown endpoint")

        def do_POST(self) -> None:
            if self.path not in INFERENCE_PATHS:
                self.fail(404, "unknown endpoint")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < size <= MAX_JSON_BYTES:
                    raise ValueError("expected a bounded Content-Length")
                self.connection.settimeout(timeout)
                raw = self.rfile.read(size)
                body = json.loads(raw)
                if len(raw) != size or not isinstance(body, dict) or body.get("model") != directory.model:
                    raise ValueError("expected a complete request for the configured model")
            except (OSError, ValueError) as error:
                self.fail(400, str(error))
                return

            deadline = time.monotonic() + timeout
            request_id = uuid.uuid4().hex
            while (budget := deadline - time.monotonic()) > 0:
                endpoint = directory.endpoint()
                if endpoint is not None and self.try_endpoint(endpoint, raw, request_id, budget):
                    return
                time.sleep(min(DIRECTORY_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
            self.fail(503, "no active execution group accepted the request")

        def try_endpoint(self, endpoint: Endpoint, raw: bytes, request_id: str, budget: float) -> bool:
            version_headers = endpoint.headers()
            headers = {**version_headers, "Content-Type": "application/json", "X-Request-Id": request_id}
            with contextlib.closing(endpoint.connect(min(2.0, budget))) as connection:
                try:
                    connection.connect()
                except HTTP_ERRORS:
                    # No request bytes have been sent.
                    return False

                try:
                    connection.sock.settimeout(budget)
                    connection.request("POST", self.path, body=raw, headers=headers)
                    response = connection.getresponse()
                except HTTP_ERRORS:
                    self.fail(502, "execution group disconnected after forwarding began")
                    return True

                with response:
                    if admission_rejected(response):
                        return False
                    if any(response.getheader(key) != value for key, value in version_headers.items()):
                        self.fail(502, "upstream response does not match the admitted group")
                        return True
                    with contextlib.suppress(*HTTP_ERRORS):
                        self.send_response(response.status, response.reason)
                        for key, value in response.getheaders():
                            if key.lower() not in HOP_HEADERS and key.lower() not in PRIVATE_HEADERS:
                                self.send_header(key, value)
                        for key, value in version_headers.items():
                            self.send_header(key, value)
                        self.end_headers()
                        while chunk := response.read1(65536):
                            self.wfile.write(chunk)
            return True

        def write_json(self, status: int, body: dict, headers: dict | None = None) -> None:
            data = json.dumps(body).encode()
            with contextlib.suppress(*HTTP_ERRORS):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

        def fail(self, status: int, message: str) -> None:
            self.write_json(status, {"error": {"message": message, "type": "plait_error"}})

    return ThreadingHTTPServer((bind, port), Entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="stable API model alias")
    parser.add_argument("--group-id", required=True, help="hex ID of the versioned aggregate group")
    parser.add_argument("--control-bin", default="plait-node")
    parser.add_argument("--control-peer", action="append", required=True)
    parser.add_argument("--control-listen", default="127.0.0.1:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        group = bytes.fromhex(args.group_id)
        if len(group) != 16 or len(args.group_id) != 32:
            raise ValueError("expected 32 hexadecimal digits for --group-id")
    except ValueError as error:
        parser.error(str(error))
    if not args.model or not 0 < args.port < 65536 or not 0 < args.timeout < float("inf"):
        parser.error("model, port and timeout must be valid")

    control = Control(args.control_bin, args.control_peer, args.control_listen)
    directory = ActiveGroupDirectory(args.model, group, control)
    stop = threading.Event()
    watcher = threading.Thread(target=directory.watch, args=(stop,))

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupt)
    watcher.start()
    try:
        with server(args.host, args.port, directory, args.timeout) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        watcher.join()
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    main()
