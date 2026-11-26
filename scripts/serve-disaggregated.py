#!/usr/bin/env python3
"""Route OpenAI completions through versioned vLLM P/D groups."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import ipaddress
import itertools
import json
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from serving_group import (
    DIRECTORY_POLL_SECONDS, DIRECTORY_READ_TTL, HANDOFF_HEADER, HTTP_ERRORS,
    INFERENCE_PATHS, MAX_JSON_BYTES, RELEASE_KV_PATH, Control, ControlError,
    Endpoint, admission_rejected, endpoint_ready,
)

class UpstreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class Route:
    prefill: Endpoint
    decode: Endpoint
    kv_host: str
    kv_port: int

    @classmethod
    def parse(cls, config: dict) -> Route:
        prefill = Endpoint.parse(config["prefill"], "prefill")
        decode = Endpoint.parse(config["decode"], "decode")
        host = ipaddress.IPv4Address(config["prefill"]["kv_host"])
        port = config["prefill"]["kv_port"]
        if type(port) is not int or not 0 < port < 65536 or prefill.version == decode.version:
            raise ValueError("invalid KV source or execution group pair")
        return cls(prefill, decode, str(host), port)


class Directory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def routes(self) -> tuple[str, list[Route]]:
        return parse_directory(json.loads(self.path.read_text()))


def parse_directory(document: dict) -> tuple[str, list[Route]]:
    model = document["model"]
    if not isinstance(model, str) or not model:
        raise ValueError("route directory requires a model")
    return model, [Route.parse(item) for item in document["paths"]]


class ControlDirectory:
    def __init__(self, model: str, control: Control) -> None:
        self.model = model
        self.control = control
        self.lock = threading.Lock()
        self.revision = 0
        self.paths: list[Route] = []
        self.until = 0.0

    def refresh(self) -> None:
        sent = time.monotonic()
        result = self.control.execute({"ObserveRoutes": {"model": self.model}}, retry=False)
        if "Ok" not in result:
            return
        observed = result["Ok"]["RoutesObserved"]["revision"]
        minimum = max(self.revision, observed)
        document = self.control.request({
            "request": "routes", "model": self.model, "minimum_revision": minimum,
        })["directory"]
        if document is None:
            return
        model, routes = parse_directory(document)
        revision = document["revision"]
        if model != self.model or type(revision) is not int or revision < 1:
            raise ValueError("unexpected control directory")
        with self.lock:
            if revision < minimum:
                return
            if revision == self.revision:
                observed = set(routes)
                # A group version that has stopped accepting cannot be active again.
                routes = [route for route in self.paths if route in observed]
            self.revision = revision
            self.paths = routes
            self.until = sent + DIRECTORY_READ_TTL

    def routes(self) -> tuple[str, list[Route]]:
        with self.lock:
            if time.monotonic() >= self.until:
                raise ValueError("routing directory is unavailable or expired")
            return self.model, self.paths

    def watch(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                with contextlib.suppress(ControlError, ValueError, KeyError, TypeError, AttributeError):
                    self.refresh()
                stop.wait(DIRECTORY_POLL_SECONDS)
        finally:
            self.control.close()


def json_object(raw: bytes) -> dict:
    item = json.loads(raw)
    if not isinstance(item, dict):
        raise ValueError("expected a JSON object")
    return item


def read_json(response: http.client.HTTPResponse) -> dict:
    body = response.read(MAX_JSON_BYTES + 1)
    if len(body) > MAX_JSON_BYTES:
        raise UpstreamError("engine response exceeds the JSON size limit")
    try:
        return json_object(body)
    except ValueError as error:
        raise UpstreamError("engine returned an invalid JSON response") from error


def prefill_payload(request: dict, plait_ticket: str | None = None) -> dict:
    payload = {**request, "stream": False}
    payload.pop("stream_options", None)
    payload["max_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    if "min_tokens" in payload:
        payload["min_tokens"] = 0
    payload["kv_transfer_params"] = (
        {"plait_handoff": {"ticket": plait_ticket}}
        if plait_ticket is not None
        else {"do_remote_decode": True, "do_remote_prefill": False}
    )
    return payload


def transfer_descriptor(response: dict, route: Route, plait_ticket: str | None = None) -> dict:
    params = response.get("kv_transfer_params")
    if plait_ticket is not None:
        receipt = params.get("plait_handoff") if isinstance(params, dict) else None
        if (
            not isinstance(receipt, dict)
            or receipt.get("ticket") != plait_ticket
            or receipt.get("store") != f"{route.kv_host}:{route.kv_port}"
            or not isinstance(receipt.get("prompt_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", receipt["prompt_sha256"]) is None
            or type(receipt.get("cached_tokens")) is not int
            or receipt["cached_tokens"] < 0
        ):
            raise UpstreamError("prefill did not publish the requested Plait KV handoff")
        return {"plait_handoff": receipt}
    if (
        not isinstance(params, dict)
        or params.get("do_remote_prefill") is not True
        or not params.get("remote_request_id") or not params.get("remote_engine_id")
        or not params.get("remote_block_ids") or params.get("remote_host") != route.kv_host
        or params.get("remote_port") != route.kv_port
    ):
        raise UpstreamError("prefill did not export KV for the configured source")
    return params


def release_transfer(endpoint: Endpoint, token: str) -> None:
    headers = {**endpoint.headers(), HANDOFF_HEADER: token}
    with contextlib.suppress(*HTTP_ERRORS), contextlib.closing(endpoint.connect(1.0)) as connection:
        connection.request("POST", RELEASE_KV_PATH, body=b"", headers=headers)
        with connection.getresponse() as response:
            response.read()


def request_engine(
    connection: http.client.HTTPConnection, endpoint: Endpoint, path: str,
    body: dict, request_id: str, handoff: str | None = None,
) -> http.client.HTTPResponse:
    headers = {**endpoint.headers(), "Content-Type": "application/json", "X-Request-Id": request_id}
    if handoff is not None:
        headers[HANDOFF_HEADER] = handoff
    connection.request("POST", path, body=json.dumps(body).encode(), headers=headers)
    return connection.getresponse()


def server(
    bind: str, port: int, directory: Directory | ControlDirectory, timeout: float,
    kv_transport: str = "nixl",
) -> ThreadingHTTPServer:
    next_path = itertools.count()

    class Entry(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            config = self.read_directory()
            if config is None:
                return
            model, routes = config
            if self.path == "/v1/models":
                self.write_json(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            elif self.path == "/health" and routes:
                self.write_json(200, {})
            elif self.path == "/health":
                self.fail(503, "no P/D routes are configured")
            else:
                self.fail(404, "unknown endpoint")

        def do_POST(self) -> None:
            if self.path not in INFERENCE_PATHS:
                self.fail(404, "unknown endpoint")
                return
            config = self.read_directory()
            if config is None:
                return
            model, routes = config
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 < size <= MAX_JSON_BYTES:
                    raise ValueError("expected a bounded Content-Length")
                self.connection.settimeout(timeout)
                body = json_object(self.rfile.read(size))
                prompt = body.get("prompt")
                if body.get("model") != model or body.get("n", 1) != 1:
                    raise ValueError("expected the configured model and one completion")
                if isinstance(prompt, list) and any(type(token) is not int for token in prompt):
                    raise ValueError("batched prompts need separate P/D requests")
                body.pop("kv_transfer_params", None)
            except (OSError, ValueError) as error:
                self.fail(400, str(error))
                return

            deadline = time.monotonic() + timeout
            if routes:
                start = next(next_path) % len(routes)
                routes = routes[start:] + routes[:start]
            for route in routes:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    break
                if not endpoint_ready(route.decode, budget):
                    continue
                try:
                    if self.try_route(route, body, deadline):
                        return
                except (OSError, http.client.HTTPException, UpstreamError) as error:
                    self.fail(502, str(error))
                    return
            self.fail(503, "no P/D route accepted the request")

        def try_route(self, route: Route, body: dict, deadline: float) -> bool:
            request_id, token = uuid.uuid4().hex, uuid.uuid4().hex
            plait_ticket = token if kv_transport == "plait" else None
            budget = max(0.001, deadline - time.monotonic())
            with contextlib.closing(route.prefill.connect(budget)) as connection:
                with request_engine(
                    connection, route.prefill, self.path, prefill_payload(body, plait_ticket), request_id, token,
                ) as response:
                    if admission_rejected(response):
                        return False
                    if response.status != 200:
                        self.write_json(response.status, read_json(response))
                        return True
                    try:
                        params = transfer_descriptor(read_json(response), route, plait_ticket)
                    except UpstreamError:
                        release_transfer(route.prefill, token)
                        raise

            budget = deadline - time.monotonic()
            if budget <= 0:
                release_transfer(route.prefill, token)
                raise UpstreamError("request deadline elapsed before decode")
            with contextlib.closing(route.decode.connect(budget)) as connection:
                with request_engine(
                    connection, route.decode, self.path, {**body, "kv_transfer_params": params}, request_id,
                ) as response:
                    if admission_rejected(response):
                        release_transfer(route.prefill, token)
                        return False
                    if self.write_decode(response):
                        release_transfer(route.prefill, token)
            return True

        def write_decode(self, response: http.client.HTTPResponse) -> bool:
            content_type = response.getheader("Content-Type", "")
            if not content_type.startswith("text/event-stream"):
                content = read_json(response)
                content.pop("kv_transfer_params", None)
                self.write_json(response.status, content)
                return response.status == 200 and "error" not in content

            done = False
            failed = False
            try:
                self.send_response(response.status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                while line := response.readline(MAX_JSON_BYTES):
                    self.wfile.write(line)
                    if line.startswith(b"data:"):
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            done = True
                        else:
                            try:
                                failed = "error" in json_object(payload) or failed
                            except ValueError:
                                failed = True
            except HTTP_ERRORS:
                return False
            return response.status == 200 and done and not failed

        def read_directory(self) -> tuple[str, list[Route]] | None:
            try:
                return directory.routes()
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                self.fail(503, "P/D route configuration is unavailable")
                return None

        def write_json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            with contextlib.suppress(*HTTP_ERRORS):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def fail(self, status: int, message: str) -> None:
            kind = "invalid_request_error" if status < 500 else "server_error"
            self.write_json(status, {"error": {"message": message, "type": kind}})

    return ThreadingHTTPServer((bind, port), Entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--routes", type=Path, help="JSON file with versioned P/D paths")
    source.add_argument("--model", help="model alias published in the replicated directory")
    parser.add_argument("--control-bin", default="plait-node")
    parser.add_argument("--control-peer", action="append", default=[])
    parser.add_argument("--control-listen", default="127.0.0.1:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--kv-transport", choices=("nixl", "plait"), default="nixl")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not 0 < args.port < 65536 or not 0 < args.timeout < float("inf"):
        parser.error("port and timeout must be valid")
    if args.routes is None and not args.control_peer:
        parser.error("--model requires at least one --control-peer")

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        with route_source(args) as directory, server(
            args.host, args.port, directory, args.timeout, args.kv_transport,
        ) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, old_handler)


@contextlib.contextmanager
def route_source(args: argparse.Namespace):
    if args.routes is not None:
        yield Directory(args.routes)
        return
    directory = ControlDirectory(args.model, Control(args.control_bin, args.control_peer, args.control_listen))
    stop = threading.Event()
    watcher = threading.Thread(target=directory.watch, args=(stop,), name="plait-route-directory")
    watcher.start()
    try:
        yield directory
    finally:
        stop.set()
        watcher.join()


if __name__ == "__main__":
    main()
