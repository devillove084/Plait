"""Regression tests for P/D handoff admission, drain, and route replacement."""

import contextlib
import http.client
import json
import runpy
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serving_group import (
    ADMISSION_HEADER, HANDOFF_HEADER, RELEASE_KV_PATH, VERSION_HEADER, Gate, group_proxy, run,
)


ROUTER = runpy.run_path(str(Path(__file__).with_name("serve-disaggregated.py")))
PREFILL, DECODE, SPARE = (f"{number:032x}@1" for number in range(1, 4))
TOKEN = "f" * 32
COMPLETIONS = "/v1/completions"


class LaunchTest(unittest.TestCase):
    def test_child_uses_the_claimed_version_even_if_parent_has_another_epoch(self):
        group, host = "a" * 32, "b" * 32
        version = {"group": list(bytes.fromhex(group)), "epoch": 7}
        digest = [2] * 32
        args = SimpleNamespace(
            control_bin="plait-node", control_listen="127.0.0.1:0", control_peer=[],
            group_id=group, group_epoch=7, host_id=host, role="aggregated", model="model",
            tp=1, port_base=8000, backend_port=9000, proxy_host="127.0.0.1",
            kv_lease_seconds=30, startup_timeout=1, verify_release=None,
        )
        control = Mock()
        control.request.return_value = {"group": {
            "phase": "Preparing", "version": version, "role": "Aggregated", "model": "model",
            "tensor_parallel": 1, "pipeline_parallel": 1, "launch_digest": digest,
            "grants": [{"device": {"node": list(bytes.fromhex(host)), "ordinal": 2}}],
        }}
        control.execute.return_value = {"Ok": {"Group": {"phase": "Preparing"}}}
        env = {"PLAIT_TP_GROUP_VERSION": f"{group}@6"}
        command = ["vllm", "serve", "model"]
        with (
            patch("serving_group.Control", return_value=control),
            patch("serving_group.group_proxy"),
            patch("serving_group.socket.socket"),
            patch("serving_group.subprocess.Popen") as process,
            patch("serving_group.stop_process_group", return_value=True),
        ):
            process.return_value.poll.return_value = 0
            run(args, command, digest, env, lambda _: True, lambda: False)

        claim = control.execute.call_args_list[0].args[0]["ClaimLaunch"]
        self.assertEqual(claim["version"], version)
        self.assertEqual(process.call_args.kwargs["env"]["PLAIT_TP_GROUP_VERSION"], f"{group}@7")
        self.assertEqual(env["PLAIT_TP_GROUP_VERSION"], f"{group}@6")


class CalibrationTest(unittest.TestCase):
    def test_requires_repeatable_slo_and_does_not_skip_an_overloaded_rate(self):
        benchmark = runpy.run_path(str(Path(__file__).with_name("benchmark-serving.py")))

        def trial(good, rps, end):
            return {
                "good_requests": good, "arrivals": 100, "finished_at_ms": end,
                "metrics_interval_s": 10, "kv_bytes": 2400,
                "benchmark": {"completed": 100, "request_goodput": rps},
            }

        points = [
            {"offered_rps": 8, "trials": [trial(98, 8.1, 100), trial(97, 8.2, 200)]},
            {"offered_rps": 16, "trials": [trial(98, 15.4, 400), trial(93, 14.1, 500)]},
            {"offered_rps": 24, "trials": [trial(97, 23.1, 600), trial(96, 22.9, 800)]},
        ]
        result = benchmark["summarize"](points, .95, 1000)
        self.assertEqual(result["selected_rate"], 8)
        self.assertEqual(result["slo_profile"], {"rps": 8, "valid_until_ms": 1100})
        self.assertEqual(result["transfer_profile"]["bytes_per_request"], 24)
        self.assertEqual([point["passes"] for point in points], [True, False, True])


class DirectoryTest(unittest.TestCase):
    def test_stale_replicas_cannot_restore_an_older_or_drained_path(self):
        def path(decode):
            return {
                "prefill": {"url": "http://127.0.0.1:8100", "version": PREFILL,
                            "kv_host": "127.0.0.1", "kv_port": 9100},
                "decode": {"url": "http://127.0.0.1:8101", "version": decode},
            }

        class Replica:
            replies = iter([
                (1, [path(DECODE)]), (2, [path(SPARE)]), (1, [path(DECODE)]),
                (2, []), (2, [path(SPARE)]),
            ])

            def execute(self, _action, *, retry):
                return {"Ok": {"RoutesObserved": {"revision": 1}}}

            def request(self, _request):
                revision, paths = next(self.replies)
                return {"directory": {"model": "test", "revision": revision, "paths": paths}}

        directory = ROUTER["ControlDirectory"]("test", Replica())
        for expected in ([DECODE], [SPARE], [SPARE], [], []):
            directory.refresh()
            self.assertEqual([route.decode.version for route in directory.routes()[1]], expected)
        directory.until = 0
        with self.assertRaises(ValueError):
            directory.routes()


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.runtime = contextlib.ExitStack()
        self.addCleanup(self.runtime.close)

    def start(self, server):
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        self.runtime.callback(server.server_close)
        self.runtime.callback(thread.join)
        self.runtime.callback(server.shutdown)
        return server.server_port

    def backend(self, respond):
        class Native(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                content_type, body = respond(body)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                pass

        return self.start(ThreadingHTTPServer(("127.0.0.1", 0), Native))

    def group(self, role, version, respond):
        gate = Gate()
        gate.allow_until(time.monotonic() + 30)
        port = self.start(group_proxy(0, self.backend(respond), gate, version, role=role, kv_lease=30))
        return port, gate

    def request(self, port, path, body=None, headers=None):
        with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port, timeout=2)) as conn:
            conn.request("POST", path, None if body is None else json.dumps(body).encode(), headers or {})
            with conn.getresponse() as response:
                return response.status, response.headers, response.read()

    def test_drain_keeps_export_alive_until_versioned_release(self):
        port, gate = self.group("prefill", PREFILL, lambda _: ("application/json", b"{}"))
        headers = {VERSION_HEADER: PREFILL, HANDOFF_HEADER: TOKEN}
        self.assertEqual(self.request(port, COMPLETIONS, {}, headers)[0], 200)
        gate.close()
        self.assertFalse(gate.drained())
        status, fields, _ = self.request(port, COMPLETIONS, {}, headers)
        self.assertEqual((status, fields[ADMISSION_HEADER]), (503, "rejected"))
        stale = {**headers, VERSION_HEADER: PREFILL.replace("@1", "@2")}
        self.assertEqual(self.request(port, RELEASE_KV_PATH, headers=stale)[0], 409)
        self.assertFalse(gate.drained())
        self.assertEqual(self.request(port, RELEASE_KV_PATH, headers=headers)[0], 204)
        self.assertTrue(gate.drained())

    def test_explicit_decode_rejection_can_use_another_route(self):
        unused_d, closing_gate = self.group("decode", DECODE, lambda _: self.fail("old D ran"))
        stream = b'data: {"choices": []}\n\ndata: [DONE]\n\n'
        exported = []
        decoded = []

        def prefill(body):
            self.assertEqual(body["kv_transfer_params"], {
                "do_remote_decode": True, "do_remote_prefill": False,
            })
            self.assertEqual(body["max_tokens"], 1)
            self.assertIs(body["stream"], False)
            closing_gate.close()
            descriptor = {
                "do_remote_prefill": True, "remote_request_id": f"request-{len(exported)}",
                "remote_engine_id": "engine", "remote_block_ids": [[1]],
                "remote_host": "127.0.0.1", "remote_port": 9100,
            }
            exported.append(descriptor)
            return "application/json", json.dumps({"kv_transfer_params": descriptor}).encode()

        def decode(body):
            decoded.append(body)
            return "text/event-stream", stream

        p_port, p_gate = self.group("prefill", PREFILL, prefill)
        d_port, _ = self.group("decode", SPARE, decode)
        directory = Path(self.runtime.enter_context(tempfile.TemporaryDirectory())) / "routes.json"
        endpoint = {"url": f"http://127.0.0.1:{p_port}", "version": PREFILL,
                    "kv_host": "127.0.0.1", "kv_port": 9100}
        directory.write_text(json.dumps({"model": "test", "paths": [
            {"prefill": endpoint, "decode": {"url": f"http://127.0.0.1:{port}", "version": version}}
            for port, version in ((unused_d, DECODE), (d_port, SPARE))
        ]}))
        router_port = self.start(ROUTER["server"]("127.0.0.1", 0, ROUTER["Directory"](directory), 5))
        status, _, body = self.request(router_port, COMPLETIONS, {
            "model": "test", "prompt": "test", "stream": True, "max_tokens": 8,
            "kv_transfer_params": {"remote_host": "untrusted"},
        })
        self.assertEqual((status, body), (200, stream))
        self.assertEqual(len(exported), 2)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["kv_transfer_params"], exported[1])
        self.assertEqual(decoded[0]["max_tokens"], 8)
        # The final ACK runs before the router closes the response socket.
        self.assertTrue(p_gate.drained())

        # vLLM may also send the terminal marker after a streaming error.
        stream = b'data: {"error": {"message": "decode failed"}}\n\ndata: [DONE]\n\n'
        status, _, body = self.request(router_port, COMPLETIONS, {
            "model": "test", "prompt": "next request", "stream": True, "max_tokens": 8,
        })
        self.assertEqual((status, body), (200, stream))
        self.assertFalse(p_gate.drained())


if __name__ == "__main__":
    unittest.main()
