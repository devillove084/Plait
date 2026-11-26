"""Regression tests for versioned serving, drain, and route replacement."""

import contextlib
import http.client
import json
import os
import runpy
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serving_group import (
    ADMISSION_HEADER, HANDOFF_HEADER, RELEASE_KV_PATH, VERSION_HEADER,
    ControlError, Endpoint, Gate, check_backend_port, group_proxy, record_host_stop, run,
)
from serving_launch import host_command, resident_spec
from serving_load import Load
from serving_replay import RECOVERY_HEADER, RECOVERY_MODE, sse


ROUTER = runpy.run_path(str(Path(__file__).with_name("serve-disaggregated.py")))
INGRESS = runpy.run_path(str(Path(__file__).with_name("serve-aggregated.py")))
PREFILL, DECODE, SPARE = (f"{number:032x}@1" for number in range(1, 4))
TOKEN = "f" * 32
COMPLETIONS = "/v1/completions"


class LaunchTest(unittest.TestCase):
    def test_amd_kfd_release_uses_the_device_identity_captured_before_launch(self):
        group, host = "a" * 32, "b" * 32
        node = list(bytes.fromhex(host))
        version = {"group": list(bytes.fromhex(group)), "epoch": 7}
        grants = [{"version": version, "device": {"node": node, "ordinal": 1},
                   "incarnation": 1, "fence": 2}]
        monitor = {"driver": "amd_kfd", "program": "/vendor/amd-smi"}
        args = SimpleNamespace(
            control_bin="plait-node", control_listen=None, control_peer=[], group_id=group,
            group_epoch=7, host_id=host, role="decode", model="model", tp=1, port_base=8000,
            backend_port=9000, proxy_host="127.0.0.1", kv_lease_seconds=30, startup_timeout=1,
            drain_timeout=1, verify_release=None, release_driver=monitor,
        )
        spec = {"api_host": host, "master_addr": "192.0.2.2", "master_port": 9008,
                "hosts": {host: {"address": "192.0.2.2", "argv": ["vllm"],
                                 "model_path": "/model", "device_env": "HIP_VISIBLE_DEVICES"}}}
        digest = [2] * 32
        control = Mock()
        control.request.return_value = {"group": {"phase": "Preparing", "version": version,
            "role": "Decode", "model": "model", "tensor_parallel": 1, "pipeline_parallel": 1,
            "launch_digest": digest, "grants": grants}}
        control.execute.return_value = {"Ok": {"Group": {"phase": "Preparing"}}}
        identities = [{"hip": 1, "management": 7, "kfd": 30, "node": 3,
                       "render": 129, "bdf": "0000:02:00.0"}]
        proof_record = {"monitor": monitor, "host": node, "version": version, "grants": grants}
        calls = []

        def verify(argv, *, input, env, **_kwargs):
            self.assertNotIn("HIP_VISIBLE_DEVICES", env)
            action = argv[-1]
            calls.append(action)
            if action == "capture-host-release":
                process.assert_not_called()
                self.assertEqual(json.loads(input), proof_record)
                return SimpleNamespace(stdout=json.dumps(identities))
            stopped.assert_called_once()
            self.assertEqual(json.loads(input), {**proof_record, "identities": identities})
            return SimpleNamespace(returncode=0)

        with (
            patch("serving_group.Control", return_value=control), patch("serving_group.group_proxy"),
            patch("serving_group.check_backend_port"), patch("serving_group.subprocess.run", side_effect=verify),
            patch("serving_group.subprocess.Popen") as process,
            patch("serving_group.stop_process_group", return_value=True) as stopped,
            patch("serving_group.record_host_stop") as retire,
        ):
            process.return_value.poll.return_value = 0
            run(args, ["vllm", "serve", "model"], digest, {"HIP_VISIBLE_DEVICES": "5"},
                lambda _: True, lambda: False, host_launch_spec=spec)
        self.assertEqual(calls, ["capture-host-release", "verify-host-release"])
        self.assertEqual(process.call_args.kwargs["env"]["HIP_VISIBLE_DEVICES"], "1")
        retire.assert_called_once()

    def test_backend_probe_accepts_a_drained_port_and_rejects_an_active_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaises(OSError):
                check_backend_port(port)
            with socket.create_connection(listener.getsockname(), timeout=2) as client:
                with listener.accept()[0] as server:
                    server.settimeout(2)
                    server.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(server.recv(1), b"")

        check_backend_port(port)

    def test_single_device_hosted_pd_keeps_vllm_default_executor(self):
        host = "b" * 32
        recipe = {"api_host": list(bytes.fromhex(host)), "master_port": 9010,
                  "hosts": [{"host": list(bytes.fromhex(host)), "address": "192.0.2.1"}]}
        profile = {"argv": ["vllm"], "model_path": "/local/model",
                   "device_env": "CUDA_VISIBLE_DEVICES", "args": [], "env": {}}
        spec = resident_spec(recipe, profile, pd=True)
        command = ["vllm", "serve", "model", "--tensor-parallel-size", "1"]
        self.assertEqual(host_command(command, spec, host, 0, [1]),
                         ["vllm", "serve", "/local/model", "--tensor-parallel-size", "1"])
        self.assertIn("--distributed-executor-backend", host_command(command, spec, host, 0, [2]))
        with self.assertRaisesRegex(ValueError, "supplies the KV connector and block size"):
            resident_spec(recipe, {**profile, "args": ["--block_size=256"]}, pd=True)

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

    def test_follower_uses_local_grants_and_reports_readiness_from_its_child(self):
        group, host, head = "a" * 32, "b" * 32, "c" * 32
        version = {"group": list(bytes.fromhex(group)), "epoch": 3}
        digest = [2] * 32
        args = SimpleNamespace(
            control_bin="plait-node", control_listen=None, control_peer=[],
            group_id=group, group_epoch=3, host_id=host, role="aggregated", model="model",
            tp=3, backend_port=9000, startup_timeout=1, verify_release=None,
        )
        spec = {
            "api_host": head, "master_addr": "192.0.2.1", "master_port": 9008,
            "hosts": {
                head: {"address": "192.0.2.1"},
                host: {"address": "192.0.2.2", "argv": ["vllm"], "model_path": "/local/model",
                       "device_env": "ASCEND_RT_VISIBLE_DEVICES"},
            },
        }
        control = Mock()
        control.request.return_value = {"group": {
            "phase": "Preparing", "version": version, "role": "Aggregated", "model": "model",
            "tensor_parallel": 3, "pipeline_parallel": 1, "launch_digest": digest,
            "grants": [
                {"device": {"node": list(bytes.fromhex(node)), "ordinal": ordinal}}
                for node, ordinal in ((host, 6), (head, 1), (host, 2))
            ],
        }}
        probe = Mock()
        stopped = False

        def execute(action, **_kwargs):
            nonlocal stopped
            if "ObserveHostLaunch" in action:
                stopped = True
            return {"Ok": {"Group": {"phase": "Preparing"}}}

        control.execute.side_effect = execute
        child = Mock()
        child.poll.return_value = None

        def start(_command, *, pass_fds, **_kwargs):
            os.write(pass_fds[0], signal)
            return child

        for signal in (b"x", b"1"):
            with (
                self.subTest(signal=signal),
                patch("serving_group.Control", return_value=control) as factory,
                patch("serving_group.group_proxy") as proxy,
                patch("serving_group.subprocess.Popen", side_effect=start) as process,
                patch("serving_group.stop_process_group", return_value=True),
                patch("serving_group.time.sleep"),
            ):
                control.execute.reset_mock()
                def call():
                    return run(
                        args, ["vllm", "serve", "model"], digest, {"CUDA_VISIBLE_DEVICES": "0"},
                        probe, lambda: stopped, host_launch_spec=spec,
                    )
                if signal == b"x":
                    with self.assertRaisesRegex(ControlError, "readiness pipe"):
                        call()
                else:
                    self.assertEqual(call(), 0)
                calls = [next(iter(item.args[0])) for item in control.execute.call_args_list]
                self.assertEqual(calls[0], "ClaimHostLaunch")
                self.assertEqual("HostLaunchReady" in calls, signal == b"1")
                factory.assert_called_once_with("plait-node", [], "192.0.2.2:0", seeds=())
                proxy.assert_not_called()
                probe.assert_not_called()
                command = process.call_args.args[0]
                self.assertEqual(command[command.index("--plait-tp-layout") + 1 :], ["1", "2", "--headless"])
                child_env = process.call_args.kwargs["env"]
                self.assertEqual(child_env["ASCEND_RT_VISIBLE_DEVICES"], "2,6")
                self.assertNotIn("CUDA_VISIBLE_DEVICES", child_env)

    def test_host_release_uses_the_committed_order_and_full_worker_set(self):
        control = Mock()
        control.execute.side_effect = [
            {"Err": "WorkersStillRunning"},
            {"Ok": {"Group": {"phase": "Draining"}}},
            {"Err": "WorkersStillRunning"},
        ]
        identity = {"version": {"group": [1] * 16, "epoch": 1}, "host": [2] * 16}
        workers = [{"node": [ordinal] * 16, "ordinal": 0} for ordinal in (2, 3)]
        with patch("serving_group.time.sleep"):
            record_host_stop(control, identity, workers, 0)
        actions = [call.args[0] for call in control.execute.call_args_list]
        self.assertEqual(actions[:2], [{"HostLaunchStopped": identity}] * 2)
        self.assertEqual(actions[2], {"Retire": {"version": identity["version"], "workers": workers}})


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
    def test_aggregate_route_never_reopens_a_replaced_or_closed_epoch(self):
        group = bytes(16)

        def view(epoch, phase="Active"):
            return {"result": {"Ok": {
                "version": {"group": list(group), "epoch": epoch}, "phase": phase,
                "role": "Aggregated", "endpoint": {"http": "127.0.0.1:12345", "api_model": "test"},
            }}}

        control = Mock()
        control.request.side_effect = [
            view(1), view(2), view(1), view(2, "Draining"), view(2), view(3),
            {"result": {"Err": {"NoActiveGroup": list(group)}}}, view(3),
        ]
        directory = INGRESS["ActiveGroupDirectory"]("test", group, control)
        for expected in (1, 2, 2, None, None, 3, None, None):
            directory.refresh()
            endpoint = directory.endpoint()
            self.assertEqual(endpoint.version if endpoint else None, (
                f"{group.hex()}@{expected}" if expected is not None else None
            ))

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

    def test_aggregate_ingress_retries_a_rejected_request_in_the_new_epoch(self):
        old_port, old_gate = self.group("aggregated", DECODE, lambda _: self.fail("closed group ran"))
        new_port, _ = self.group("aggregated", SPARE, lambda _: ("application/json", b'{"ok":true}'))
        old_gate.close()
        directory = SimpleNamespace(model="test", endpoint=Mock(side_effect=[
            Endpoint("127.0.0.1", old_port, DECODE, "aggregated"),
            Endpoint("127.0.0.1", new_port, SPARE, "aggregated"),
        ]))
        ingress = self.start(INGRESS["server"]("127.0.0.1", 0, directory, 2))
        status, headers, body = self.request(ingress, COMPLETIONS, {"model": "test"})
        self.assertEqual((status, headers[VERSION_HEADER], json.loads(body)), (200, SPARE, {"ok": True}))

    def test_aggregate_ingress_does_not_replay_after_the_upstream_receives_bytes(self):
        received = threading.Event()
        new_port, _ = self.group("aggregated", SPARE, lambda _: self.fail("request was replayed"))

        class Dropped(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                received.set()

        old_port = self.start(ThreadingHTTPServer(("127.0.0.1", 0), Dropped))
        directory = SimpleNamespace(model="test", endpoint=Mock(side_effect=[
            Endpoint("127.0.0.1", old_port, DECODE, "aggregated"),
            Endpoint("127.0.0.1", new_port, SPARE, "aggregated"),
        ]))
        ingress = self.start(INGRESS["server"]("127.0.0.1", 0, directory, 2))
        status, _, _ = self.request(ingress, COMPLETIONS, {"model": "test"})
        self.assertEqual(status, 502)
        self.assertTrue(received.is_set())
        directory.endpoint.assert_called_once()

    def test_drain_keeps_export_alive_until_versioned_release(self):
        port, gate = self.group("prefill", PREFILL, lambda _: ("application/json", b"{}"))
        headers = {VERSION_HEADER: PREFILL, HANDOFF_HEADER: TOKEN}
        status, fields, _ = self.request(port, COMPLETIONS, {}, headers)
        self.assertEqual((status, fields[VERSION_HEADER]), (200, PREFILL))
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

    def test_plait_handoff_rejects_an_unexpected_store_before_decode(self):
        store = "127.0.0.1:9101"
        decoded = []

        def prefill(body):
            handoff = body["kv_transfer_params"]["plait_handoff"]
            self.assertEqual(set(handoff), {"ticket"})
            receipt = {
                **handoff, "store": store, "prompt_sha256": "a" * 64,
                "cached_tokens": 128,
            }
            return "application/json", json.dumps({
                "kv_transfer_params": {"plait_handoff": receipt},
            }).encode()

        def decode(body):
            decoded.append(body)
            return "application/json", b'{"choices": []}'

        p_port, p_gate = self.group("prefill", PREFILL, prefill)
        d_port, _ = self.group("decode", DECODE, decode)
        directory = Path(self.runtime.enter_context(tempfile.TemporaryDirectory())) / "routes.json"
        directory.write_text(json.dumps({"model": "test", "paths": [{
            "prefill": {"url": f"http://127.0.0.1:{p_port}", "version": PREFILL,
                        "kv_host": "127.0.0.1", "kv_port": 9100},
            "decode": {"url": f"http://127.0.0.1:{d_port}", "version": DECODE},
        }]}))
        router_port = self.start(ROUTER["server"](
            "127.0.0.1", 0, ROUTER["Directory"](directory), 5, kv_transport="plait",
        ))
        payload = {"model": "test", "prompt": "test", "max_tokens": 8}
        self.assertEqual(self.request(router_port, COMPLETIONS, payload)[0], 502)
        self.assertEqual(decoded, [])
        self.assertTrue(p_gate.drained())

        store = "127.0.0.1:9100"
        self.assertEqual(self.request(router_port, COMPLETIONS, payload)[0], 200)
        self.assertEqual(decoded[0]["kv_transfer_params"]["plait_handoff"]["store"], store)

    def test_plait_pd_replay_matches_the_existing_stream_before_emitting_a_suffix(self):
        for mismatched in (False, True):
            with self.subTest(mismatched=mismatched):
                tickets = []

                def prefill(body):
                    ticket = body["kv_transfer_params"]["plait_handoff"]["ticket"]
                    tickets.append(ticket)
                    receipt = {"ticket": ticket, "store": "127.0.0.1:9100",
                               "prompt_sha256": "c" * 64, "cached_tokens": 128}
                    return "application/json", json.dumps({
                        "kv_transfer_params": {"plait_handoff": receipt},
                    }).encode()

                def chunk(identity, text, ids, prompt=None, terminal=False):
                    choice = {"index": 0, "text": text, "token_ids": ids,
                              "finish_reason": "length" if terminal else None}
                    if prompt is not None:
                        choice["prompt_token_ids"] = prompt
                    return sse({"id": identity, "model": "test", "choices": [choice]})

                def old_decode(body):
                    self.assertTrue(body["return_token_ids"])
                    self.assertEqual(body["kv_transfer_params"]["plait_handoff"]["ticket"], tickets[-1])
                    return "text/event-stream", chunk("first", "a", [8], [4])

                def new_decode(body):
                    self.assertEqual(body["kv_transfer_params"]["plait_handoff"]["ticket"], tickets[-1])
                    prefix = "x" if mismatched else "a"
                    stream = chunk("second", prefix + "b", [8, 9], [4])
                    stream += chunk("second", "", [], terminal=True) + b"data: [DONE]\n\n"
                    return "text/event-stream", stream

                p_port, gate = self.group("prefill", PREFILL, prefill)
                old_port, _ = self.group("decode", DECODE, old_decode)
                new_port, _ = self.group("decode", SPARE, new_decode)
                source = Endpoint("127.0.0.1", p_port, PREFILL, "prefill")
                old_path, new_path = [ROUTER["Route"](source, Endpoint("127.0.0.1", port, version, "decode"),
                                                      "127.0.0.1", 9100)
                                      for port, version in ((old_port, DECODE), (new_port, SPARE))]
                directory = SimpleNamespace(routes=Mock(side_effect=[("test", [old_path]), ("test", [new_path])]))
                router = self.start(ROUTER["server"]("127.0.0.1", 0, directory, 2, kv_transport="plait"))
                status, headers, body = self.request(router, COMPLETIONS, {
                    "model": "test", "prompt": [4], "max_tokens": 2, "stream": True,
                    "temperature": 0, "return_token_ids": True,
                }, {RECOVERY_HEADER: RECOVERY_MODE})
                self.assertEqual((status, headers[VERSION_HEADER], headers[RECOVERY_HEADER]),
                                 (200, DECODE, RECOVERY_MODE))
                data = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
                frames = [json.loads(line) for line in data if line != b"[DONE]"]
                choices = [choice for frame in frames for choice in frame.get("choices", [])]
                self.assertEqual({frame["id"] for frame in frames if "id" in frame}, {"first"})
                self.assertEqual([token for choice in choices for token in choice["token_ids"]],
                                 [8] if mismatched else [8, 9])
                self.assertEqual("".join(choice["text"] for choice in choices), "a" if mismatched else "ab")
                self.assertEqual(data.count(b"[DONE]"), 0 if mismatched else 1)
                self.assertEqual("error" in frames[-1], mismatched)
                self.assertEqual(len(set(tickets)), 2)
                self.assertTrue(gate.drained())

    def test_load_counts_valid_arrivals_when_routes_are_unavailable(self):
        directory = Mock()
        directory.routes.side_effect = ValueError("directory unavailable")
        load = Load("test")
        port = self.start(ROUTER["server"](
            "127.0.0.1", 0, directory, 5, kv_transport="plait", load=load,
        ))
        request = {"model": "test", "prompt": "input", "max_tokens": 8}
        self.assertEqual(self.request(port, COMPLETIONS, request)[0], 503)
        self.assertEqual(self.request(port, COMPLETIONS, {**request, "model": "other"})[0], 400)
        snapshot = load.snapshot()
        self.assertEqual((snapshot["arrivals"], snapshot["completed"],
                          snapshot["failed"], snapshot["in_flight"]), (1, 0, 1, 0))


if __name__ == "__main__":
    unittest.main()
