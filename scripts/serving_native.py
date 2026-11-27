"""Bind a vLLM HTTP endpoint to the native TP group in its launch descriptor."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


_VERSION = b"x-plait-group-version"
_ROLE = b"x-plait-group-role"
_ADMISSION = b"x-plait-admission"
_BODY = b'{"error":{"message":"execution group version does not match"}}'


class NativeGroupHeaders:
    """ASGI middleware for an engine launched with a single committed TP group."""

    def __init__(self, app) -> None:
        self.app = app
        path = os.environ.get("VLLM_PLAIT_TP_GROUP_CONFIG")
        if os.environ.get("VLLM_USE_PLAIT") != "1" or not path:
            raise ValueError("native group middleware requires a committed Plait launch")
        groups = json.loads(Path(path).read_text(encoding="utf-8"))["tp_groups"]
        if len(groups) != 1 or not re.fullmatch(r"[a-f0-9]{32}@[1-9][0-9]*", groups[0]["version"]):
            raise ValueError("native group middleware requires one valid TP group version")
        self.version = groups[0]["version"].encode("ascii")
        self.headers = [(_VERSION, self.version), (_ROLE, b"aggregated")]

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        versions = [value for key, value in scope["headers"] if key.lower() == _VERSION]
        if versions and versions != [self.version]:
            await send({"type": "http.response.start", "status": 409, "headers": [
                *self.headers, (_ADMISSION, b"rejected"),
                (b"content-type", b"application/json"), (b"content-length", str(len(_BODY)).encode()),
            ]})
            await send({"type": "http.response.body", "body": _BODY})
            return

        async def send_versioned(message):
            if message["type"] == "http.response.start":
                headers = [
                    (key, value) for key, value in message.get("headers", [])
                    if key.lower() not in (_VERSION, _ROLE, _ADMISSION)
                ]
                message = {**message, "headers": headers + self.headers}
            await send(message)

        await self.app(scope, receive, send_versioned)
