"""Capability-tiered routing for heterogeneous Plait/vLLM deployments.

The router is deliberately model-agnostic: a tier advertises its context,
quality class, quantization format, and an OpenAI-compatible endpoint. The
default tiers match the measured 4-local/8-remote layout:

* local 4x3090: Qwen3.5-9B W4A16 (Marlin), cheap short prompts
* remote 8xPRO: Qwen3.5-35B-A3B-FP8, long prompts
* remote 8xPRO: Qwen3.5-122B-A10B-NVFP4, high-quality/long-context requests

No cross-host tensor parallel is implied by routing; each tier is an
independent replica set.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    endpoint: str
    model: str
    max_context: int
    quality: int
    quant: str
    throughput: float = 1.0
    enabled: bool = True

    def can_serve(self, prompt_tokens: int, max_tokens: int, quality: int) -> bool:
        return (
            self.enabled
            and self.quality >= quality
            and prompt_tokens + max_tokens <= self.max_context
        )


class CapabilityRouter:
    """Choose the cheapest tier that satisfies context and quality."""

    def __init__(self, tiers: list[Tier]) -> None:
        if not tiers:
            raise ValueError("router needs at least one tier")
        self.tiers = sorted(tiers, key=lambda tier: (tier.quality, -tier.throughput))

    def choose(
        self,
        prompt_tokens: int,
        *,
        max_tokens: int = 256,
        required_quality: int = 0,
    ) -> Tier:
        eligible = [
            tier
            for tier in self.tiers
            if tier.can_serve(prompt_tokens, max_tokens, required_quality)
        ]
        if not eligible:
            raise ValueError(
                f"no tier serves prompt={prompt_tokens} max_tokens={max_tokens} "
                f"quality={required_quality}"
            )
        return min(eligible, key=lambda tier: (tier.quality, -tier.throughput))

    def generate(
        self,
        prompt: str,
        *,
        prompt_tokens: int,
        max_tokens: int = 256,
        required_quality: int = 0,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> dict:
        tier = self.choose(
            prompt_tokens,
            max_tokens=max_tokens,
            required_quality=required_quality,
        )
        payload = json.dumps(
            {
                "model": tier.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode()
        request = urllib.request.Request(
            tier.endpoint.rstrip("/") + "/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        body["_plait_tier"] = tier.name
        return body


def default_tiers(local_endpoint: str, remote_endpoint: str) -> list[Tier]:
    return [
        Tier(
            name="local-9b-w4a16",
            endpoint=local_endpoint,
            model="/mnt/nas1/hf/Qwen3.5-9B-W4A16",
            max_context=4096,
            quality=1,
            quant="w4a16",
            throughput=1.0,
        ),
        Tier(
            name="remote-35b-fp8",
            endpoint=remote_endpoint,
            model="/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8",
            max_context=32768,
            quality=2,
            quant="fp8",
            throughput=0.8,
        ),
        Tier(
            name="remote-122b-nvfp4",
            endpoint=remote_endpoint,
            model="/mnt/nas1/hf/Qwen3.5-122B-A10B-NVFP4",
            max_context=131072,
            quality=3,
            quant="nvfp4",
            throughput=0.25,
            # Upstream vLLM 0.28 Qwen3.5 TP produces degenerate output; keep
            # the tier out of routing until a validated TP path lands.
            enabled=False,
        ),
    ]
