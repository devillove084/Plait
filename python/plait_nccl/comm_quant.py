"""Optional FP8 communication quantization for Plait allreduce.

This is intentionally opt-in. It is only useful when a model must be tensor
parallel across hosts; data-parallel and single-host TP should keep bf16/fp16
activations. The allreduce itself is done by the existing Plait collective:
the payload is quantized to E4M3, reduced in FP32 inside Plait, and written
back as FP8. A common per-tensor scale is negotiated with a one-element FP32
allreduce before the payload.
"""

from __future__ import annotations

import math

import torch

FP8_E4M3 = 10
FLOAT32 = 7
E4M3_MAX = 448.0


def quantize_fp8(
    tensor: torch.Tensor, scale: torch.Tensor | float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to E4M3 with a symmetric per-tensor scale."""
    values = tensor.float()
    if scale is None:
        scale_t = values.abs().max().clamp_min(1e-8) / E4M3_MAX
    else:
        scale_t = torch.as_tensor(scale, dtype=torch.float32, device=tensor.device)
    quantized = (values / scale_t).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return quantized, scale_t


def dequantize_fp8(quantized: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=quantized.device)
    return quantized.float() * scale_t


def relative_error(reference: torch.Tensor, approximate: torch.Tensor) -> float:
    numerator = (reference.float() - approximate.float()).norm()
    denominator = reference.float().norm().clamp_min(1e-12)
    return float((numerator / denominator).item())


def should_quantize(
    cross_host: bool,
    measured_relative_error: float,
    *,
    max_relative_error: float = 0.01,
    min_bandwidth_gain: float = 1.5,
    achieved_bandwidth_gain: float = 1.0,
) -> bool:
    """The gate for enabling FP8 communication.

    Both conditions are mandatory: the model must actually cross hosts, and
    the measured error/bandwidth must clear the configured thresholds.
    """
    return (
        cross_host
        and math.isfinite(measured_relative_error)
        and measured_relative_error <= max_relative_error
        and achieved_bandwidth_gain >= min_bandwidth_gain
    )


def all_reduce_fp8(comm, tensor: torch.Tensor) -> torch.Tensor:
    """Quantized allreduce for a Plait ``CommHandle``.

    Only use this for cross-host tensor parallelism. The common scale is the
    maximum absolute value across ranks, so every rank encodes with the same
    scale before the FP8 sum.
    """
    if tensor.numel() == 0:
        return tensor
    if not tensor.is_cuda:
        raise ValueError("all_reduce_fp8 needs a CUDA tensor")
    local_max = tensor.float().abs().max().reshape(1)
    global_max = torch.empty_like(local_max)
    result = comm.all_reduce(
        local_max.data_ptr(), global_max.data_ptr(), 1, FLOAT32, 0
    )
    if result != 0:
        raise RuntimeError(f"Plait NCCL scale allreduce failed: {result}")
    scale = global_max.clamp_min(1e-8) / E4M3_MAX
    quantized, _ = quantize_fp8(tensor, scale)
    reduced = torch.empty(tensor.numel(), dtype=torch.float32, device=tensor.device)
    result = comm.all_reduce_fp8_to_f32(
        quantized.data_ptr(), reduced.data_ptr(), quantized.numel()
    )
    if result != 0:
        raise RuntimeError(f"Plait NCCL FP8 allreduce failed: {result}")
    return (reduced * scale).to(tensor.dtype)
