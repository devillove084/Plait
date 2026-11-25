"""vLLM compressed-tensors W4A16 adapter for the Plait device ABI."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import torch
from compressed_tensors.quantization import ActivationOrdering
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
    CompressedTensorsWNA16,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform

from . import NativeW4A16
from .torch import linear


QUANTIZATION = "plait-w4a16"


@lru_cache(maxsize=1)
def _load_backend(path: Path, expected_digest: str | None) -> NativeW4A16:
    if expected_digest is not None and sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("Plait W4A16 library differs from the launch plan digest")
    backend = NativeW4A16(path)
    if not backend.supports_device_launch:
        raise ValueError("Plait W4A16 requires a library with the device launch entrypoint")
    return backend


class PlaitCompressedTensorsW4A16(CompressedTensorsScheme):
    supports_pre_processed_weights = False

    def __init__(self, group_size: int, symmetric: bool, backend: NativeW4A16):
        self.group_size = group_size
        self.symmetric = symmetric
        self.backend = backend

    @classmethod
    def get_min_capability(cls) -> int:
        return 75  # Match vLLM's compressed-tensors scheme.

    def create_weights(
        self, layer, output_size, input_size, output_partition_sizes,
        input_size_per_partition, params_dtype, weight_loader, **kwargs,
    ) -> None:
        if params_dtype != torch.float16:
            raise ValueError("Plait W4A16 requires --dtype half")
        k = input_size_per_partition
        n = sum(output_partition_sizes)
        group_size = input_size if self.group_size == -1 else self.group_size
        if group_size < 1 or input_size % group_size or k % group_size or k % 8:
            raise ValueError("Plait W4A16 requires K partitions aligned to groups and INT4 words")
        if not self.symmetric and any(size % 8 for size in output_partition_sizes):
            raise ValueError("Plait W4A16 requires zero-point output partitions aligned to INT4 words")
        self.group_size = group_size
        groups = k // group_size

        def packed_parameter(dim, shape):
            return PackedvLLMParameter(
                input_dim=1, output_dim=0, packed_dim=dim, packed_factor=8,
                data=torch.empty(shape, dtype=torch.int32), weight_loader=weight_loader,
            )

        layer.params_dtype = params_dtype
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.output_partition_sizes = output_partition_sizes
        layer.register_parameter("weight_packed", packed_parameter(1, (n, k // 8)))
        layer.register_parameter("weight_scale", GroupQuantScaleParameter(
            input_dim=1, output_dim=0, data=torch.empty((n, groups), dtype=params_dtype),
            weight_loader=weight_loader,
        ))
        layer.register_parameter("weight_shape", BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader,
        ))
        if not self.symmetric:
            layer.register_parameter("weight_zero_point", packed_parameter(0, (n // 8, groups)))

    def process_weights_after_loading(self, layer) -> None:
        packed = layer.weight_packed.detach()
        if packed.device.type != "cuda" or sys.byteorder != "little":
            raise ValueError("Plait W4A16 needs little-endian CUDA device weights")
        layer.register_buffer("plait_packed", packed.contiguous().view(torch.uint8), persistent=False)
        scale = layer.weight_scale.detach().contiguous()
        layer.register_buffer("plait_scale", scale, persistent=False)

        if self.symmetric:
            # Compressed-tensors stores signed INT4 in biased unsigned codes.
            zero = torch.full_like(scale, 8, dtype=torch.uint8)
        else:
            # Checkpoint zero points pack eight output channels per INT32 word.
            words = layer.weight_zero_point.detach()
            shifts = torch.arange(0, 32, 4, device=words.device, dtype=torch.int32)
            zero = ((words[:, None, :] >> shifts[None, :, None]) & 15).to(torch.uint8)
            zero = zero.flatten(0, 1).contiguous()
        layer.register_buffer("plait_zero", zero, persistent=False)

    def apply_weights(self, layer, x, bias=None):
        return linear(
            self.backend, x.contiguous(), layer.plait_packed, layer.plait_scale,
            layer.plait_zero, group_size=self.group_size, bias=bias,
        )


class PlaitW4A16Config(CompressedTensorsConfig):
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant == QUANTIZATION and hf_quant_cfg.get("quant_method") == "compressed-tensors":
            return QUANTIZATION
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("quant_method") != "compressed-tensors" or config.get("format") != "pack-quantized":
            raise ValueError("Plait W4A16 requires a compressed-tensors pack-quantized checkpoint")
        parsed = super().from_config(config)
        for recipe in parsed.target_scheme_map.values():
            weights = recipe.get("weights") if recipe else None
            if getattr(weights, "actorder", None) == ActivationOrdering.GROUP:
                raise ValueError("Plait W4A16 requires sequential weight groups")
        return parsed

    def get_name(self) -> str:
        return QUANTIZATION

    def get_scheme(self, layer, layer_name=None):
        scheme = super().get_scheme(layer, layer_name)
        if scheme is None:
            return None
        if not isinstance(scheme, CompressedTensorsWNA16) or scheme.num_bits != 4:
            raise ValueError(f"Plait W4A16 cannot load the quantization scheme for {layer_name}")
        if not current_platform.is_cuda():
            raise ValueError("Plait W4A16 currently supports vLLM's CUDA platform")
        config = get_current_vllm_config_or_none()
        if config is not None and config.model_config is not None and not config.model_config.enforce_eager:
            raise ValueError("Plait W4A16 requires --enforce-eager")
        path = Path(os.environ["PLAIT_W4A16_LIBRARY"]).resolve(strict=True)
        backend = _load_backend(path, os.environ.get("PLAIT_W4A16_SHA256"))
        return PlaitCompressedTensorsW4A16(scheme.group_size, scheme.symmetric, backend)


@lru_cache(maxsize=1)
def register() -> None:
    register_quantization_config(QUANTIZATION)(PlaitW4A16Config)
