"""Portable asymmetric W4A16 linear operator and native dispatch."""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
from dataclasses import dataclass
from os import PathLike


UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class W4A16:
    """A[M,K] × (dequantized W[N,K]).T, with FP32 accumulation."""

    m: int
    n: int
    k: int
    group_size: int
    output_bits: int = 16

    def __post_init__(self) -> None:
        dimensions = (self.m, self.n, self.k, self.group_size)
        if any(type(value) is not int or not 1 <= value <= UINT32_MAX for value in dimensions):
            raise ValueError("dimensions and group size must be positive uint32 values")
        if self.m * self.n > UINT32_MAX or max(self.m, self.n) * self.k > sys.maxsize // 2:
            raise ValueError("W4A16 tensor is too large")
        if type(self.output_bits) is not int or self.output_bits not in (16, 32):
            raise ValueError("output_bits must be 16 or 32")

    @property
    def packed_stride(self) -> int:
        return (self.k + 1) // 2

    @property
    def groups(self) -> int:
        return (self.k + self.group_size - 1) // self.group_size

    @property
    def output_bytes(self) -> int:
        return self.m * self.n * (self.output_bits // 8)

    def _store(self, output: bytearray, index: int, value: float) -> None:
        if self.output_bits == 16:
            if abs(value) >= 65520:
                value = math.copysign(math.inf, value)
            struct.pack_into("<e", output, index * 2, value)
        else:
            struct.pack_into("<f", output, index * 4, ctypes.c_float(value).value)

    def views(self, a, w, scale, zero, bias=None) -> tuple[memoryview, ...]:
        """Validate contiguous host bytes. Half values use little-endian IEEE FP16."""
        values = (a, w, scale, zero)
        sizes = (2 * self.m * self.k, self.n * self.packed_stride, 2 * self.n * self.groups,
                 self.n * self.groups)
        if bias is not None:
            values += (bias,)
            sizes += (2 * self.n,)
        views = tuple(memoryview(value).cast("B") for value in values)
        if any(len(view) != size for view, size in zip(views, sizes)):
            raise ValueError("operand buffer sizes do not match the W4A16 layout")
        if any(value > 15 for value in views[3]):
            raise ValueError("zero points must fit an unsigned nibble")
        return views

    def reference(self, a, w, scale, zero, bias=None) -> bytes:
        """CPU oracle with high-precision accumulation for error comparisons."""
        a, w, scale, zero, *bias = self.views(a, w, scale, zero, bias)
        input_values = struct.unpack(f"<{self.m * self.k}e", a)
        scales = struct.unpack(f"<{self.n * self.groups}e", scale)
        biases = struct.unpack(f"<{self.n}e", bias[0]) if bias else (0.0,) * self.n
        output = bytearray(self.output_bytes)
        for channel in range(self.n):
            weights = []
            for inner in range(self.k):
                pair = w[channel * self.packed_stride + inner // 2]
                code = (pair >> ((inner % 2) * 4)) & 15
                group = channel * self.groups + inner // self.group_size
                weights.append((code - zero[group]) * scales[group])
            for row in range(self.m):
                start = row * self.k
                total = math.fsum(input_values[start + inner] * weight for inner, weight in enumerate(weights))
                self._store(output, row * self.n + channel, total + biases[channel])
        return bytes(output)

    def merge_partials(self, partials, bias=None) -> bytes:
        """Combine already-aligned FP32 K-shard outputs and apply the layer bias once."""
        views = [memoryview(value).cast("B") for value in partials]
        elements = self.m * self.n
        if not views or any(len(view) != 4 * elements for view in views):
            raise ValueError("K-shard partials must have the same FP32 [M,N] layout")
        biases = struct.unpack(f"<{self.n}e", bias) if bias is not None else (0.0,) * self.n
        output = bytearray(self.output_bytes)
        for index in range(elements):
            value = 0.0
            for view in views:
                partial, = struct.unpack_from("<f", view, index * 4)
                value = ctypes.c_float(value + partial).value
            self._store(output, index, ctypes.c_float(value + biases[index % self.n]).value)
        return bytes(output)


class _Shape(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("m", "n", "k", "group_size", "output_bits")]

    @classmethod
    def from_operator(cls, shape: W4A16) -> _Shape:
        return cls(shape.m, shape.n, shape.k, shape.group_size, shape.output_bits)


class NativeW4A16:
    """A library implementing the Plait ABI in kernels/w4a16.h."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.library = ctypes.CDLL(str(path))
        self._host = self.library.plait_w4a16_run_host
        shape = ctypes.POINTER(_Shape)
        pointer = (ctypes.c_void_p,) * 6
        self._host.argtypes = (shape, ctypes.c_int) + pointer
        self._host.restype = ctypes.c_int
        self._launch = getattr(self.library, "plait_w4a16_launch", None)
        if self._launch is not None:
            self._launch.argtypes = (shape,) + pointer + (ctypes.c_void_p,)
            self._launch.restype = ctypes.c_int

    @property
    def supports_device_launch(self) -> bool:
        return self._launch is not None

    def run_host(self, shape: W4A16, device: int, a, w, scale, zero, bias=None) -> bytes:
        source = shape.views(a, w, scale, zero, bias)
        operands = [(ctypes.c_uint8 * len(view)).from_buffer_copy(view) for view in source]
        if bias is None:
            operands.append(None)
        output_type = ctypes.c_float if shape.output_bits == 32 else ctypes.c_uint16
        output = (output_type * (shape.m * shape.n))()
        status = self._host(ctypes.byref(_Shape.from_operator(shape)), device, *operands, output)
        if status:
            raise RuntimeError(f"native W4A16 host execution failed with status {status}")
        return bytes(output)

    def launch_device(self, shape: W4A16, a: int, w: int, scale: int, zero: int, bias: int,
                      out: int, stream: int) -> None:
        """Submit to a caller-owned stream. Pointers must reference valid device storage."""
        if self._launch is None:
            raise RuntimeError("this library provides host-staged execution")
        status = self._launch(ctypes.byref(_Shape.from_operator(shape)), a, w, scale, zero, bias, out, stream)
        if status:
            raise RuntimeError(f"native W4A16 submission failed with status {status}")


def register_vllm() -> None:
    """vLLM general plugin entrypoint. Enabling the candidate still requires opt-in."""
    if os.environ.get("PLAIT_W4A16_LIBRARY"):
        from .vllm import register

        register()
