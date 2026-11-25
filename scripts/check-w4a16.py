#!/usr/bin/env python3
"""Verify the asymmetric Plait W4A16 operator on native accelerators."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from plait_ops import NativeW4A16, W4A16


def fp16(values) -> bytes:
    return b"".join(struct.pack("<e", value) for value in values)


def pack_codes(codes: list[int]) -> bytes:
    if len(codes) % 2:
        codes = [*codes, 15]  # The padding nibble must be ignored.
    return bytes(low | (high << 4) for low, high in zip(codes[::2], codes[1::2]))


def inputs(shape: W4A16, rng: random.Random, bias: bool) -> tuple[bytes, ...]:
    activation = fp16(rng.randint(-63, 63) / 32 for _ in range(shape.m * shape.k))
    packed = b"".join(pack_codes([rng.randrange(16) for _ in range(shape.k)]) for _ in range(shape.n))
    scales = fp16(rng.randint(1, 31) / 128 for _ in range(shape.n * shape.groups))
    zeros = bytes(rng.randrange(16) for _ in range(shape.n * shape.groups))
    args = (activation, packed, scales, zeros)
    if bias:
        args += (fp16(rng.randint(-15, 15) / 16 for _ in range(shape.n)),)
    return args


def compare(expected: bytes, output: bytes, output_bits: int = 16) -> float:
    if len(expected) != len(output):
        raise AssertionError("W4A16 output byte length mismatch")
    dtype = "<e" if output_bits == 16 else "<f"
    rel_tol, abs_tol = (0.002, 0.001) if output_bits == 16 else (2e-6, 1e-5)
    errors = []
    for index, ((reference,), (observed,)) in enumerate(zip(struct.iter_unpack(dtype, expected),
                                                            struct.iter_unpack(dtype, output))):
        if not math.isclose(reference, observed, rel_tol=rel_tol, abs_tol=abs_tol):
            raise AssertionError(f"W4A16 output {index}: expected {reference}, got {observed}")
        errors.append(abs(reference - observed))
    return max(errors, default=0.0)


def check_torch(backend: NativeW4A16, ordinal: int, shape: W4A16, args: tuple[bytes, ...], expected: bytes) -> float:
    import torch
    from plait_ops.torch import linear

    gpu = torch.device("cuda", ordinal)
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        def tensor(value: bytes, dtype, dimensions):
            return torch.frombuffer(bytearray(value), dtype=dtype).reshape(dimensions).to(gpu)

        a = tensor(args[0], torch.float16, (1, shape.m, shape.k))
        w = tensor(args[1], torch.uint8, (shape.n, shape.packed_stride))
        scale = tensor(args[2], torch.float16, (shape.n, shape.groups))
        zero = tensor(args[3], torch.uint8, (shape.n, shape.groups))
        bias = tensor(args[4], torch.float16, (shape.n,)) if len(args) == 5 else None
        options = dict(group_size=shape.group_size, output_bits=shape.output_bits, bias=bias)
        output = linear(backend, a, w, scale, zero, **options)
        output_dtype = torch.float16 if shape.output_bits == 16 else torch.float32
        assert output.shape == (1, shape.m, shape.n) and output.dtype == output_dtype
        codec = "e" if shape.output_bits == 16 else "f"
        values = struct.pack(f"<{output.numel()}{codec}", *output.flatten().tolist())
        empty = linear(backend, a[:, :0], w, scale, zero, **options)
        assert empty.shape == (1, 0, shape.n) and empty.dtype == output_dtype
    return compare(expected, values, shape.output_bits)


def shard_case(index: int):
    full = W4A16(3, 5, 17, 3)
    full_args = inputs(full, random.Random(90125), bias=True)
    a, w, scale, zero, _ = full_args
    start, end = ((0, 3), (3, 9), (9, 17))[index]
    shard = W4A16(full.m, full.n, end - start, full.group_size, 32)
    activations = b"".join(a[2 * (row * full.k + start):2 * (row * full.k + end)]
                           for row in range(full.m))
    packed = bytearray()
    scales = bytearray()
    zeros = bytearray()
    for channel in range(full.n):
        offset = channel * full.packed_stride
        codes = [(w[offset + inner // 2] >> (4 * (inner % 2))) & 15 for inner in range(start, end)]
        packed.extend(pack_codes(codes))
        first_group = channel * full.groups + start // full.group_size
        end_group = first_group + shard.groups
        scales.extend(scale[2 * first_group:2 * end_group])
        zeros.extend(zero[first_group:end_group])
    return full, full_args, shard, (activations, bytes(packed), bytes(scales), bytes(zeros)), (start, end)


def build(args: argparse.Namespace, directory: Path) -> Path:
    kernels = ROOT / "kernels"
    if args.backend == "metal":
        library = directory / "libplait-w4a16.dylib"
        command = [args.compiler or "clang", "-std=c11", "-O2", "-dynamiclib", "-framework", "Metal",
                   "-framework", "Foundation", "-lobjc", str(kernels / "w4a16_metal.c"), "-o", str(library)]
    else:
        library = directory / "libplait-w4a16.so"
        command = [args.compiler or "nvcc", "-std=c++14", "-O2", "-shared", "-Xcompiler", "-fPIC",
                   str(kernels / "w4a16_cuda.cu"), "-o", str(library)]
        if args.arch:
            command.insert(1, f"-arch={args.arch}")
    subprocess.run(command, check=True)
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "ppu", "metal"), required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--arch", help="CUDA compiler architecture, such as sm_86")
    parser.add_argument("--compiler", help="nvcc or clang override")
    parser.add_argument("--library", type=Path, help="reuse a native library implementing the Plait ABI")
    parser.add_argument("--torch", action="store_true", help="also validate the device-tensor adapter")
    parser.add_argument("--scratch", type=Path, help="parent folder for temporary build files")
    parser.add_argument("--shard", type=int, choices=(0, 1, 2), help="emit a specific FP32 K-shard result")
    parser.add_argument("--partial-stdout", action="store_true", help="write only the selected shard as FP32 bytes")
    args = parser.parse_args()
    if any(index < 0 for index in args.devices):
        parser.error("device indices must be nonnegative")
    if args.torch and args.backend == "metal":
        parser.error("the device-tensor adapter needs a CUDA-compatible backend")
    if args.partial_stdout and (args.shard is None or len(args.devices) != 1 or args.torch):
        parser.error("--partial-stdout requires one device, a shard, and no --torch")

    rng = random.Random(23)
    hand = (W4A16(1, 1, 3, 2), (fp16((1, 2, 3)), bytes((0xE1, 0xF4)), fp16((.5, 1.5)),
                                 bytes((5, 3)), fp16((.25,))))
    assert hand[0].reference(*hand[1]) == fp16((11.75,))
    cases = [hand]
    for dims in ((1, 1, 1, 1), (2, 3, 7, 3), (3, 17, 64, 32), (4, 33, 256, 64),
                 (2, 65, 385, 128), (3, 7, 35, 8, 32)):
        operator = W4A16(*dims)
        cases.append((operator, inputs(operator, rng, bias=len(cases) % 2 == 0)))

    results = []
    report = {"backend": args.backend, "seed": 23, "cases": results}
    with tempfile.TemporaryDirectory(prefix="plait-w4a16-", dir=args.scratch) as work:
        library = args.library.resolve() if args.library else build(args, Path(work))
        backend = NativeW4A16(library)
        if args.partial_stdout:
            _, _, shape, operands, _ = shard_case(args.shard)
            partial = backend.run_host(shape, args.devices[0], *operands)
            compare(shape.reference(*operands), partial, shape.output_bits)
            sys.stdout.buffer.write(partial)
            return
        for ordinal in args.devices:
            for shape, operands in cases:
                reference = shape.reference(*operands)
                native = backend.run_host(shape, ordinal, *operands)
                result = {"device": ordinal, "shape": [shape.m, shape.n, shape.k, shape.group_size],
                          "bias": len(operands) == 5, "output_bits": shape.output_bits,
                          "max_abs_error": compare(reference, native, shape.output_bits)}
                if args.torch:
                    result["torch_max_abs_error"] = check_torch(backend, ordinal, shape, operands, reference)
                results.append(result)
        if args.shard is not None:
            _, _, shape, operands, interval = shard_case(args.shard)
            ordinal = args.devices[0]
            partial = backend.run_host(shape, ordinal, *operands)
            error = compare(shape.reference(*operands), partial, shape.output_bits)
            report["k_partial"] = {"range": interval, "shape": [shape.m, shape.n], "device": ordinal,
                                   "output_bits": shape.output_bits, "max_abs_error": error, "data": partial.hex()}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
