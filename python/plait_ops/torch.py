"""Eager device-tensor adapter for a CUDA-compatible Plait W4A16 library."""

from __future__ import annotations

from . import NativeW4A16, W4A16


def linear(backend: NativeW4A16, activations, packed, scale, zero, *, group_size: int, bias=None,
           output_bits: int = 16):
    import torch

    if activations.ndim < 1 or packed.ndim != 2:
        raise ValueError("expected activations ending in K and row-major packed weights")
    k, n = activations.shape[-1], packed.shape[0]
    rows = activations.numel() // k if k else 0
    shape = W4A16(max(1, rows), n, k, group_size, output_bits)
    if activations.device.type != "cuda":
        raise ValueError("this adapter needs a CUDA-compatible device tensor")
    operands = ((activations, torch.float16), (packed, torch.uint8), (scale, torch.float16),
                (zero, torch.uint8))
    if bias is not None:
        operands += ((bias, torch.float16),)
    if any(tensor.dtype != dtype or tensor.device != activations.device or not tensor.is_contiguous()
           for tensor, dtype in operands):
        raise ValueError("W4A16 operands must use their specified dtypes and share one contiguous device")
    if (packed.shape != (n, shape.packed_stride) or scale.shape != (n, shape.groups)
            or zero.shape != scale.shape or (bias is not None and bias.shape != (n,))):
        raise ValueError("packed weights, scales, zero points, and bias do not match W4A16 dimensions")
    dtype = torch.float16 if shape.output_bits == 16 else torch.float32
    output = torch.empty((*activations.shape[:-1], n), dtype=dtype, device=activations.device)
    if rows == 0:
        return output
    with torch.cuda.device(activations.device):
        stream = torch.cuda.current_stream()
        # The external kernel may still be reading when Python drops an operand.
        for tensor, _ in operands:
            tensor.record_stream(stream)
        backend.launch_device(shape, activations.data_ptr(), packed.data_ptr(), scale.data_ptr(),
                              zero.data_ptr(), 0 if bias is None else bias.data_ptr(),
                              output.data_ptr(), stream.cuda_stream)
    return output
