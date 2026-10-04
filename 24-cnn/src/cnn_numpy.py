"""Educational NCHW convolution and pooling implemented with NumPy loops."""

from __future__ import annotations

import numpy as np


def conv_output_size(
    input_size: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    if min(input_size, kernel_size, stride, dilation) <= 0 or padding < 0:
        raise ValueError("invalid convolution dimensions")
    effective = dilation * (kernel_size - 1) + 1
    output = (input_size + 2 * padding - effective) // stride + 1
    if output <= 0:
        raise ValueError("kernel is larger than padded input")
    return output


def conv2d_nchw(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    inputs = np.asarray(x, dtype=float)
    kernels = np.asarray(weight, dtype=float)

    if inputs.ndim != 4 or kernels.ndim != 4:
        raise ValueError("x and weight must be rank-4 NCHW/OIHW arrays")

    n, c_in, h, w = inputs.shape
    c_out, kernel_c_in, kh, kw = kernels.shape

    if c_in != kernel_c_in:
        raise ValueError("input channel count does not match kernel")
    if stride <= 0 or padding < 0:
        raise ValueError("invalid stride/padding")

    if bias is None:
        biases = np.zeros(c_out, dtype=float)
    else:
        biases = np.asarray(bias, dtype=float).reshape(-1)
        if biases.shape != (c_out,):
            raise ValueError("bias must have shape (out_channels,)")

    h_out = conv_output_size(h, kh, stride=stride, padding=padding)
    w_out = conv_output_size(w, kw, stride=stride, padding=padding)

    padded = np.pad(
        inputs,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    output = np.empty((n, c_out, h_out, w_out), dtype=float)

    for batch in range(n):
        for out_channel in range(c_out):
            for out_y in range(h_out):
                y0 = out_y * stride
                for out_x in range(w_out):
                    x0 = out_x * stride
                    window = padded[
                        batch,
                        :,
                        y0 : y0 + kh,
                        x0 : x0 + kw,
                    ]
                    output[batch, out_channel, out_y, out_x] = (
                        np.sum(window * kernels[out_channel])
                        + biases[out_channel]
                    )

    return output


def max_pool2d_nchw(
    x: np.ndarray,
    *,
    kernel_size: int = 2,
    stride: int | None = None,
) -> np.ndarray:
    inputs = np.asarray(x, dtype=float)
    if inputs.ndim != 4:
        raise ValueError("x must be rank-4 NCHW")

    if stride is None:
        stride = kernel_size

    if kernel_size <= 0 or stride <= 0:
        raise ValueError("kernel_size and stride must be positive")

    n, channels, h, w = inputs.shape
    h_out = conv_output_size(h, kernel_size, stride=stride)
    w_out = conv_output_size(w, kernel_size, stride=stride)

    output = np.empty((n, channels, h_out, w_out), dtype=float)

    for batch in range(n):
        for channel in range(channels):
            for out_y in range(h_out):
                y0 = out_y * stride
                for out_x in range(w_out):
                    x0 = out_x * stride
                    window = inputs[
                        batch,
                        channel,
                        y0 : y0 + kernel_size,
                        x0 : x0 + kernel_size,
                    ]
                    output[batch, channel, out_y, out_x] = np.max(window)

    return output
