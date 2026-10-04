"""Vision Transformer patch/token preparation implemented with NumPy."""

from __future__ import annotations

import numpy as np


def patch_count(
    height: int,
    width: int,
    patch_size: int,
) -> int:
    if min(height, width, patch_size) <= 0:
        raise ValueError("dimensions must be positive")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")
    return (height // patch_size) * (width // patch_size)


def patchify_nchw(
    images: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 4:
        raise ValueError("expected NCHW images")

    batch, channels, height, width = x.shape
    count = patch_count(height, width, patch_size)

    grid_h = height // patch_size
    grid_w = width // patch_size

    patches = (
        x.reshape(
            batch,
            channels,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(
            batch,
            count,
            channels * patch_size * patch_size,
        )
    )
    return patches


def linear_patch_embedding(
    patches: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(patches, dtype=float)
    w = np.asarray(weight, dtype=float)

    if x.ndim != 3 or w.ndim != 2:
        raise ValueError("expected patches (B,N,P) and weight (P,D)")
    if x.shape[-1] != w.shape[0]:
        raise ValueError("patch dimension and projection weight mismatch")

    output = x @ w

    if bias is not None:
        b = np.asarray(bias, dtype=float)
        if b.shape != (w.shape[1],):
            raise ValueError("bias shape mismatch")
        output = output + b

    return output


def prepend_cls_token(
    patch_tokens: np.ndarray,
    cls_token: np.ndarray,
) -> np.ndarray:
    x = np.asarray(patch_tokens, dtype=float)
    token = np.asarray(cls_token, dtype=float)

    if x.ndim != 3:
        raise ValueError("patch_tokens must have shape (B,N,D)")

    if token.shape == (x.shape[-1],):
        token = token.reshape(1, 1, -1)
    elif token.shape == (1, 1, x.shape[-1]):
        pass
    else:
        raise ValueError("cls_token must have shape (D,) or (1,1,D)")

    repeated = np.broadcast_to(
        token,
        (x.shape[0], 1, x.shape[-1]),
    )
    return np.concatenate([repeated, x], axis=1)


def add_learned_positions(
    tokens: np.ndarray,
    position_embedding: np.ndarray,
) -> np.ndarray:
    x = np.asarray(tokens, dtype=float)
    position = np.asarray(position_embedding, dtype=float)

    if x.ndim != 3:
        raise ValueError("tokens must have shape (B,T,D)")

    if position.shape == (x.shape[1], x.shape[2]):
        position = position[None, :, :]
    elif position.shape == (1, x.shape[1], x.shape[2]):
        pass
    else:
        raise ValueError("position embedding shape mismatch")

    return x + position
