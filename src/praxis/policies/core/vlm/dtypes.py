"""VLM dtype parsing."""

from __future__ import annotations

import torch

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_vlm_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported vlm_dtype {dtype_str!r}. Expected one of {list(_DTYPE_MAP)}."
        )
    return _DTYPE_MAP[dtype_str]
