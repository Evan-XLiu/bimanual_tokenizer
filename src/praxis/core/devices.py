"""Device validation shared by training, eval, and serving entrypoints.

The public CLIs accept explicit device strings. If a user asks for CUDA, we
should fail at the boundary when CUDA is unavailable instead of silently running
the requested job on CPU.
"""

from __future__ import annotations

import torch


def require_available_device(
    requested_device: object,
    *,
    purpose: str,
) -> torch.device:
    """Return a torch device, preserving explicit CUDA intent."""
    device = torch.device(str(requested_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{purpose} requested device={device}, but this Python environment "
            "does not have CUDA available through PyTorch. Set device=cpu or "
            "launch with a CUDA-enabled torch build."
        )
    return device
