"""KI-owned helpers for one-forward CE/FM training."""

from __future__ import annotations

from typing import Protocol, cast

from praxis.policies.core.action_expert.integrated import (
    VlmActionExpertPrefixAdapter,
)
from praxis.policies.core.vlm.prefix import VlmPrefixAdapter, VlmPrefixLogitsAdapter


class KiVlmPrefixAdapter(
    VlmActionExpertPrefixAdapter,
    VlmPrefixLogitsAdapter,
    Protocol,
):
    """Prefix adapter surface required by KI's shared CE/FM training."""


def require_ki_one_forward_adapter(
    adapter: VlmPrefixAdapter,
) -> VlmPrefixLogitsAdapter:
    """Return an adapter that can project final VLM hidden states to logits."""
    final_hidden_to_logits = getattr(adapter, "final_hidden_to_logits", None)
    if not callable(final_hidden_to_logits):
        raise RuntimeError(
            "KI one-forward training requires the VLM prefix adapter to implement "
            "final_hidden_to_logits(...)."
        )
    return cast(VlmPrefixLogitsAdapter, adapter)
