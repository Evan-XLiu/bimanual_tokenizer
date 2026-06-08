# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared state-prefix conditioning helpers for VLM policy families."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from praxis.policies.core.observations import (
    concatenate_state_features,
)
from praxis.policies.core.vlm.prefix import VlmInputBatch
from praxis.policies.core.vlm.vocab import resolve_or_add_special_tokens

StatePrefixEncoding = Literal["none", "discrete_text", "continuous_projection"]
StatePrefixEmbedInit = Literal["normal", "avg", "zero"]

STATE_PREFIX_EMBEDDINGS_KEY = "state_prefix_embeddings"
STATE_PREFIX_PLACEHOLDER_MASK_KEY = "state_prefix_placeholder_mask"
DEFAULT_STATE_PREFIX_TOKEN = "<|state|>"

_VALID_STATE_PREFIX_ENCODINGS = {
    "none",
    "discrete_text",
    "continuous_projection",
}
_VALID_STATE_PREFIX_EMBED_INITS = {"normal", "avg", "zero"}


@dataclass(frozen=True)
class PreparedStatePrefix:
    """Rendered prompt text plus optional projected-state source values."""

    texts: list[str]
    state_vector: Tensor | None = None


def validate_state_prefix_encoding(value: str) -> StatePrefixEncoding:
    """Validate and normalize the configured VLM state-prefix encoding."""
    if value not in _VALID_STATE_PREFIX_ENCODINGS:
        allowed = ", ".join(sorted(_VALID_STATE_PREFIX_ENCODINGS))
        raise ValueError(
            f"state_prefix_encoding must be one of {{{allowed}}}, got {value!r}."
        )
    return value  # type: ignore[return-value]


def validate_state_prefix_num_bins(value: int) -> int:
    """Validate the bin count used by discrete-text state prefixes."""
    value = int(value)
    if value < 2:
        raise ValueError(f"state_prefix_num_bins must be >= 2, got {value}.")
    return value


def validate_state_prefix_placeholder_token(value: str) -> str:
    """Validate the special token used as the continuous-state placeholder."""
    token = str(value)
    if not token:
        raise ValueError("state_prefix_placeholder_token must be a non-empty string.")
    return token


def validate_state_prefix_embed_init(value: str) -> StatePrefixEmbedInit:
    """Validate the placeholder-token embedding initialization mode."""
    value = str(value)
    if value not in _VALID_STATE_PREFIX_EMBED_INITS:
        raise ValueError(
            "state_prefix_embed_init must be one of {'normal', 'avg', 'zero'}, "
            f"got {value!r}."
        )
    return value  # type: ignore[return-value]


def prepare_state_prefix(
    obs: dict[str, Tensor],
    *,
    state_features: Sequence[Any],
    encoding: StatePrefixEncoding,
    num_bins: int,
    placeholder_token: str,
    batch_size: int | None = None,
) -> PreparedStatePrefix:
    """Prepare the state-prefix payload consumed by VLM input builders."""
    encoding = validate_state_prefix_encoding(str(encoding))
    if encoding == "none":
        return PreparedStatePrefix(texts=[])
    missing = [str(feat.key) for feat in state_features if feat.key not in obs]
    if missing:
        raise KeyError(
            "State-prefix conditioning requires declared state feature(s) in the "
            f"batch: {', '.join(missing)}."
        )
    if encoding == "discrete_text":
        num_bins = validate_state_prefix_num_bins(num_bins)
        state_vec = concatenate_state_features(obs, state_features=state_features)
        if state_vec is None:
            return PreparedStatePrefix(texts=[])
        _validate_state_prefix_batch_size(
            state_vec,
            batch_size=batch_size,
            context="state prefix text",
        )
        return PreparedStatePrefix(
            texts=_discretize_state_vector(state_vec, num_bins=num_bins)
        )
    if encoding == "continuous_projection":
        placeholder_token = validate_state_prefix_placeholder_token(placeholder_token)
        state_vec = concatenate_state_features(obs, state_features=state_features)
        if state_vec is None:
            return PreparedStatePrefix(texts=[])
        _validate_state_prefix_batch_size(
            state_vec,
            batch_size=batch_size,
            context="state prefix projection",
        )
        return PreparedStatePrefix(
            texts=[
                f"State: {placeholder_token}" for _ in range(int(state_vec.shape[0]))
            ],
            state_vector=state_vec,
        )
    raise AssertionError(f"Unhandled state_prefix_encoding={encoding!r}.")


def resolve_state_dim(state_features: Sequence[Any]) -> int:
    """Return the flattened state dimension represented by PolicyIO features."""
    return sum(
        math.prod(tuple(int(dim) for dim in getattr(feat, "shape", ())))
        for feat in state_features
    )


def setup_state_prefix_token(
    *,
    tokenizer: Any,
    vlm: Any,
    token: str,
    init_mode: str,
) -> int:
    """Add one newly-reserved special token used as a continuous state placeholder."""
    token = validate_state_prefix_placeholder_token(token)
    init_mode = validate_state_prefix_embed_init(init_mode)
    vocab = tokenizer.get_vocab()
    if token in vocab:
        raise ValueError(
            f"state_prefix_placeholder_token={token!r} is already present in the "
            "tokenizer vocab; choose a newly-reserved token."
        )
    token_id = int(
        resolve_or_add_special_tokens(
            tokenizer=tokenizer,
            vlm=vlm,
            tokens=[token],
            init_mode=init_mode,
        )[0]
    )
    _validate_single_token_encoding(tokenizer, token=token, token_id=token_id)
    return token_id


def replace_state_prefix_placeholder_embeddings(
    *,
    input_ids: Tensor,
    inputs_embeds: Tensor,
    vlm_inputs: VlmInputBatch,
) -> Tensor:
    """Replace state placeholder token embeddings with projected continuous state."""
    state_emb = vlm_inputs.get(STATE_PREFIX_EMBEDDINGS_KEY)
    if state_emb is None:
        return inputs_embeds
    if not isinstance(state_emb, Tensor):
        raise TypeError(
            f"{STATE_PREFIX_EMBEDDINGS_KEY} must be a Tensor, got {type(state_emb)!r}."
        )
    placeholder_mask = vlm_inputs.get(STATE_PREFIX_PLACEHOLDER_MASK_KEY)
    if placeholder_mask is None:
        raise ValueError(
            f"{STATE_PREFIX_PLACEHOLDER_MASK_KEY} is required when "
            f"{STATE_PREFIX_EMBEDDINGS_KEY} is present."
        )
    if state_emb.ndim == 2:
        state_emb = state_emb[:, None, :]
    if state_emb.ndim != 3 or int(state_emb.shape[1]) != 1:
        raise ValueError(
            f"{STATE_PREFIX_EMBEDDINGS_KEY} must have shape (B, D) or (B, 1, D), "
            f"got {tuple(state_emb.shape)}."
        )
    if int(state_emb.shape[0]) != int(input_ids.shape[0]):
        raise ValueError(
            "State prefix embedding batch size must match input_ids batch size, "
            f"got {int(state_emb.shape[0])} and {int(input_ids.shape[0])}."
        )
    if int(state_emb.shape[-1]) != int(inputs_embeds.shape[-1]):
        raise ValueError(
            "State prefix embedding width must match VLM embedding width, got "
            f"{int(state_emb.shape[-1])} and {int(inputs_embeds.shape[-1])}."
        )

    mask = placeholder_mask.to(device=input_ids.device, dtype=torch.bool)
    if mask.shape != input_ids.shape:
        raise ValueError(
            "State prefix placeholder mask must match input_ids shape, got "
            f"{tuple(mask.shape)} and {tuple(input_ids.shape)}."
        )
    counts = mask.sum(dim=1)
    if not bool(counts.eq(1).all()):
        raise ValueError(
            "Each sample must contain exactly one state prefix placeholder token; "
            f"got counts={counts.detach().cpu().tolist()}."
        )
    state_emb = state_emb.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    mask_3d = mask[:, :, None].expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(mask_3d, state_emb.reshape(-1))


class StatePrefixPolicyMixin:
    """Lifecycle and input-building helpers for VLM state prefixes."""

    config: Any
    io: Any | None
    processor: Any
    vlm: Any
    state_prefix_token_id: int | None
    state_prefix_projector: nn.Linear | None
    _continuous_state_prefix_embeddings_enabled: bool

    def _initialize_state_prefix_members(self) -> None:
        self.state_prefix_token_id = None
        self.state_prefix_projector = None
        self._continuous_state_prefix_embeddings_enabled = True

    def _setup_state_prefix_token(self) -> None:
        if self._state_prefix_encoding() != "continuous_projection":
            return
        self.state_prefix_token_id = setup_state_prefix_token(
            tokenizer=self.processor.tokenizer,
            vlm=self.vlm,
            token=str(self.config.state_prefix_placeholder_token),
            init_mode=str(self.config.state_prefix_embed_init),
        )

    def _build_state_prefix_projector(
        self,
        *,
        hidden_size: int | None = None,
    ) -> None:
        self.state_prefix_projector = None
        if self.io is None:
            raise RuntimeError("PolicyIO must be set before building state prefix.")
        if self._state_prefix_encoding() != "continuous_projection":
            return
        state_dim = resolve_state_dim(self.io.state_features)
        if state_dim <= 0:
            return
        if hidden_size is None:
            raise RuntimeError(
                "continuous state-prefix projection requires the VLM hidden size."
            )
        self.state_prefix_projector = nn.Linear(int(state_dim), int(hidden_size))

    def _prepare_state_prefix(
        self,
        obs: dict[str, Tensor],
        *,
        batch_size: int | None = None,
    ) -> PreparedStatePrefix:
        if self.io is None:
            raise RuntimeError("PolicyIO must be set before building state prefix.")
        return prepare_state_prefix(
            obs,
            state_features=self.io.state_features,
            encoding=self._state_prefix_encoding(),
            num_bins=int(self.config.state_prefix_num_bins),
            placeholder_token=str(
                getattr(
                    self.config,
                    "state_prefix_placeholder_token",
                    DEFAULT_STATE_PREFIX_TOKEN,
                )
            ),
            batch_size=(
                _infer_state_prefix_batch_size(obs)
                if batch_size is None
                else batch_size
            ),
        )

    def _add_continuous_state_prefix_embeddings(
        self,
        vlm_inputs: VlmInputBatch,
        state_prefix: PreparedStatePrefix,
    ) -> VlmInputBatch:
        if (
            self._state_prefix_encoding() != "continuous_projection"
            or not self._continuous_state_prefix_embeddings_enabled
        ):
            return dict(vlm_inputs)
        state_vec = state_prefix.state_vector
        if state_vec is None:
            return dict(vlm_inputs)
        if self.state_prefix_projector is None:
            raise RuntimeError(
                "state_prefix_projector is required for "
                "state_prefix_encoding='continuous_projection'."
            )
        if self.state_prefix_token_id is None:
            raise RuntimeError(
                "state_prefix_token_id is required for "
                "state_prefix_encoding='continuous_projection'."
            )
        param = self.state_prefix_projector.weight
        state_vec = state_vec.to(device=param.device, dtype=param.dtype)
        state_emb = self.state_prefix_projector(state_vec)
        input_ids = vlm_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError(
                "continuous state-prefix projection requires VLM input_ids."
            )
        out: VlmInputBatch = dict(vlm_inputs)
        out[STATE_PREFIX_EMBEDDINGS_KEY] = state_emb
        out[STATE_PREFIX_PLACEHOLDER_MASK_KEY] = input_ids.eq(
            int(self.state_prefix_token_id)
        )
        return out

    @contextmanager
    def _without_continuous_state_prefix_embeddings(self) -> Iterator[None]:
        previous = self._continuous_state_prefix_embeddings_enabled
        self._continuous_state_prefix_embeddings_enabled = False
        try:
            yield
        finally:
            self._continuous_state_prefix_embeddings_enabled = previous

    def _observations_for_state_prefix(
        self,
        raw_batch: dict[str, Tensor],
        *,
        norm_batch: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Return the batch whose state values should feed VLM-prefix conditioning."""
        if self._state_prefix_encoding() == "none":
            return raw_batch
        if norm_batch is not None:
            return norm_batch
        normalize = getattr(self, "normalize", None)
        if normalize is None:
            return raw_batch
        return normalize(dict(raw_batch))

    def _state_prefix_encoding(self) -> StatePrefixEncoding:
        return validate_state_prefix_encoding(str(self.config.state_prefix_encoding))


def _discretize_state_vector(state_vec: Tensor, *, num_bins: int) -> list[str]:
    bins = (
        ((state_vec + 1) / 2 * (int(num_bins) - 1))
        .round()
        .clamp(0, int(num_bins) - 1)
        .long()
    )
    return ["State: " + " ".join(str(token) for token in row) for row in bins.tolist()]


def _validate_state_prefix_batch_size(
    state_vec: Tensor,
    *,
    batch_size: int | None,
    context: str,
) -> None:
    if batch_size is None:
        return
    actual = int(state_vec.shape[0])
    if actual != int(batch_size):
        raise ValueError(
            f"{context} batch size must match VLM input batch size, "
            f"got {actual} and {int(batch_size)}."
        )


def _infer_state_prefix_batch_size(obs: dict[str, Tensor]) -> int | None:
    for value in obs.values():
        if isinstance(value, Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _validate_single_token_encoding(
    tokenizer: Any, *, token: str, token_id: int
) -> None:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return
    try:
        ids = cast(Any, encode)(token, add_special_tokens=False)
    except TypeError:
        ids = cast(Any, encode)(token)
    if len(ids) != 1 or int(ids[0]) != int(token_id):
        raise RuntimeError(
            f"state_prefix_placeholder_token={token!r} must encode to exactly its "
            f"reserved id {token_id}, got {ids!r}."
        )
