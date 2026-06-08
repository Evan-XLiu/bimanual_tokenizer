# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Token adapter for action autoencoders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch.nn.functional as F
from torch import Tensor

from praxis.codecs.latents.core import AutoencoderEncoding
from praxis.codecs.latents.core.training import (
    masked_autoencoder_loss,
    split_decay_param_groups,
)
from praxis.codecs.tokens.core.tokenizer import BaseTokenizer


@dataclass(frozen=True, kw_only=True)
class AutoencoderTokenization(AutoencoderEncoding):
    """Latents, tokens, and optional auxiliary losses from an encoder path."""

    token_ids: Tensor


class AutoencoderTokenizer(BaseTokenizer):
    """Token mixin for latent autoencoders with discrete latent codes.

    Learned token codecs should inherit this mixin together with the concrete
    latent autoencoder family, for example
    ``class OATTokenizer(AutoencoderTokenizer, OATAutoencoder)``.
    """

    def _tokenize_norm_actions(self, norm_actions: Tensor) -> AutoencoderTokenization:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _tokenize_norm_actions()."
        )

    def _tokens_to_latents(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _tokens_to_latents()."
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        actions = batch[self.action_key]
        norm_actions = self._normalize_action_tensor(actions)
        encoded = self._tokenize_norm_actions(norm_actions)
        recon = cast(Any, self)._decode_to_norm_actions(encoded.latents)
        raw_loss = F.mse_loss(recon, norm_actions, reduction="none")
        primary_loss = masked_autoencoder_loss(
            raw_loss,
            batch.get(f"{self.action_key}_is_pad"),
        )
        loss = primary_loss + sum(encoded.loss_terms)
        return loss, dict(encoded.metrics)

    def tokenize_with_latents(self, actions: Tensor) -> AutoencoderTokenization:
        """Tokenize actions while also returning the quantized latent path."""
        norm_actions = self._normalize_action_tensor(actions)
        return self._tokenize_norm_actions(norm_actions)

    def tokenize(self, actions: Tensor) -> Tensor:
        return self.tokenize_with_latents(actions).token_ids

    def detokenize(self, tokens: Tensor, **kwargs: Any) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must be rank-2 [B, L], got shape={tuple(tokens.shape)}."
            )
        latents = self._tokens_to_latents(tokens.long())
        return cast(Any, self).decode(latents, **kwargs)

    def autoencode(self, actions: Tensor, **kwargs: Any) -> Tensor:
        encoded = self.tokenize_with_latents(actions)
        return cast(Any, self).decode(encoded.latents, **kwargs)

    def get_optim_params(self) -> list[dict]:
        decay_weight_decay: float | None = None
        decay_fn = getattr(self, "_decay_weight_decay", None)
        if callable(decay_fn):
            decay_weight_decay = cast(Callable[[], float | None], decay_fn)()
        return split_decay_param_groups(
            self,
            decay_weight_decay=decay_weight_decay,
            no_decay_weight_decay=0.0,
        )
