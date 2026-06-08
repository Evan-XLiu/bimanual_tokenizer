# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.core.vlm.prefix import VlmPrefixState


class DummyFixedTokenizer:
    def __init__(self) -> None:
        self.max_code_seq_len = 4
        self.last_detokenize_tokens = None

    def tokenize(self, actions):
        del actions
        return torch.tensor([[1, 2, 3, 4]])

    def detokenize(self, tokens):
        self.last_detokenize_tokens = tokens.clone()
        return tokens.float()


class DummyVariableTokenizer:
    def __init__(self) -> None:
        self.max_code_seq_len = 4
        self._pad_token_id = 9
        self.last_detokenize_tokens = None

    def tokenize(self, actions, *, pad=False):
        del actions, pad
        return [torch.tensor([1, 2]), [3]]

    def detokenize(self, tokens, strict_decode=None):
        del strict_decode
        self.last_detokenize_tokens = tokens.clone()
        return tokens.float()


class DummyFeature:
    def __init__(self, key: str) -> None:
        self.key = key


class BasicPrefixAdapter:
    """Small VLM prefix adapter for BAR tests that patch lightweight dummy VLMs."""

    def __init__(self, vlm: Any) -> None:
        self.vlm = vlm

    @property
    def num_layers(self) -> int:
        return len(getattr(self.vlm, "layers", ()))

    @property
    def hidden_size(self) -> int:
        return int(self.vlm.get_input_embeddings().weight.shape[1])

    def prepare_prefix(
        self,
        vlm_inputs: dict[str, torch.Tensor],
        *,
        layer_attention_mask: torch.Tensor | None = None,
    ) -> VlmPrefixState:
        input_ids = vlm_inputs["input_ids"]
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.vlm.get_input_embeddings()(input_ids)
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )
        return VlmPrefixState(
            hidden=hidden,
            mask=attention_mask.bool(),
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            cache_position=position_ids[0],
            position_embeddings=(hidden, hidden),
        )

    def forward_prefix_layer(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
    ) -> VlmPrefixState:
        layer = getattr(self.vlm, "layers", ())[layer_idx]
        hidden = layer(state.hidden)
        return VlmPrefixState(
            hidden=hidden,
            mask=state.mask,
            attention_mask=state.attention_mask,
            position_ids=state.position_ids,
            cache_position=state.cache_position,
            position_embeddings=state.position_embeddings,
            suffix_position_offsets=state.suffix_position_offsets,
            extra=state.extra,
        )

    def final_hidden_to_logits(
        self,
        hidden: torch.Tensor,
        *,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.vlm.get_input_embeddings().weight
        if candidate_token_ids is not None:
            weight = weight.index_select(0, candidate_token_ids.to(weight.device))
        return hidden @ weight.T


def make_basic_policy_io(*, with_image: bool = False) -> PolicyIO:
    obs_features: list[FeatureSpec] = [
        FeatureSpec(
            key="observation.state",
            type=FeatureType.STATE,
            shape=(4,),
            norm_mode=NormMode.MIN_MAX,
        )
    ]
    if with_image:
        obs_features.append(
            FeatureSpec(
                key="observation.image",
                type=FeatureType.IMAGE,
                shape=(3, 4, 4),
                norm_mode=NormMode.MEAN_STD,
            )
        )

    return PolicyIO(
        obs_features=obs_features,
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(2,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=1,
        n_action_steps=1,
        fps=10,
    )


def basic_stats_for_io(io: PolicyIO) -> dict[str, dict[str, torch.Tensor]]:
    stats: dict[str, dict[str, torch.Tensor]] = {
        "observation.state": {
            "min": torch.zeros(4),
            "max": torch.full((4,), 10.0),
            "mean": torch.full((4,), 5.0),
            "std": torch.ones(4),
        },
        "action": {
            "min": torch.full((2,), -1.0),
            "max": torch.full((2,), 1.0),
            "mean": torch.zeros(2),
            "std": torch.ones(2),
        },
    }
    if any(feat.key == "observation.image" for feat in io.obs_features):
        stats["observation.image"] = {
            "mean": torch.full((3, 2, 2), 0.5),
            "std": torch.full((3, 2, 2), 0.25),
        }
    return stats


def bounded_minmax_stats_for_io(
    io: PolicyIO,
    *,
    state_min: float = -1.0,
    state_max: float = 1.0,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> dict[str, dict[str, torch.Tensor]]:
    state_dim = int(io.state_features[0].shape[0])
    action_dim = int(io.action_dim)
    return {
        "observation.state": {
            "min": torch.full((state_dim,), float(state_min)),
            "max": torch.full((state_dim,), float(state_max)),
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim),
        },
        "action": {
            "min": torch.full((action_dim,), float(action_min)),
            "max": torch.full((action_dim,), float(action_max)),
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
        },
    }


def mock_build_vlm_inputs_basic(
    self, obs, *, action_token_strings=None, instructions=None
):
    del self, action_token_strings, instructions
    batch_size = next(v.shape[0] for v in obs.values() if isinstance(v, torch.Tensor))
    return {
        "input_ids": torch.zeros(batch_size, 10, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
    }


def mock_build_vlm_inputs_all_ignore_basic(
    self, obs, *, action_token_strings=None, instructions=None
):
    del self, action_token_strings, instructions
    batch_size = next(v.shape[0] for v in obs.values() if isinstance(v, torch.Tensor))
    seq_len = 10
    return {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.full((batch_size, seq_len), -100, dtype=torch.long),
    }
