# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo backbone implementation for knowledge-insulated policies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from torch import Tensor

from praxis.policies.backbones.molmo.prefix import (
    MolmoPrefixAdapter,
)
from praxis.policies.backbones.molmo.vlm import (
    ensure_molmo_vision_eval,
    freeze_molmo_vision,
    load_molmo_vlm,
    run_molmo_processor,
)
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.hf_chatprompt import render_chat_texts
from praxis.policies.core.vlm.prefix import VlmInputBatch, with_compact_position_ids
from praxis.policies.ki.backbones.molmo.molmo_config import MolmoPolicyConfig
from praxis.policies.ki.core.one_forward import KiVlmPrefixAdapter
from praxis.policies.ki.core.policy import PolicyBase


class MolmoPolicyBase(PolicyBase):
    """Common implementation for Molmo-backed KI policies."""

    CONFIG_CLASS = MolmoPolicyConfig
    _PREFIX_ATTENTION: ClassVar[Literal["bidirectional", "causal"]] = "causal"
    _PREFILL_EXTRA_KEYS: ClassVar[tuple[str, ...]] = (
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
        "token_type_ids",
    )
    vlm: Any
    processor: Any

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        self.vlm, self.processor = load_molmo_vlm(model_name, dtype_str)

    def _freeze_vision_encoder_modules(self) -> None:
        freeze_molmo_vision(self.vlm)

    def _ensure_frozen_vision_eval(self) -> None:
        ensure_molmo_vision_eval(self.vlm)

    def _build_vlm_prefix_adapter(self) -> KiVlmPrefixAdapter:
        return MolmoPrefixAdapter(self.vlm)

    def _prepare_vlm_prefill_inputs_for_action_expert(
        self,
        vlm_inputs: VlmInputBatch,
    ) -> VlmInputBatch:
        return with_compact_position_ids(vlm_inputs, first_position=0)

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        if self.io is None:
            raise RuntimeError("Policy IO must be set before building Molmo inputs.")
        image_batch = stack_observation_images(
            obs,
            image_features=self.io.image_features,
            as_uint8=True,
        )
        bsz = int(image_batch.shape[0])
        num_images = int(image_batch.shape[1])

        if instructions is None:
            instructions = [self.config.default_instruction] * bsz
        elif len(instructions) != bsz:
            raise ValueError(
                f"Instructions batch length mismatch: got {len(instructions)} "
                f"instructions for batch size {bsz}."
            )

        state_prefix = self._prepare_state_prefix(obs, batch_size=bsz)
        texts = render_chat_texts(
            processor=self.processor,
            num_images=num_images,
            state_texts=state_prefix.texts,
            instructions=instructions,
            action_token_strings=action_token_strings,
            bsz=bsz,
        )
        batch_input = run_molmo_processor(
            processor=self.processor,
            text=texts,
            image_batch=image_batch,
            padding=True,
            return_tensors="pt",
        )
        batch_input = batch_input.to(self.vlm.device)
        vlm_inputs = dict(batch_input)
        return self._add_continuous_state_prefix_embeddings(vlm_inputs, state_prefix)
