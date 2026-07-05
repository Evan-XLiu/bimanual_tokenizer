# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone shared base for Pali VLM action-token policies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from torch import Tensor

from praxis.policies.backbones.pali.prefix import PaliGemmaPrefixAdapter
from praxis.policies.backbones.pali.vlm import (
    PaliProcessorLike,
    ensure_paligemma_vision_eval,
    freeze_paligemma_vision,
    load_paligemma_vlm,
    render_paligemma_texts,
    run_paligemma_processor,
)
from praxis.policies.bar.backbones.pali.pali_config import (
    PaliPolicyConfig,
)
from praxis.policies.bar.core.vlm.base import VlmActionTokenPolicyBase
from praxis.policies.core.observations import (
    stack_observation_images,
)
from praxis.policies.core.vlm.prefix import VlmInputBatch, VlmPrefixLogitsAdapter


class PaliPolicyBase(VlmActionTokenPolicyBase):
    """Common implementation for Pali-VLM action-token policies."""

    CONFIG_CLASS: ClassVar[type[PaliPolicyConfig] | None] = None
    ACTION_TOKEN_OWNER_NAME: ClassVar[str] = "Pali"
    _PREFIX_ATTENTION: ClassVar[Literal["bidirectional", "causal"]] = "bidirectional"
    _SCHEDULE_STRIP_KEYS: ClassVar[tuple[str, ...]] = ("labels", "token_type_ids")
    _PREFILL_EXTRA_KEYS: ClassVar[tuple[str, ...]] = ("pixel_values",)
    vlm: Any
    processor: PaliProcessorLike

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        self.vlm, self.processor = load_paligemma_vlm(model_name, dtype_str)

    def _freeze_vision_encoder_modules(self) -> None:
        freeze_paligemma_vision(self.vlm)

    def _ensure_frozen_vision_eval(self) -> None:
        ensure_paligemma_vision_eval(self.vlm)

    def _build_vlm_prefix_adapter(self) -> VlmPrefixLogitsAdapter:
        return PaliGemmaPrefixAdapter(self.vlm)

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        if self.io is None:
            raise RuntimeError(
                "Policy IO must be set before building image batch tensor."
            )
        image_batch = stack_observation_images(
            obs,
            image_features=self.io.image_features,
            as_uint8=True,
        )
        bsz = int(image_batch.shape[0])
        num_images = int(image_batch.shape[1])
        state_prefix = self._prepare_state_prefix(obs, batch_size=bsz)

        if instructions is None:
            instructions = [self.config.default_instruction] * bsz
        elif len(instructions) != bsz:
            raise ValueError(
                f"Instructions batch length mismatch: got {len(instructions)} "
                f"instructions for batch size {bsz}."
            )

        texts = render_paligemma_texts(
            processor=self.processor,
            state_texts=state_prefix.texts,
            instructions=instructions,
            action_token_strings=None,
            bsz=bsz,
            num_images=num_images,
        )
        batch_input = run_paligemma_processor(
            processor=self.processor,
            text=texts,
            suffix=action_token_strings,
            image_batch=image_batch,
            padding=True,
            return_tensors="pt",
        )
        batch_input = batch_input.to(self.vlm.device)
        return dict(batch_input)
