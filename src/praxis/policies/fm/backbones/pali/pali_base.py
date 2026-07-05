# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""PaliGemma backbone implementation for flow-matching policies."""

from __future__ import annotations

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
from praxis.policies.core.action_expert import VlmActionExpertPrefixAdapter
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.prefix import (
    VlmInputBatch,
    with_compact_position_ids,
)
from praxis.policies.fm.backbones.pali.pali_config import PaliPolicyConfig
from praxis.policies.fm.core.policy import PolicyBase


class PaliPolicyBase(PolicyBase):
    """Common implementation for PaliGemma-backed FM policies."""

    CONFIG_CLASS = PaliPolicyConfig
    processor: PaliProcessorLike

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        self.vlm, self.processor = load_paligemma_vlm(model_name, dtype_str)

    def _freeze_vision_encoder_modules(self) -> None:
        freeze_paligemma_vision(self.vlm)

    def _ensure_frozen_vision_eval(self) -> None:
        ensure_paligemma_vision_eval(self.vlm)

    def _build_vlm_prefix_adapter(self) -> VlmActionExpertPrefixAdapter:
        return PaliGemmaPrefixAdapter(self.vlm)

    def _prepare_vlm_prefill_inputs_for_action_expert(
        self,
        vlm_inputs: VlmInputBatch,
    ) -> VlmInputBatch:
        return with_compact_position_ids(vlm_inputs, first_position=1)

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        if self.io is None:
            raise RuntimeError("Policy IO must be set before building Pali inputs.")
        image_batch = stack_observation_images(
            obs,
            image_features=self.io.image_features,
            as_uint8=True,
        )
        bsz = int(image_batch.shape[0])
        if instructions is None:
            instructions = [self.config.default_instruction] * bsz
        elif len(instructions) != bsz:
            raise ValueError(
                f"Instructions batch length mismatch: got {len(instructions)} "
                f"instructions for batch size {bsz}."
            )
        state_prefix = self._prepare_state_prefix(obs, batch_size=bsz)
        texts = render_paligemma_texts(
            processor=self.processor,
            state_texts=state_prefix.texts,
            instructions=instructions,
            action_token_strings=None,
            bsz=bsz,
            num_images=int(image_batch.shape[1]),
        )
        batch_input = run_paligemma_processor(
            processor=self.processor,
            text=texts,
            suffix=None,
            image_batch=image_batch,
        )
        batch_input = batch_input.to(self.vlm.device)
        vlm_inputs = dict(batch_input)
        return self._add_continuous_state_prefix_embeddings(vlm_inputs, state_prefix)
