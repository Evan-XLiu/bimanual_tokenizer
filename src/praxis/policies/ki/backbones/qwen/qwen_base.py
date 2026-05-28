"""Qwen backbone implementation for knowledge-insulated policies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from torch import Tensor

from praxis.policies.backbones.qwen.prefix import (
    QwenPrefixAdapter,
    with_qwen_position_ids,
)
from praxis.policies.backbones.qwen.vlm import (
    ensure_qwen_vision_eval,
    freeze_qwen_vision,
    load_qwen_vlm,
    run_qwen_processor,
)
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.hf_chatprompt import render_chat_texts
from praxis.policies.core.vlm.prefix import VlmInputBatch
from praxis.policies.ki.backbones.qwen.qwen_config import QwenPolicyConfig
from praxis.policies.ki.core.one_forward import KiVlmPrefixAdapter
from praxis.policies.ki.core.policy import PolicyBase


class QwenPolicyBase(PolicyBase):
    """Common implementation for Qwen-backed KI policies."""

    CONFIG_CLASS = QwenPolicyConfig
    _PREFIX_ATTENTION: ClassVar[Literal["bidirectional", "causal"]] = "causal"
    _PREFILL_EXTRA_KEYS: ClassVar[tuple[str, ...]] = (
        "pixel_values",
        "image_grid_thw",
    )
    vlm: Any
    processor: Any

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        self.vlm, self.processor = load_qwen_vlm(model_name, dtype_str)

    def _freeze_vision_encoder_modules(self) -> None:
        freeze_qwen_vision(self.vlm)

    def _ensure_frozen_vision_eval(self) -> None:
        ensure_qwen_vision_eval(self.vlm)

    def _build_vlm_prefix_adapter(self) -> KiVlmPrefixAdapter:
        return QwenPrefixAdapter(self.vlm)

    def _prepare_vlm_prefill_inputs_for_action_expert(
        self,
        vlm_inputs: VlmInputBatch,
    ) -> VlmInputBatch:
        return with_qwen_position_ids(self.vlm, vlm_inputs)

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        if self.io is None:
            raise RuntimeError("Policy IO must be set before building Qwen inputs.")
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
        batch_input = run_qwen_processor(
            processor=self.processor,
            text=texts,
            image_batch=image_batch,
            padding=True,
            return_tensors="pt",
        )
        batch_input = batch_input.to(self.vlm.device)
        vlm_inputs = dict(batch_input)
        return self._add_continuous_state_prefix_embeddings(vlm_inputs, state_prefix)
