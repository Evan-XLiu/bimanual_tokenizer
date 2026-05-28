"""Smol backbone implementation for knowledge-insulated policies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

from torch import Tensor

from praxis.policies.backbones.smol.prefix import SmolPrefixAdapter
from praxis.policies.backbones.smol.preprocessing import (
    run_processor as run_smol_processor,
)
from praxis.policies.backbones.smol.vlm import (
    ensure_smol_vision_eval,
    freeze_smol_vision,
    load_smol_vlm,
)
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.hf_chatprompt import render_chat_texts
from praxis.policies.core.vlm.prefix import VlmInputBatch, with_compact_position_ids
from praxis.policies.ki.backbones.smol.smol_config import SmolPolicyConfig
from praxis.policies.ki.core.one_forward import KiVlmPrefixAdapter
from praxis.policies.ki.core.policy import PolicyBase


class SmolPolicyBase(PolicyBase):
    """Common implementation for Smol-backed KI policies."""

    CONFIG_CLASS = SmolPolicyConfig
    _PREFIX_ATTENTION: ClassVar[Literal["bidirectional", "causal"]] = "causal"
    _PREFILL_EXTRA_KEYS: ClassVar[tuple[str, ...]] = (
        "pixel_values",
        "pixel_attention_mask",
        "image_hidden_states",
    )
    vlm: Any
    processor: Any

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        self.vlm, self.processor = load_smol_vlm(model_name, dtype_str)

    def _freeze_vision_encoder_modules(self) -> None:
        freeze_smol_vision(self.vlm)

    def _ensure_frozen_vision_eval(self) -> None:
        ensure_smol_vision_eval(self.vlm)

    def _build_vlm_prefix_adapter(self) -> KiVlmPrefixAdapter:
        return SmolPrefixAdapter(self.vlm)

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
            raise RuntimeError("Policy IO must be set before building Smol inputs.")
        image_batch = stack_observation_images(
            obs,
            image_features=self.io.image_features,
            as_uint8=False,
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
        batch_input = run_smol_processor(
            self.processor,
            text=texts,
            image_batch=image_batch,
            resize_imgs_with_padding=cast(
                SmolPolicyConfig,
                self.config,
            ).resize_imgs_with_padding,
            padding=True,
            return_tensors="pt",
        )
        batch_input = batch_input.to(self.vlm.device)
        vlm_inputs = dict(batch_input)
        return self._add_continuous_state_prefix_embeddings(vlm_inputs, state_prefix)
