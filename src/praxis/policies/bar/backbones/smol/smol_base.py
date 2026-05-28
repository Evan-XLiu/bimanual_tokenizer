"""Shared base for SmolVLM action-token policies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

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
from praxis.policies.bar.backbones.smol.smol_config import SmolPolicyConfig
from praxis.policies.bar.core.vlm.base import VlmActionTokenPolicyBase
from praxis.policies.core.observations import (
    stack_observation_images,
)
from praxis.policies.core.vlm.hf_chatprompt import render_chat_texts
from praxis.policies.core.vlm.prefix import VlmInputBatch, VlmPrefixLogitsAdapter


class SmolPolicyBase(VlmActionTokenPolicyBase):
    """Common implementation for SmolVLM action-token policies.

    Handles VLM loading, action token vocabulary setup, state discretization,
    image conversion, chat template building, label masking, action-chunking
    queue, optional LoRA, and Smol-specific checkpoint filtering in LoRA mode.
    """

    CONFIG_CLASS: ClassVar[type[SmolPolicyConfig] | None] = None
    ACTION_TOKEN_OWNER_NAME: ClassVar[str] = "Smol"
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

    def _build_vlm_prefix_adapter(self) -> VlmPrefixLogitsAdapter:
        return SmolPrefixAdapter(self.vlm)

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        """Build SmolVLM chat-template inputs from observation dict.

        Args:
            obs: Observation tensors.
            action_token_strings: Per-sample action token strings (training only).
            instructions: Per-sample text instructions (uses default if absent).

        Returns:
            Dict with ``input_ids``, ``attention_mask``, and ``pixel_values``.
        """
        if self.io is None:
            raise RuntimeError(
                "Policy IO must be set before building image batch tensor."
            )
        image_batch = stack_observation_images(
            obs,
            image_features=self.io.image_features,
            as_uint8=False,
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
            resize_imgs_with_padding=self.config.resize_imgs_with_padding,
            padding=True,
            return_tensors="pt",
        )
        batch_input = batch_input.to(self.vlm.device)
        return dict(batch_input)
