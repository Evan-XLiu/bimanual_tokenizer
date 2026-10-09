# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo backbone implementation for knowledge-insulated policies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from torch import Tensor

from praxis.core.types import NormMode
from praxis.policies.backbones.molmo.prefix import (
    MolmoPrefixAdapter,
)
from praxis.policies.backbones.molmo.vlm import (
    configure_molmo_vision_checkpointing,
    ensure_molmo_vision_eval,
    freeze_molmo_vision,
    load_molmo_vlm,
    run_molmo_processor,
)
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.dtypes import parse_vlm_dtype
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

    def __init__(self, config: MolmoPolicyConfig | dict[str, Any], **kwargs) -> None:
        super().__init__(config, **kwargs)
        configure_molmo_vision_checkpointing(
            self.vlm, bool(self.config.gradient_checkpointing)
        )
        # Include codec and optional LoRA parameters created after VLM loading.
        self.to(dtype=parse_vlm_dtype(self.config.vlm_dtype))

    def _post_set_io(self) -> None:
        super()._post_set_io()
        # IO creates the expert, state projector, and normalization buffers.
        self.to(dtype=parse_vlm_dtype(self.config.vlm_dtype))

    def _actions_from_batch(self, batch: dict[str, Tensor]) -> Tensor:
        return (
            super()
            ._actions_from_batch(batch)
            .to(dtype=parse_vlm_dtype(self.config.vlm_dtype))
        )

    def set_normalization_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        modes: dict[str, NormMode] | None = None,
    ) -> None:
        super().set_normalization_stats(stats, modes=modes)
        # Stats can be installed or replaced after IO setup and model loading.
        for module in (self.normalize, self.unnormalize_action):
            if module is not None:
                module.to(dtype=parse_vlm_dtype(self.config.vlm_dtype))

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
            channels_last=True,
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
            # Molmo2's template only accepts alternating user/assistant roles;
            # it rejects a leading system message.
            include_system=False,
        )
        batch_input = run_molmo_processor(
            processor=self.processor,
            text=texts,
            image_batch=image_batch,
            channels_last=True,
            padding=True,
            return_tensors="pt",
        )
        batch_input = batch_input.to(self.vlm.device)
        vlm_inputs = dict(batch_input)
        dtype = parse_vlm_dtype(self.config.vlm_dtype)
        vlm_inputs = {
            key: value.to(dtype=dtype)
            if isinstance(value, Tensor) and value.is_floating_point()
            else value
            for key, value in vlm_inputs.items()
        }
        return self._add_continuous_state_prefix_embeddings(vlm_inputs, state_prefix)
