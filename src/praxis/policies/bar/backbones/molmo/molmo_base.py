"""Molmo2-ER image/text path for BAR action-token policies."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from torch import Tensor

from praxis.policies.backbones.molmo.prefix import MolmoPrefixAdapter
from praxis.policies.backbones.molmo.vlm import (
    configure_molmo_vision_checkpointing,
    ensure_molmo_vision_eval,
    freeze_molmo_vision,
    load_molmo_vlm,
    run_molmo_processor,
)
from praxis.policies.bar.backbones.molmo.molmo_config import MolmoPolicyConfig
from praxis.policies.bar.core.vlm.base import VlmActionTokenPolicyBase
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.dtypes import parse_vlm_dtype
from praxis.policies.core.vlm.hf_chatprompt import render_chat_texts
from praxis.policies.core.vlm.prefix import VlmInputBatch, VlmPrefixLogitsAdapter


def _molmo_channels_last_observations(
    obs: dict[str, Tensor], image_features: Any
) -> dict[str, Tensor]:
    """Resolve camera layout from PolicyIO before the shared image stacker.

    PolicyIO image shapes are CHW. Explicit conversion avoids treating a narrow
    CHW image (for example 3x1x1) as HWC because its width is also 1.
    """
    converted = dict(obs)
    for feature in image_features:
        images = obs[feature.key]
        if images.ndim not in (4, 5):
            raise ValueError(f"{feature.key}: expected BCHW/BHWC with optional history")
        shape = tuple(int(x) for x in images.shape[-3:])
        expected = tuple(int(x) for x in getattr(feature, "shape", ()))
        if len(expected) != 3:
            raise ValueError(f"{feature.key}: PolicyIO requires a CHW image shape")
        chw = expected
        hwc = (expected[1], expected[2], expected[0])
        if shape == chw and shape != hwc:
            channels_first = True
        elif shape == hwc and shape != chw:
            channels_first = False
        elif shape[0] == expected[0] and shape[-1] != expected[0]:
            channels_first = True
        elif shape[-1] == expected[0] and shape[0] != expected[0]:
            channels_first = False
        else:
            raise ValueError(
                f"{feature.key}: ambiguous image layout {shape}; "
                f"PolicyIO CHW shape is {expected}"
            )
        if channels_first:
            converted[feature.key] = images.movedim(-3, -1)
    return converted


class MolmoPolicyBase(VlmActionTokenPolicyBase):
    """BAR token CE with the native Molmo2-ER vision and text backbone."""

    CONFIG_CLASS: ClassVar[type[MolmoPolicyConfig] | None] = None
    ACTION_TOKEN_OWNER_NAME: ClassVar[str] = "Molmo"
    _PREFIX_ATTENTION: ClassVar[Literal["bidirectional", "causal"]] = "causal"
    _PREFILL_BIDIRECTIONAL_IMAGE_TOKENS: ClassVar[bool] = True
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
        self.to(dtype=parse_vlm_dtype(self.config.vlm_dtype))

    def _post_set_io(self) -> None:
        super()._post_set_io()
        self.to(dtype=parse_vlm_dtype(self.config.vlm_dtype))

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        self.vlm, self.processor = load_molmo_vlm(model_name, dtype_str)

    def _freeze_vision_encoder_modules(self) -> None:
        freeze_molmo_vision(self.vlm)

    def _ensure_frozen_vision_eval(self) -> None:
        ensure_molmo_vision_eval(self.vlm)

    def _build_vlm_prefix_adapter(self) -> VlmPrefixLogitsAdapter:
        return MolmoPrefixAdapter(self.vlm)

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
            _molmo_channels_last_observations(obs, self.io.image_features),
            image_features=self.io.image_features,
            as_uint8=True,
            channels_last=True,
        )
        bsz, num_images = int(image_batch.shape[0]), int(image_batch.shape[1])
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
            include_system=False,
        )
        batch_input = run_molmo_processor(
            processor=self.processor,
            text=texts,
            image_batch=image_batch,
            channels_last=True,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)
        dtype = parse_vlm_dtype(self.config.vlm_dtype)
        return {
            key: value.to(dtype=dtype)
            if isinstance(value, Tensor) and value.is_floating_point()
            else value
            for key, value in dict(batch_input).items()
        }
