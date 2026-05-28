"""Shared config dataclass for VLM action-token policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal, cast

from praxis.policies.bar.core.stop import (
    DecodeStopMode,
    validate_decode_keep_k_for_stop_mode,
    validate_decode_stop_mode,
)
from praxis.policies.core.state_prefix import (
    validate_state_prefix_encoding,
    validate_state_prefix_num_bins,
)
from praxis.policies.core.vlm.loss import (
    validate_block_ce_weighting,
    validate_ce_vocab_scope,
)

BarStatePrefixEncoding = Literal["none", "discrete_text"]


@dataclass
class VlmActionTokenPolicyConfig:
    """Hyperparameters shared by Pali/Smol/Qwen action-token policies."""

    PRETRAINED_RUNTIME_OVERRIDE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "block_ce_weighting",
            "ce_vocab_scope",
            "decode_keep_k",
            "decode_temperature",
            "decode_top_k",
            "default_instruction",
            "max_new_tokens",
            "n_action_steps",
        }
    )

    name: str = "vlm_action_token"
    pretrained_path: str | None = None

    # --- VLM backbone ---
    vlm_model_name: str = "google/paligemma2-3b-mix-224"
    vlm_dtype: str = "bfloat16"

    # --- Freezing ---
    freeze_vision_encoder: bool = False

    # --- LoRA (optional, off = full fine-tuning) ---
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_dropout: float = 0.05
    lora_modules_to_save: tuple[str, ...] = ("lm_head", "embed_tokens")
    gradient_checkpointing: bool = False

    # --- Architecture dims (derived from PolicyIO at construction) ---
    action_dim: int | None = None

    # --- Action tokenizer ---
    action_tokenizer: dict[str, Any] = field(default_factory=dict)
    freeze_action_tokenizer: bool = True

    # --- Action tokens in VLM vocabulary ---
    num_action_tokens: int | None = None
    action_token_template: str = "<|action_{i}|>"
    action_embed_init: str = "normal"  # "normal" | "avg" | "zero"

    # --- VLM state prefix ---
    state_prefix_encoding: BarStatePrefixEncoding = "discrete_text"
    state_prefix_num_bins: int = 256

    # --- Generation ---
    max_new_tokens: int | None = None
    decode_temperature: float | None = None
    decode_top_k: int | None = None
    decode_keep_k: int | None = None
    decode_stop_mode: DecodeStopMode = "fixed_budget"
    block_ce_weighting: str = "block_mean"
    ce_vocab_scope: str = "full"
    action_mask_token_template: str = "<|action_mask|>"

    # --- Instruction ---
    default_instruction: str = "Analyze the input image(s) and predict robot actions."

    # --- Policy IO ---
    n_obs_steps: int = 1
    action_horizon: int = 32
    n_action_steps: int = 16

    def __post_init__(self) -> None:
        self.decode_stop_mode = validate_decode_stop_mode(self.decode_stop_mode)
        validate_decode_keep_k_for_stop_mode(
            decode_stop_mode=self.decode_stop_mode,
            decode_keep_k=self.decode_keep_k,
        )
        self.block_ce_weighting = validate_block_ce_weighting(self.block_ce_weighting)
        self.ce_vocab_scope = validate_ce_vocab_scope(self.ce_vocab_scope)
        state_prefix_encoding = validate_state_prefix_encoding(
            str(self.state_prefix_encoding)
        )
        if state_prefix_encoding == "continuous_projection":
            raise ValueError(
                "BAR does not support state_prefix_encoding='continuous_projection' "
                "because BAR inference uses cached HF decode. Use "
                "state_prefix_encoding='discrete_text'."
            )
        self.state_prefix_encoding = cast(
            BarStatePrefixEncoding,
            state_prefix_encoding,
        )
        self.state_prefix_num_bins = validate_state_prefix_num_bins(
            self.state_prefix_num_bins
        )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> VlmActionTokenPolicyConfig:
        if "supervise_eos" in cfg:
            raise ValueError(
                "policy.supervise_eos is not configurable. Use "
                "policy.decode_stop_mode='eos' to enable EOS stopping."
            )
        if "block_schedule" in cfg:
            raise ValueError(
                "policy.block_schedule is not configurable. The action tokenizer "
                "is the only source for block_schedule; tokenizers without an "
                "explicit schedule use the standard all-ones schedule."
            )
        known = {f.name for f in cls.__dataclass_fields__.values()}
        cfg = dict(cfg)
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(
                "Unknown policy config key(s) for "
                f"{cls.__name__}: {', '.join(unknown)}."
            )
        filtered = {k: v for k, v in cfg.items() if k in known}
        for key in ("lora_target_modules", "lora_modules_to_save"):
            if key in filtered and isinstance(filtered[key], list):
                filtered[key] = tuple(filtered[key])
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
