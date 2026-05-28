"""Shared config for VLM-conditioned continuous-action policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal

from praxis.policies.core.config import (
    non_negative_float,
    positive_float,
    positive_int,
    tuple_of_str,
)
from praxis.policies.core.state_prefix import (
    DEFAULT_STATE_PREFIX_TOKEN,
    StatePrefixEmbedInit,
    StatePrefixEncoding,
    validate_state_prefix_embed_init,
    validate_state_prefix_encoding,
    validate_state_prefix_num_bins,
    validate_state_prefix_placeholder_token,
)

FlowTimeSampling = Literal["beta", "uniform"]


@dataclass
class ActionExpertPolicyConfig:
    """Hyperparameters shared by FM-style VLM/action-expert policies."""

    PRETRAINED_RUNTIME_OVERRIDE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "default_instruction",
            "flow_loss_weight",
            "flow_sampling_steps",
            "n_action_steps",
        }
    )

    name: str = "fm_pali"
    pretrained_path: str | None = None

    vlm_model_name: str = "google/paligemma2-3b-mix-224"
    vlm_dtype: str = "bfloat16"
    freeze_vision_encoder: bool = False

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
    lora_modules_to_save: tuple[str, ...] = ()
    gradient_checkpointing: bool = False

    action_dim: int | None = None
    n_obs_steps: int = 1
    action_horizon: int = 32
    n_action_steps: int = 16

    state_prefix_encoding: StatePrefixEncoding = "continuous_projection"
    state_prefix_num_bins: int = 256
    state_prefix_placeholder_token: str = DEFAULT_STATE_PREFIX_TOKEN
    state_prefix_embed_init: StatePrefixEmbedInit = "normal"

    flow_loss_weight: float = 1.0
    flow_sampling_steps: int = 10
    flow_time_sampling: FlowTimeSampling = "beta"
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    flow_min_time: float = 0.001

    expert_hidden_size: int | None = None
    expert_width_multiplier: float = 0.75
    expert_num_layers: int | None = None
    expert_self_attn_every_n_layers: int = 2
    expert_dropout: float = 0.0
    expert_time_min_period: float = 4e-3
    expert_time_max_period: float = 4.0

    default_instruction: str = "Analyze the input image(s) and predict robot actions."

    def __post_init__(self) -> None:
        self.flow_loss_weight = non_negative_float(
            self.flow_loss_weight,
            "flow_loss_weight",
        )
        self.flow_sampling_steps = positive_int(
            self.flow_sampling_steps, "flow_sampling_steps"
        )
        self.action_horizon = positive_int(self.action_horizon, "action_horizon")
        self.n_action_steps = positive_int(self.n_action_steps, "n_action_steps")
        self.n_obs_steps = positive_int(self.n_obs_steps, "n_obs_steps")
        if self.expert_hidden_size is not None:
            self.expert_hidden_size = positive_int(
                self.expert_hidden_size, "expert_hidden_size"
            )
        self.expert_width_multiplier = positive_float(
            self.expert_width_multiplier, "expert_width_multiplier"
        )
        if self.expert_num_layers is not None:
            self.expert_num_layers = positive_int(
                self.expert_num_layers, "expert_num_layers"
            )
        self.expert_self_attn_every_n_layers = int(self.expert_self_attn_every_n_layers)
        if (
            self.expert_self_attn_every_n_layers == 0
            or self.expert_self_attn_every_n_layers < -1
        ):
            raise ValueError(
                "expert_self_attn_every_n_layers must be -1 or a positive integer, "
                f"got {self.expert_self_attn_every_n_layers}."
            )
        self.expert_dropout = non_negative_float(self.expert_dropout, "expert_dropout")
        self.expert_time_min_period = positive_float(
            self.expert_time_min_period, "expert_time_min_period"
        )
        self.expert_time_max_period = positive_float(
            self.expert_time_max_period, "expert_time_max_period"
        )
        if self.expert_time_min_period > self.expert_time_max_period:
            raise ValueError(
                "expert_time_min_period must be <= expert_time_max_period, "
                f"got {self.expert_time_min_period} and {self.expert_time_max_period}."
            )
        if self.flow_time_sampling not in {"beta", "uniform"}:
            raise ValueError(
                "flow_time_sampling must be one of {'beta', 'uniform'}, "
                f"got {self.flow_time_sampling!r}."
            )
        self.flow_beta_alpha = positive_float(self.flow_beta_alpha, "flow_beta_alpha")
        self.flow_beta_beta = positive_float(self.flow_beta_beta, "flow_beta_beta")
        self.flow_min_time = float(self.flow_min_time)
        if not 0.0 <= self.flow_min_time < 1.0:
            raise ValueError(
                f"flow_min_time must satisfy 0 <= value < 1, got {self.flow_min_time}."
            )
        if self.action_dim is not None:
            self.action_dim = positive_int(self.action_dim, "action_dim")
        self.state_prefix_encoding = validate_state_prefix_encoding(
            str(self.state_prefix_encoding)
        )
        self.state_prefix_num_bins = validate_state_prefix_num_bins(
            self.state_prefix_num_bins
        )
        self.state_prefix_placeholder_token = validate_state_prefix_placeholder_token(
            self.state_prefix_placeholder_token
        )
        self.state_prefix_embed_init = validate_state_prefix_embed_init(
            self.state_prefix_embed_init
        )
        for key in ("lora_target_modules", "lora_modules_to_save"):
            setattr(self, key, tuple_of_str(getattr(self, key), key))

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]):
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(
                f"Unknown policy config key(s) for {cls.__name__}: {', '.join(unknown)}."
            )
        payload = dict(cfg)
        for key in ("lora_target_modules", "lora_modules_to_save"):
            if key in payload:
                payload[key] = tuple_of_str(payload[key], key)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
