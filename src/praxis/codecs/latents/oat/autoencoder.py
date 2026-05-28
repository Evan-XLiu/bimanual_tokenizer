"""OAT-family continuous action-latent autoencoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from praxis.codecs.core import pad_code_sequence
from praxis.codecs.latents.core import Autoencoder, AutoencoderEncoding
from praxis.codecs.latents.oat.config import OATAutoencoderConfig, OATModelConfig
from praxis.codecs.latents.oat.modeling import RegisterQueryEncoder, SinglePassDecoder
from praxis.codecs.latents.oat.schedule import normalize_decode_keep_k


class OATAutoencoder(Autoencoder):
    """OAT encoder/decoder without latent quantization."""

    config: OATModelConfig

    def __init__(
        self, config: dict[str, Any] | OATModelConfig | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = OATAutoencoderConfig()
        elif isinstance(config, OATModelConfig):
            cfg = config
        else:
            cfg = OATAutoencoderConfig.from_dict(config)

        self.config = cfg
        if cfg.action_dim is None:
            raise ValueError(
                f"{self.__class__.__name__} requires action_dim at construction time. "
                "Set codec.action_dim explicitly or build via dataset/PolicyIO-aware path."
            )

        self.action_dim = int(cfg.action_dim)
        self.latent_dim = int(cfg.latent_dim)
        self.encoder = self._build_encoder(cfg, self.latent_dim)
        self.decoder = self._build_decoder(
            cfg,
            self.latent_dim,
            self.encoder.latent_horizon,
        )
        self.latent_horizon = int(self.decoder.latent_horizon)

    @property
    def supports_prefix_decode(self) -> bool:
        return self.config.latent_dropout_mode != "disable"

    def _build_encoder(self, cfg: OATModelConfig, latent_dim: int):
        action_dim = cfg.action_dim
        if action_dim is None:
            raise ValueError(
                "OATAutoencoder requires action_dim before encoder construction."
            )
        return RegisterQueryEncoder(
            sample_dim=int(action_dim),
            sample_horizon=cfg.action_horizon,
            emb_dim=cfg.emb_dim,
            head_dim=cfg.head_dim,
            depth=cfg.encoder_depth,
            pdropout=cfg.dropout,
            latent_dim=latent_dim,
            register_schedule=cfg.register_schedule,
        )

    def _build_decoder(self, cfg: OATModelConfig, latent_dim: int, latent_horizon: int):
        action_dim = cfg.action_dim
        if action_dim is None:
            raise ValueError(
                "OATAutoencoder requires action_dim before decoder construction."
            )
        return SinglePassDecoder(
            sample_dim=int(action_dim),
            sample_horizon=cfg.action_horizon,
            emb_dim=cfg.emb_dim,
            head_dim=cfg.head_dim,
            depth=cfg.decoder_depth,
            pdropout=cfg.dropout,
            latent_dropout_mode=cfg.latent_dropout_mode,
            latent_dim=latent_dim,
            latent_horizon=latent_horizon,
        )

    def _encode_norm_actions_to_latents(
        self, norm_actions: Tensor
    ) -> AutoencoderEncoding:
        return AutoencoderEncoding(latents=self.encoder(norm_actions))

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs: Any) -> Tensor:
        latents, decode_keep_k = self._prepare_decode_latents(
            latents,
            decode_keep_k=kwargs.get("decode_keep_k"),
        )
        return self.decoder(latents, decode_keep_k=decode_keep_k)

    def _prepare_decode_latents(
        self, latents: Tensor, *, decode_keep_k: Sequence[int] | None
    ) -> tuple[Tensor, list[int] | None]:
        if latents.ndim != 3:
            raise ValueError(
                f"latents must be rank-3 [B, L, D], got shape={tuple(latents.shape)}."
            )
        provided_horizon = int(latents.shape[1])
        if provided_horizon > self.latent_horizon:
            raise ValueError(
                f"latent length mismatch: expected <= {self.latent_horizon}, got {provided_horizon}."
            )
        prepared_keep_k: list[int] | None = None
        if decode_keep_k is not None:
            prepared_keep_k = normalize_decode_keep_k(
                decode_keep_k,
                batch_size=int(latents.shape[0]),
                latent_horizon=self.latent_horizon,
                default_keep_k=self.latent_horizon,
            )
        elif provided_horizon < self.latent_horizon:
            prepared_keep_k = [provided_horizon] * int(latents.shape[0])

        uses_prefix_decode = provided_horizon < self.latent_horizon or (
            prepared_keep_k is not None
            and any(k < self.latent_horizon for k in prepared_keep_k)
        )
        if uses_prefix_decode and not self.supports_prefix_decode:
            raise ValueError(
                f"{self.__class__.__name__} does not support prefix decode when "
                "latent_dropout_mode='disable'."
            )
        if prepared_keep_k is not None and any(
            k > provided_horizon for k in prepared_keep_k
        ):
            raise ValueError(
                "decode_keep_k values must be <= provided latent length "
                f"{provided_horizon}, got {prepared_keep_k}."
            )
        if provided_horizon < self.latent_horizon:
            latents = pad_code_sequence(latents, self.latent_horizon)
        return latents, prepared_keep_k

    def _decay_weight_decay(self) -> float | None:
        return 0.0
