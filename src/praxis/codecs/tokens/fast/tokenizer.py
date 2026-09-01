# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""FAST action tokenizer: DCT + BPE compression (Physical Intelligence).

Non-trainable, deterministic tokenizer that converts continuous action
sequences to discrete tokens via:
  1. DCT-II along the time axis
  2. Quantise (round, scale, offset)
  3. Map integers to Unicode characters
  4. BPE-encode the character string
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerFast

from praxis.codecs.tokens.core import BaseTokenizer
from praxis.codecs.tokens.fast.config import FASTTokenizerConfig
from praxis.codecs.tokens.fast.dct import dct, idct
from praxis.core.config import config_values_equal

logger = logging.getLogger(__name__)


class FASTTokenizer(BaseTokenizer):
    """FAST tokenizer with PI-aligned byte-level BPE compression."""

    _CHECKPOINT_OWNED_RUNTIME_FIELDS = frozenset(
        {
            "action_dim",
            "action_horizon",
            "action_normalization_mode",
            "clip_quantile_normalized_actions",
            "scale",
            "min_token",
            "max_token_length",
            "strict_decode",
            "vocab_size",
        }
    )

    def __init__(
        self, config: dict[str, Any] | FASTTokenizerConfig | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = FASTTokenizerConfig()
        elif isinstance(config, FASTTokenizerConfig):
            cfg = config
        else:
            cfg = FASTTokenizerConfig.from_dict(config)

        self.config = cfg
        self.action_dim = int(cfg.action_dim) if cfg.action_dim is not None else 0
        self.action_horizon = int(cfg.action_horizon)
        self.action_normalization_mode = str(cfg.action_normalization_mode)
        if self.action_normalization_mode not in {
            "dataset_min_max",
            "quantile_01_99",
        }:
            raise ValueError(
                "FAST action_normalization_mode must be 'dataset_min_max' or "
                f"'quantile_01_99', got {self.action_normalization_mode!r}."
            )
        self.clip_quantile_normalized_actions = bool(
            cfg.clip_quantile_normalized_actions
        )
        self.scale = float(cfg.scale)
        self.min_token = int(cfg.min_token)
        self.max_token_length = int(cfg.max_token_length)
        self.strict_decode = bool(cfg.strict_decode)

        self._clear_bpe()
        if cfg.bpe_tokenizer_path is not None:
            self._set_bpe(
                PreTrainedTokenizerFast.from_pretrained(cfg.bpe_tokenizer_path)
            )
            self._maybe_load_processor_metadata(cfg.bpe_tokenizer_path)

    # ------------------------------------------------------------------
    # BaseTokenizer interface
    # ------------------------------------------------------------------

    @property
    def codebook_size(self) -> int:
        return self._bpe_vocab_size + 1  # +1 for pad

    @property
    def variable_length_codes(self) -> bool:
        return True

    @property
    def max_code_seq_len(self) -> int:
        return int(self.max_token_length)

    def _clear_bpe(self) -> None:
        self._bpe: PreTrainedTokenizerFast | None = None
        self._bpe_vocab_size = 0
        self._pad_token_id = 0

    def _set_bpe(self, bpe: PreTrainedTokenizerFast) -> None:
        self._validate_bpe_decoder(bpe)
        self._bpe = bpe
        self._bpe_vocab_size = int(bpe.vocab_size)
        self._pad_token_id = self._bpe_vocab_size

    def _maybe_load_processor_metadata(self, processor_id_or_path: str) -> None:
        """Load FAST processor metadata (e.g., min_token) when available.

        The PI universal FAST release stores quantization metadata in
        ``processor_config.json``. When using only ``bpe_tokenizer_path``, we
        still need these values to keep tokenization compatible with the
        published tokenizer.
        """
        cfg_path: Path | None = None
        candidate = Path(processor_id_or_path)
        if candidate.is_dir():
            local_cfg = candidate / "processor_config.json"
            if local_cfg.exists():
                cfg_path = local_cfg
        if cfg_path is None:
            try:
                from huggingface_hub import hf_hub_download

                cfg_path = Path(
                    hf_hub_download(
                        repo_id=processor_id_or_path,
                        filename="processor_config.json",
                    )
                )
            except Exception:
                return
        try:
            data = json.loads(cfg_path.read_text())
        except Exception:
            return

        min_token = data.get("min_token")
        scale = data.get("scale")
        vocab_size = data.get("vocab_size")
        if min_token is not None:
            self.min_token = int(min_token)
            self.config.min_token = int(min_token)
        if scale is not None:
            self.scale = float(scale)
            self.config.scale = float(scale)
        if vocab_size is not None:
            self.config.vocab_size = int(vocab_size)

        logger.info(
            "Loaded FAST processor metadata from %s (min_token=%s, scale=%s, vocab_size=%s).",
            cfg_path,
            self.min_token,
            self.scale,
            self.config.vocab_size,
        )

    def set_io(self, io) -> None:
        super().set_io(io)
        io_action_dim = int(io.action_dim)
        if self.action_dim <= 0:
            self.action_dim = io_action_dim
            self.config.action_dim = io_action_dim
        elif self.action_dim != io_action_dim:
            raise ValueError(
                "FAST tokenizer action_dim mismatch: "
                f"tokenizer action_dim={self.action_dim}, io action_dim={io_action_dim}."
            )

    def set_normalization_stats(self, stats, modes=None) -> None:
        """Install dataset min/max or robust q01/q99 action statistics."""
        if self.action_normalization_mode != "quantile_01_99":
            super().set_normalization_stats(stats, modes=modes)
            return

        source_key = self.action_key if self.action_key in stats else None
        if source_key is None and len(stats) == 1:
            source_key = next(iter(stats))
        if source_key is None:
            super().set_normalization_stats(stats, modes=modes)
            return
        source_stats = dict(stats[source_key])
        if "q01" not in source_stats or "q99" not in source_stats:
            raise ValueError(
                "FAST quantile_01_99 normalization requires q01 and q99 action "
                f"statistics for {source_key!r}."
            )
        robust_stats = dict(source_stats)
        robust_stats["min"] = source_stats["q01"]
        robust_stats["max"] = source_stats["q99"]
        robust_payload = dict(stats)
        robust_payload[source_key] = robust_stats
        super().set_normalization_stats(robust_payload, modes=modes)

    def _normalize_action_tensor(self, actions: Tensor) -> Tensor:
        normalized = super()._normalize_action_tensor(actions)
        if (
            self.action_normalization_mode == "quantile_01_99"
            and self.clip_quantile_normalized_actions
        ):
            normalized = normalized.clamp(-1.0, 1.0)
        return normalized

    def _unnormalize_action_tensor(self, actions: Tensor) -> Tensor:
        if (
            self.action_normalization_mode == "quantile_01_99"
            and self.clip_quantile_normalized_actions
        ):
            actions = actions.clamp(-1.0, 1.0)
        return super()._unnormalize_action_tensor(actions)

    @staticmethod
    def _validate_bpe_decoder(bpe: PreTrainedTokenizerFast) -> None:
        """Fail fast when BPE tokenizer lacks a decoder."""
        if bpe.backend_tokenizer.decoder is None:
            raise RuntimeError("FAST BPE tokenizer has no decoder configured.")

    def _require_bpe(self) -> PreTrainedTokenizerFast:
        if self._bpe is None:
            raise RuntimeError(
                "FAST BPE tokenizer is not initialized. Provide a pretrained tokenizer "
                "path or call fit() first."
            )
        return self._bpe

    def tokenize(
        self,
        actions: Tensor,
        *,
        pad: bool = True,
        pad_value: int | None = None,
        return_lengths: bool = False,
    ) -> Tensor | list[Tensor] | tuple[Tensor, Tensor] | tuple[list[Tensor], Tensor]:
        """Tokenize actions into FAST BPE ids.

        Args:
            actions: Continuous action chunks ``(B, T, D)``.
            pad: Whether to pad outputs to a dense tensor.
            pad_value: Optional pad value override when ``pad=True``.
                Defaults to FAST internal pad token id.
            return_lengths: Whether to also return per-sample token lengths.

        Returns:
            If ``pad=True``: ``LongTensor[B, max_token_length]`` (and optional lengths).
            If ``pad=False``: ``list[LongTensor[L_i]]`` (and optional lengths).
        """
        if self.action_dim <= 0:
            raise RuntimeError(
                "FAST tokenizer action_dim is unresolved. "
                "Set tokenizer.action_dim or call set_io(...) before tokenization."
            )
        bpe = self._require_bpe()
        actions = self._normalize_action_tensor(actions)
        # 1. DCT along time axis
        coeffs = dct(actions, dim=-2)  # (B, T, D)
        # 2. Quantise
        quantised = torch.round(coeffs * self.scale)  # (B, T, D)
        # 3. Flatten last two dims -> (B, T*D), convert to char string per batch
        flat = quantised.reshape(actions.shape[0], -1)  # (B, T*D)
        flat = torch.clamp(flat - self.min_token, min=0).long()

        flat_list = flat.tolist()
        char_strings = ["".join(map(chr, row)) for row in flat_list]
        batch_encoding = cast(BatchEncoding, bpe(char_strings))
        all_ids = cast(list[list[int]], batch_encoding["input_ids"])

        lengths = torch.tensor(
            [len(ids) for ids in all_ids],
            dtype=torch.long,
            device=actions.device,
        )

        if not pad:
            tokens_list = [
                torch.tensor(ids, dtype=torch.long, device=actions.device)
                for ids in all_ids
            ]
            if return_lengths:
                return tokens_list, lengths
            return tokens_list

        # 4. Pad to max_token_length
        used_pad_value = self._pad_token_id if pad_value is None else int(pad_value)
        padded = torch.full(
            (actions.shape[0], self.max_token_length),
            used_pad_value,
            dtype=torch.long,
            device=actions.device,
        )
        # Truncate, build flat values + scatter indices in one pass
        flat_vals: list[int] = []
        row_indices: list[int] = []
        col_indices: list[int] = []
        for i, ids in enumerate(all_ids):
            if len(ids) > self.max_token_length:
                logger.warning(
                    "BPE produced %d tokens for sample %d, truncating to max_token_length=%d",
                    len(ids),
                    i,
                    self.max_token_length,
                )
            length = min(len(ids), self.max_token_length)
            flat_vals.extend(ids[:length])
            row_indices.extend([i] * length)
            col_indices.extend(range(length))
            lengths[i] = length
        if flat_vals:
            padded[row_indices, col_indices] = torch.tensor(
                flat_vals,
                dtype=torch.long,
                device=actions.device,
            )
        if return_lengths:
            return padded, lengths
        return padded

    def detokenize(
        self,
        tokens: Tensor,
        *,
        strict_decode: bool | None = None,
        pad_value: int | None = None,
        **kwargs,
    ) -> Tensor:
        """``(B, L)`` BPE token ids back to ``(B, T, D)`` continuous actions.

        Args:
            strict_decode: If ``True``, raise on length mismatch. If ``None``
                (default), falls back to ``self.strict_decode`` from config.
            pad_value: Optional pad id override matching ``tokenize(...,
                pad_value=...)``. Defaults to FAST internal pad token id.
        """
        del kwargs
        if self.action_dim <= 0:
            raise RuntimeError(
                "FAST tokenizer action_dim is unresolved. "
                "Set tokenizer.action_dim or call set_io(...) before detokenization."
            )
        bpe = self._require_bpe()
        strict = self.strict_decode if strict_decode is None else strict_decode
        batch_size = tokens.shape[0]
        device = tokens.device
        # Vectorized pad stripping: compute per-row valid lengths, then
        # flatten non-pad ids for batch_decode.
        used_pad_value = self._pad_token_id if pad_value is None else int(pad_value)
        not_pad = tokens != used_pad_value  # (B, L)
        valid_lengths = not_pad.sum(dim=1)  # (B,)
        # Mask out pad tokens with 0 so .tolist() gives clean rows, then trim
        masked = tokens * not_pad  # pads become 0
        masked_list = masked.tolist()
        valid_id_lists = [
            row[:vl]
            for row, vl in zip(masked_list, valid_lengths.tolist(), strict=False)
        ]
        # Batch BPE decode (Rust-parallelised)
        decoded_strs = bpe.batch_decode(valid_id_lists)

        expected_len = self.action_horizon * self.action_dim

        # Fast path: if every decoded string has the right length, build one
        # tensor for the whole batch and run a single batched IDCT.
        ord_lists = [list(map(ord, s)) for s in decoded_strs]
        all_match = all(len(ol) == expected_len for ol in ord_lists)

        if not all_match and strict:
            # Find first mismatch for the error message
            for i, ol in enumerate(ord_lists):
                if len(ol) != expected_len:
                    raise ValueError(
                        f"BPE decoded {len(ol)} coefficients for sample {i}, "
                        f"expected {expected_len} (horizon={self.action_horizon} * dim={self.action_dim}). "
                        f"Set strict_decode=False to allow pad/truncate."
                    )

        if not all_match:
            zero_coeff_code = -self.min_token
            # Lenient decode keeps a valid decoded prefix/suffix contract:
            # preserve decoded coefficients, truncate overflow, and pad missing
            # coefficients as real zero DCT coefficients.
            for i, ol in enumerate(ord_lists):
                if len(ol) < expected_len:
                    ol.extend([zero_coeff_code] * (expected_len - len(ol)))
                elif len(ol) > expected_len:
                    ord_lists[i] = ol[:expected_len]

        # Single tensor creation + min_token offset (vectorized)
        coeffs = torch.tensor(ord_lists, dtype=torch.float32, device=device)  # (B, T*D)
        coeffs = (coeffs + self.min_token) / self.scale
        coeffs = coeffs.reshape(batch_size, self.action_horizon, self.action_dim)
        # Batched IDCT: one matmul for the whole batch
        out = idct(coeffs, dim=-2)  # (B, T, D)
        return self._unnormalize_action_tensor(out)

    # ------------------------------------------------------------------
    # fit(): train BPE from scratch
    # ------------------------------------------------------------------

    def fit(self, actions: Tensor) -> None:
        """Train a BPE tokenizer from scratch on action data.

        Args:
            actions: ``(N, T, D)`` tensor of action chunks.
        """
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.trainers import BpeTrainer

        actions = self._normalize_action_tensor(actions)

        # DCT along time axis → flatten → quantize
        coeffs = dct(actions, dim=-2)  # (N, T, D)
        quantised = torch.round(coeffs * self.scale)  # (N, T, D)
        flat = quantised.reshape(actions.shape[0], -1)  # (N, T*D)

        min_token = int(flat.min().item())
        max_token = int(flat.max().item())
        token_span = max_token - min_token + 1
        if token_span > int(self.config.vocab_size):
            raise ValueError(
                f"FAST BPE vocab_size={self.config.vocab_size} is too small for token range "
                f"[{min_token}, {max_token}] (needs at least {token_span})."
            )
        if token_span + 100 > int(self.config.vocab_size):
            logger.warning(
                "FAST BPE initial alphabet size %d is close to vocab_size=%d; consider increasing vocab_size.",
                token_span,
                self.config.vocab_size,
            )

        self.min_token = min_token
        self.config.min_token = self.min_token

        flat_shifted = (flat - self.min_token).clamp(min=0).long()

        corpus = ["".join(chr(int(v)) for v in row) for row in flat_shifted]

        # Match PI's byte-level BPE setup.
        bpe = ByteLevelBPETokenizer()
        alphabet = [chr(i) for i in range(token_span)]
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=2,
            show_progress=True,
            initial_alphabet=alphabet,
            special_tokens=[],
            max_token_length=10000,
        )
        bpe._tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Wrap as PreTrainedTokenizerFast
        trained_bpe = PreTrainedTokenizerFast(
            tokenizer_object=bpe,
            clean_up_tokenization_spaces=False,
        )
        self._set_bpe(trained_bpe)
        self.config.vocab_size = self._bpe_vocab_size

        # BPE is now locally trained and will be saved alongside the checkpoint;
        # clear the remote path so config.json doesn't misleadingly point to the
        # original pretrained repo.
        self.config.bpe_tokenizer_path = None

        logger.info(
            "FAST fit complete: vocab_size=%d, min_token=%d",
            self._bpe_vocab_size,
            self.min_token,
        )

    # ------------------------------------------------------------------
    # Save / Load overrides for BPE persistence
    # ------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save model weights + BPE tokenizer."""
        super()._save_pretrained(save_directory)
        if self._bpe is not None:
            self._bpe.save_pretrained(save_directory)

    @classmethod
    def validate_pretrained_runtime_overrides(
        cls,
        *,
        build_cfg: dict[str, Any],
        source_cfg: dict[str, Any],
        load_id: str,
    ) -> None:
        """Reject explicit FAST metadata overrides owned by the checkpoint."""
        for key in sorted(cls._CHECKPOINT_OWNED_RUNTIME_FIELDS):
            if key not in build_cfg or key not in source_cfg:
                continue
            if config_values_equal(build_cfg[key], source_cfg[key]):
                continue
            raise ValueError(
                f"Cannot override tokenizer.{key} when loading "
                f"tokenizer.pretrained_path={load_id!r}. FAST shape and decode "
                "metadata are checkpoint-owned; use a matching tokenizer checkpoint "
                "or build the tokenizer from scratch."
            )

    def load_pretrained_runtime_state(self, source: BaseTokenizer) -> None:
        """Copy non-state-dict runtime assets when factory rebuilds from config."""
        source_bpe = getattr(source, "_bpe", None)
        if source_bpe is not None:
            self._set_bpe(source_bpe)
        source_config = getattr(source, "config", None)
        for attr in (
            "action_dim",
            "action_horizon",
            "scale",
            "min_token",
            "max_token_length",
            "strict_decode",
        ):
            if hasattr(source, attr):
                value = getattr(source, attr)
                setattr(self, attr, value)
                if hasattr(self.config, attr):
                    setattr(self.config, attr, value)
            elif source_config is not None and hasattr(source_config, attr):
                value = getattr(source_config, attr)
                setattr(self, attr, value)
                if hasattr(self.config, attr):
                    setattr(self.config, attr, value)
        if source_config is not None:
            self.config.vocab_size = int(
                self._bpe_vocab_size
                if source_bpe is not None
                else source_config.vocab_size
            )
            self.config.bpe_tokenizer_path = getattr(
                source_config,
                "bpe_tokenizer_path",
                None,
            )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str | Path,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        proxies: dict | None = None,
        resume_download: bool | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        **kwargs,
    ) -> FASTTokenizer:
        """Load model and restore the co-located tokenizer when available."""
        config_bpe_path = cls._extract_config_bpe_tokenizer_path(kwargs.get("config"))
        constructor_kwargs = cls._suppress_constructor_bpe_load(kwargs)
        hub_kwargs = {
            "revision": revision,
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "token": token,
        }
        instance = super()._from_pretrained(
            model_id=model_id,
            **hub_kwargs,
            **constructor_kwargs,
        )
        model_dir = Path(model_id)
        if model_dir.is_dir() and (model_dir / "tokenizer.json").exists():
            instance._set_bpe(PreTrainedTokenizerFast.from_pretrained(str(model_dir)))
            instance.config.bpe_tokenizer_path = None
        elif instance._bpe is None:
            loaded_bpe_path: str | None = None
            if instance._try_set_bpe_from_pretrained(str(model_id), hub_kwargs):
                loaded_bpe_path = str(model_id)
            elif config_bpe_path is not None:
                instance._set_bpe(
                    PreTrainedTokenizerFast.from_pretrained(
                        str(config_bpe_path),
                        **hub_kwargs,
                    )
                )
                loaded_bpe_path = str(config_bpe_path)
            if loaded_bpe_path is not None:
                instance.config.bpe_tokenizer_path = (
                    config_bpe_path if loaded_bpe_path == str(config_bpe_path) else None
                )
                instance._maybe_load_processor_metadata(loaded_bpe_path)
            else:
                raise FileNotFoundError(
                    "FAST checkpoint is missing BPE tokenizer assets. Expected "
                    f"co-located tokenizer files under {model_id!r} or "
                    "config.bpe_tokenizer_path."
                )
        elif instance._bpe is not None:
            instance._set_bpe(instance._bpe)
        return instance

    def _try_set_bpe_from_pretrained(
        self,
        model_id: str,
        hub_kwargs: dict[str, Any],
    ) -> bool:
        try:
            bpe = PreTrainedTokenizerFast.from_pretrained(model_id, **hub_kwargs)
        except OSError:
            return False
        self._set_bpe(bpe)
        return True

    @staticmethod
    def _extract_config_bpe_tokenizer_path(config: Any) -> str | None:
        if config is None:
            return None
        if isinstance(config, dict):
            value = config.get("bpe_tokenizer_path")
        else:
            value = getattr(config, "bpe_tokenizer_path", None)
        return None if value is None else str(value)

    @staticmethod
    def _suppress_constructor_bpe_load(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Defer BPE loading until co-located checkpoint files are tried first."""
        out = dict(kwargs)
        config = out.get("config")
        if config is None:
            return out
        if isinstance(config, dict):
            adjusted = dict(config)
            adjusted["bpe_tokenizer_path"] = None
            out["config"] = adjusted
            return out
        if isinstance(config, FASTTokenizerConfig):
            adjusted = FASTTokenizerConfig.from_dict(config.to_dict())
            adjusted.bpe_tokenizer_path = None
            out["config"] = adjusted
        return out
