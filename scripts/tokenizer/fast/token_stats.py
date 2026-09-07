#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Compute FAST token statistics on a dataset.

Tokenizes action chunks from a LeRobot dataset and reports:
1) BPE token sequence length statistics
2) reconstruction quality after tokenize->detokenize (MSE/MAE/RMSE)

Usage:
    python scripts/tokenizer/fast/token_stats.py \
        --dataset-repo-id lerobot/libero \
        --checkpoint /path/to/fast_tokenizer/pretrained_model \
        [--dataset-root /local/path] \
        [--dataset-episodes 0 1 2] \
        [--split all|val] \
        [--batch-size 512] \
        [--num-workers 4] \
        [--max-batches 200] \
        [--action-horizon 32]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast

from praxis.codecs.factory import make_codec
from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer
from praxis.core.types import (
    CANONICAL_ACTION_KEY,
    FeatureSpec,
    FeatureType,
    NormMode,
    PolicyIO,
)
from praxis.data.lerobot_adapter import LeRobotActionOnlyDatasetAdapter
from praxis.data.utils import resolve_feature_norm_mode
from praxis.training.core.episode_splits import (
    split_validation_episodes as _random_split_validation_episodes,
)
from praxis.training.core.trainer import resolve_sampler_frame_drops

_NON_ACTION_FEATURE_KEYS = {
    "index",
    "frame_index",
    "episode_index",
    "timestamp",
    "task",
    "task_index",
    "reward",
    "done",
    "next.reward",
    "next.done",
}


def _feature_shape(feature_info: object) -> tuple[int, ...] | None:
    raw_shape = (
        feature_info.get("shape")
        if isinstance(feature_info, dict)
        else getattr(feature_info, "shape", None)
    )
    if raw_shape is None:
        return None
    return tuple(cast(Any, raw_shape))


def _infer_action_key(meta_features: Mapping[str, object]) -> str:
    """Infer the action feature key from LeRobot metadata."""
    candidates = [
        key
        for key, info in meta_features.items()
        if not key.startswith(("observation.", "next."))
        and not key.endswith("_is_pad")
        and key not in _NON_ACTION_FEATURE_KEYS
        and (shape := _feature_shape(info)) is not None
        and len(shape) == 1
    ]

    if CANONICAL_ACTION_KEY in candidates:
        return CANONICAL_ACTION_KEY
    if len(candidates) == 1:
        return candidates[0]

    action_named = [key for key in candidates if "action" in key.lower()]
    if len(action_named) == 1:
        return action_named[0]

    raise ValueError(f"cannot infer action key. Candidates={sorted(candidates)}")


def _reconstruction_error_sums(
    *,
    recons: torch.Tensor,
    actions: torch.Tensor,
    action_is_pad: torch.Tensor | None,
) -> tuple[float, float, float]:
    sqerr = (recons - actions).pow(2)
    abserr = (recons - actions).abs()

    if action_is_pad is None:
        return (
            float(sqerr.sum().item()),
            float(abserr.sum().item()),
            float(sqerr.numel()),
        )

    valid = ~action_is_pad.bool()
    while valid.ndim < sqerr.ndim:
        valid = valid.unsqueeze(-1)
    valid_f = valid.to(device=sqerr.device, dtype=sqerr.dtype)
    valid_elements = valid_f.expand_as(sqerr)
    return (
        float((sqerr * valid_f).sum().item()),
        float((abserr * valid_f).sum().item()),
        float(valid_elements.sum().item()),
    )


def _load_train_split_config(
    *, checkpoint_dir: Path
) -> tuple[list[int] | None, float, int | None, bool, int, int]:
    train_cfg_path = checkpoint_dir / "train_config.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(
            "Expected train_config.yaml inside tokenizer checkpoint directory for "
            f"validation split reproduction, but missing: {train_cfg_path}"
        )
    cfg = OmegaConf.to_container(OmegaConf.load(train_cfg_path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"train_config must be a mapping, got {type(cfg)!r}")
    dataset_cfg = cfg.get("dataset", {})
    trainer_cfg = cfg.get("trainer", {})
    if not isinstance(dataset_cfg, dict) or not isinstance(trainer_cfg, dict):
        raise TypeError("train_config dataset/trainer sections must be mappings.")

    raw_episodes = dataset_cfg.get("episodes")
    dataset_episodes = (
        [int(ep) for ep in raw_episodes]
        if isinstance(raw_episodes, list) and len(raw_episodes) > 0
        else None
    )
    val_ratio = float(trainer_cfg.get("val_ratio", 0.05))
    raw_split_seed = trainer_cfg.get("split_seed")
    split_seed = int(raw_split_seed) if raw_split_seed is not None else None
    use_episode_aware_sampler = bool(trainer_cfg.get("use_episode_aware_sampler", True))
    drop_n_first_frames = int(trainer_cfg.get("drop_n_first_frames", 0))
    drop_n_last_frames = int(trainer_cfg.get("drop_n_last_frames", 0))
    return (
        dataset_episodes,
        val_ratio,
        split_seed,
        use_episode_aware_sampler,
        drop_n_first_frames,
        drop_n_last_frames,
    )


def _resolve_validation_episodes(
    *,
    all_episodes: list[int],
    val_ratio: float,
    split_seed: int | None,
) -> list[int]:
    if split_seed is None:
        if val_ratio > 0 and len(all_episodes) > 1:
            n_val = max(1, round(len(all_episodes) * val_ratio))
            return all_episodes[-n_val:]
        return all_episodes
    return _random_split_validation_episodes(
        all_episodes,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FAST token-length and reconstruction statistics on a dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained FAST tokenizer pretrained_model directory.",
    )
    parser.add_argument(
        "--bpe-tokenizer-path",
        type=str,
        default=None,
        help=(
            "Optionally replace the checkpoint's fitted BPE with another FAST "
            "BPE (for example physical-intelligence/fast), while retaining the "
            "checkpoint's dataset normalization statistics and action geometry."
        ),
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repo_id (e.g. lerobot/libero).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Local dataset root path. None = default HF cache.",
    )
    parser.add_argument(
        "--dataset-episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episode subset. None = all episodes.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "val"),
        default="all",
        help=(
            "Dataset split to evaluate. 'all' preserves the historical behavior. "
            "'val' reproduces the tokenizer validation split from "
            "<checkpoint>/train_config.yaml."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap number of batches. None = full dataset.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Override action horizon. None = use tokenizer's configured value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load tokenizer ---
    print(f"Loading FAST tokenizer checkpoint from {args.checkpoint} ...")
    tokenizer = make_codec(
        {"kind": "tokens", "name": "fast", "pretrained_path": args.checkpoint}
    )
    if not isinstance(tokenizer, FASTTokenizer):
        print(
            f"Error: loaded tokenizer is {type(tokenizer).__name__}, expected FASTTokenizer.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.bpe_tokenizer_path is not None:
        print(f"  replacing BPE with {args.bpe_tokenizer_path} ...")
        tokenizer._set_bpe(
            PreTrainedTokenizerFast.from_pretrained(args.bpe_tokenizer_path)
        )
        tokenizer.config.bpe_tokenizer_path = args.bpe_tokenizer_path
        tokenizer._maybe_load_processor_metadata(args.bpe_tokenizer_path)

    action_horizon = int(args.action_horizon or tokenizer.action_horizon)
    # Keep tokenizer geometry aligned with the sampled chunk horizon.
    tokenizer.action_horizon = action_horizon
    if hasattr(tokenizer, "config"):
        tokenizer.config.action_horizon = action_horizon
    action_dim = tokenizer.action_dim
    print(f"  action_dim={action_dim}, action_horizon={action_horizon}")
    print(f"  current max_token_length={tokenizer.max_token_length}")
    print(f"  codebook_size={tokenizer.codebook_size}")
    print(f"  bpe_vocab_size={getattr(tokenizer, '_bpe_vocab_size', 'n/a')}")
    print(f"  pad_token_id={getattr(tokenizer, '_pad_token_id', 'n/a')}")
    print(f"  min_token={tokenizer.min_token}")
    print(f"  bpe_tokenizer_path={tokenizer.config.bpe_tokenizer_path}")

    split_base_episodes = args.dataset_episodes
    use_episode_aware_sampler = False
    cfg_drop_n_first_frames = 0
    cfg_drop_n_last_frames = 0
    if args.split == "val":
        (
            train_dataset_episodes,
            val_ratio,
            split_seed,
            use_episode_aware_sampler,
            cfg_drop_n_first_frames,
            cfg_drop_n_last_frames,
        ) = _load_train_split_config(checkpoint_dir=Path(args.checkpoint))
        if args.dataset_episodes is None:
            split_base_episodes = train_dataset_episodes
        else:
            print(
                "  WARNING: --dataset-episodes was provided with --split val; "
                "using that episode list as the base before applying val_ratio."
            )

    # --- Discover dataset action feature ---
    print(f"\nLoading dataset {args.dataset_repo_id} ...")
    temp_ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=split_base_episodes,
    )
    meta_features = temp_ds.meta.features
    try:
        action_key = _infer_action_key(meta_features)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    action_info = meta_features[action_key]
    action_shape = _feature_shape(action_info)
    if action_shape is None:
        print(
            f"Error: action feature {action_key!r} has no shape metadata.",
            file=sys.stderr,
        )
        sys.exit(1)
    dataset_action_dim = int(action_shape[0])

    if dataset_action_dim != action_dim:
        print(
            f"  WARNING: dataset action_dim={dataset_action_dim} != tokenizer action_dim={action_dim}"
        )
        print(
            "  Aligning tokenizer action_dim to dataset action_dim for stats/reconstruction."
        )
        tokenizer.action_dim = dataset_action_dim
        if hasattr(tokenizer, "config"):
            tokenizer.config.action_dim = dataset_action_dim
        action_dim = dataset_action_dim

    io = PolicyIO(
        obs_features=[],
        action_feature=FeatureSpec(
            key=action_key,
            type=FeatureType.ACTION,
            shape=action_shape,
            norm_mode=(
                resolve_feature_norm_mode(action_info, NormMode.MIN_MAX)
                if isinstance(action_info, dict)
                else NormMode.MIN_MAX
            ),
        ),
        n_obs_steps=0,
        n_action_steps=action_horizon,
        fps=temp_ds.fps,
    )
    tokenizer.set_io(io)

    if args.split == "val":
        all_episodes = (
            list(split_base_episodes)
            if split_base_episodes is not None
            else list(range(int(temp_ds.meta.total_episodes)))
        )
        dataset_episodes = _resolve_validation_episodes(
            all_episodes=all_episodes,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
        split_seed_label = "legacy_tail" if split_seed is None else str(split_seed)
        print(
            f"  validation split: {len(dataset_episodes)}/{len(all_episodes)} "
            f"episodes (val_ratio={val_ratio}, split_seed={split_seed_label})"
        )
    else:
        dataset_episodes = args.dataset_episodes

    drop_n_first_frames, drop_n_last_frames = resolve_sampler_frame_drops(
        io,
        configured_drop_n_first_frames=cfg_drop_n_first_frames,
        configured_drop_n_last_frames=cfg_drop_n_last_frames,
    )
    print(
        "  dataloader settings: "
        f"split={args.split}, "
        f"use_episode_aware_sampler={use_episode_aware_sampler}, "
        f"drop_n_first_frames={drop_n_first_frames}, "
        f"drop_n_last_frames={drop_n_last_frames}"
    )
    del temp_ds

    # --- Build dataset + dataloader ---
    dataset = LeRobotActionOnlyDatasetAdapter(
        repo_id=args.dataset_repo_id,
        io=io,
        root=args.dataset_root,
        episodes=dataset_episodes,
    )
    if tokenizer.normalize_action is None:
        print(
            "Error: checkpoint did not provide normalization stats.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("  normalization stats: checkpoint")

    dataloader = dataset.make_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_episode_aware_sampler=use_episode_aware_sampler,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
    )
    print(f"  {len(dataset)} samples, action_key={action_key!r}, shape={action_shape}")

    # --- Tokenize and collect lengths ---
    total_batches = len(dataloader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    print(f"\nTokenizing ({total_batches} batches) ...")

    all_lengths: list[torch.Tensor] = []
    n_processed = 0
    sqerr_sum = 0.0
    abserr_sum = 0.0
    n_valid = 0.0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            actions = batch[action_key]  # (B, T, D)
            token_list, lengths = tokenizer.tokenize(
                actions, pad=False, return_lengths=True
            )
            all_lengths.append(lengths)

            # Reconstruct from padded/truncated tokens under current max_token_length.
            max_tl = int(tokenizer.max_token_length)
            pad_id = int(
                getattr(tokenizer, "_pad_token_id", tokenizer.codebook_size - 1)
            )
            tokens = torch.full(
                (actions.shape[0], max_tl),
                pad_id,
                dtype=torch.long,
                device=actions.device,
            )
            for i, seq in enumerate(token_list):
                keep = min(int(seq.numel()), max_tl)
                if keep > 0:
                    tokens[i, :keep] = seq[:keep]

            recons = tokenizer.detokenize(tokens)
            pad_key = f"{action_key}_is_pad"
            batch_sqerr, batch_abserr, batch_valid = _reconstruction_error_sums(
                recons=recons,
                actions=actions,
                action_is_pad=batch.get(pad_key),
            )
            sqerr_sum += batch_sqerr
            abserr_sum += batch_abserr
            n_valid += batch_valid

            n_processed += actions.shape[0]

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                print(
                    f"  batch {batch_idx + 1}/{total_batches} ({n_processed} samples)"
                )

    # --- Compute and report stats ---
    lengths = torch.cat(all_lengths)
    lengths_f = lengths.float()

    print("\n" + "=" * 50)
    print(f"  FAST Token-Length Statistics  (N = {len(lengths)})")
    print("=" * 50)
    print(f"  min:      {lengths.min().item()}")
    print(f"  max:      {lengths.max().item()}")
    print(f"  mean:     {lengths_f.mean().item():.1f}")
    print(f"  median:   {lengths_f.median().item():.1f}")
    print(f"  std:      {lengths_f.std().item():.1f}")
    print(f"  p5:       {lengths_f.quantile(0.05).item():.0f}")
    print(f"  p25:      {lengths_f.quantile(0.25).item():.0f}")
    print(f"  p75:      {lengths_f.quantile(0.75).item():.0f}")
    print(f"  p95:      {lengths_f.quantile(0.95).item():.0f}")
    print(f"  p99:      {lengths_f.quantile(0.99).item():.0f}")
    print()

    max_tl = tokenizer.max_token_length
    n_truncated = (lengths > max_tl).sum().item()
    pct_truncated = n_truncated / len(lengths) * 100
    print(f"  current max_token_length = {max_tl}")
    print(
        f"  samples > max_token_length: {n_truncated}/{len(lengths)} ({pct_truncated:.1f}%)"
    )

    # Suggest a safe max_token_length
    p100 = lengths.max().item()
    suggested = ((p100 + 15) // 16) * 16  # round up to next multiple of 16
    print(f"  suggested max_token_length: {suggested}  (max + headroom, rounded to 16)")

    if n_valid <= 0:
        print("  WARNING: no valid reconstruction elements to compute MSE/MAE.")
    else:
        reconst_mse = sqerr_sum / n_valid
        reconst_mae = abserr_sum / n_valid
        reconst_rmse = reconst_mse**0.5
        print()
        print("  Reconstruction (tokenize -> detokenize):")
        print(f"  mse:      {reconst_mse:.6e}")
        print(f"  rmse:     {reconst_rmse:.6e}")
        print(f"  mae:      {reconst_mae:.6e}")
    print()
    print("  FAST tokenizer metadata:")
    print(f"  codebook_size:      {tokenizer.codebook_size}")
    print(f"  bpe_vocab_size:     {getattr(tokenizer, '_bpe_vocab_size', 'n/a')}")
    print(f"  pad_token_id:       {getattr(tokenizer, '_pad_token_id', 'n/a')}")
    print(f"  max_token_length:   {tokenizer.max_token_length}")
    print(f"  min_token:          {tokenizer.min_token}")
    print(f"  bpe_tokenizer_path: {tokenizer.config.bpe_tokenizer_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
