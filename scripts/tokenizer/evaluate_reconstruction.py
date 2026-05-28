#!/usr/bin/env python3
"""Evaluate tokenizer reconstruction MSE on a LeRobot dataset split.

The headline metric intentionally matches ``CodecsTrainer.reconstruct``:
tokenize raw action chunks, detokenize them back to raw action space, then
compute MSE against the original raw actions.

Usage:
    uv run python scripts/tokenizer/evaluate_reconstruction.py \
        --checkpoint outputs/train_codecs/20260422/mshab_boat_codec/checkpoints/latest \
        --keep-ks 1 2 4 8 16 \
        [--dataset-repo-id chaoqi-liu/mshab_settable] \
        [--split val] \
        [--batch-size 512] \
        [--num-workers 4] \
        [--max-batches 100]
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf

from praxis.codecs.factory import make_codec
from praxis.codecs.tokens.core import BaseTokenizer
from praxis.data.lerobot_adapter import LeRobotActionOnlyDatasetAdapter
from praxis.training.core.episode_splits import (
    split_validation_episodes as _split_validation_episodes,
)
from praxis.training.core.trainer import resolve_sampler_frame_drops


@dataclass(frozen=True)
class CheckpointPaths:
    checkpoint: Path
    pretrained_model: Path
    train_config: Path


@dataclass(frozen=True)
class TrainSplitConfig:
    dataset_repo_id: str
    dataset_root: str | None
    dataset_episodes: list[int] | None
    val_ratio: float
    split_seed: int | None
    use_episode_aware_sampler: bool
    drop_n_first_frames: int
    drop_n_last_frames: int
    codec_name: str


@dataclass(frozen=True)
class EvalMetrics:
    mse: float
    rmse: float
    mae: float
    samples: int
    batches: int


@dataclass
class MetricAccumulator:
    sqerr_sum: float = 0.0
    abserr_sum: float = 0.0
    samples: int = 0
    batches: int = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        batch_size = int(gt.shape[0])
        self.sqerr_sum += float(F.mse_loss(pred, gt).item()) * batch_size
        self.abserr_sum += float(F.l1_loss(pred, gt).item()) * batch_size
        self.samples += batch_size
        self.batches += 1

    def finalize(self) -> EvalMetrics:
        if self.samples == 0:
            raise RuntimeError("No samples were evaluated.")
        mse = self.sqerr_sum / self.samples
        mae = self.abserr_sum / self.samples
        return EvalMetrics(
            mse=mse,
            rmse=mse**0.5,
            mae=mae,
            samples=self.samples,
            batches=self.batches,
        )


@dataclass(frozen=True)
class EvalResult:
    full: EvalMetrics
    keep_k: dict[int, EvalMetrics]


def resolve_checkpoint_paths(checkpoint: str | Path) -> CheckpointPaths:
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            "Generic tokenizer evaluation requires a local checkpoint directory "
            f"with train_config.yaml, but path does not exist: {path}"
        )

    if path.name == "pretrained_model":
        pretrained_model = path
        checkpoint_root = path.parent
    elif (path / "pretrained_model").is_dir():
        pretrained_model = path / "pretrained_model"
        checkpoint_root = path
    else:
        pretrained_model = path
        checkpoint_root = path.parent

    train_config = pretrained_model / "train_config.yaml"
    if not train_config.exists():
        raise FileNotFoundError(
            "Expected train_config.yaml in pretrained_model directory for exact "
            f"split reproduction, but missing: {train_config}"
        )
    return CheckpointPaths(
        checkpoint=checkpoint_root,
        pretrained_model=pretrained_model,
        train_config=train_config,
    )


def _as_int_list(value: object) -> list[int] | None:
    if isinstance(value, list) and len(value) > 0:
        return [int(ep) for ep in value]
    return None


def load_train_split_config(train_config_path: Path) -> TrainSplitConfig:
    cfg = OmegaConf.to_container(OmegaConf.load(train_config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"train_config must be a mapping, got {type(cfg)!r}")
    dataset_cfg = cfg.get("dataset", {})
    trainer_cfg = cfg.get("trainer", {})
    codec_cfg = cfg.get("codec", {})
    if (
        not isinstance(dataset_cfg, dict)
        or not isinstance(trainer_cfg, dict)
        or not isinstance(codec_cfg, dict)
    ):
        raise TypeError("train_config dataset/trainer/codec sections must be mappings.")

    repo_id = dataset_cfg.get("repo_id")
    codec_name = codec_cfg.get("name")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("train_config dataset.repo_id must be a non-empty string.")
    if not isinstance(codec_name, str) or not codec_name:
        raise ValueError("train_config codec.name must be a non-empty string.")

    root = dataset_cfg.get("root")
    if root is not None and not isinstance(root, str):
        raise TypeError(
            f"train_config dataset.root must be a string or null, got {type(root)!r}."
        )

    raw_split_seed = trainer_cfg.get("split_seed")
    return TrainSplitConfig(
        dataset_repo_id=repo_id,
        dataset_root=root,
        dataset_episodes=_as_int_list(dataset_cfg.get("episodes")),
        val_ratio=float(trainer_cfg.get("val_ratio", 0.05)),
        split_seed=int(raw_split_seed) if raw_split_seed is not None else None,
        use_episode_aware_sampler=bool(
            trainer_cfg.get("use_episode_aware_sampler", True)
        ),
        drop_n_first_frames=int(trainer_cfg.get("drop_n_first_frames", 0)),
        drop_n_last_frames=int(trainer_cfg.get("drop_n_last_frames", 0)),
        codec_name=codec_name,
    )


def split_validation_episodes(
    *, all_episodes: list[int], val_ratio: float, split_seed: int | None = 0
) -> list[int]:
    if split_seed is None:
        if val_ratio > 0 and len(all_episodes) > 1:
            n_val = max(1, round(len(all_episodes) * val_ratio))
            return all_episodes[-n_val:]
        return all_episodes
    return _split_validation_episodes(
        all_episodes,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )


def resolve_eval_episodes(
    *,
    split: str,
    all_episodes: list[int],
    val_ratio: float,
    split_seed: int | None = 0,
) -> list[int]:
    if split == "all":
        return all_episodes
    if split == "val":
        return split_validation_episodes(
            all_episodes=all_episodes,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
    raise ValueError(f"Unsupported split {split!r}. Expected 'all' or 'val'.")


def _load_tokenizer(
    *, tokenizer_name: str, pretrained_model: Path, device: str
) -> BaseTokenizer:
    tokenizer = make_codec(
        {
            "kind": "tokens",
            "name": tokenizer_name,
            "pretrained_path": str(pretrained_model),
        }
    )
    if not isinstance(tokenizer, BaseTokenizer):
        raise TypeError(
            f"Checkpoint resolved to {type(tokenizer).__name__}, expected a token codec."
        )
    tokenizer.to(device)
    tokenizer.eval()
    if tokenizer.io is None:
        raise RuntimeError("Tokenizer checkpoint has no PolicyIO sidecar.")
    if tokenizer.normalize_action is None:
        raise RuntimeError("Tokenizer checkpoint has no action normalization stats.")
    return tokenizer


def supported_keep_ks(
    tokenizer: BaseTokenizer, *, tokenizer_name: str
) -> list[int] | None:
    if not bool(getattr(tokenizer, "supports_prefix_decode", False)):
        return None

    name = tokenizer_name.strip().lower()
    if name == "boat":
        schedule = tokenizer.block_schedule
        if schedule is None:
            return None
        return [int(k) for k in itertools.accumulate(schedule)]
    if name == "oat":
        latent_horizon = getattr(tokenizer, "latent_horizon", None)
        if latent_horizon is None:
            return None
        return list(range(1, int(latent_horizon) + 1))
    return None


def validate_keep_ks(
    requested_keep_ks: list[int] | None,
    *,
    tokenizer: BaseTokenizer,
    tokenizer_name: str,
) -> list[int]:
    if not requested_keep_ks:
        return []

    valid_keep_ks = supported_keep_ks(tokenizer, tokenizer_name=tokenizer_name)
    if valid_keep_ks is None:
        raise ValueError(
            f"Tokenizer {tokenizer_name!r} does not expose supported prefix decode budgets."
        )

    valid_set = set(valid_keep_ks)
    invalid = [k for k in requested_keep_ks if k not in valid_set]
    if invalid:
        raise ValueError(
            f"Invalid keep_k values for tokenizer {tokenizer_name!r}: {invalid}. "
            f"Supported values: {valid_keep_ks}."
        )
    return sorted({int(k) for k in requested_keep_ks})


def evaluate_reconstruction(
    *,
    tokenizer: BaseTokenizer,
    dataloader: Any,
    action_key: str,
    device: str,
    max_batches: int | None,
    keep_ks: list[int],
) -> EvalResult:
    full_accum = MetricAccumulator()
    keep_accum = {k: MetricAccumulator() for k in keep_ks}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            actions = batch[action_key].to(device)
            tokens = tokenizer.tokenize(actions)
            recon_actions = tokenizer.detokenize(tokens)
            full_accum.update(recon_actions, actions)

            if keep_ks:
                encoded = cast(Any, tokenizer).tokenize_with_latents(actions)
                latents = encoded.latents
                batch_size = int(actions.shape[0])
                for k in keep_ks:
                    keep_recon = cast(Any, tokenizer).decode(
                        latents, decode_keep_k=[k] * batch_size
                    )
                    keep_accum[k].update(keep_recon, actions)

            if full_accum.batches % 50 == 0:
                print(
                    f"  processed {full_accum.samples} samples "
                    f"({full_accum.batches} batches)"
                )

    return EvalResult(
        full=full_accum.finalize(),
        keep_k={k: accum.finalize() for k, accum in keep_accum.items()},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tokenizer raw action-space reconstruction MSE."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local tokenizer checkpoint dir, either checkpoints/latest or pretrained_model.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Override dataset repo id. Defaults to checkpoint train_config.yaml.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Override local dataset root. Defaults to checkpoint train_config.yaml.",
    )
    parser.add_argument(
        "--dataset-episodes",
        type=int,
        nargs="+",
        default=None,
        help="Override base episode subset before split selection.",
    )
    parser.add_argument(
        "--codec",
        default=None,
        help="Override action-token codec. Defaults to checkpoint train_config.yaml.",
    )
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap number of batches. None = full selected split.",
    )
    parser.add_argument(
        "--keep-ks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional prefix decode budgets. Supported for OAT (1..latent_horizon) "
            "and BOAT (cumulative block_schedule values)."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a compact JSON result after the human-readable summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_paths = resolve_checkpoint_paths(args.checkpoint)
    split_cfg = load_train_split_config(checkpoint_paths.train_config)
    tokenizer_name = str(args.codec or split_cfg.codec_name)
    dataset_repo_id = str(args.dataset_repo_id or split_cfg.dataset_repo_id)
    dataset_root = (
        args.dataset_root if args.dataset_root is not None else split_cfg.dataset_root
    )
    base_episodes = (
        [int(ep) for ep in args.dataset_episodes]
        if args.dataset_episodes is not None
        else split_cfg.dataset_episodes
    )

    print(
        f"Loading tokenizer {tokenizer_name!r} from {checkpoint_paths.pretrained_model} ..."
    )
    tokenizer = _load_tokenizer(
        tokenizer_name=tokenizer_name,
        pretrained_model=checkpoint_paths.pretrained_model,
        device=args.device,
    )
    assert tokenizer.io is not None
    if tokenizer.io.action_feature is None:
        raise RuntimeError("Tokenizer PolicyIO has no action feature.")
    action_key = tokenizer.action_key
    keep_ks = validate_keep_ks(
        args.keep_ks,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
    )
    if keep_ks:
        print(f"  keep_ks={keep_ks}")

    print(f"Loading dataset {dataset_repo_id} ...")
    temp_ds = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
        episodes=base_episodes,
    )
    try:
        all_episodes = (
            list(base_episodes)
            if base_episodes is not None
            else list(range(int(temp_ds.meta.total_episodes)))
        )
        eval_episodes = resolve_eval_episodes(
            split=args.split,
            all_episodes=all_episodes,
            val_ratio=split_cfg.val_ratio,
            split_seed=split_cfg.split_seed,
        )
        fps = temp_ds.fps
    finally:
        del temp_ds

    drop_n_first_frames, drop_n_last_frames = resolve_sampler_frame_drops(
        tokenizer.io,
        configured_drop_n_first_frames=split_cfg.drop_n_first_frames,
        configured_drop_n_last_frames=split_cfg.drop_n_last_frames,
    )
    split_seed_label = (
        "legacy_tail" if split_cfg.split_seed is None else str(split_cfg.split_seed)
    )
    print(
        f"  split={args.split}: {len(eval_episodes)}/{len(all_episodes)} episodes "
        f"(val_ratio={split_cfg.val_ratio}, split_seed={split_seed_label})"
    )
    print(
        "  dataloader: "
        f"use_episode_aware_sampler={split_cfg.use_episode_aware_sampler}, "
        f"drop_n_first_frames={drop_n_first_frames}, "
        f"drop_n_last_frames={drop_n_last_frames}, "
        f"fps={fps}"
    )
    print("  normalization stats: checkpoint")

    dataset = LeRobotActionOnlyDatasetAdapter(
        repo_id=dataset_repo_id,
        io=tokenizer.io,
        root=dataset_root,
        episodes=cast(list[int] | None, eval_episodes),
    )
    dataloader = dataset.make_dataloader(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        use_episode_aware_sampler=split_cfg.use_episode_aware_sampler,
        drop_n_first_frames=drop_n_first_frames,
        drop_n_last_frames=drop_n_last_frames,
        drop_last=False,
    )
    print(f"  samples={len(dataset)}, action_key={action_key!r}")

    total_batches = len(dataloader)
    if args.max_batches is not None:
        total_batches = min(total_batches, int(args.max_batches))
    print(f"\nEvaluating {total_batches} batches ...")
    result = evaluate_reconstruction(
        tokenizer=tokenizer,
        dataloader=dataloader,
        action_key=action_key,
        device=args.device,
        max_batches=args.max_batches,
        keep_ks=keep_ks,
    )

    metrics = result.full
    print("\n" + "=" * 52)
    print("Tokenizer Reconstruction")
    print("=" * 52)
    print(f"checkpoint: {checkpoint_paths.pretrained_model}")
    print(f"codec:      {tokenizer_name}")
    print(f"dataset:    {dataset_repo_id}")
    print(f"split:      {args.split}")
    print(f"samples:    {metrics.samples}")
    print(f"batches:    {metrics.batches}")
    print()
    print("Raw action-space reconstruction:")
    print(f"mse:        {metrics.mse:.9e}")
    print(f"rmse:       {metrics.rmse:.9e}")
    print(f"mae:        {metrics.mae:.9e}")
    if result.keep_k:
        print()
        print("Prefix decode reconstruction:")
        print(f"{'keep_k':>8} | {'mse':>14} | {'rmse':>14} | {'mae':>14}")
        print("-" * 62)
        for k, keep_metrics in result.keep_k.items():
            print(
                f"{k:>8} | {keep_metrics.mse:>14.9e} | "
                f"{keep_metrics.rmse:>14.9e} | {keep_metrics.mae:>14.9e}"
            )
    print("=" * 52)

    if args.json:
        payload: dict[str, Any] = {
            "checkpoint": str(checkpoint_paths.pretrained_model),
            "codec": tokenizer_name,
            "dataset_repo_id": dataset_repo_id,
            "split": args.split,
            "samples": metrics.samples,
            "batches": metrics.batches,
            "mse": metrics.mse,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "keep_k": {
                str(k): {
                    "mse": keep_metrics.mse,
                    "rmse": keep_metrics.rmse,
                    "mae": keep_metrics.mae,
                    "samples": keep_metrics.samples,
                    "batches": keep_metrics.batches,
                }
                for k, keep_metrics in result.keep_k.items()
            },
        }
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
