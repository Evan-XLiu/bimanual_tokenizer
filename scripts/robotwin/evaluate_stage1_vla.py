#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Training-free VLA-side evaluation of action-token semantics.

For each already-trained policy this script measures:

* teacher-forced action-token NLL on matched observations;
* the NLL increase after independently shuffling visual, language, or all
  observation context while keeping the action-token target fixed;
* greedy generated-token accuracy/edit distance and decoded action MSE;
* oracle tokenizer reconstruction MSE on the exact same generated subset.

The matched NLL answers "how easily can the existing VLA predict these
tokens?".  The shuffle gaps test whether that predictability actually comes
from task context instead of only token priors.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from praxis.data.lerobot_adapter import LeRobotDatasetAdapter
from praxis.policies.factory import _load_pretrained_policy
from praxis.training.core.episode_splits import split_validation_episodes

DEFAULT_DATASET_REPO_ID = "local/robotwin_new_demo_clean"
DEFAULT_DATASET_ROOT = (
    "/scratch/xuanweiliu/bimanual_tokenizer/data/lerobot/local/robotwin_new_demo_clean"
)

ROBOTWIN5: tuple[tuple[str, str, str], ...] = (
    (
        "OAT",
        "bar_oat_pali",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/robotwin_eval/"
        "checkpoint_snapshots/bar_oat_pali_fft_step13400/pretrained_model",
    ),
    (
        "FAST",
        "bar_fast_pali",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_bar_fast_pali_fft/600669.hopper-m-02/train/checkpoints/"
        "best/step_5200/pretrained_model",
    ),
    (
        "BOAT",
        "bar_boat_pali",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_bar_boat_pali_robotwin_new_30k/616455.hopper-m-02/"
        "train/checkpoints/best/step_30000/pretrained_model",
    ),
    (
        "BTT",
        "bar_btt_pali",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_bar_btt_pali_robotwin_new_30k/622038.hopper-m-02/"
        "train/checkpoints/latest/pretrained_model",
    ),
    (
        "QueST",
        "bar_quest_pali",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_bar_quest_pali_robotwin_new_30k/613537.hopper-m-02/"
        "train/checkpoints/best/step_26000/pretrained_model",
    ),
)


@dataclass(frozen=True)
class PolicySpec:
    label: str
    policy_name: str
    checkpoint: Path


@dataclass
class ScalarMean:
    weighted_sum: float = 0.0
    weight: int = 0

    def add(self, value: float, weight: int) -> None:
        self.weighted_sum += float(value) * int(weight)
        self.weight += int(weight)

    def mean(self) -> float:
        return self.weighted_sum / self.weight if self.weight else float("nan")


def _parse_spec(raw: str) -> PolicySpec:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --model {raw!r}; expected LABEL:POLICY_NAME:CHECKPOINT."
        )
    return PolicySpec(parts[0], parts[1], Path(parts[2]).expanduser())


def _resolve_specs(args: argparse.Namespace) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    if args.suite == "robotwin5":
        specs.extend(PolicySpec(a, b, Path(c)) for a, b, c in ROBOTWIN5)
    specs.extend(_parse_spec(raw) for raw in args.model)
    if not specs:
        raise ValueError("Pass --suite robotwin5 or at least one --model.")
    return specs


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _slice_batch(batch: dict[str, Any], size: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, (torch.Tensor, list, tuple)):
            sliced[key] = value[:size]
        else:
            sliced[key] = value
    return sliced


def _permute_value(value: Any, permutation: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == permutation.numel():
            return value.index_select(0, permutation.to(value.device))
        return value
    order = permutation.cpu().tolist()
    if isinstance(value, list) and len(value) == len(order):
        return [value[index] for index in order]
    if isinstance(value, tuple) and len(value) == len(order):
        return tuple(value[index] for index in order)
    return value


def _derangement(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        return torch.arange(batch_size, device=device)
    shift = max(1, batch_size // 2)
    return torch.roll(torch.arange(batch_size, device=device), shifts=shift)


def _shuffle_context(
    policy: Any, batch: dict[str, Any], *, mode: str
) -> dict[str, Any]:
    action_key = policy.io.action_feature.key
    batch_size = int(batch[action_key].shape[0])
    permutation = _derangement(batch_size, batch[action_key].device)
    image_keys = {feature.key for feature in policy.io.image_features}
    obs_keys = set(policy.io.obs_keys)
    language_key = policy.io.language_key
    if mode == "vision":
        keys = image_keys
    elif mode == "language":
        keys = {language_key} if language_key is not None else set()
    elif mode == "all":
        keys = obs_keys | ({language_key} if language_key is not None else set())
    else:
        raise ValueError(f"Unsupported shuffle mode: {mode!r}")
    output = dict(batch)
    for key in keys:
        if key in output:
            output[key] = _permute_value(output[key], permutation)
    return output


def _levenshtein(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        current = [i]
        for j, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _token_agreement(
    predicted: list[list[int]], target: list[list[int]]
) -> dict[str, float]:
    exact = 0
    edit_sum = 0.0
    prefix_correct = 0
    prefix_total = 0
    length_abs_error = 0.0
    for pred, gold in zip(predicted, target, strict=True):
        exact += int(pred == gold)
        edit_sum += _levenshtein(pred, gold) / max(1, len(pred), len(gold))
        overlap = min(len(pred), len(gold))
        prefix_correct += sum(
            a == b for a, b in zip(pred[:overlap], gold[:overlap], strict=True)
        )
        prefix_total += overlap
        length_abs_error += abs(len(pred) - len(gold))
    count = len(target)
    return {
        "exact_sequence_rate": exact / count if count else float("nan"),
        "normalized_edit_distance": edit_sum / count if count else float("nan"),
        "aligned_prefix_token_accuracy": (
            prefix_correct / prefix_total if prefix_total else float("nan")
        ),
        "mean_absolute_length_error": (
            length_abs_error / count if count else float("nan")
        ),
    }


def _make_dataloader(
    *,
    policy: Any,
    dataset_repo_id: str,
    dataset_root: str,
    episodes: list[int],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[LeRobotDatasetAdapter, Any]:
    dataset = LeRobotDatasetAdapter(
        repo_id=dataset_repo_id,
        io=policy.io,
        root=dataset_root,
        episodes=episodes,
    )
    torch.manual_seed(seed)
    loader = dataset.make_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        use_episode_aware_sampler=False,
        drop_last=False,
    )
    return dataset, loader


def _evaluate_policy(
    *,
    spec: PolicySpec,
    dataset_repo_id: str,
    dataset_root: str,
    episodes: list[int],
    device: str,
    batch_size: int,
    generation_batch_size: int,
    num_workers: int,
    max_batches: int,
    max_generation_samples: int,
    seed: int,
) -> dict[str, Any]:
    print(f"Loading {spec.policy_name} from {spec.checkpoint} ...", flush=True)
    policy = _load_pretrained_policy(spec.policy_name, str(spec.checkpoint))
    policy.to(device)
    policy.eval()
    if policy.io is None or policy.io.action_feature is None:
        raise RuntimeError(f"{spec.label} checkpoint has no PolicyIO action feature.")
    dataset, loader = _make_dataloader(
        policy=policy,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        episodes=episodes,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    nll = ScalarMean()
    shuffled = {mode: ScalarMean() for mode in ("vision", "language", "all")}
    pred_sq_sum = 0.0
    oracle_sq_sum = 0.0
    action_elements = 0
    token_exact = 0.0
    token_edit = 0.0
    token_prefix_correct = 0.0
    token_length_error = 0.0
    generated_samples = 0
    evaluated_batches = 0
    evaluated_samples = 0

    # Action-token codecs may be stored in the policy's VLM dtype (bf16), while
    # LeRobot batches arrive as float32.  Use the same autocast context as
    # deployment/training so tokenization and forward passes have compatible
    # dtypes on CUDA.
    with torch.inference_mode(), policy._autocast_context():
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = _move_batch(raw_batch, device)
            action_key = policy.io.action_feature.key
            current_batch_size = int(batch[action_key].shape[0])
            matched_loss, _ = policy(batch)
            nll.add(float(matched_loss.item()), current_batch_size)
            for mode, accumulator in shuffled.items():
                shuffled_batch = _shuffle_context(policy, batch, mode=mode)
                shuffled_loss, _ = policy(shuffled_batch)
                accumulator.add(float(shuffled_loss.item()), current_batch_size)
            evaluated_batches += 1
            evaluated_samples += current_batch_size

            remaining = max_generation_samples - generated_samples
            if remaining > 0:
                take = min(current_batch_size, remaining, generation_batch_size)
                generation_batch = _slice_batch(batch, take)
                target_actions = generation_batch[action_key]
                predicted_actions, info = policy.predict_action_chunk(generation_batch)
                target_rows = policy.tokenize_actions_to_policy_tokens(target_actions)
                predicted_rows = [
                    [int(token) for token in row]
                    for row in info["pred_action_token_ids"]
                ]
                oracle_actions = policy._decode_generated_action_tokens(
                    target_rows, device=target_actions.device
                )
                pred_sq_sum += float(
                    (predicted_actions - target_actions).square().sum().item()
                )
                oracle_sq_sum += float(
                    (oracle_actions - target_actions).square().sum().item()
                )
                action_elements += int(target_actions.numel())
                agreement = _token_agreement(predicted_rows, target_rows)
                token_exact += agreement["exact_sequence_rate"] * take
                token_edit += agreement["normalized_edit_distance"] * take
                token_prefix_correct += (
                    agreement["aligned_prefix_token_accuracy"] * take
                )
                token_length_error += agreement["mean_absolute_length_error"] * take
                generated_samples += take

            print(
                f"  batch {evaluated_batches}/{max_batches}: "
                f"matched_nll={nll.mean():.5f}, generated={generated_samples}",
                flush=True,
            )

    matched = nll.mean()
    shuffle_results: dict[str, Any] = {}
    for mode, accumulator in shuffled.items():
        value = accumulator.mean()
        shuffle_results[mode] = {
            "nll": value,
            "gap": value - matched,
            "relative_gap": (value - matched) / matched if matched != 0 else None,
        }
    generation: dict[str, Any] | None = None
    if generated_samples > 0:
        pred_mse = pred_sq_sum / action_elements
        oracle_mse = oracle_sq_sum / action_elements
        generation = {
            "samples": generated_samples,
            "predicted_action_mse": pred_mse,
            "oracle_tokenizer_reconstruction_mse": oracle_mse,
            "prediction_excess_mse": pred_mse - oracle_mse,
            "exact_sequence_rate": token_exact / generated_samples,
            "normalized_edit_distance": token_edit / generated_samples,
            "aligned_prefix_token_accuracy": (token_prefix_correct / generated_samples),
            "mean_absolute_length_error": token_length_error / generated_samples,
        }
    result = {
        "label": spec.label,
        "policy_name": spec.policy_name,
        "checkpoint": str(spec.checkpoint),
        "evaluated_batches": evaluated_batches,
        "evaluated_samples": evaluated_samples,
        "teacher_forced": {
            "matched_nll": matched,
            "matched_perplexity": math.exp(min(matched, 80.0)),
            "shuffle": shuffle_results,
        },
        "generation": generation,
        "dataset_samples": len(dataset),
    }
    del loader, dataset, policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _write_summary(results: list[dict[str, Any]], path: Path) -> None:
    header = (
        "label,matched_nll,perplexity,vision_shuffle_gap,language_shuffle_gap,"
        "all_shuffle_gap,predicted_action_mse,oracle_reconstruction_mse,"
        "prediction_excess_mse,exact_sequence_rate,normalized_edit_distance,"
        "aligned_prefix_token_accuracy\n"
    )
    rows = [header]
    for result in results:
        teacher = result["teacher_forced"]
        generation = result.get("generation") or {}
        values = (
            result["label"],
            teacher["matched_nll"],
            teacher["matched_perplexity"],
            teacher["shuffle"]["vision"]["gap"],
            teacher["shuffle"]["language"]["gap"],
            teacher["shuffle"]["all"]["gap"],
            generation.get("predicted_action_mse", ""),
            generation.get("oracle_tokenizer_reconstruction_mse", ""),
            generation.get("prediction_excess_mse", ""),
            generation.get("exact_sequence_rate", ""),
            generation.get("normalized_edit_distance", ""),
            generation.get("aligned_prefix_token_accuracy", ""),
        )
        rows.append(",".join(str(value) for value in values) + "\n")
    path.write_text("".join(rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("robotwin5",), default=None)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL:POLICY_NAME:CHECKPOINT",
    )
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-generation-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    specs = _resolve_specs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        download_videos=False,
    )
    try:
        all_episodes = list(range(int(metadata.meta.total_episodes)))
    finally:
        del metadata
    episodes = split_validation_episodes(
        all_episodes,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    dataset_info = {
        "repo_id": args.dataset_repo_id,
        "root": args.dataset_root,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "episodes": episodes,
    }

    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.label}", flush=True)
        result = _evaluate_policy(
            spec=spec,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            episodes=episodes,
            device=args.device,
            batch_size=args.batch_size,
            generation_batch_size=args.generation_batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            max_generation_samples=args.max_generation_samples,
            seed=args.seed,
        )
        results.append(result)
        payload = {"dataset": dataset_info, "models": results}
        (args.output_dir / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_summary(results, args.output_dir / "summary.csv")
        teacher = result["teacher_forced"]
        print(
            f"  done: nll={teacher['matched_nll']:.5f}, "
            f"all_shuffle_gap={teacher['shuffle']['all']['gap']:.5f}",
            flush=True,
        )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
