#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Training-free stage-1 diagnostics for action tokenizers.

The script evaluates every tokenizer on exactly the same raw action chunks.  It
reports reconstruction, code usage, and the causal effect of replacing one
token at a time with a token from another trajectory.  For BTT it additionally
checks the architectural left/right arm-swap equivariance.

Example:
    uv run python scripts/tokenizer/evaluate_stage1_semantics.py \
      --suite robotwin5 --device cuda \
      --output-dir outputs/tokenizer_stage1/robotwin5
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf

from praxis.codecs.factory import make_codec
from praxis.codecs.tokens.btt.tokenizer import BTTTokenizer
from praxis.codecs.tokens.core import BaseTokenizer
from praxis.core.types import PolicyIO
from praxis.data.lerobot_adapter import LeRobotActionOnlyDatasetAdapter
from praxis.training.core.episode_splits import split_validation_episodes

DEFAULT_DATASET_REPO_ID = "local/robotwin_new_demo_clean"
DEFAULT_DATASET_ROOT = (
    "/scratch/xuanweiliu/bimanual_tokenizer/data/lerobot/local/robotwin_new_demo_clean"
)

ROBOTWIN5: tuple[tuple[str, str, str], ...] = (
    (
        "OAT",
        "oat",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_oat_robotwin_new_2500/600693.hopper-m-02/train/checkpoints/"
        "best/step_296000/pretrained_model",
    ),
    (
        "FAST",
        "fast",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_fast_q01_q99_noclip_v1024_filtered/598729.hopper-m-02/"
        "train/checkpoints/latest/pretrained_model",
    ),
    (
        "BOAT",
        "boat",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_boat_robotwin_new/616167.hopper-m-02/train/checkpoints/"
        "best/step_300000/pretrained_model",
    ),
    (
        "BTT",
        "btt",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_btt_robotwin_new/621340.hopper-m-02/train/checkpoints/"
        "best/step_297000/pretrained_model",
    ),
    (
        "QueST",
        "quest",
        "/scratch/xuanweiliu/bimanual_tokenizer/outputs/"
        "hopper_quest_robotwin_new/613294.hopper-m-02/train/checkpoints/"
        "best/step_299000/pretrained_model",
    ),
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    codec: str
    checkpoint: Path


@dataclass(frozen=True)
class CheckpointInfo:
    pretrained_model: Path
    train_config: Path


def _resolve_checkpoint(checkpoint: Path) -> CheckpointInfo:
    path = checkpoint.expanduser()
    if path.name == "pretrained_model":
        pretrained_model = path
    elif (path / "pretrained_model").is_dir():
        pretrained_model = path / "pretrained_model"
    else:
        pretrained_model = path
    train_config = pretrained_model / "train_config.yaml"
    if not train_config.is_file():
        raise FileNotFoundError(f"Missing tokenizer train config: {train_config}")
    return CheckpointInfo(pretrained_model, train_config)


def _load_io(pretrained_model: Path) -> PolicyIO:
    payload = json.loads(
        (pretrained_model / "policy_io.json").read_text(encoding="utf-8")
    )
    return PolicyIO.from_dict(payload)


def _load_codec(spec: ModelSpec, device: str) -> BaseTokenizer:
    checkpoint = _resolve_checkpoint(spec.checkpoint)
    tokenizer = make_codec(
        {
            "kind": "tokens",
            "name": spec.codec,
            "pretrained_path": str(checkpoint.pretrained_model),
        }
    )
    if not isinstance(tokenizer, BaseTokenizer):
        raise TypeError(f"{spec.label} did not resolve to a token codec.")
    tokenizer.to(device)
    tokenizer.eval()
    return tokenizer


def _load_sampler_config(train_config: Path) -> tuple[bool, int, int]:
    payload = OmegaConf.to_container(OmegaConf.load(train_config), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid train config: {train_config}")
    trainer = payload.get("trainer", {})
    if not isinstance(trainer, dict):
        trainer = {}
    return (
        bool(trainer.get("use_episode_aware_sampler", True)),
        int(trainer.get("drop_n_first_frames", 0)),
        int(trainer.get("drop_n_last_frames", 0)),
    )


def _parse_model_spec(raw: str) -> ModelSpec:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --model {raw!r}; expected LABEL:CODEC:CHECKPOINT.")
    return ModelSpec(parts[0], parts[1], Path(parts[2]).expanduser())


def _resolve_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if args.suite == "robotwin5":
        specs.extend(ModelSpec(a, b, Path(c)) for a, b, c in ROBOTWIN5)
    specs.extend(_parse_model_spec(raw) for raw in args.model)
    if not specs:
        raise ValueError("Pass --suite robotwin5 or at least one --model.")
    return specs


def _collect_actions(
    *,
    reference: ModelSpec,
    dataset_repo_id: str,
    dataset_root: str,
    split: str,
    val_ratio: float,
    split_seed: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    paths = _resolve_checkpoint(reference.checkpoint)
    use_episode_sampler, configured_drop_first, configured_drop_last = (
        _load_sampler_config(paths.train_config)
    )
    io = _load_io(paths.pretrained_model)
    if io.action_feature is None:
        raise RuntimeError("Reference tokenizer has no action feature in PolicyIO.")

    temp_ds = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root)
    try:
        all_episodes = list(range(int(temp_ds.meta.total_episodes)))
        episodes = (
            all_episodes
            if split == "all"
            else split_validation_episodes(
                all_episodes, val_ratio=val_ratio, split_seed=split_seed
            )
        )
    finally:
        del temp_ds

    drop_first = max(int(io.n_obs_steps) - 1, configured_drop_first)
    drop_last = max(int(io.n_action_steps) - 1, configured_drop_last)
    dataset = LeRobotActionOnlyDatasetAdapter(
        repo_id=dataset_repo_id,
        io=io,
        root=dataset_root,
        episodes=cast(list[int] | None, episodes),
    )
    loader = dataset.make_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        use_episode_aware_sampler=use_episode_sampler,
        drop_n_first_frames=drop_first,
        drop_n_last_frames=drop_last,
        drop_last=False,
    )
    chunks: list[torch.Tensor] = []
    n = 0
    for batch in loader:
        chunk = batch[io.action_feature.key].cpu()
        take = min(int(chunk.shape[0]), max_samples - n)
        chunks.append(chunk[:take])
        n += take
        if n >= max_samples:
            break
    if not chunks:
        raise RuntimeError("No action chunks were loaded.")
    actions = torch.cat(chunks, dim=0)
    metadata = {
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": dataset_root,
        "split": split,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "episodes": episodes,
        "available_samples": len(dataset),
        "evaluated_samples": int(actions.shape[0]),
        "action_shape": list(actions.shape[1:]),
    }
    return actions, metadata


def _tokenize_dense(
    tokenizer: BaseTokenizer, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(getattr(tokenizer, "variable_length_codes", False)):
        result = cast(Any, tokenizer).tokenize(actions, return_lengths=True)
        tokens, lengths = result
        if not isinstance(tokens, torch.Tensor):
            pad = int(getattr(tokenizer, "_pad_token_id", 0))
            width = int(tokenizer.max_code_seq_len)
            dense = torch.full(
                (len(tokens), width), pad, dtype=torch.long, device=actions.device
            )
            for row, values in enumerate(tokens):
                dense[row, : len(values)] = values
            tokens = dense
        return tokens.long(), lengths.long()
    tokens = tokenizer.tokenize(actions)
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Expected dense fixed tokens, got {type(tokens)!r}.")
    lengths = torch.full(
        (tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=tokens.device
    )
    return tokens.long(), lengths


def _detokenize(tokenizer: BaseTokenizer, tokens: torch.Tensor) -> torch.Tensor:
    if bool(getattr(tokenizer, "variable_length_codes", False)):
        return cast(Any, tokenizer).detokenize(tokens, strict_decode=False)
    return tokenizer.detokenize(tokens)


def _entropy_from_counts(counts: torch.Tensor) -> tuple[float, float]:
    total = float(counts.sum().item())
    if total <= 0:
        return float("nan"), float("nan")
    probs = counts[counts > 0].double() / total
    entropy_nats = float(-(probs * probs.log()).sum().item())
    return entropy_nats / math.log(2.0), math.exp(entropy_nats)


def _code_metrics(
    tokens: torch.Tensor, lengths: torch.Tensor, codebook_size: int
) -> dict[str, Any]:
    width = int(tokens.shape[1])
    valid = torch.arange(width, device=tokens.device)[None, :] < lengths[:, None]
    values = tokens[valid].detach().cpu()
    counts = torch.bincount(values, minlength=codebook_size)
    entropy_bits, perplexity = _entropy_from_counts(counts)
    positional_entropy: list[float | None] = []
    positional_utilization: list[float] = []
    for pos in range(width):
        pos_values = tokens[:, pos][valid[:, pos]].detach().cpu()
        pos_counts = torch.bincount(pos_values, minlength=codebook_size)
        pos_entropy, _ = _entropy_from_counts(pos_counts)
        positional_entropy.append(None if math.isnan(pos_entropy) else pos_entropy)
        positional_utilization.append(float((pos_counts > 0).sum().item()))
    used = int((counts > 0).sum().item())
    return {
        "codebook_size": codebook_size,
        "used_codes": used,
        "utilization": used / codebook_size,
        "entropy_bits": entropy_bits,
        "effective_codebook_perplexity": perplexity,
        "mean_token_length": float(lengths.float().mean().item()),
        "min_token_length": int(lengths.min().item()),
        "max_token_length": int(lengths.max().item()),
        "position_entropy_bits": positional_entropy,
        "position_used_codes": positional_utilization,
    }


def _intervention_metrics(
    *,
    tokenizer: BaseTokenizer,
    actions: torch.Tensor,
    max_samples: int,
) -> tuple[dict[str, Any], np.ndarray]:
    actions = actions[:max_samples]
    tokens, lengths = _tokenize_dense(tokenizer, actions)
    baseline = _detokenize(tokenizer, tokens)
    batch, width = tokens.shape
    horizon = int(baseline.shape[1])
    action_dim = int(baseline.shape[2])
    arm_dim = action_dim // 2
    influence = torch.full((width, horizon, 2), float("nan"), dtype=torch.float64)
    valid_pairs: list[int] = []
    changed_pairs: list[int] = []
    donor = torch.roll(torch.arange(batch, device=tokens.device), shifts=1)
    for pos in range(width):
        keep = (lengths > pos) & (lengths[donor] > pos)
        indices = keep.nonzero(as_tuple=False).flatten()
        valid_pairs.append(int(indices.numel()))
        if int(indices.numel()) == 0:
            changed_pairs.append(0)
            continue
        edited = tokens[indices].clone()
        replacements = tokens[donor[indices], pos]
        actually_changed = edited[:, pos] != replacements
        changed_pairs.append(int(actually_changed.sum().item()))
        edited[:, pos] = replacements
        try:
            decoded = _detokenize(tokenizer, edited)
        except Exception:
            continue
        delta = (decoded - baseline[indices]).square().double()
        influence[pos, :, 0] = delta[:, :, :arm_dim].mean(dim=(0, 2)).cpu()
        influence[pos, :, 1] = delta[:, :, arm_dim:].mean(dim=(0, 2)).cpu()

    finite = torch.nan_to_num(influence, nan=0.0)
    flat = finite.flatten(1)
    row_energy = flat.sum(dim=1)
    usable = row_energy > 0
    locality: list[float] = []
    arm_selectivity: list[float] = []
    peak_time: list[int | None] = []
    for pos in range(width):
        energy = row_energy[pos]
        if float(energy.item()) <= 0:
            peak_time.append(None)
            continue
        probs = flat[pos] / energy
        probs = probs[probs > 0]
        entropy = float(-(probs * probs.log()).sum().item())
        locality.append(1.0 - entropy / math.log(max(2, horizon * 2)))
        arm_energy = finite[pos].sum(dim=0)
        arm_selectivity.append(
            float(
                (arm_energy.max() - arm_energy.min()).item()
                / max(float(arm_energy.sum().item()), 1e-12)
            )
        )
        peak_time.append(int(finite[pos].sum(dim=1).argmax().item()))
    summary = {
        "samples": batch,
        "valid_replacement_pairs_per_position": valid_pairs,
        "changed_replacement_pairs_per_position": changed_pairs,
        "mean_locality": float(np.mean(locality)) if locality else None,
        "mean_arm_selectivity": (
            float(np.mean(arm_selectivity)) if arm_selectivity else None
        ),
        "peak_time_per_position": peak_time,
        "mean_effect_mse": (
            float(finite[usable].mean().item()) if bool(usable.any()) else None
        ),
    }
    return summary, influence.numpy()


def _swap_physical_arms(actions: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    arm_dim = int(actions.shape[-1] // 2)
    left, right = actions.split(arm_dim, dim=-1)
    sign = sign.to(device=actions.device, dtype=actions.dtype)
    return torch.cat((right * sign, left * sign), dim=-1)


def _btt_swap_metrics(tokenizer: BTTTokenizer, actions: torch.Tensor) -> dict[str, Any]:
    swapped_actions = _swap_physical_arms(actions, tokenizer.encoder.right_arm_sign)
    tokens = tokenizer.tokenize(actions)
    swapped_tokens = tokenizer.tokenize(swapped_actions)
    half = int(tokens.shape[1] // 2)
    expected_tokens = torch.cat((tokens[:, half:], tokens[:, :half]), dim=1)
    matches = swapped_tokens.eq(expected_tokens)
    reconstruction = tokenizer.detokenize(tokens)
    swapped_reconstruction = tokenizer.detokenize(swapped_tokens)
    expected_reconstruction = _swap_physical_arms(
        reconstruction, tokenizer.encoder.right_arm_sign
    )
    return {
        "samples": int(actions.shape[0]),
        "token_accuracy": float(matches.float().mean().item()),
        "exact_sequence_rate": float(matches.all(dim=1).float().mean().item()),
        "decoder_equivariance_mse": float(
            F.mse_loss(swapped_reconstruction, expected_reconstruction).item()
        ),
        "decoder_equivariance_max_abs": float(
            (swapped_reconstruction - expected_reconstruction).abs().max().item()
        ),
    }


def _plot_heatmap(path: Path, influence: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    data = np.concatenate((influence[:, :, 0], influence[:, :, 1]), axis=1)
    fig, ax = plt.subplots(figsize=(10, max(3.5, data.shape[0] * 0.18)))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="magma")
    ax.axvline(influence.shape[1] - 0.5, color="white", linewidth=1)
    ax.set_xlabel("output time (left arm | right arm)")
    ax.set_ylabel("intervened token position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean squared action change")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _evaluate_one(
    *,
    spec: ModelSpec,
    actions_cpu: torch.Tensor,
    device: str,
    batch_size: int,
    intervention_samples: int,
    output_dir: Path,
) -> dict[str, Any]:
    paths = _resolve_checkpoint(spec.checkpoint)
    tokenizer = _load_codec(spec, device)
    recon_sq = 0.0
    recon_abs = 0.0
    arm_sq = torch.zeros(2, dtype=torch.float64)
    element_count = 0
    arm_element_count = 0
    all_tokens: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(actions_cpu.shape[0]), batch_size):
            actions = actions_cpu[start : start + batch_size].to(device)
            tokens, lengths = _tokenize_dense(tokenizer, actions)
            reconstruction = _detokenize(tokenizer, tokens)
            error = reconstruction - actions
            recon_sq += float(error.square().sum().item())
            recon_abs += float(error.abs().sum().item())
            arm_dim = int(error.shape[-1] // 2)
            arm_sq[0] += error[:, :, :arm_dim].square().sum().double().cpu()
            arm_sq[1] += error[:, :, arm_dim:].square().sum().double().cpu()
            element_count += int(error.numel())
            arm_element_count += int(error[:, :, :arm_dim].numel())
            all_tokens.append(tokens.cpu())
            all_lengths.append(lengths.cpu())

        tokens_cpu = torch.cat(all_tokens, dim=0)
        lengths_cpu = torch.cat(all_lengths, dim=0)
        code = _code_metrics(tokens_cpu, lengths_cpu, int(tokenizer.codebook_size))
        intervention_actions = actions_cpu[:intervention_samples].to(device)
        intervention, influence = _intervention_metrics(
            tokenizer=tokenizer,
            actions=intervention_actions,
            max_samples=intervention_samples,
        )
        swap = None
        if isinstance(tokenizer, BTTTokenizer):
            swap = _btt_swap_metrics(tokenizer, intervention_actions)

    safe_name = spec.label.lower().replace(" ", "_")
    np.savez_compressed(
        output_dir / f"{safe_name}_intervention.npz", influence=influence
    )
    _plot_heatmap(
        output_dir / f"{safe_name}_intervention.png",
        influence,
        f"{spec.label}: single-token replacement influence",
    )
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mse = recon_sq / element_count
    result: dict[str, Any] = {
        "label": spec.label,
        "codec": spec.codec,
        "checkpoint": str(paths.pretrained_model),
        "reconstruction": {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": recon_abs / element_count,
            "left_arm_mse": float(arm_sq[0].item()) / arm_element_count,
            "right_arm_mse": float(arm_sq[1].item()) / arm_element_count,
        },
        "code_usage": code,
        "intervention": intervention,
    }
    if swap is not None:
        result["arm_swap_equivariance"] = swap
    return result


def _write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    columns = (
        "label,codec,mse,mae,used_codes,codebook_size,utilization,"
        "entropy_bits,effective_codebook_perplexity,mean_token_length,"
        "intervention_locality,arm_selectivity,swap_token_accuracy,"
        "swap_decoder_mse\n"
    )
    rows = [columns]
    for result in results:
        recon = result["reconstruction"]
        usage = result["code_usage"]
        intervention = result["intervention"]
        swap = result.get("arm_swap_equivariance", {})
        rows.append(
            ",".join(
                str(value)
                for value in (
                    result["label"],
                    result["codec"],
                    recon["mse"],
                    recon["mae"],
                    usage["used_codes"],
                    usage["codebook_size"],
                    usage["utilization"],
                    usage["entropy_bits"],
                    usage["effective_codebook_perplexity"],
                    usage["mean_token_length"],
                    intervention["mean_locality"],
                    intervention["mean_arm_selectivity"],
                    swap.get("token_accuracy", ""),
                    swap.get("decoder_equivariance_mse", ""),
                )
            )
            + "\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("robotwin5",), default=None)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL:CODEC:CHECKPOINT",
    )
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--intervention-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = _resolve_specs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading the shared held-out action set ...", flush=True)
    actions, dataset_metadata = _collect_actions(
        reference=specs[0],
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        split=args.split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"Loaded {actions.shape[0]} chunks shaped {tuple(actions.shape[1:])}.")
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] Evaluating {spec.label} ...", flush=True)
        result = _evaluate_one(
            spec=spec,
            actions_cpu=actions,
            device=args.device,
            batch_size=args.batch_size,
            intervention_samples=args.intervention_samples,
            output_dir=args.output_dir,
        )
        results.append(result)
        partial = {"dataset": dataset_metadata, "models": results}
        (args.output_dir / "results.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_summary_csv(results, args.output_dir / "summary.csv")
        print(
            f"  mse={result['reconstruction']['mse']:.6g}, "
            f"usage={result['code_usage']['utilization']:.2%}, "
            f"locality={result['intervention']['mean_locality']}",
            flush=True,
        )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
