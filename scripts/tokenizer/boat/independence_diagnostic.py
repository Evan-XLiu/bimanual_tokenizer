#!/usr/bin/env python3
"""Probe BOAT intra-block token dependence from encoded action tokens.

This script measures how much extra signal same-block context provides beyond
earlier blocks when predicting each BOAT token position.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class BlockSpec:
    """Describe one block in a block schedule."""

    block_index: int
    start: int
    size: int
    positions: tuple[int, ...]
    earlier_positions: tuple[int, ...]


@dataclass(frozen=True)
class TargetProbeSpec:
    """Describe one per-target independence probe."""

    block_index: int
    block_size: int
    target_position: int
    earlier_positions: tuple[int, ...]
    other_block_positions: tuple[int, ...]

    @property
    def baseline_positions(self) -> tuple[int, ...]:
        return self.earlier_positions

    @property
    def full_positions(self) -> tuple[int, ...]:
        return self.earlier_positions + self.other_block_positions


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Optimization settings for one classification probe."""

    hidden_dim: int
    emb_dim: int
    epochs: int
    lr: float
    batch_size: int
    seed: int
    device: str


@dataclass(frozen=True)
class ProbeStepPlan:
    """Concrete optimization budget for the configured probes."""

    train_examples: int
    val_examples: int
    steps_per_epoch: int
    steps_per_probe: int
    total_probes: int
    total_optimizer_steps: int


@dataclass(frozen=True)
class ProbeMetrics:
    """Best validation metrics for a trained probe."""

    best_val_accuracy: float
    best_val_loss: float


@dataclass(frozen=True)
class TargetProbeResult:
    """Reported result for one target position."""

    block_index: int
    block_size: int
    target_position: int
    earlier_positions: tuple[int, ...]
    other_block_positions: tuple[int, ...]
    baseline_accuracy: float
    full_accuracy: float
    delta_accuracy: float
    baseline_loss: float
    full_loss: float
    verdict: str


class IndependenceProbe(nn.Module):
    """Token classifier probe over a small set of discrete context tokens."""

    def __init__(
        self,
        *,
        num_input_tokens: int,
        codebook_size: int,
        emb_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_input_tokens = int(num_input_tokens)
        self.codebook_size = int(codebook_size)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)

        if self.num_input_tokens > 0:
            self.emb = nn.Embedding(self.codebook_size, self.emb_dim)
            self.mlp = nn.Sequential(
                nn.Linear(self.num_input_tokens * self.emb_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.codebook_size),
            )
            self.logits_bias = None
        else:
            self.emb = None
            self.mlp = None
            self.logits_bias = nn.Parameter(torch.zeros(self.codebook_size))

    def forward(self, input_token_ids: Tensor) -> Tensor:
        batch_size = int(input_token_ids.shape[0])
        if self.num_input_tokens == 0:
            assert self.logits_bias is not None
            return self.logits_bias.unsqueeze(0).expand(batch_size, -1)
        assert self.emb is not None
        assert self.mlp is not None
        x = self.emb(input_token_ids).flatten(1)
        return self.mlp(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure BOAT intra-block token dependence with per-target MLP probes."
        )
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="BOAT tokenizer checkpoint path or HF Hub repo id.",
    )
    parser.add_argument(
        "--data-repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repo id (for example lerobot/libero_10_no_noops).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional local LeRobot cache root override.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000,
        help="Maximum number of action samples to encode into the probe dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for dataset encoding.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers for dataset encoding.",
    )
    parser.add_argument(
        "--probe-hidden",
        type=int,
        default=128,
        help="Hidden size of the MLP probe.",
    )
    parser.add_argument(
        "--probe-emb-dim",
        type=int,
        default=32,
        help="Embedding size used inside the probe.",
    )
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=100,
        help="Training epochs for each baseline/full probe.",
    )
    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the probe optimizer.",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=512,
        help="Mini-batch size used while training the probes.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.01,
        help="Accuracy-delta threshold used for the text verdict.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device used for encoding and probe training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def build_block_specs(block_schedule: tuple[int, ...]) -> list[BlockSpec]:
    specs: list[BlockSpec] = []
    cursor = 0
    earlier: list[int] = []
    for block_index, block_size in enumerate(block_schedule):
        positions = tuple(range(cursor, cursor + int(block_size)))
        specs.append(
            BlockSpec(
                block_index=block_index,
                start=cursor,
                size=int(block_size),
                positions=positions,
                earlier_positions=tuple(earlier),
            )
        )
        earlier.extend(positions)
        cursor += int(block_size)
    return specs


def build_target_probe_specs(block_schedule: tuple[int, ...]) -> list[TargetProbeSpec]:
    specs: list[TargetProbeSpec] = []
    for block in build_block_specs(block_schedule):
        if block.size < 2:
            continue
        for target_position in block.positions:
            other_positions = tuple(
                pos for pos in block.positions if pos != target_position
            )
            specs.append(
                TargetProbeSpec(
                    block_index=block.block_index,
                    block_size=block.size,
                    target_position=target_position,
                    earlier_positions=block.earlier_positions,
                    other_block_positions=other_positions,
                )
            )
    return specs


def split_train_val(
    tokens: Tensor,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 2:
        raise ValueError(
            f"Expected tokens to have shape (N, L), got {tuple(tokens.shape)}."
        )
    num_rows = int(tokens.shape[0])
    if num_rows < 2:
        raise ValueError(
            f"Need at least 2 samples for a train/val split, got {num_rows}."
        )
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(num_rows, generator=generator)
    val_count = max(1, int(round(num_rows * val_fraction)))
    val_count = min(val_count, num_rows - 1)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return tokens[train_idx], tokens[val_idx]


def _slice_positions(tokens: Tensor, positions: tuple[int, ...]) -> Tensor:
    if not positions:
        return torch.empty((tokens.shape[0], 0), dtype=torch.long)
    idx = torch.tensor(positions, dtype=torch.long)
    return tokens.index_select(1, idx).long()


def _mini_batch_indices(
    num_rows: int,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> list[Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    if shuffle:
        order = torch.randperm(num_rows, generator=generator)
    else:
        order = torch.arange(num_rows)
    return [order[i : i + batch_size] for i in range(0, num_rows, batch_size)]


def build_probe_step_plan(
    *,
    train_examples: int,
    val_examples: int,
    probe_batch_size: int,
    probe_epochs: int,
    total_probes: int,
) -> ProbeStepPlan:
    if train_examples <= 0:
        raise ValueError(f"train_examples must be > 0, got {train_examples}.")
    if val_examples <= 0:
        raise ValueError(f"val_examples must be > 0, got {val_examples}.")
    if probe_batch_size <= 0:
        raise ValueError(f"probe_batch_size must be > 0, got {probe_batch_size}.")
    if probe_epochs <= 0:
        raise ValueError(f"probe_epochs must be > 0, got {probe_epochs}.")
    if total_probes <= 0:
        raise ValueError(f"total_probes must be > 0, got {total_probes}.")

    steps_per_epoch = math.ceil(train_examples / probe_batch_size)
    steps_per_probe = steps_per_epoch * probe_epochs
    return ProbeStepPlan(
        train_examples=train_examples,
        val_examples=val_examples,
        steps_per_epoch=steps_per_epoch,
        steps_per_probe=steps_per_probe,
        total_probes=total_probes,
        total_optimizer_steps=steps_per_probe * total_probes,
    )


def _evaluate_probe(
    model: IndependenceProbe,
    *,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    device: str,
    generator: torch.Generator,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = int(targets.shape[0])
    with torch.inference_mode():
        for batch_idx in _mini_batch_indices(
            total_examples,
            batch_size=batch_size,
            shuffle=False,
            generator=generator,
        ):
            batch_inputs = inputs[batch_idx].to(device, non_blocking=True)
            batch_targets = targets[batch_idx].to(device, non_blocking=True)
            logits = model(batch_inputs)
            loss = nn.functional.cross_entropy(logits, batch_targets)
            total_loss += float(loss.item()) * int(batch_targets.shape[0])
            total_correct += int((logits.argmax(dim=-1) == batch_targets).sum().item())
    return total_correct / total_examples, total_loss / total_examples


def train_probe(
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    codebook_size: int,
    cfg: ProbeTrainingConfig,
) -> ProbeMetrics:
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    model = IndependenceProbe(
        num_input_tokens=int(train_inputs.shape[1]),
        codebook_size=int(codebook_size),
        emb_dim=int(cfg.emb_dim),
        hidden_dim=int(cfg.hidden_dim),
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc = -math.inf
    best_loss = math.inf
    for _epoch in range(cfg.epochs):
        model.train()
        for batch_idx in _mini_batch_indices(
            int(train_targets.shape[0]),
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
        ):
            batch_inputs = train_inputs[batch_idx].to(cfg.device, non_blocking=True)
            batch_targets = train_targets[batch_idx].to(cfg.device, non_blocking=True)
            logits = model(batch_inputs)
            loss = nn.functional.cross_entropy(logits, batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_acc, val_loss = _evaluate_probe(
            model,
            inputs=val_inputs,
            targets=val_targets,
            batch_size=cfg.batch_size,
            device=cfg.device,
            generator=generator,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss

    return ProbeMetrics(
        best_val_accuracy=float(best_acc), best_val_loss=float(best_loss)
    )


def run_target_probe(
    *,
    spec: TargetProbeSpec,
    train_tokens: Tensor,
    val_tokens: Tensor,
    codebook_size: int,
    cfg: ProbeTrainingConfig,
    delta_threshold: float,
) -> TargetProbeResult:
    train_targets = train_tokens[:, spec.target_position].long()
    val_targets = val_tokens[:, spec.target_position].long()

    baseline_train = _slice_positions(train_tokens, spec.baseline_positions)
    baseline_val = _slice_positions(val_tokens, spec.baseline_positions)
    full_train = _slice_positions(train_tokens, spec.full_positions)
    full_val = _slice_positions(val_tokens, spec.full_positions)

    baseline_metrics = train_probe(
        train_inputs=baseline_train,
        train_targets=train_targets,
        val_inputs=baseline_val,
        val_targets=val_targets,
        codebook_size=codebook_size,
        cfg=cfg,
    )
    full_metrics = train_probe(
        train_inputs=full_train,
        train_targets=train_targets,
        val_inputs=full_val,
        val_targets=val_targets,
        codebook_size=codebook_size,
        cfg=cfg,
    )
    delta = full_metrics.best_val_accuracy - baseline_metrics.best_val_accuracy
    verdict = "DEPENDENT" if delta >= delta_threshold else "~INDEPENDENT"
    return TargetProbeResult(
        block_index=spec.block_index,
        block_size=spec.block_size,
        target_position=spec.target_position,
        earlier_positions=spec.earlier_positions,
        other_block_positions=spec.other_block_positions,
        baseline_accuracy=baseline_metrics.best_val_accuracy,
        full_accuracy=full_metrics.best_val_accuracy,
        delta_accuracy=delta,
        baseline_loss=baseline_metrics.best_val_loss,
        full_loss=full_metrics.best_val_loss,
        verdict=verdict,
    )


def format_results_table(results: list[TargetProbeResult]) -> str:
    header = (
        f"{'Block':>5}  {'Size':>4}  {'Target':>6}  "
        f"{'Baseline':>9}  {'Full':>9}  {'Delta':>8}  {'Verdict':>13}"
    )
    rows = [header, "-" * len(header)]
    rows.extend(
        f"{result.block_index:>5}  {result.block_size:>4}  "
        f"{result.target_position:>6}  "
        f"{result.baseline_accuracy:>9.4f}  "
        f"{result.full_accuracy:>9.4f}  "
        f"{result.delta_accuracy:>+8.4f}  "
        f"{result.verdict:>13}"
        for result in results
    )
    return "\n".join(rows)


def summarize_results(results: list[TargetProbeResult]) -> dict[str, Any]:
    grouped: dict[int, list[float]] = {}
    for result in results:
        grouped.setdefault(result.block_index, []).append(result.delta_accuracy)

    per_block = [
        {
            "block_index": block_index,
            "num_targets": len(deltas),
            "mean_delta_accuracy": sum(deltas) / len(deltas),
            "max_delta_accuracy": max(deltas),
            "min_delta_accuracy": min(deltas),
        }
        for block_index, deltas in sorted(grouped.items())
    ]
    overall_deltas = [result.delta_accuracy for result in results]
    return {
        "num_probes": len(results),
        "mean_delta_accuracy": (
            sum(overall_deltas) / len(overall_deltas) if overall_deltas else 0.0
        ),
        "max_delta_accuracy": max(overall_deltas) if overall_deltas else 0.0,
        "min_delta_accuracy": min(overall_deltas) if overall_deltas else 0.0,
        "per_block": per_block,
    }


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested device 'cuda' but CUDA is not available. "
            "Run this on the interactive GPU node or pass --device cpu for a tiny smoke run."
        )
    return device


def collect_token_dataset(
    *,
    tokenizer_path: str,
    data_repo_id: str,
    data_root: str | None,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[Tensor, tuple[int, ...], int]:
    from praxis.codecs.factory import make_codec
    from praxis.codecs.tokens.boat.tokenizer import BOATTokenizer
    from praxis.data.lerobot_adapter import LeRobotActionOnlyDatasetAdapter

    tokenizer = make_codec(
        {"kind": "tokens", "name": "boat", "pretrained_path": tokenizer_path}
    )
    if not isinstance(tokenizer, BOATTokenizer):
        raise TypeError(
            "BOAT independence diagnostic requires a BOAT tokenizer checkpoint, "
            f"got {type(tokenizer).__name__}."
        )
    if tokenizer.io is None:
        raise RuntimeError(
            f"Tokenizer checkpoint at {tokenizer_path} is missing PolicyIO sidecar data."
        )
    tokenizer.to(device)
    tokenizer.eval()

    dataset = LeRobotActionOnlyDatasetAdapter(
        repo_id=data_repo_id,
        io=tokenizer.io,
        root=data_root,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    action_key = tokenizer.action_key
    collected: list[Tensor] = []
    total = 0
    with torch.inference_mode():
        for batch in dataloader:
            actions = batch[action_key]
            remaining = num_samples - total
            if remaining <= 0:
                break
            if int(actions.shape[0]) > remaining:
                actions = actions[:remaining]
            encoded = tokenizer.tokenize_with_latents(
                actions.to(device, non_blocking=True)
            )
            tokens_cpu = encoded.token_ids.detach().to("cpu", non_blocking=False).long()
            collected.append(tokens_cpu)
            total += int(tokens_cpu.shape[0])
            print(f"Collected {total}/{num_samples} encoded action samples.")

    if not collected:
        raise RuntimeError(
            "Failed to collect any encoded actions from the dataset. "
            "Check the dataset repo id and sample limit."
        )
    token_tensor = torch.cat(collected, dim=0)
    return (
        token_tensor,
        tuple(int(x) for x in tokenizer.block_schedule),
        int(tokenizer.codebook_size),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    print(
        f"Loading tokens from tokenizer={args.tokenizer_path} dataset={args.data_repo_id}"
    )
    tokens, block_schedule, codebook_size = collect_token_dataset(
        tokenizer_path=args.tokenizer_path,
        data_repo_id=args.data_repo_id,
        data_root=args.data_root,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(
        f"Encoded token dataset shape={tuple(tokens.shape)}, "
        f"block_schedule={block_schedule}, codebook_size={codebook_size}"
    )

    train_tokens, val_tokens = split_train_val(
        tokens,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        f"Train/val split: train={int(train_tokens.shape[0])}, "
        f"val={int(val_tokens.shape[0])}"
    )

    probe_specs = build_target_probe_specs(block_schedule)
    print(f"Running {len(probe_specs)} per-target probes.")
    probe_cfg = ProbeTrainingConfig(
        hidden_dim=args.probe_hidden,
        emb_dim=args.probe_emb_dim,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        batch_size=args.probe_batch_size,
        seed=args.seed,
        device=device,
    )
    probe_step_plan = build_probe_step_plan(
        train_examples=int(train_tokens.shape[0]),
        val_examples=int(val_tokens.shape[0]),
        probe_batch_size=args.probe_batch_size,
        probe_epochs=args.probe_epochs,
        total_probes=len(probe_specs) * 2,
    )
    print(
        "Probe optimization budget: "
        f"{probe_step_plan.steps_per_epoch} steps/epoch, "
        f"{probe_step_plan.steps_per_probe} steps/probe, "
        f"{probe_step_plan.total_optimizer_steps} optimizer steps total."
    )

    results: list[TargetProbeResult] = []
    for probe_index, probe_spec in enumerate(probe_specs, start=1):
        print(
            f"[{probe_index}/{len(probe_specs)}] "
            f"block={probe_spec.block_index} target={probe_spec.target_position} "
            f"baseline_ctx={len(probe_spec.baseline_positions)} "
            f"full_ctx={len(probe_spec.full_positions)}"
        )
        results.append(
            run_target_probe(
                spec=probe_spec,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                codebook_size=codebook_size,
                cfg=probe_cfg,
                delta_threshold=args.delta_threshold,
            )
        )

    summary = summarize_results(results)
    print()
    print(format_results_table(results))
    print()
    print(json.dumps(summary, indent=2))

    payload = {
        "tokenizer_path": args.tokenizer_path,
        "data_repo_id": args.data_repo_id,
        "num_samples": int(tokens.shape[0]),
        "token_seq_len": int(tokens.shape[1]),
        "block_schedule": list(block_schedule),
        "codebook_size": codebook_size,
        "probe_config": asdict(probe_cfg),
        "probe_step_plan": asdict(probe_step_plan),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote JSON results to {output_path}")


if __name__ == "__main__":
    main()
