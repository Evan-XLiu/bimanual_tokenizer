#!/usr/bin/env python3
"""Backfill eval monitor results from eval_results.json to W&B.

This script attaches to the training run recorded in ``wandb_run_info.json`` and
logs historical eval metrics using ``sim_eval/eval_step`` as the x-axis metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WANDB_RUN_INFO_FILENAME = "wandb_run_info.json"
EVAL_RESULTS_FILENAME = "eval_results.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill eval_results.json into the corresponding W&B run.",
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=None,
        help="Path to eval_results.json (default: <train-dir>/eval/eval_results.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads without calling wandb.log.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {path}, got {type(payload).__name__}"
        )
    return payload


def _normalize_metrics(metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        mapped_key = key if key.startswith("sim_eval/") else f"sim_eval/{key}"
        out[mapped_key] = float(value)
    return out


def main() -> None:
    args = _parse_args()
    train_dir = args.train_dir.resolve()
    run_info_path = train_dir / WANDB_RUN_INFO_FILENAME
    eval_results_path = (
        args.eval_results or (train_dir / "eval" / EVAL_RESULTS_FILENAME)
    ).resolve()

    run_info = _load_json(run_info_path)
    eval_results = _load_json(eval_results_path)

    entries: list[tuple[int, dict[str, float]]] = []
    for step_raw, metrics_raw in eval_results.items():
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics_raw, dict):
            continue
        metrics = _normalize_metrics(metrics_raw)
        if metrics:
            entries.append((step, metrics))
    entries.sort(key=lambda x: x[0])

    if not entries:
        print(f"No numeric eval metrics found in {eval_results_path}")
        return

    if args.dry_run:
        print(
            f"[dry-run] Would backfill {len(entries)} entries to run {run_info.get('id')}"
        )
        for step, metrics in entries:
            payload = {"sim_eval/eval_step": step, **metrics}
            print(payload)
        return

    import wandb

    wandb.init(
        id=run_info["id"],
        project=run_info.get("project"),
        entity=run_info.get("entity"),
        resume="allow",
    )
    try:
        wandb.define_metric("sim_eval/eval_step")
        wandb.define_metric("sim_eval/*", step_metric="sim_eval/eval_step")

        print(f"Backfilling {len(entries)} entries to run {run_info['id']} ...")
        for step, metrics in entries:
            payload = {"sim_eval/eval_step": step, **metrics}
            wandb.log(payload)
            print(f"  logged eval_step={step} keys={sorted(metrics.keys())}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
