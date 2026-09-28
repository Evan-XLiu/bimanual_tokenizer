#!/usr/bin/env python3
"""Plot validation reconstruction loss for completed tokenizer runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "tokenizer_comparison"
RUNS = {
    "BTT": ROOT / "outputs/hopper_btt_robotwin_new/621340.hopper-m-02/train/train_metrics.jsonl",
    "OAT": ROOT / "outputs/hopper_oat_robotwin_new_2500/600693.hopper-m-02/train/train_metrics.jsonl",
    "QueST": ROOT / "outputs/hopper_quest_robotwin_new/613294.hopper-m-02/train/train_metrics.jsonl",
    "ActionCodec": ROOT / "outputs/hopper_acodec_robotwin_new/627717.hopper-m-02/train/train_metrics.jsonl",
    "BOAT": ROOT / "outputs/hopper_boat_robotwin_new/616167.hopper-m-02/train/train_metrics.jsonl",
}
COLORS = {
    "BTT": "#E45756",
    "OAT": "#4C78A8",
    "QueST": "#54A24B",
    "ActionCodec": "#B279A2",
    "BOAT": "#F58518",
}


def load(path: Path) -> list[tuple[int, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "eval/reconst_loss" in record:
            rows.append((int(record["step"]), float(record["eval/reconst_loss"])))
    return rows


def main() -> None:
    histories = {name: load(path) for name, path in RUNS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    steps = sorted({step for rows in histories.values() for step, _ in rows})
    values_by_run = {
        name: {step: value for step, value in rows}
        for name, rows in histories.items()
    }
    with (OUT / "tokenizer_reconst_curves_5methods.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *RUNS])
        for step in steps:
            writer.writerow([step, *(values_by_run[name].get(step, "") for name in RUNS)])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for name, rows in histories.items():
        x, y = zip(*rows)
        ax.plot(x, y, label=name, color=COLORS[name], linewidth=2.2)
        best_step, best_value = min(rows, key=lambda item: item[1])
        ax.scatter([best_step], [best_value], color=COLORS[name], s=28, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(0, 300000)
    ax.set_xticks([0, 50000, 100000, 150000, 200000, 250000, 300000])
    ax.set_xticklabels(["0", "50k", "100k", "150k", "200k", "250k", "300k"])
    ax.set_ylim(5e-4, 3e-1)
    ax.set_xlabel("Tokenizer training step")
    ax.set_ylabel("Validation reconstruction loss (eval/reconst_loss, log scale)")
    ax.set_title("Tokenizer reconstruction loss during pretraining")
    ax.legend(ncol=3, frameon=True)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.savefig(OUT / "tokenizer_reconst_curves_5methods.png", dpi=220)
    fig.savefig(OUT / "tokenizer_reconst_curves_5methods.pdf")

    selected_steps = [10_000, 20_000, 50_000, 150_000, 300_000]
    selected_csv = OUT / "tokenizer_reconst_selected_5methods.csv"
    with selected_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *RUNS])
        for step in selected_steps:
            writer.writerow([step, *(values_by_run[name][step] for name in RUNS)])

    fig_selected, ax_selected = plt.subplots(
        figsize=(9.5, 5.8), constrained_layout=True
    )
    for name in RUNS:
        values = [values_by_run[name][step] for step in selected_steps]
        ax_selected.plot(
            selected_steps,
            values,
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=name,
            color=COLORS[name],
        )
    ax_selected.set_yscale("log")
    # Express loss in 1e-3 units while retaining logarithmic separation across
    # the early ActionCodec point and the late-stage differences.
    for line in ax_selected.lines:
        line.set_ydata([value * 1e3 for value in line.get_ydata()])
    ax_selected.set_xlim(8_000, 302_000)
    ax_selected.set_xticks(selected_steps)
    ax_selected.set_xticklabels(["10k", "20k", "50k", "150k", "300k"])
    ax_selected.set_ylim(0.5, 40.0)
    ax_selected.set_xlabel("Tokenizer training step")
    ax_selected.set_ylabel(r"Validation reconstruction loss ($\times 10^{-3}$, log scale)")
    ax_selected.set_title("Tokenizer reconstruction loss at selected steps")
    ax_selected.legend(ncol=3, frameon=True)
    ax_selected.grid(True, which="both", axis="y", alpha=0.25)
    fig_selected.savefig(
        OUT / "tokenizer_reconst_selected_5methods.png", dpi=220
    )
    fig_selected.savefig(OUT / "tokenizer_reconst_selected_5methods.pdf")


if __name__ == "__main__":
    main()
