from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from praxis.scripts.resilient_simpler import (
    evaluate_resilient_batches,
    resilient_simpler_commands,
)


def _fake_simpler_command(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_simpler.py"
    script.write_text(
        """
import argparse
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num-episodes", type=int, required=True)
parser.add_argument("--metrics-output-path", required=True)
args = parser.parse_args()

seeds = range(args.seed, args.seed + args.num_episodes)
if 42 in seeds:
    time.sleep(60)

payload = {
    "n_episodes": args.num_episodes,
    "success_rate": 1.0,
    "ever_success_rate": 1.0,
    "final_success_rate": 1.0,
    "avg_episode_length": 10.0,
    "avg_reward": 0.0,
    "avg_max_reward": 0.0,
    "eval_s": 0.01,
}
with open(args.metrics_output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
print(f"completed seed={args.seed} count={args.num_episodes}", flush=True)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return [
        sys.executable,
        str(script),
        "--seed",
        "42",
        "--num-episodes",
        "4",
        "--metrics-output-path",
        str(tmp_path / "metrics.json"),
    ]


def test_resilient_batches_isolate_a_stalled_episode(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"

    evaluate_resilient_batches(
        command=_fake_simpler_command(tmp_path),
        output_path=output_path,
        work_dir=tmp_path / "batches",
        inactivity_timeout_sec=0.1,
    )

    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["n_episodes"] == 4
    assert metrics["success_rate"] == 0.75
    assert metrics["ever_success_rate"] == 0.75
    assert metrics["final_success_rate"] == 0.75
    assert metrics["timed_out_episodes"] == [42]
    assert sorted(metrics["completed_episode_seeds"]) == [43, 44, 45]

    batch_records = sorted((tmp_path / "batches").glob("*/result.json"))
    assert batch_records
    assert any(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "timeout"
        for path in batch_records
    )


def test_resilient_command_context_wraps_and_restores_builder(
    monkeypatch, tmp_path: Path
) -> None:
    from praxis_eval.envs.simpler import eval as simpler_eval

    metrics_path = tmp_path / "metrics.json"
    def original_builder(**_kwargs):
        return [
            "python",
            "-m",
            "simpler",
            "--seed",
            "42",
            "--num-episodes",
            "50",
            "--metrics-output-path",
            str(metrics_path),
        ]
    monkeypatch.setattr(simpler_eval, "build_simpler_eval_command", original_builder)

    with resilient_simpler_commands(inactivity_timeout_sec=120.0):
        wrapped = simpler_eval.build_simpler_eval_command()

    assert wrapped[:3] == [sys.executable, "-m", "praxis.scripts.resilient_simpler"]
    assert wrapped[wrapped.index("--output-path") + 1] == str(metrics_path)
    assert wrapped[wrapped.index("--inactivity-timeout-sec") + 1] == "120.0"
    assert wrapped[-len(original_builder()) :] == original_builder()
    assert simpler_eval.build_simpler_eval_command is original_builder


def test_resilient_batches_do_not_hide_process_errors(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"
    command = [
        sys.executable,
        "-c",
        "import sys; sys.exit(3)",
        "--seed",
        "42",
        "--num-episodes",
        "1",
        "--metrics-output-path",
        str(output_path),
    ]

    with pytest.raises(RuntimeError, match="SIMPLER subprocess failed"):
        evaluate_resilient_batches(
            command=command,
            output_path=output_path,
            work_dir=tmp_path / "batches",
            inactivity_timeout_sec=1.0,
        )

    assert not output_path.exists()
