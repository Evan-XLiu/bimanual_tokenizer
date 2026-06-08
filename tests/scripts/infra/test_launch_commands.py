# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from scripts.infra.launch import commands


def test_python_bin_routes_training_through_same_interpreter() -> None:
    cmd = commands.build_tokenizer_command(
        codec="oat",
        dataset="libero",
        gpus=2,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="outputs/run",
        skip_codec_fit=False,
        extra_overrides=[],
        python_bin="/env/bin/python",
    )

    assert cmd[:4] == [
        "/env/bin/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
    ]
    assert "uv" not in cmd


def test_python_bin_routes_eval_monitor_to_same_env_bin() -> None:
    cmd = commands.build_eval_monitor_command(
        train_dir="outputs/train",
        num_eval_per_task=1,
        num_parallel_env=1,
        record_episodes_per_task=0,
        stale_timeout_sec=60,
        eval_timeout_sec=120,
        eval_overrides=[],
        python_bin="/env/bin/python",
    )

    assert cmd[0] == "/env/bin/praxis-eval-monitor"
    assert "uv" not in cmd


def test_python_bin_routes_eval_to_same_env_bin(tmp_path: Path, monkeypatch) -> None:
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "train_config.yaml").write_text("dataset:\n  name: libero\n")
    monkeypatch.setattr(
        commands,
        "build_eval_overrides_from_train_config",
        lambda *_args, **_kwargs: ["env.type=libero"],
    )

    cmd = commands.build_eval_command(
        str(ckpt),
        "outputs/eval",
        num_eval_per_task=1,
        num_parallel_env=1,
        record_episodes_per_task=0,
        eval_overrides=[],
        python_bin="/env/bin/python",
    )

    assert cmd[:2] == ["/env/bin/praxis-eval", "env.type=libero"]
    assert "uv" not in cmd
