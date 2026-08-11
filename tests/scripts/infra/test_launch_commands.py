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


def test_robotwin_bin_bootstrap_uses_read_only_dataset_config() -> None:
    cmd = commands.build_tokenizer_command(
        codec="bin",
        dataset="robotwin2_clean",
        gpus=8,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="robotwin-bin",
        output_dir="/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/bin",
        skip_codec_fit=True,
        extra_overrides=["codec.action_horizon=32", "trainer.max_steps=1"],
        python_bin="/env/bin/python",
    )

    assert "--config-name" in cmd
    assert "train_codecs" in cmd
    assert "codecs@codec=tokens/bin" in cmd
    assert "trainer.skip_codec_fit=true" in cmd
    assert "dataset=robotwin2_clean" in cmd
    assert "codec.action_horizon=32" in cmd


def test_robotwin_full_ar_uses_bin_artifact_and_explicit_smoke_geometry() -> None:
    tokenizer = "/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/bin/pretrained_model"
    cmd = commands.build_policy_command(
        method="bar",
        codec="bin",
        backbone="qwen",
        dataset="robotwin2_clean",
        gpus=8,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="robotwin-ar-smoke",
        output_dir="/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/ar",
        tokenizer_ckpt=tokenizer,
        extra_overrides=[
            "policy.action_horizon=32",
            "policy.n_action_steps=16",
            "trainer.max_steps=10",
        ],
        python_bin="/env/bin/python",
    )

    assert "policy=bar_bin_qwen" in cmd
    assert "train_recipe/policy@_global_=bar" in cmd
    assert f"policy.action_tokenizer.pretrained_path={tokenizer}" in cmd
    assert "dataset=robotwin2_clean" in cmd
    assert "policy.action_horizon=32" in cmd
    assert "policy.n_action_steps=16" in cmd
    assert "trainer.max_steps=10" in cmd
