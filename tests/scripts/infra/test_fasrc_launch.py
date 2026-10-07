# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for command construction in scripts/infra/fasrc_launch.py."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from praxis.policies.catalog import iter_policy_specs
from tests._paths import repo_root


@pytest.fixture(autouse=True)
def _set_public_launcher_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRAXIS_FASRC_ACCOUNT", "fasrc_account")
    monkeypatch.setenv("PRAXIS_FASRC_KEMPNER_ACCOUNT", "fasrc_kempner_account")


def _load_fasrc_launch_module():
    # Import launcher internals directly; the entrypoint stays thin.
    args_mod = importlib.import_module("scripts.infra.launch.args")
    commands_mod = importlib.import_module("scripts.infra.launch.commands")
    fasrc_launch_mod = importlib.import_module("scripts.infra.fasrc_launch")
    slurm_mod = importlib.import_module("scripts.infra.launch.slurm")
    stages_mod = importlib.import_module("scripts.infra.launch.stages")

    class _LaunchAPI:
        build_tokenizer_command = staticmethod(commands_mod.build_tokenizer_command)
        build_policy_command = staticmethod(commands_mod.build_policy_command)
        build_eval_monitor_command = staticmethod(
            commands_mod.build_eval_monitor_command
        )
        build_eval_command = staticmethod(commands_mod.build_eval_command)
        _submit_or_print_stage = staticmethod(slurm_mod.submit_or_print_stage)
        _build_arg_parser = staticmethod(args_mod.build_arg_parser)
        _validate_args = staticmethod(args_mod.validate_args)
        _parse_gres_gpu_count = staticmethod(args_mod._parse_gres_gpu_count)
        _flatten_eval_overrides = staticmethod(args_mod.flatten_eval_overrides)
        _parse_partition_inputs = staticmethod(args_mod.parse_partition_inputs)
        _resolve_stage_run_name_and_output_dir = staticmethod(
            stages_mod.resolve_stage_run_name_and_output_dir
        )
        _strip_resolved_run_overrides = staticmethod(
            stages_mod.strip_resolved_run_overrides
        )
        _validate_partition_list = staticmethod(
            fasrc_launch_mod.validate_partition_list
        )
        _is_requeue_partition_spec = staticmethod(
            fasrc_launch_mod._is_requeue_partition_spec
        )
        _resolve_account_for_partition_spec = staticmethod(
            fasrc_launch_mod._resolve_account_for_partition_spec
        )
        run_training_job = staticmethod(slurm_mod.run_training_job)
        make_executor = staticmethod(slurm_mod.make_executor)
        _link_worktree_outputs = staticmethod(slurm_mod._link_worktree_outputs)
        os = slurm_mod.os
        subprocess = slurm_mod.subprocess
        _query_slurm_job_state = staticmethod(slurm_mod.query_slurm_job_state)

        @staticmethod
        def _build_eval_monitor_dependency(
            train_job_id: int | None,
        ) -> tuple[dict[str, object], str, str]:
            return slurm_mod.build_eval_monitor_dependency(
                train_job_id,
                query_job_state=_LaunchAPI._query_slurm_job_state,
            )

    return _LaunchAPI


def _make_executor_with_captured_parameters(
    monkeypatch: pytest.MonkeyPatch,
    mod: Any,
    tmp_path: Path,
    **overrides: Any,
) -> dict[str, Any]:
    import submitit

    created: list[Any] = []

    class _FakeAutoExecutor:
        def __init__(self, *, folder: str) -> None:
            self.folder = folder
            self.parameters: dict[str, Any] = {}
            created.append(self)

        def update_parameters(self, **kwargs: Any) -> None:
            self.parameters.update(kwargs)

    monkeypatch.setattr(submitit, "AutoExecutor", _FakeAutoExecutor)

    defaults: dict[str, Any] = {
        "project_root": str(tmp_path),
        "run_name": "submitit-test",
        "gpus": 1,
        "mem": "32G",
        "partition": "seas_gpu",
        "account": "fasrc_account",
        "num_nodes": 1,
        "cpus_per_task": 4,
        "time_minutes": 30,
        "constraint": None,
        "gres": None,
        "requeue_policy": "allow",
    }
    defaults.update(overrides)
    executor = mod.make_executor(**defaults)

    assert created == [executor]
    return executor.parameters


def test_build_tokenizer_command_single_gpu_uses_torchrun_standalone() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_tokenizer_command(
        codec="oat",
        dataset="libero_10",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        skip_codec_fit=False,
        extra_overrides=[],
    )
    assert cmd[:6] == [
        "uv",
        "run",
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
    ]
    assert "--nproc_per_node=1" in cmd
    assert "--config-name" in cmd
    assert "train_codecs" in cmd
    assert "codecs@codec=tokens/oat" in cmd


def test_build_tokenizer_command_multi_gpu_sets_nproc_per_node() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_tokenizer_command(
        codec="oat",
        dataset="libero_10",
        gpus=4,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        skip_codec_fit=False,
        extra_overrides=[],
    )
    assert cmd[:6] == [
        "uv",
        "run",
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
    ]
    assert "--nproc_per_node=4" in cmd


def test_build_tokenizer_command_fast_default_fits_dataset_bpe() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_tokenizer_command(
        codec="fast",
        dataset="libero",
        gpus=2,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        skip_codec_fit=False,
        extra_overrides=[],
    )
    assert "codec.bpe_tokenizer_path=null" in cmd
    assert "trainer.skip_codec_fit=true" not in cmd


def test_build_tokenizer_command_fast_skip_fit_uses_default_bootstrap_bpe() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_tokenizer_command(
        codec="fast",
        dataset="libero",
        gpus=2,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        skip_codec_fit=True,
        extra_overrides=[],
    )
    assert "codec.bpe_tokenizer_path=null" not in cmd
    assert not any(arg.startswith("codec.bpe_tokenizer_path=") for arg in cmd)
    assert "trainer.skip_codec_fit=true" in cmd


def test_build_tokenizer_command_bin_skip_fit_bootstraps_stats_checkpoint() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_tokenizer_command(
        codec="bin",
        dataset="libero",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        skip_codec_fit=True,
        extra_overrides=[],
    )
    assert "codecs@codec=tokens/bin" in cmd
    assert "trainer.skip_codec_fit=true" in cmd
    assert "codec.bpe_tokenizer_path=null" not in cmd


def test_build_tokenizer_command_rejects_skip_fit_for_learned_codecs() -> None:
    mod = _load_fasrc_launch_module()
    with pytest.raises(ValueError, match="skip_codec_fit is only supported"):
        mod.build_tokenizer_command(
            codec="oat",
            dataset="libero",
            gpus=1,
            num_nodes=1,
            main_process_ip=None,
            main_process_port=29500,
            run_name="run",
            output_dir="out",
            skip_codec_fit=True,
            extra_overrides=[],
        )


def test_build_policy_command_multi_gpu_sets_nproc_per_node() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_policy_command(
        method="bar",
        codec="oat",
        backbone="qwen",
        dataset="libero_10",
        gpus=4,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        tokenizer_ckpt="tokenizer_path",
        extra_overrides=[],
    )
    assert cmd[:6] == [
        "uv",
        "run",
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
    ]
    assert "--nproc_per_node=4" in cmd
    assert "policy.action_tokenizer.pretrained_path=tokenizer_path" in cmd
    assert "train_recipe/policy@_global_=bar" in cmd
    assert not any(arg.startswith("trainer.lr=") for arg in cmd)


def test_build_policy_command_multi_node_includes_rendezvous_flags() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_policy_command(
        method="bar",
        codec="oat",
        backbone="qwen",
        dataset="libero_10",
        gpus=4,
        num_nodes=2,
        main_process_ip="10.0.0.1",
        main_process_port=29501,
        run_name="run",
        output_dir="out",
        tokenizer_ckpt="tokenizer_path",
        extra_overrides=[],
    )
    assert "--nproc_per_node=4" in cmd
    assert "--nnodes=2" in cmd
    assert "--node_rank=__SLURM_PROCID__" in cmd
    assert "--master_addr=10.0.0.1" in cmd
    assert "--master_port=29501" in cmd


def test_build_policy_command_rejects_codec_backed_policy_without_checkpoint() -> None:
    mod = _load_fasrc_launch_module()

    with pytest.raises(ValueError, match="requires an action-token checkpoint"):
        mod.build_policy_command(
            method="bar",
            codec="fast",
            backbone="pali",
            dataset="libero",
            gpus=4,
            num_nodes=1,
            main_process_ip=None,
            main_process_port=29500,
            run_name="run",
            output_dir="out",
            tokenizer_ckpt=None,
            extra_overrides=[],
        )


def test_build_policy_command_round_trips_all_existing_policy_configs() -> None:
    mod = _load_fasrc_launch_module()
    policy_config_root = repo_root() / "src" / "praxis" / "configs" / "policy"
    config_names = sorted(p.stem for p in policy_config_root.glob("*.yaml"))

    assert config_names == sorted(spec.name for spec in iter_policy_specs())
    for spec in iter_policy_specs():
        cmd = mod.build_policy_command(
            method=spec.method,
            codec=spec.codec,
            backbone=spec.backbone,
            dataset="libero_10",
            gpus=1,
            num_nodes=1,
            main_process_ip=None,
            main_process_port=29500,
            run_name="run",
            output_dir="out",
            tokenizer_ckpt="tokenizer_path" if spec.codec is not None else None,
            extra_overrides=[],
        )
        assert f"policy={spec.name}" in cmd
        assert f"train_recipe/policy@_global_={spec.method}" in cmd


def test_launcher_policy_matrix_matches_existing_policy_configs() -> None:
    config_mod = importlib.import_module("scripts.infra.launch.config")
    policy_config_root = repo_root() / "src" / "praxis" / "configs" / "policy"
    config_names = sorted(p.stem for p in policy_config_root.glob("*.yaml"))

    assert sorted(config_mod.supported_policy_config_names()) == config_names


def test_build_policy_command_sets_canonical_bar_policy_name() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_policy_command(
        method="bar",
        codec="fast",
        backbone="qwen",
        dataset="libero_10",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        tokenizer_ckpt="tokenizer_path",
        extra_overrides=[],
    )

    assert "policy=bar_fast_qwen" in cmd
    assert "train_recipe/policy@_global_=bar" in cmd
    assert "policy.action_tokenizer.pretrained_path=tokenizer_path" in cmd


def test_build_policy_command_sets_canonical_ki_policy_name() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_policy_command(
        method="ki",
        codec="oat",
        backbone="pali",
        dataset="libero_10",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        tokenizer_ckpt="tokenizer_path",
        extra_overrides=[],
    )

    assert "policy=ki_oat_pali" in cmd
    assert "train_recipe/policy@_global_=ki" in cmd
    assert "policy.action_tokenizer.pretrained_path=tokenizer_path" in cmd
    assert "policy.gradient_checkpointing=true" not in cmd


@pytest.mark.parametrize("codec", ["fast", "oat", "btt"])
def test_build_policy_command_supports_molmo_ki_specs(codec: str) -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_policy_command(
        method="ki",
        codec=codec,
        backbone="molmo",
        dataset="libero_10",
        gpus=4,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        tokenizer_ckpt="tokenizer_path",
        extra_overrides=[],
    )

    assert f"policy=ki_{codec}_molmo" in cmd
    assert "train_recipe/policy@_global_=ki" in cmd
    assert "policy.action_tokenizer.pretrained_path=tokenizer_path" in cmd


def test_build_policy_command_sets_canonical_fm_policy_name() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_policy_command(
        method="fm",
        codec=None,
        backbone="pali",
        dataset="libero_10",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        tokenizer_ckpt=None,
        extra_overrides=[],
    )

    assert "policy=fm_pali" in cmd
    assert "train_recipe/policy@_global_=fm" in cmd
    assert "policy.gradient_checkpointing=true" not in cmd
    assert not any(arg.startswith("policy.action_tokenizer.") for arg in cmd)


def test_build_policy_command_rejects_fm_tokenizer_checkpoint() -> None:
    mod = _load_fasrc_launch_module()

    with pytest.raises(ValueError, match="does not use action-token checkpoints"):
        mod.build_policy_command(
            method="fm",
            codec=None,
            backbone="pali",
            dataset="libero_10",
            gpus=1,
            num_nodes=1,
            main_process_ip=None,
            main_process_port=29500,
            run_name="run",
            output_dir="out",
            tokenizer_ckpt="tokenizer_path",
            extra_overrides=[],
        )


def test_build_policy_command_rejects_unsupported_ki_boat_policy() -> None:
    mod = _load_fasrc_launch_module()

    with pytest.raises(ValueError, match="Unsupported policy codec 'boat'"):
        mod.build_policy_command(
            method="ki",
            codec="boat",
            backbone="pali",
            dataset="libero_10",
            gpus=1,
            num_nodes=1,
            main_process_ip=None,
            main_process_port=29500,
            run_name="run",
            output_dir="out",
            tokenizer_ckpt="tokenizer_path",
            extra_overrides=[],
        )


def test_build_policy_command_selects_dataset_config_for_bridge_single_view() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_policy_command(
        method="bar",
        codec="oat",
        backbone="qwen",
        dataset="bridge_single_view",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        tokenizer_ckpt="tokenizer_path",
        extra_overrides=[],
    )
    assert "dataset=bridge_single_view" in cmd
    assert "dataset.name=bridge_single_view" not in cmd
    assert "dataset.repo_id=chaoqi-liu/bridge_single_view" not in cmd


def test_build_tokenizer_command_selects_dataset_config_for_bridge_single_view() -> (
    None
):
    mod = _load_fasrc_launch_module()
    cmd = mod.build_tokenizer_command(
        codec="boat",
        dataset="bridge_single_view",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        skip_codec_fit=False,
        extra_overrides=[],
    )
    assert "dataset=bridge_single_view" in cmd
    assert "dataset.name=bridge_single_view" not in cmd
    assert "dataset.repo_id=chaoqi-liu/bridge_single_view" not in cmd


def test_build_tokenizer_command_falls_back_to_scalar_dataset_overrides() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_tokenizer_command(
        codec="oat",
        dataset="libero_spatial",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="run",
        output_dir="out",
        skip_codec_fit=False,
        extra_overrides=[],
    )
    assert "dataset=libero_spatial" not in cmd
    assert "dataset.name=libero_spatial" in cmd
    assert "dataset.repo_id=lerobot/libero_spatial" in cmd


def test_build_eval_monitor_command_passes_record_episodes_per_task() -> None:
    mod = _load_fasrc_launch_module()
    cmd = mod.build_eval_monitor_command(
        train_dir="outputs/train_policy/run",
        num_eval_per_task=10,
        num_parallel_env=80,
        record_episodes_per_task=0,
        stale_timeout_sec=7200,
        eval_timeout_sec=1800,
        eval_overrides=["policy.decode_strict=true"],
    )
    assert "--record-episodes-per-task" in cmd
    assert "0" in cmd
    assert "--stale-timeout-sec" in cmd
    assert "7200" in cmd
    assert "--eval-timeout-sec" in cmd
    assert "1800" in cmd
    assert "--eval-override" in cmd


def test_eval_override_repeated_flags_are_preserved() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "eval",
            "--policy-ckpt=outputs/checkpoint/pretrained_model",
            "--eval-output-dir=outputs/eval/run",
            "--eval-override",
            "env.python_bin=/tmp/simpler/bin/python",
            "--eval-override",
            "eval.policy_kwargs.decode_keep_k=1",
        ]
    )

    assert mod._flatten_eval_overrides(args.eval_override) == [
        "env.python_bin=/tmp/simpler/bin/python",
        "eval.policy_kwargs.decode_keep_k=1",
    ]


def test_stage_run_name_honors_top_level_overrides() -> None:
    mod = _load_fasrc_launch_module()

    run_name, output_dir = mod._resolve_stage_run_name_and_output_dir(
        default_run_name="libero_bar_boat_pali",
        default_output_dir="outputs/train_policy/20260509/libero_bar_boat_pali",
        overrides=[
            "policy.decode_keep_k=8",
            "run_name=libero_bar_oat_pali_probe",
            "output_dir=outputs/train_policy/20260509/libero_bar_oat_pali_probe",
        ],
    )

    assert run_name == "libero_bar_oat_pali_probe"
    assert output_dir == "outputs/train_policy/20260509/libero_bar_oat_pali_probe"


def test_strip_resolved_run_overrides_keeps_unrelated_overrides() -> None:
    mod = _load_fasrc_launch_module()

    assert mod._strip_resolved_run_overrides(
        [
            "run_name=custom",
            "trainer.max_steps=1000",
            "output_dir=outputs/run",
            "policy.decode_keep_k=8",
        ]
    ) == ["trainer.max_steps=1000", "policy.decode_keep_k=8"]


def test_build_eval_command_keeps_declared_policy_kwargs_unprefixed(
    tmp_path: Path,
) -> None:
    mod = _load_fasrc_launch_module()
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.yaml").write_text(
        "\n".join(
            [
                "policy:",
                "  name: bar_boat_pali",
                "dataset:",
                "  name: libero",
                "device: cuda",
            ]
        )
    )

    cmd = mod.build_eval_command(
        str(ckpt),
        "outputs/eval/run",
        num_eval_per_task=10,
        num_parallel_env=40,
        record_episodes_per_task=0,
        eval_overrides=[
            "eval.policy_kwargs.decode_keep_k=8",
            "eval.policy_kwargs.decode_temperature=0.7",
            "eval.policy_kwargs.decode_top_k=8",
            "eval.policy_kwargs.decode_strict=true",
        ],
    )

    assert "eval.policy_kwargs.decode_keep_k=8" in cmd
    assert "+eval.policy_kwargs.decode_keep_k=8" not in cmd
    assert "eval.policy_kwargs.decode_temperature=0.7" in cmd
    assert "+eval.policy_kwargs.decode_temperature=0.7" not in cmd
    assert "eval.policy_kwargs.decode_top_k=8" in cmd
    assert "+eval.policy_kwargs.decode_top_k=8" not in cmd
    assert "+eval.policy_kwargs.decode_strict=true" in cmd


def test_build_eval_command_derives_policy_and_env_from_train_config(
    tmp_path: Path,
) -> None:
    mod = _load_fasrc_launch_module()
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.yaml").write_text(
        "\n".join(
            [
                "policy:",
                "  name: bar_boat_pali",
                "dataset:",
                "  name: libero",
                "device: cuda",
            ]
        )
    )

    cmd = mod.build_eval_command(
        str(ckpt),
        "outputs/eval/run",
        num_eval_per_task=10,
        num_parallel_env=40,
        record_episodes_per_task=0,
        eval_overrides=["eval.policy_kwargs.decode_keep_k=8"],
    )

    assert cmd[:2] == ["uv", "run"]
    assert "praxis-eval" in cmd
    assert "policy=bar_boat_pali" in cmd
    assert "env.type=libero" in cmd
    assert "env.task='libero_spatial,libero_object,libero_goal,libero_10'" in cmd
    assert "policy.pretrained_path=" + str(ckpt) in cmd
    assert "eval.output_dir=outputs/eval/run" in cmd
    assert "eval.policy_kwargs.decode_keep_k=8" in cmd
    assert "+eval.policy_kwargs.decode_keep_k=8" not in cmd


def test_build_eval_command_resolves_train_config_interpolations(
    tmp_path: Path,
) -> None:
    mod = _load_fasrc_launch_module()
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.yaml").write_text(
        "\n".join(
            [
                "device_name: cuda",
                "policy:",
                "  name: bar_boat_pali",
                "dataset:",
                "  name: bridge_single_view",
                "device: ${device_name}",
            ]
        )
    )

    cmd = mod.build_eval_command(
        str(ckpt),
        "outputs/eval/run",
        num_eval_per_task=10,
        num_parallel_env=40,
        record_episodes_per_task=0,
        eval_overrides=[],
    )

    assert "env.type=simpler" in cmd
    assert "env.task=bridge" in cmd
    assert "device=cuda" in cmd


def test_submit_or_print_stage_calls_keyword_only_helpers() -> None:
    mod = _load_fasrc_launch_module()
    calls: dict[str, Any] = {}

    class _DummyJob:
        job_id = "12345"

    def _print_slurm_params(
        *,
        gpus: int,
        mem: str,
        partition: str,
        account: str,
        num_nodes: int,
        extra: str = "",
    ) -> None:
        calls["print"] = {
            "gpus": gpus,
            "mem": mem,
            "partition": partition,
            "account": account,
            "num_nodes": num_nodes,
            "extra": extra,
        }

    def _make_executor(
        *,
        run_name: str,
        gpus: int,
        mem: str,
        partition: str,
        account: str,
        num_nodes: int,
    ) -> str:
        calls["executor"] = {
            "run_name": run_name,
            "gpus": gpus,
            "mem": mem,
            "partition": partition,
            "account": account,
            "num_nodes": num_nodes,
        }
        return "executor"

    def _submit_job(executor: str) -> _DummyJob:
        calls["submit"] = executor
        return _DummyJob()

    job = mod._submit_or_print_stage(
        label="eval",
        run_name="eval_keepk_8",
        command=["uv", "run", "praxis-eval"],
        gpus=1,
        mem="64G",
        partition="kempner_h100",
        account="fasrc_kempner_account",
        num_nodes=1,
        dry_run=False,
        print_slurm_params=_print_slurm_params,
        make_executor=_make_executor,
        submit_job=_submit_job,
    )

    assert job is not None
    assert job.job_id == "12345"
    assert calls["submit"] == "executor"
    assert calls["executor"]["run_name"] == "eval_keepk_8"
    assert calls["print"]["partition"] == "kempner_h100"


def test_arg_parser_requires_policy_ckpt_and_eval_output_dir_for_eval() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()

    with pytest.raises(SystemExit) as exc1:
        parser.parse_args(["eval"])
    assert exc1.value.code == 2

    with pytest.raises(SystemExit) as exc2:
        parser.parse_args(["eval", "--policy-ckpt", "ckpt/pretrained_model"])
    assert exc2.value.code == 2


def test_arg_parser_rejects_removed_stage_cli() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()

    for argv in (
        ["--stage", "both"],
        ["both", "--dataset", "libero"],
        ["train-policy", "--stage", "policy", "--dataset", "libero"],
    ):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 2


def test_validate_args_eval_monitor_rejects_non_positive_train_job_id() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "eval-monitor",
            "--train-dir",
            "outputs/train_policy/run",
            "--train-job-id",
            "0",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        mod._validate_args(parser, args)
    assert exc.value.code == 2


def test_build_arg_parser_sets_git_worktree_defaults() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(["train-policy", "--codec", "boat", "--dataset", "libero"])
    assert args.git_ref is None
    assert args.git_worktree_dir == "outputs/worktree"


def test_validate_args_defaults_to_bar_oat_pali() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "train-policy",
            "--dataset",
            "libero",
            "--tokenizer-ckpt",
            "tokenizer_path",
        ]
    )

    mod._validate_args(parser, args)

    assert args.method == "bar"
    assert args.codec == "oat"
    assert args.backbone == "pali"


def test_validate_args_allows_codecless_fm_policy_stage() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "train-policy",
            "--method",
            "fm",
            "--backbone",
            "pali",
            "--dataset",
            "libero",
        ]
    )

    mod._validate_args(parser, args)

    assert args.method == "fm"
    assert args.codec is None
    assert args.backbone == "pali"


def test_validate_args_rejects_codec_for_fm_policy() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "train-policy",
            "--method",
            "fm",
            "--codec",
            "oat",
            "--backbone",
            "pali",
            "--dataset",
            "libero",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        mod._validate_args(parser, args)

    assert exc.value.code == 2


def test_validate_args_rejects_fm_tokenizer_checkpoint() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "train-policy",
            "--method",
            "fm",
            "--backbone",
            "pali",
            "--dataset",
            "libero",
            "--tokenizer-ckpt",
            "outputs/train_codecs/run/checkpoints/latest/pretrained_model",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        mod._validate_args(parser, args)

    assert exc.value.code == 2


def test_build_eval_monitor_dependency_skips_dependency_when_training_running(
    monkeypatch,
) -> None:
    mod = _load_fasrc_launch_module()
    monkeypatch.setattr(mod, "_query_slurm_job_state", lambda _job_id: "RUNNING")
    extra_slurm, dry_run_extra, submitted_suffix = mod._build_eval_monitor_dependency(
        12345
    )
    assert extra_slurm == {}
    assert "no dependency" in dry_run_extra
    assert submitted_suffix == ""


def test_build_eval_monitor_dependency_adds_after_when_training_not_running(
    monkeypatch,
) -> None:
    mod = _load_fasrc_launch_module()
    monkeypatch.setattr(mod, "_query_slurm_job_state", lambda _job_id: "PENDING")
    extra_slurm, dry_run_extra, submitted_suffix = mod._build_eval_monitor_dependency(
        67890
    )
    assert extra_slurm == {"dependency": "after:67890"}
    assert "dependency=after:67890" in dry_run_extra
    assert "PENDING" in dry_run_extra
    assert "depends on 67890" in submitted_suffix


def test_run_training_job_multi_node_uses_srun(monkeypatch, tmp_path: Path) -> None:
    mod = _load_fasrc_launch_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        return None

    monkeypatch.setattr(mod.os, "chdir", lambda _: None)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    mod.run_training_job(
        command=[
            "uv",
            "run",
            "torchrun",
            "--node_rank=__SLURM_PROCID__",
            "--master_addr=__MASTER_ADDR__",
            "-m",
            "praxis.scripts.train",
        ],
        project_root=str(tmp_path),
        num_nodes=2,
    )

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[0] == "srun"
    assert "--nodes=2" in cmd
    assert "--ntasks=2" in cmd
    assert "--ntasks-per-node=1" in cmd
    assert cmd[-2:] and cmd[-2] == "-lc"
    assert '--node_rank="${SLURM_PROCID}"' in cmd[-1]
    assert '--master_addr="${MASTER_ADDR}"' in cmd[-1]
    assert "'--node_rank=${SLURM_PROCID}'" not in cmd[-1]
    assert "'--master_addr=${MASTER_ADDR}'" not in cmd[-1]
    assert kwargs.get("check") is True


def test_validate_args_requires_gpus_when_gres_is_set() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "train-tokenizer",
            "--codec",
            "oat",
            "--dataset",
            "libero",
            "--gres",
            "gpu:nvidia_a100-sxm4-80gb:1",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        mod._validate_args(parser, args)

    assert exc.value.code == 2


def test_validate_args_rejects_gres_gpu_count_mismatch() -> None:
    mod = _load_fasrc_launch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "train-tokenizer",
            "--codec",
            "oat",
            "--dataset",
            "libero",
            "--gpus",
            "2",
            "--gres",
            "gpu:nvidia_a100-sxm4-80gb:1",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        mod._validate_args(parser, args)

    assert exc.value.code == 2


def test_parse_gres_gpu_count_handles_slurm_default_one_gpu_forms() -> None:
    mod = _load_fasrc_launch_module()

    assert mod._parse_gres_gpu_count("gpu") == 1
    assert mod._parse_gres_gpu_count("gpu:h100") == 1
    assert mod._parse_gres_gpu_count("gpu:h100:2,gpu:a100") == 3


def test_make_executor_requests_one_task_per_node_for_multinode_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_fasrc_launch_module()

    parameters = _make_executor_with_captured_parameters(
        monkeypatch,
        mod,
        tmp_path,
        run_name="multi-node-test",
        gpus=2,
        mem="64G",
        num_nodes=2,
        cpus_per_task=8,
    )

    assert parameters["slurm_nodes"] == 2
    assert parameters["slurm_ntasks_per_node"] == 1
    assert parameters["slurm_use_srun"] is False
    assert "ntasks" not in parameters["slurm_additional_parameters"]


def test_make_executor_adds_mail_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_fasrc_launch_module()

    parameters = _make_executor_with_captured_parameters(
        monkeypatch,
        mod,
        tmp_path,
        run_name="mail-test",
    )
    additional = parameters["slurm_additional_parameters"]
    assert "mail_type" not in additional
    assert "mail_user" not in additional

    parameters = _make_executor_with_captured_parameters(
        monkeypatch,
        mod,
        tmp_path,
        run_name="mail-test",
        mail_type="END",
        mail_user="user@example.com",
    )
    additional = parameters["slurm_additional_parameters"]
    assert additional["mail_type"] == "END"
    assert additional["mail_user"] == "user@example.com"


def test_run_training_job_auto_appends_resume_checkpoint_when_latest_exists(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_fasrc_launch_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    run_dir = tmp_path / "run"
    latest = run_dir / "checkpoints" / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        return None

    monkeypatch.setattr(mod.os, "chdir", lambda _: None)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    mod.run_training_job(
        command=[
            "uv",
            "run",
            "torchrun",
            "-m",
            "praxis.scripts.train",
            f"output_dir={run_dir}",
        ],
        project_root=str(tmp_path),
        num_nodes=1,
        auto_resume_on_requeue=True,
    )

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert "run_mode=resume" in cmd
    assert f"resume_checkpoint={latest}" in cmd
    assert kwargs.get("check") is True


def test_run_training_job_respects_explicit_resume_override(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_fasrc_launch_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    run_dir = tmp_path / "run"
    latest = run_dir / "checkpoints" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    explicit = tmp_path / "manual_ckpt"

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        return None

    monkeypatch.setattr(mod.os, "chdir", lambda _: None)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    mod.run_training_job(
        command=[
            "uv",
            "run",
            "torchrun",
            "-m",
            "praxis.scripts.train",
            f"output_dir={run_dir}",
            f"resume_checkpoint={explicit}",
        ],
        project_root=str(tmp_path),
        num_nodes=1,
        auto_resume_on_requeue=True,
    )

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert "run_mode=resume" in cmd
    assert f"resume_checkpoint={explicit}" in cmd
    assert f"resume_checkpoint={latest}" not in cmd
    assert kwargs.get("check") is True


def test_run_training_job_no_auto_resume_on_non_requeue(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_fasrc_launch_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    run_dir = tmp_path / "run"
    latest = run_dir / "checkpoints" / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        return None

    monkeypatch.setattr(mod.os, "chdir", lambda _: None)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    mod.run_training_job(
        command=[
            "uv",
            "run",
            "torchrun",
            "-m",
            "praxis.scripts.train",
            f"output_dir={run_dir}",
        ],
        project_root=str(tmp_path),
        num_nodes=1,
        auto_resume_on_requeue=False,
    )

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert f"resume_checkpoint={latest}" not in cmd
    assert kwargs.get("check") is True


def test_run_training_job_with_git_ref_uses_unique_worktree_and_cleans_up(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_fasrc_launch_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []
    chdirs: list[str] = []

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        return None

    monkeypatch.setattr(mod.os, "chdir", lambda p: chdirs.append(str(p)))
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    mod.run_training_job(
        command=[
            "uv",
            "run",
            "torchrun",
            "-m",
            "praxis.scripts.train",
            "output_dir=outputs/train_policy/run",
        ],
        project_root=str(tmp_path),
        num_nodes=1,
        git_ref="main~1",
        git_worktree_dir="outputs/worktree",
    )

    assert len(calls) == 3
    git_add, _, git_remove = calls
    assert git_add[0][:6] == [
        "git",
        "-C",
        str(tmp_path),
        "worktree",
        "add",
        "--detach",
    ]
    worktree_path = Path(git_add[0][6])
    assert git_add[0][7] == "main~1"
    assert "outputs/worktree" in str(worktree_path)
    assert git_remove[0][:6] == [
        "git",
        "-C",
        str(tmp_path),
        "worktree",
        "remove",
        "--force",
    ]
    assert git_remove[0][6] == str(worktree_path)
    assert any(str(worktree_path) == p for p in chdirs)
    assert any(str(tmp_path) == p for p in chdirs)
    run_cmd = calls[1][0]
    assert f"output_dir={tmp_path / 'outputs' / 'train_policy' / 'run'}" in run_cmd


def test_link_worktree_outputs_exposes_repo_artifacts(tmp_path: Path) -> None:
    mod = _load_fasrc_launch_module()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    mod._link_worktree_outputs(worktree_path=worktree, project_root=tmp_path)

    linked_outputs = worktree / "outputs"
    assert linked_outputs.is_symlink()
    assert linked_outputs.resolve() == outputs.resolve()


def test_run_training_job_preserves_relative_tokenizer_override_path(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_fasrc_launch_module()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        return None

    monkeypatch.setattr(mod.os, "chdir", lambda _: None)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    mod.run_training_job(
        command=[
            "uv",
            "run",
            "torchrun",
            "-m",
            "praxis.scripts.train",
            "output_dir=outputs/train_policy/run",
            "policy.action_tokenizer.pretrained_path=outputs/train_codecs/run/pretrained_model",
        ],
        project_root=str(tmp_path),
        num_nodes=1,
    )

    assert len(calls) == 1
    cmd, _ = calls[0]
    assert (
        "policy.action_tokenizer.pretrained_path=outputs/train_codecs/run/pretrained_model"
        in cmd
    )
    assert f"output_dir={tmp_path / 'outputs' / 'train_policy' / 'run'}" in cmd


def test_parse_partition_inputs_supports_csv_and_repeated() -> None:
    mod = _load_fasrc_launch_module()
    out = mod._parse_partition_inputs(["kempner_h100,gpu_requeue", "kempner"])
    assert out == ["kempner_h100", "gpu_requeue", "kempner"]


def test_parse_partition_inputs_rejects_empty_tokens() -> None:
    mod = _load_fasrc_launch_module()

    for raw in ([""], ["gpu,"], [",gpu"], ["gpu,,kempner"]):
        with pytest.raises(ValueError, match="cannot be empty"):
            mod._parse_partition_inputs(raw)


def test_is_requeue_partition_spec_detects_any_requeue_partition() -> None:
    mod = _load_fasrc_launch_module()
    assert mod._is_requeue_partition_spec("gpu_requeue") is True
    assert mod._is_requeue_partition_spec("kempner_h100,kempner_requeue") is True
    assert mod._is_requeue_partition_spec("kempner_h100") is False


def test_validate_partition_list_rejects_unknown() -> None:
    mod = _load_fasrc_launch_module()
    with pytest.raises(ValueError, match="Unsupported partition"):
        mod._validate_partition_list(["kempner_h100", "not_real_partition"])


def test_validate_partition_list_rejects_kempner_requeue_mixed() -> None:
    mod = _load_fasrc_launch_module()
    with pytest.raises(ValueError, match="standalone partition"):
        mod._validate_partition_list(["kempner_requeue", "kempner"])


def test_validate_partition_list_rejects_gpu_test_mixed() -> None:
    mod = _load_fasrc_launch_module()
    with pytest.raises(ValueError, match="standalone partition"):
        mod._validate_partition_list(["gpu_test", "gpu_requeue"])


def test_resolve_account_for_kempner_partition() -> None:
    mod = _load_fasrc_launch_module()
    account = mod._resolve_account_for_partition_spec("kempner_h100")
    assert account == "fasrc_kempner_account"


def test_resolve_account_for_non_kempner_partition() -> None:
    mod = _load_fasrc_launch_module()
    account = mod._resolve_account_for_partition_spec("gpu_requeue")
    assert account == "fasrc_account"


def test_resolve_account_rejects_mixed_partition_families() -> None:
    mod = _load_fasrc_launch_module()
    with pytest.raises(ValueError, match="mixes account families"):
        mod._resolve_account_for_partition_spec("kempner_h100,gpu_requeue")
