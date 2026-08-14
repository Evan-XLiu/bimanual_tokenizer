# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from praxis.serving.model_identity import build_policy_serving_identity
from scripts.robotwin.stable2_evaluator import (
    _load_policy_plugin,
    _resolve_ffmpeg_executable,
    _validate_server_model_info,
    run_controlled_rollouts,
    run_stable2_eval,
)


class _UnstableError(Exception):
    pass


class _FakeStdin:
    def __init__(self) -> None:
        self.closed = False
        self.writes = 0

    def write(self, _data: bytes) -> None:
        self.writes += 1

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, command: list[str]) -> None:
        self.stdin = _FakeStdin()
        self.waited = False
        Path(command[-1]).write_bytes(b"encoded-video")

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.waited = True
        return 0


class _TaskEnv:
    def __init__(self) -> None:
        self.plan_success = True
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 2
        self.render_freq = 0
        self.suc = 0
        self.test_num = 0
        self.eval_video_path = None
        self.setup_seeds: list[int] = []
        self.instruction = ""
        self.close_calls = 0
        self.unstable_seeds = {100000}
        self.unexpected_error: Exception | None = None
        self.expert_errors: dict[int, Exception] = {}
        self.current_seed: int | None = None
        self.expert_success = True

    def setup_demo(self, *, now_ep_num, seed, **args):
        del now_ep_num
        self.setup_seeds.append(seed)
        if self.unexpected_error is not None:
            raise self.unexpected_error
        if seed in self.unstable_seeds:
            raise _UnstableError("rejected seed")
        self.current_seed = seed
        self.eval_video_path = Path(args["eval_video_save_dir"])
        self.take_action_cnt = 0
        self.eval_success = False

    def play_once(self):
        if self.current_seed in self.expert_errors:
            raise self.expert_errors[self.current_seed]
        return {"info": {"object": "apple"}}

    def close_env(self, **_kwargs):
        self.close_calls += 1

    def check_success(self):
        return self.expert_success

    def set_instruction(self, *, instruction):
        self.instruction = instruction

    def _set_eval_video_ffmpeg(self, process):
        self.eval_video_ffmpeg = process

    def _del_eval_video_ffmpeg(self):
        return None

    def get_obs(self):
        return {
            "observation": {
                "head_camera": {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                }
            }
        }


def _write_checkpoint_artifacts(root: Path, config: dict, policy_io: dict) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config))
    (root / "policy_io.json").write_text(json.dumps(policy_io))
    (root / "norm_stats.pt").write_bytes(b"stats")
    (root / "model.safetensors").write_bytes(b"weights")


def _write_robotwin_policy(tmp_path: Path) -> Path:
    tokenizer = tmp_path / "bin" / "pretrained_model"
    _write_checkpoint_artifacts(
        tokenizer,
        {
            "name": "bin",
            "action_dim": 14,
            "action_horizon": 32,
            "num_bins": 256,
        },
        {
            "obs_features": [],
            "action_feature": {"shape": [14]},
            "n_action_steps": 32,
        },
    )
    policy = tmp_path / "policy" / "pretrained_model"
    _write_checkpoint_artifacts(
        policy,
        {
            "name": "bar_bin_qwen",
            "action_dim": 14,
            "action_horizon": 32,
            "n_action_steps": 16,
            "state_prefix_num_bins": 256,
            "use_lora": False,
            "freeze_vision_encoder": False,
            "vlm_model_name": "Qwen/Qwen3-VL-2B-Instruct",
            "action_tokenizer": {
                "name": "bin",
                "kind": "tokens",
                "pretrained_path": str(tokenizer),
            },
        },
        {
            "obs_features": [
                {"key": "observation.state", "shape": [14]},
                {"key": "observation.images.cam_high", "shape": [3, 240, 320]},
                {
                    "key": "observation.images.cam_left_wrist",
                    "shape": [3, 240, 320],
                },
                {
                    "key": "observation.images.cam_right_wrist",
                    "shape": [3, 240, 320],
                },
            ],
            "action_feature": {"shape": [14]},
            "n_action_steps": 32,
        },
    )
    return policy


def test_ffmpeg_resolver_uses_bundled_binary_when_path_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / "ffmpeg"
    bundled.write_text("binary")
    bundled.chmod(0o755)
    monkeypatch.setattr("scripts.robotwin.stable2_evaluator.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "imageio_ffmpeg.get_ffmpeg_exe", lambda: str(bundled)
    )

    assert _resolve_ffmpeg_executable() == str(bundled.resolve())


def test_policy_plugin_loads_required_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = tmp_path / "policy" / "praxis_ar"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        "from .deploy_policy import close_model, eval, get_model, reset_model\n"
    )
    (plugin / "deploy_policy.py").write_text(
        "def get_model(*args): pass\n"
        "def eval(*args): pass\n"
        "def reset_model(*args): pass\n"
        "def close_model(*args): pass\n"
    )
    config = plugin / "deploy_policy.yml"
    config.write_text("instruction_type: unseen\n")
    monkeypatch.delitem(sys.modules, "praxis_ar", raising=False)
    monkeypatch.delitem(sys.modules, "praxis_ar.deploy_policy", raising=False)

    _load_policy_plugin(config)
    assert callable(sys.modules["praxis_ar"].get_model)


def test_controlled_rollout_records_accepted_seed_and_terminal_outcome(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "eval"
    resets: list[bool] = []

    def policy_eval(task_env, _model, _observation):
        frame = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
        task_env.eval_video_ffmpeg.stdin.write(frame)
        task_env.take_action_cnt += 1
        task_env.eval_success = True
        task_env.eval_video_ffmpeg.stdin.write(frame)

    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda *_args: [
            {"unseen": ["put the apple in the bowl"]}
        ],
        eval_function_decorator=lambda _policy, name: (
            policy_eval if name == "eval" else lambda _model: resets.append(True)
        ),
    )
    env = _TaskEnv()
    args = {
        "task_name": "pick_apple",
        "policy_name": "praxis_ar",
        "task_config": "demo_clean",
        "ckpt_setting": "smoke",
        "clear_cache_freq": 5,
        "render_freq": 0,
        "eval_video_log": True,
    }

    _, successes = run_controlled_rollouts(
        official=official,
        task_name="pick_apple",
        task_env=env,
        args=args,
        model=object(),
        start_seed=100000,
        episodes=1,
        video_size="2x2",
        instruction_type="unseen",
        output_dir=output_dir,
        max_seed_attempts=50,
        popen_factory=_FakeProcess,
    )

    assert successes == 1
    assert resets == [True]
    assert env.setup_seeds == [100000, 100001, 100001]
    record = json.loads((output_dir / "episode_records.jsonl").read_text())
    assert record["episode_index"] == 0
    assert record["seed"] == 100001
    assert record["success"] is True
    assert record["terminal_reason"] == "success"
    assert Path(record["video_path"]).name == "episode_0000_success.mp4"
    assert Path(record["video_path"]).is_file()
    attempts = [
        json.loads(line)
        for line in (output_dir / "seed_attempts.jsonl").read_text().splitlines()
    ]
    assert [(item["seed"], item["accepted"]) for item in attempts] == [
        (100000, False),
        (100001, True),
    ]


def test_instruction_selection_is_deterministic_and_seeded(tmp_path: Path) -> None:
    candidates = [f"instruction-{index}" for index in range(5)]

    def policy_eval(task_env, _model, _observation):
        frame = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
        task_env.eval_video_ffmpeg.stdin.write(frame)
        task_env.take_action_cnt += 1
        task_env.eval_success = True
        task_env.eval_video_ffmpeg.stdin.write(frame)

    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda *_args: [{"unseen": candidates}],
        eval_function_decorator=lambda _policy, name: (
            policy_eval if name == "eval" else lambda _model: None
        ),
    )

    def selected_index(name: str, instruction_seed: int) -> int:
        env = _TaskEnv()
        env.unstable_seeds = set()
        output = tmp_path / name
        run_controlled_rollouts(
            official=official,
            task_name="pick_apple",
            task_env=env,
            args={
                "task_name": "pick_apple",
                "policy_name": "praxis_ar",
                "clear_cache_freq": 5,
                "render_freq": 0,
                "eval_video_log": True,
            },
            model=object(),
            start_seed=100000,
            episodes=1,
            video_size="2x2",
            instruction_type="unseen",
            instruction_rng_seed=instruction_seed,
            output_dir=output,
            max_seed_attempts=1,
            popen_factory=_FakeProcess,
        )
        record = json.loads((output / "episode_records.jsonl").read_text())
        assert record["instruction_rng_seed"] == instruction_seed
        assert record["instruction"] == candidates[record["instruction_index"]]
        return int(record["instruction_index"])

    assert selected_index("same-a", 0) == selected_index("same-b", 0)
    assert selected_index("different", 1) != selected_index("same-c", 0)


def test_controlled_rollout_fails_at_seed_attempt_limit(tmp_path: Path) -> None:
    official = SimpleNamespace(
        UnStableError=_UnstableError,
        eval_function_decorator=lambda *_args: lambda *_unused: None,
    )
    env = _TaskEnv()
    env.unstable_seeds = set()
    env.expert_success = False

    with pytest.raises(RuntimeError, match="exhausted max_seed_attempts=2"):
        run_controlled_rollouts(
            official=official,
            task_name="pick_apple",
            task_env=env,
            args={
                "task_name": "pick_apple",
                "policy_name": "praxis_ar",
                "clear_cache_freq": 5,
                "render_freq": 0,
                "eval_video_log": True,
            },
            model=object(),
            start_seed=100000,
            episodes=1,
            video_size="2x2",
            instruction_type="unseen",
            output_dir=tmp_path / "eval",
            max_seed_attempts=2,
            popen_factory=_FakeProcess,
        )

    attempts = [
        json.loads(line)
        for line in (tmp_path / "eval" / "seed_attempts.jsonl").read_text().splitlines()
    ]
    assert [item["seed"] for item in attempts] == [100000, 100001]
    assert all(item["reason"] == "expert_plan_failed" for item in attempts)


def test_controlled_rollout_skips_generic_official_precheck_error(
    tmp_path: Path,
) -> None:
    official = SimpleNamespace(
        UnStableError=_UnstableError,
        eval_function_decorator=lambda *_args: lambda *_unused: None,
    )
    env = _TaskEnv()
    env.unstable_seeds = set()
    env.unexpected_error = OSError("renderer failed")

    with pytest.raises(RuntimeError, match="exhausted max_seed_attempts=50"):
        run_controlled_rollouts(
            official=official,
            task_name="pick_apple",
            task_env=env,
            args={
                "task_name": "pick_apple",
                "policy_name": "praxis_ar",
                "clear_cache_freq": 5,
                "render_freq": 0,
                "eval_video_log": True,
            },
            model=object(),
            start_seed=100000,
            episodes=1,
            video_size="2x2",
            instruction_type="unseen",
            output_dir=tmp_path / "eval",
            max_seed_attempts=50,
            popen_factory=_FakeProcess,
        )

    assert env.setup_seeds == list(range(100000, 100050))
    attempts = [
        json.loads(line)
        for line in (tmp_path / "eval" / "seed_attempts.jsonl").read_text().splitlines()
    ]
    assert len(attempts) == 50
    assert all(item["reason"] == "expert_precheck_error" for item in attempts)


def test_controlled_rollout_skips_official_expert_play_error(tmp_path: Path) -> None:
    def policy_eval(task_env, _model, _observation):
        frame = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
        task_env.eval_video_ffmpeg.stdin.write(frame)
        task_env.take_action_cnt += 1
        task_env.eval_success = True
        task_env.eval_video_ffmpeg.stdin.write(frame)

    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda *_args: [
            {"unseen": ["put the apple in the bowl"]}
        ],
        eval_function_decorator=lambda _policy, name: (
            policy_eval if name == "eval" else lambda _model: None
        ),
    )
    env = _TaskEnv()
    env.unstable_seeds = set()
    env.expert_errors[100000] = TypeError("grasp pose is unavailable")
    output_dir = tmp_path / "eval"

    _, successes = run_controlled_rollouts(
        official=official,
        task_name="pick_apple",
        task_env=env,
        args={
            "task_name": "pick_apple",
            "policy_name": "praxis_ar",
            "clear_cache_freq": 5,
            "render_freq": 0,
            "eval_video_log": True,
        },
        model=object(),
        start_seed=100000,
        episodes=1,
        video_size="2x2",
        instruction_type="unseen",
        output_dir=output_dir,
        max_seed_attempts=2,
        popen_factory=_FakeProcess,
    )

    assert successes == 1
    assert env.setup_seeds == [100000, 100001, 100001]
    attempts = [
        json.loads(line)
        for line in (output_dir / "seed_attempts.jsonl").read_text().splitlines()
    ]
    assert attempts[0] == {
        "accepted": False,
        "attempt_index": 0,
        "detail": "grasp pose is unavailable",
        "error_type": "TypeError",
        "reason": "expert_precheck_error",
        "seed": 100000,
    }
    assert attempts[1]["accepted"] is True


def test_rollout_failure_reaps_video_and_closes_environment(tmp_path: Path) -> None:
    processes: list[_FakeProcess] = []

    def popen_factory(command: list[str]) -> _FakeProcess:
        process = _FakeProcess(command)
        processes.append(process)
        return process

    def failing_eval(*_args) -> None:
        raise ConnectionError("policy RPC failed")

    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda *_args: [
            {"unseen": ["put the apple in the bowl"]}
        ],
        eval_function_decorator=lambda _policy, name: (
            failing_eval if name == "eval" else lambda _model: None
        ),
    )
    env = _TaskEnv()
    env.unstable_seeds = set()
    output_dir = tmp_path / "eval"

    with pytest.raises(ConnectionError, match="policy RPC failed"):
        run_controlled_rollouts(
            official=official,
            task_name="pick_apple",
            task_env=env,
            args={
                "task_name": "pick_apple",
                "policy_name": "praxis_ar",
                "clear_cache_freq": 5,
                "render_freq": 0,
                "eval_video_log": True,
            },
            model=object(),
            start_seed=100000,
            episodes=1,
            video_size="2x2",
            instruction_type="unseen",
            output_dir=output_dir,
            max_seed_attempts=50,
            popen_factory=popen_factory,
        )

    assert len(processes) == 1
    assert processes[0].stdin.closed is True
    assert processes[0].waited is True
    assert env.close_calls == 2
    assert (output_dir / "episode_0000_incomplete.mp4").is_file()


def test_video_start_failure_still_closes_accepted_environment(tmp_path: Path) -> None:
    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda *_args: [
            {"unseen": ["put the apple in the bowl"]}
        ],
        eval_function_decorator=lambda *_args: lambda *_unused: None,
    )
    env = _TaskEnv()
    env.unstable_seeds = set()

    with pytest.raises(RuntimeError, match="ffmpeg could not start"):
        run_controlled_rollouts(
            official=official,
            task_name="pick_apple",
            task_env=env,
            args={
                "task_name": "pick_apple",
                "policy_name": "praxis_ar",
                "clear_cache_freq": 5,
                "render_freq": 0,
                "eval_video_log": True,
            },
            model=object(),
            start_seed=100000,
            episodes=1,
            video_size="2x2",
            instruction_type="unseen",
            output_dir=tmp_path / "eval",
            max_seed_attempts=1,
            popen_factory=lambda _command: (_ for _ in ()).throw(
                RuntimeError("ffmpeg could not start")
            ),
        )

    assert env.close_calls == 2


def test_video_attachment_failure_reaps_ffmpeg_and_closes_environment(
    tmp_path: Path,
) -> None:
    processes: list[_FakeProcess] = []

    def popen_factory(command: list[str]) -> _FakeProcess:
        process = _FakeProcess(command)
        processes.append(process)
        return process

    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda *_args: [
            {"unseen": ["put the apple in the bowl"]}
        ],
        eval_function_decorator=lambda *_args: lambda *_unused: None,
    )
    env = _TaskEnv()
    env.unstable_seeds = set()
    env._set_eval_video_ffmpeg = lambda _process: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("video attachment failed")
    )

    with pytest.raises(RuntimeError, match="video attachment failed"):
        run_controlled_rollouts(
            official=official,
            task_name="pick_apple",
            task_env=env,
            args={
                "task_name": "pick_apple",
                "policy_name": "praxis_ar",
                "clear_cache_freq": 5,
                "render_freq": 0,
                "eval_video_log": True,
            },
            model=object(),
            start_seed=100000,
            episodes=1,
            video_size="2x2",
            instruction_type="unseen",
            output_dir=tmp_path / "eval",
            max_seed_attempts=1,
            popen_factory=popen_factory,
        )

    assert env.close_calls == 2
    assert processes[0].stdin.closed is True
    assert processes[0].waited is True
    assert (tmp_path / "eval" / "episode_0000_incomplete.mp4").is_file()


def test_stable2_eval_uses_official_runtime_and_matches_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robotwin_root = tmp_path / "RoboTwin"
    (robotwin_root / "script").mkdir(parents=True)
    (robotwin_root / "script" / "eval_policy.py").write_text("# pinned\n")
    (robotwin_root / "task_config").mkdir()
    (robotwin_root / "task_config" / "demo_clean.yml").write_text(
        "eval_video_log: true\n"
        "render_freq: 1\n"
        "clear_cache_freq: 5\n"
        "camera:\n"
        "  head_camera_type: default\n"
        "embodiment: [aloha]\n"
    )
    configs = robotwin_root / "envs" / "configs"
    configs.mkdir(parents=True)
    (configs / "_camera_config.yml").write_text("default: {h: 2, w: 2}\n")
    (configs / "_embodiment_config.yml").write_text(
        "aloha: {file_path: embodiment/aloha}\n"
    )
    plugin_config = tmp_path / "deploy_policy.yml"
    plugin_config.write_text("instruction_type: unseen\n")

    checkpoint = _write_robotwin_policy(tmp_path)
    checkpoint_identity = build_policy_serving_identity(checkpoint)
    model = SimpleNamespace(
        server_model_info={
            "serving_artifact": checkpoint_identity,
            "source": {"git_commit": "a" * 40, "git_dirty": False},
        }
    )
    main_calls: list[dict] = []
    close_calls: list[object] = []

    def decorated(_policy, name):
        if name == "get_model":
            return lambda _config: model
        if name == "close_model":
            return lambda value: close_calls.append(value)
        return lambda *_args: None

    official = SimpleNamespace(
        CONFIGS_PATH="envs/configs",
        get_embodiment_config=lambda _path: {
            "arm_joints_name": [["left"], ["right"]]
        },
        class_decorator=lambda _task: SimpleNamespace(step_lim=None),
        eval_function_decorator=decorated,
        main=lambda config: main_calls.append(config),
    )
    monkeypatch.setattr(
        "scripts.robotwin.stable2_evaluator._load_official_module",
        lambda _root: official,
    )
    monkeypatch.setattr(
        "scripts.robotwin.stable2_evaluator._load_policy_plugin", lambda _path: None
    )

    def fake_rollouts(**kwargs):
        output = Path(kwargs["output_dir"])
        kwargs["task_env"].step_lim = 400
        assert kwargs["output_dir_precreated"] is True
        assert output.is_dir()
        (output / "episode_records.jsonl").write_text("")
        return kwargs["start_seed"] + 1, 0

    monkeypatch.setattr(
        "scripts.robotwin.stable2_evaluator.run_controlled_rollouts",
        fake_rollouts,
    )

    result = run_stable2_eval(
        robotwin_root=robotwin_root,
        plugin_config=plugin_config,
        task_name="pick_apple",
        task_config="demo_clean",
        episodes_per_task=1,
        record_episodes_per_task=1,
        seed=0,
        policy_host="localhost",
        policy_port=50051,
        output_dir=tmp_path / "eval",
        run_id="smoke",
        max_seed_attempts=50,
        expected_checkpoint=checkpoint,
    )

    assert main_calls == []
    assert close_calls == [model]
    manifest = json.loads(Path(result["run_manifest_path"]).read_text())
    assert manifest["status"] == "rollout_completed"
    assert manifest["expected_checkpoint"]["checkpoint"]["path"] == str(
        checkpoint.resolve()
    )
    assert manifest["server_model_info"]["serving_artifact"][
        "artifact_sha256"
    ] == checkpoint_identity["artifact_sha256"]


def test_stable2_eval_preserves_failed_manifest_and_closes_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robotwin_root = tmp_path / "RoboTwin"
    (robotwin_root / "script").mkdir(parents=True)
    (robotwin_root / "script" / "eval_policy.py").write_text("# pinned\n")
    (robotwin_root / "task_config").mkdir()
    (robotwin_root / "task_config" / "demo_clean.yml").write_text(
        "eval_video_log: true\n"
    )
    plugin_config = tmp_path / "deploy_policy.yml"
    plugin_config.write_text("instruction_type: unseen\n")
    checkpoint = _write_robotwin_policy(tmp_path)
    checkpoint_identity = build_policy_serving_identity(checkpoint)
    model = SimpleNamespace(
        server_model_info={
            "serving_artifact": checkpoint_identity,
            "source": {"git_commit": "a" * 40, "git_dirty": False},
        }
    )
    close_calls: list[object] = []
    official = SimpleNamespace(
        eval_function_decorator=lambda _policy, name: (
            (lambda value: close_calls.append(value))
            if name == "close_model"
            else (lambda *_args: None)
        )
    )
    monkeypatch.setattr(
        "scripts.robotwin.stable2_evaluator._load_official_module",
        lambda _root: official,
    )
    monkeypatch.setattr(
        "scripts.robotwin.stable2_evaluator._load_policy_plugin", lambda _path: None
    )
    monkeypatch.setattr(
        "scripts.robotwin.stable2_evaluator._prepare_official_runtime",
        lambda **_kwargs: (
            SimpleNamespace(step_lim=None),
            {"eval_video_log": True},
            model,
            "2x2",
            100000,
        ),
    )
    monkeypatch.setattr(
        "scripts.robotwin.stable2_evaluator.run_controlled_rollouts",
        lambda **_kwargs: (
            setattr(_kwargs["task_env"], "step_lim", 400),
            (_ for _ in ()).throw(ConnectionError("RPC failed")),
        )[1],
    )

    with pytest.raises(ConnectionError, match="RPC failed"):
        run_stable2_eval(
            robotwin_root=robotwin_root,
            plugin_config=plugin_config,
            task_name="pick_apple",
            task_config="demo_clean",
            episodes_per_task=1,
            record_episodes_per_task=1,
            seed=0,
            policy_host="localhost",
            policy_port=50051,
            output_dir=tmp_path / "eval",
            run_id="failure",
            max_seed_attempts=50,
            expected_checkpoint=checkpoint,
        )

    task_output = tmp_path / "eval" / "pick_apple"
    manifest = json.loads((task_output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["error_type"] == "builtins.ConnectionError"
    assert manifest["failure"]["message"] == "RPC failed"
    assert close_calls == [model]


def test_server_identity_rejects_different_checkpoint(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    with pytest.raises(RuntimeError, match="different checkpoint"):
        _validate_server_model_info(
            {
                "serving_artifact": {"checkpoint": {"path": str(tmp_path / "other")}},
            },
            expected_checkpoint=expected,
        )
