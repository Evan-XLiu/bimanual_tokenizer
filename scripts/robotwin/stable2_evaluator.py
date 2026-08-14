# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Auditable overlay for the pinned official RoboTwin stable_2.0 evaluator."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml


def _load_policy_plugin(plugin_config_path: Path) -> None:
    package_root = plugin_config_path.parent
    required = {
        "__init__.py": package_root / "__init__.py",
        "deploy_policy.py": package_root / "deploy_policy.py",
        "deploy_policy.yml": plugin_config_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Praxis RoboTwin policy plugin: {missing}.")

    expected_init = required["__init__.py"].resolve()
    existing = sys.modules.get("praxis_ar")
    if existing is not None:
        existing_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_file != expected_init:
            raise RuntimeError(
                "RoboTwin policy plugin is already shadowed by another praxis_ar: "
                f"{existing_file} != {expected_init}."
            )
    plugin_parent = str(package_root.parent.resolve())
    if plugin_parent in sys.path:
        sys.path.remove(plugin_parent)
    sys.path.insert(0, plugin_parent)
    importlib.invalidate_caches()
    module = ModuleType("praxis_ar")
    module.__file__ = str(expected_init)
    module.__package__ = "praxis_ar"
    module.__path__ = [str(package_root.resolve())]
    module.__spec__ = importlib.util.spec_from_file_location(
        "praxis_ar", expected_init, submodule_search_locations=module.__path__
    )
    deploy_module = ModuleType("praxis_ar.deploy_policy")
    deploy_module.__file__ = str(required["deploy_policy.py"].resolve())
    deploy_module.__package__ = "praxis_ar"
    deploy_module.__spec__ = importlib.util.spec_from_file_location(
        "praxis_ar.deploy_policy", required["deploy_policy.py"]
    )
    sys.modules["praxis_ar"] = module
    sys.modules["praxis_ar.deploy_policy"] = deploy_module
    try:
        exec(
            compile(
                required["deploy_policy.py"].read_bytes(),
                str(required["deploy_policy.py"]),
                "exec",
            ),
            deploy_module.__dict__,
        )
        exec(
            compile(
                required["__init__.py"].read_bytes(),
                str(required["__init__.py"]),
                "exec",
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop("praxis_ar.deploy_policy", None)
        sys.modules.pop("praxis_ar", None)
        raise
    resolved_init = Path(str(getattr(module, "__file__", ""))).resolve()
    if resolved_init != expected_init:
        raise RuntimeError(
            f"Loaded praxis_ar from {resolved_init}, expected {expected_init}."
        )
    resolved_deploy = Path(str(getattr(deploy_module, "__file__", ""))).resolve()
    if resolved_deploy != required["deploy_policy.py"].resolve():
        raise RuntimeError(
            "Loaded praxis_ar.deploy_policy from an unexpected path: "
            f"{resolved_deploy}."
        )
    for function_name in ("get_model", "eval", "reset_model", "close_model"):
        if not callable(getattr(module, function_name, None)):
            raise RuntimeError(
                f"Praxis RoboTwin plugin does not expose {function_name}()."
            )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _default_popen(command: list[str]):
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _resolve_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as exc:
        raise FileNotFoundError(
            "No ffmpeg executable is on PATH and imageio-ffmpeg is not installed."
        ) from exc
    bundled_ffmpeg = Path(get_ffmpeg_exe()).expanduser().resolve()
    if not bundled_ffmpeg.is_file() or not os.access(bundled_ffmpeg, os.X_OK):
        raise FileNotFoundError(
            "No executable ffmpeg binary is available from PATH or imageio-ffmpeg."
        )
    return str(bundled_ffmpeg)


def _start_video(
    *,
    path: Path,
    video_size: str,
    popen_factory: Callable[[list[str]], Any],
):
    return popen_factory(
        [
            _resolve_ffmpeg_executable(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(path),
        ]
    )


def _close_video_process(process: Any, *, timeout: float = 30.0) -> None:
    stdin = getattr(process, "stdin", None)
    if stdin is not None and not bool(getattr(stdin, "closed", False)):
        stdin.close()
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        return_code = wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
        try:
            return_code = wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            return_code = wait(timeout=5.0)
    if return_code not in {None, 0}:
        raise RuntimeError(f"ffmpeg exited with code {return_code}.")


def run_controlled_rollouts(
    *,
    official: ModuleType | Any,
    task_name: str,
    task_env: Any,
    args: dict[str, Any],
    model: Any,
    start_seed: int,
    episodes: int,
    video_size: str | None,
    instruction_type: str,
    output_dir: str | Path,
    max_seed_attempts: int,
    instruction_rng_seed: int | None = None,
    output_dir_precreated: bool = False,
    resume: bool = False,
    popen_factory: Callable[[list[str]], Any] = _default_popen,
) -> tuple[int, int]:
    """Run stable_2.0 episodes while recording accepted seeds and outcomes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if max_seed_attempts < episodes:
        raise ValueError(
            "max_seed_attempts must be at least the requested episode count."
        )
    if not bool(args.get("eval_video_log")):
        raise ValueError("RoboTwin task config must set eval_video_log=true.")
    if not video_size:
        raise ValueError(
            "video_size is required when full episode recording is enabled."
        )

    root = Path(output_dir).expanduser().resolve()
    if output_dir_precreated:
        if not root.is_dir():
            raise FileNotFoundError(f"Pre-created eval output directory is missing: {root}.")
    else:
        root.mkdir(parents=True, exist_ok=resume)
    args["eval_video_save_dir"] = root
    records_path = root / "episode_records.jsonl"
    attempts_path = root / "seed_attempts.jsonl"
    eval_func = official.eval_function_decorator(args["policy_name"], "eval")
    reset_func = official.eval_function_decorator(args["policy_name"], "reset_model")

    existing_records: list[dict[str, Any]] = []
    if resume and records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid episode record in {records_path}.")
                existing_records.append(value)
    completed_indices = {int(item["episode_index"]) for item in existing_records}
    if len(completed_indices) != len(existing_records) or any(
        index < 0 or index >= episodes for index in completed_indices
    ):
        raise ValueError(f"Duplicate or out-of-range episode index in {records_path}.")
    task_env.suc = sum(bool(item.get("success")) for item in existing_records)
    task_env.test_num = len(existing_records)
    accepted = len(existing_records)
    episode_index = next(
        (index for index in range(episodes) if index not in completed_indices), episodes
    )
    if resume and attempts_path.is_file():
        attempt_lines = [
            line for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        attempted_seeds = [
            int(json.loads(line)["seed"])
            for line in attempt_lines
        ]
        now_seed = max(attempted_seeds, default=int(start_seed) - 1) + 1
        seed_attempts = len(attempt_lines)
    else:
        now_seed = int(start_seed)
        seed_attempts = 0
    pending_path = root / "pending_episode.json"
    if resume and pending_path.is_file():
        pending = _load_json_mapping(pending_path)
        if pending.get("task") != task_name:
            raise ValueError(f"Pending episode belongs to another task: {pending_path}.")
        pending_index = int(pending["episode_index"])
        if pending_index in completed_indices:
            pending_path.unlink()
        else:
            episode_index = pending_index
            now_seed = int(pending["seed"])
    resolved_instruction_rng_seed = (
        int(start_seed) if instruction_rng_seed is None else int(instruction_rng_seed)
    )
    instruction_rng = np.random.default_rng(resolved_instruction_rng_seed)
    args["eval_mode"] = True

    records_mode = "a" if resume and records_path.exists() else "x"
    attempts_mode = "a" if resume and attempts_path.exists() else "x"
    with (
        records_path.open(records_mode, encoding="utf-8") as records,
        attempts_path.open(attempts_mode, encoding="utf-8") as attempts,
    ):
        while accepted < episodes:
            if seed_attempts >= max_seed_attempts:
                raise RuntimeError(
                    "RoboTwin expert precheck exhausted max_seed_attempts="
                    f"{max_seed_attempts} after accepting {accepted}/{episodes} episodes. "
                    f"See {attempts_path}."
                )
            attempted_seed = now_seed
            seed_attempts += 1
            render_freq = args["render_freq"]
            args["render_freq"] = 0
            try:
                task_env.setup_demo(
                    now_ep_num=episode_index,
                    seed=now_seed,
                    is_test=True,
                    **args,
                )
                episode_info = task_env.play_once()
            except official.UnStableError as exc:
                attempts.write(
                    json.dumps(
                        {
                            "attempt_index": seed_attempts - 1,
                            "seed": attempted_seed,
                            "accepted": False,
                            "reason": "unstable_error",
                            "detail": str(exc),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                attempts.flush()
                os.fsync(attempts.fileno())
                now_seed += 1
                continue
            finally:
                task_env.close_env()
                args["render_freq"] = render_freq

            if not (task_env.plan_success and task_env.check_success()):
                attempts.write(
                    json.dumps(
                        {
                            "attempt_index": seed_attempts - 1,
                            "seed": attempted_seed,
                            "accepted": False,
                            "reason": "expert_plan_failed",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                attempts.flush()
                os.fsync(attempts.fileno())
                now_seed += 1
                continue
            attempts.write(
                json.dumps(
                    {
                        "attempt_index": seed_attempts - 1,
                        "seed": attempted_seed,
                        "accepted": True,
                        "reason": "expert_plan_succeeded",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            attempts.flush()
            os.fsync(attempts.fileno())
            accepted += 1
            pending_video = root / f"episode_{episode_index:04d}_pending.mp4"
            _atomic_json(
                root / "pending_episode.json",
                {
                    "task": task_name,
                    "episode_index": episode_index,
                    "seed": now_seed,
                    "server_generation": args.get("server_generation"),
                },
            )
            ffmpeg: Any | None = None
            try:
                task_env.setup_demo(
                    now_ep_num=episode_index,
                    seed=now_seed,
                    is_test=True,
                    **args,
                )
                episode_info_list = [episode_info["info"]]
                descriptions = official.generate_episode_descriptions(
                    args["task_name"], episode_info_list, episodes
                )
                instruction_candidates = descriptions[0][instruction_type]
                if (
                    not isinstance(instruction_candidates, list)
                    or not instruction_candidates
                ):
                    raise ValueError(
                        "RoboTwin generated no instruction candidates for "
                        f"instruction_type={instruction_type!r}."
                    )
                instruction_index = int(
                    instruction_rng.integers(0, len(instruction_candidates))
                )
                instruction = str(instruction_candidates[instruction_index])
                task_env.set_instruction(instruction=instruction)

                ffmpeg = _start_video(
                    path=pending_video,
                    video_size=video_size,
                    popen_factory=popen_factory,
                )
                task_env._set_eval_video_ffmpeg(ffmpeg)
                assert ffmpeg is not None
                reset_func(model)
                success = False
                while task_env.take_action_cnt < task_env.step_lim:
                    observation = task_env.get_obs()
                    eval_func(task_env, model, observation)
                    if task_env.eval_success:
                        success = True
                        break

                # stable_2.0 take_action writes the terminal success frame itself.
                if not success:
                    terminal_observation = task_env.get_obs()
                    terminal_rgb = terminal_observation["observation"]["head_camera"][
                        "rgb"
                    ]
                    if ffmpeg.stdin is None:
                        raise RuntimeError(
                            "ffmpeg stdin is unavailable for terminal frame."
                        )
                    ffmpeg.stdin.write(
                        np.asarray(terminal_rgb, dtype=np.uint8).tobytes()
                    )
                _close_video_process(ffmpeg)
                ffmpeg = None
                if hasattr(task_env, "eval_video_ffmpeg"):
                    del task_env.eval_video_ffmpeg

                outcome = "success" if success else "failure"
                final_video = root / f"episode_{episode_index:04d}_{outcome}.mp4"
                if not pending_video.is_file():
                    raise FileNotFoundError(
                        "RoboTwin did not produce the pending episode video: "
                        f"{pending_video}"
                    )
                pending_video.rename(final_video)
                if success:
                    task_env.suc += 1
                terminal_reason = "success" if success else "step_limit"
                width, height = (int(value) for value in video_size.split("x", 1))
                record = {
                    "task": task_name,
                    "episode_index": episode_index,
                    "seed": now_seed,
                    "instruction": instruction,
                    "instruction_index": instruction_index,
                    "instruction_rng_seed": resolved_instruction_rng_seed,
                    "success": success,
                    "terminal_reason": terminal_reason,
                    "policy_steps": int(task_env.take_action_cnt),
                    "video_width": width,
                    "video_height": height,
                    "video_path": str(final_video),
                }
                records.write(json.dumps(record, sort_keys=True) + "\n")
                records.flush()
                os.fsync(records.fileno())
                completed_indices.add(episode_index)
                pending_path = root / "pending_episode.json"
                if pending_path.exists():
                    pending_path.unlink()
            except BaseException:
                try:
                    if ffmpeg is not None:
                        _close_video_process(ffmpeg)
                finally:
                    if hasattr(task_env, "eval_video_ffmpeg"):
                        del task_env.eval_video_ffmpeg
                    incomplete_video = root / (
                        f"episode_{episode_index:04d}_incomplete.mp4"
                    )
                    if pending_video.is_file() and not incomplete_video.exists():
                        pending_video.rename(incomplete_video)
                raise
            finally:
                task_env.close_env(
                    clear_cache=((accepted + 1) % int(args["clear_cache_freq"]) == 0)
                )
                if getattr(task_env, "render_freq", 0):
                    task_env.viewer.close()

            episode_index += 1
            task_env.test_num += 1
            now_seed += 1

    _atomic_json(
        root / "evaluator_result.json",
        {
            "task": task_name,
            "episodes": episodes,
            "successes": int(task_env.suc),
            "success_rate": float(task_env.suc / episodes),
            "records_path": str(records_path),
            "seed_attempts_path": str(attempts_path),
            "seed_attempts": seed_attempts,
            "max_seed_attempts": max_seed_attempts,
            "instruction_rng_seed": resolved_instruction_rng_seed,
        },
    )
    return now_seed, int(task_env.suc)


def _load_official_module(robotwin_root: Path) -> ModuleType:
    source = robotwin_root / "script" / "eval_policy.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official RoboTwin evaluator not found: {source}")
    for path in (
        robotwin_root,
        robotwin_root / "policy",
        robotwin_root / "description" / "utils",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "robotwin_official_eval_policy", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official RoboTwin evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}.")
    return payload


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Expected valid JSON at {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping at {path}.")
    return payload


def _prepare_official_task(
    *,
    official: ModuleType | Any,
    root: Path,
    task_config_path: Path,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], str, int]:
    """Build the official task state without creating a policy client."""
    task_args = _load_yaml_mapping(task_config_path)
    task_args.update(
        {
            "task_name": config["task_name"],
            "task_config": config["task_config"],
            "ckpt_setting": config["ckpt_setting"],
            "policy_name": config["policy_name"],
        }
    )

    configs_path = Path(official.CONFIGS_PATH)
    if not configs_path.is_absolute():
        configs_path = root / configs_path
    embodiment_types = _load_yaml_mapping(configs_path / "_embodiment_config.yml")
    camera_types = _load_yaml_mapping(configs_path / "_camera_config.yml")

    embodiment = task_args.get("embodiment")
    if not isinstance(embodiment, list) or len(embodiment) not in {1, 3}:
        raise ValueError("RoboTwin embodiment must contain one or three entries.")

    def embodiment_file(name: str) -> str:
        try:
            value = embodiment_types[name]["file_path"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing embodiment file_path for {name!r}.") from exc
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid embodiment file_path for {name!r}: {value!r}.")
        return value

    head_camera_type = task_args["camera"]["head_camera_type"]
    head_camera = camera_types[head_camera_type]
    task_args["head_camera_h"] = int(head_camera["h"])
    task_args["head_camera_w"] = int(head_camera["w"])
    video_size = f"{task_args['head_camera_w']}x{task_args['head_camera_h']}"

    task_args["left_robot_file"] = embodiment_file(str(embodiment[0]))
    if len(embodiment) == 1:
        task_args["right_robot_file"] = task_args["left_robot_file"]
        task_args["dual_arm_embodied"] = True
    else:
        task_args["right_robot_file"] = embodiment_file(str(embodiment[1]))
        task_args["embodiment_dis"] = embodiment[2]
        task_args["dual_arm_embodied"] = False

    task_args["left_embodiment_config"] = official.get_embodiment_config(
        task_args["left_robot_file"]
    )
    task_args["right_embodiment_config"] = official.get_embodiment_config(
        task_args["right_robot_file"]
    )
    config["left_arm_dim"] = len(
        task_args["left_embodiment_config"]["arm_joints_name"][0]
    )
    config["right_arm_dim"] = len(
        task_args["right_embodiment_config"]["arm_joints_name"][1]
    )

    task_env = official.class_decorator(config["task_name"])
    start_seed = 100000 * (1 + int(config["seed"]))
    return task_env, task_args, video_size, start_seed


def _prepare_official_runtime(
    *,
    official: ModuleType | Any,
    root: Path,
    task_config_path: Path,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], Any, str, int]:
    """Build stable_2.0 task and policy state without its write-heavy outer main."""
    task_env, task_args, video_size, start_seed = _prepare_official_task(
        official=official,
        root=root,
        task_config_path=task_config_path,
        config=config,
    )
    get_model = official.eval_function_decorator(config["policy_name"], "get_model")
    try:
        model = get_model(config)
    except BaseException:
        task_env.close_env()
        raise
    return task_env, task_args, model, video_size, start_seed


def _validate_server_model_info(
    server_model_info: Any,
    *,
    expected_checkpoint: Path,
) -> dict[str, Any]:
    if not isinstance(server_model_info, dict):
        raise RuntimeError(
            "RoboTwin policy plugin did not expose structured server_model_info."
        )
    server_artifact = server_model_info.get("serving_artifact")
    if not isinstance(server_artifact, dict):
        raise RuntimeError(
            "Praxis policy server health response has no serving artifact identity."
        )
    checkpoint = server_artifact.get("checkpoint")
    loaded_path = checkpoint.get("path") if isinstance(checkpoint, dict) else None
    if loaded_path is None or Path(loaded_path).resolve() != expected_checkpoint.resolve():
        raise RuntimeError(
            "Praxis policy server loaded a different checkpoint: "
            f"server={loaded_path!r}, expected={str(expected_checkpoint)!r}."
        )
    return server_model_info


def _run_stable2_eval_impl(
    *,
    robotwin_root: str | Path,
    plugin_config: str | Path,
    task_name: str,
    task_config: str,
    episodes_per_task: int,
    record_episodes_per_task: int,
    seed: int,
    policy_host: str,
    policy_port: int,
    output_dir: str | Path,
    run_id: str,
    max_seed_attempts: int,
    expected_checkpoint: str | Path,
    resume: bool,
) -> dict[str, Any]:
    if record_episodes_per_task != episodes_per_task:
        raise ValueError(
            "record_episodes_per_task must equal episodes_per_task: "
            f"{record_episodes_per_task} != {episodes_per_task}."
        )
    root = Path(robotwin_root).expanduser().resolve()
    plugin_config_path = Path(plugin_config).expanduser().resolve()
    _load_policy_plugin(plugin_config_path)
    official = _load_official_module(root)
    task_config_path = root / "task_config" / f"{task_config}.yml"
    task_args = _load_yaml_mapping(task_config_path)
    if not bool(task_args.get("eval_video_log")):
        raise ValueError(f"{task_config_path} must set eval_video_log: true.")

    config = _load_yaml_mapping(plugin_config_path)
    instruction_type = config.get("instruction_type")
    if not isinstance(instruction_type, str) or not instruction_type:
        raise ValueError("Praxis RoboTwin plugin must set instruction_type.")
    official_instruction_type = task_args.get("eval_instruction")
    if (
        official_instruction_type is not None
        and instruction_type != official_instruction_type
    ):
        raise ValueError(
            "Praxis RoboTwin instruction_type must match the official task config: "
            f"{instruction_type!r} != {official_instruction_type!r}."
        )
    config.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "ckpt_setting": run_id,
            "policy_name": "praxis_ar",
            "instruction_type": instruction_type,
            "seed": int(seed),
            "host": policy_host,
            "port": int(policy_port),
        }
    )
    task_output = Path(output_dir).expanduser().resolve() / task_name
    expected_checkpoint_path = Path(expected_checkpoint).expanduser().resolve()
    instruction_rng_seed = 100000 * (1 + int(seed))
    run_manifest_path = task_output / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "status": "initializing",
        "failure": None,
        "robotwin_root": str(root),
        "official_evaluator_path": str(root / "script" / "eval_policy.py"),
        "task_config_path": str(task_config_path),
        "plugin_config_path": str(plugin_config_path),
        "task": task_name,
        "task_config": task_config,
        "episodes_per_task": episodes_per_task,
        "record_episodes_per_task": record_episodes_per_task,
        "seed": seed,
        "instruction_type": instruction_type,
        "official_eval_instruction": official_instruction_type,
        "task_step_limit": None,
        "instruction_selection": {
            "algorithm": "numpy.default_rng/PCG64",
            "rng_seed": instruction_rng_seed,
            "selection": "uniform candidate index per accepted episode",
        },
        "policy_host": policy_host,
        "policy_port": policy_port,
        "run_id": run_id,
        "max_seed_attempts": max_seed_attempts,
        "expected_checkpoint": {
            "checkpoint": {"path": str(expected_checkpoint_path)}
        },
        "server_model_info": None,
    }
    _atomic_json(run_manifest_path, run_manifest)

    previous_cwd = Path.cwd()
    model: Any | None = None
    try:
        os.chdir(root)
        task_env, task_args, model, video_size, start_seed = _prepare_official_runtime(
            official=official,
            root=root,
            task_config_path=task_config_path,
            config=config,
        )
        if start_seed != instruction_rng_seed:
            raise RuntimeError(
                "RoboTwin instruction seed derivation disagrees with simulator seed: "
                f"{instruction_rng_seed} != {start_seed}."
            )
        server_model_info = _validate_server_model_info(
            getattr(model, "server_model_info", None),
            expected_checkpoint=expected_checkpoint_path,
        )
        run_manifest.update(
            {"status": "running", "server_model_info": server_model_info}
        )
        _atomic_json(run_manifest_path, run_manifest)
        run_controlled_rollouts(
            official=official,
            task_name=task_name,
            task_env=task_env,
            args=task_args,
            model=model,
            start_seed=start_seed,
            episodes=episodes_per_task,
            video_size=video_size,
            instruction_type=str(config["instruction_type"]),
            output_dir=task_output,
            max_seed_attempts=max_seed_attempts,
            instruction_rng_seed=instruction_rng_seed,
            output_dir_precreated=True,
            resume=resume,
        )
        task_step_limit = getattr(task_env, "step_lim", None)
        if not isinstance(task_step_limit, int) or task_step_limit <= 0:
            raise RuntimeError(
                f"RoboTwin task {task_name!r} has invalid step limit "
                f"{task_step_limit!r}."
            )
        run_manifest["task_step_limit"] = task_step_limit
        close_model = official.eval_function_decorator("praxis_ar", "close_model")
        close_model(model)
        model = None

        run_manifest["status"] = "rollout_completed"
        _atomic_json(run_manifest_path, run_manifest)
    except BaseException as exc:
        close_error: str | None = None
        if model is not None:
            try:
                close_model = official.eval_function_decorator(
                    "praxis_ar", "close_model"
                )
                close_model(model)
            except Exception as cleanup_exc:
                close_error = (
                    f"{type(cleanup_exc).__module__}."
                    f"{type(cleanup_exc).__qualname__}: {cleanup_exc}"
                )
        run_manifest.update(
            {
                "status": "failed",
                "failure": {
                    "error_type": (
                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                    ),
                    "message": str(exc),
                    "cleanup_error": close_error,
                },
            }
        )
        _atomic_json(run_manifest_path, run_manifest)
        raise
    finally:
        os.chdir(previous_cwd)

    return {
        "task_output_dir": str(task_output),
        "episode_records_path": str(task_output / "episode_records.jsonl"),
        "run_manifest_path": str(run_manifest_path),
    }


def run_stable2_eval(
    *,
    robotwin_root: str | Path,
    plugin_config: str | Path,
    task_name: str,
    task_config: str,
    episodes_per_task: int,
    record_episodes_per_task: int,
    seed: int,
    policy_host: str,
    policy_port: int,
    output_dir: str | Path,
    run_id: str,
    max_seed_attempts: int,
    expected_checkpoint: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Create a durable preflight manifest before entering the heavy evaluator setup."""
    task_output = Path(output_dir).expanduser().resolve() / task_name
    run_manifest_path = task_output / "run_manifest.json"
    if task_output.exists():
        if not run_manifest_path.is_file():
            raise FileExistsError(
                f"Use a fresh task eval output directory: {task_output}."
            )
        preflight = _load_json_mapping(run_manifest_path)
        allowed_statuses = {"preflight", "failed", "running", "rollout_completed"}
        if (
            not resume
            or preflight.get("status") not in allowed_statuses
            or preflight.get("task") != task_name
            or preflight.get("run_id") != run_id
        ):
            raise FileExistsError(
                f"Existing task output is not this run's preflight: {task_output}."
            )
    else:
        task_output.parent.mkdir(parents=True, exist_ok=True)
        task_output.mkdir(exist_ok=False)
        preflight = {
            "status": "preflight",
            "failure": None,
            "task": task_name,
            "task_config": task_config,
            "run_id": run_id,
            "robotwin_root": str(Path(robotwin_root).expanduser().resolve()),
            "expected_checkpoint_path": str(
                Path(expected_checkpoint).expanduser().resolve()
            ),
        }
        _atomic_json(run_manifest_path, preflight)

    try:
        return _run_stable2_eval_impl(
            robotwin_root=robotwin_root,
            plugin_config=plugin_config,
            task_name=task_name,
            task_config=task_config,
            episodes_per_task=episodes_per_task,
            record_episodes_per_task=record_episodes_per_task,
            seed=seed,
            policy_host=policy_host,
            policy_port=policy_port,
            output_dir=output_dir,
            run_id=run_id,
            max_seed_attempts=max_seed_attempts,
            expected_checkpoint=expected_checkpoint,
            resume=resume,
        )
    except BaseException as exc:
        current = _load_json_mapping(run_manifest_path)
        if current.get("status") != "failed":
            current.update(
                {
                    "status": "failed",
                    "failure": {
                        "stage": "preflight",
                        "error_type": (
                            f"{type(exc).__module__}.{type(exc).__qualname__}"
                        ),
                        "message": str(exc),
                    },
                }
            )
            _atomic_json(run_manifest_path, current)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--plugin-config", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episodes-per-task", required=True, type=int)
    parser.add_argument("--record-episodes-per-task", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=50051)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-seed-attempts", required=True, type=int)
    parser.add_argument("--expected-checkpoint", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    result = run_stable2_eval(
        robotwin_root=cli.robotwin_root,
        plugin_config=cli.plugin_config,
        task_name=cli.task,
        task_config=cli.task_config,
        episodes_per_task=cli.episodes_per_task,
        record_episodes_per_task=cli.record_episodes_per_task,
        seed=cli.seed,
        policy_host=cli.policy_host,
        policy_port=cli.policy_port,
        output_dir=cli.output_dir,
        run_id=cli.run_id,
        max_seed_attempts=cli.max_seed_attempts,
        expected_checkpoint=cli.expected_checkpoint,
        resume=cli.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
