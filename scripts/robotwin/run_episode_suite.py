# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Plan official RoboTwin seeds, run episodes in parallel, and aggregate results."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from scripts.robotwin.run_eval import DEFAULT_PLUGIN_SOURCE
from scripts.robotwin.run_suite import (
    DEFAULT_TASKS_PATH,
    aggregate_suite,
    assigned_tasks,
    load_official_tasks,
)
from scripts.robotwin.stable2_evaluator import (
    _atomic_json,
    _close_video_process,
    _load_official_module,
    _load_policy_plugin,
    _load_yaml_mapping,
    _prepare_official_runtime,
    _prepare_official_task,
    _start_video,
    _validate_server_model_info,
)
from scripts.robotwin.verify_eval import verify_task_eval

TASK_CONFIG = "demo_clean"
SEED_GROUP = 0
INSTRUCTION_TYPE = "seen"


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
        values.append(value)
    return values


def _runtime_config(
    *,
    plugin_config: Path,
    task_name: str,
    run_id: str,
    policy_host: str | None = None,
    policy_port: int | None = None,
    episode_namespace: str | None = None,
) -> dict[str, Any]:
    config = _load_yaml_mapping(plugin_config)
    if config.get("instruction_type") != INSTRUCTION_TYPE:
        raise ValueError(
            f"RoboTwin {TASK_CONFIG} requires instruction_type={INSTRUCTION_TYPE!r}."
        )
    config.update(
        {
            "task_name": task_name,
            "task_config": TASK_CONFIG,
            "ckpt_setting": run_id,
            "policy_name": "praxis_ar",
            "seed": SEED_GROUP,
        }
    )
    if policy_host is not None:
        config["host"] = policy_host
    if policy_port is not None:
        config["port"] = int(policy_port)
    if episode_namespace is not None:
        config["episode_namespace"] = episode_namespace
    return config


def _validate_task_config(robotwin_root: Path, plugin_config: Path) -> Path:
    task_config_path = robotwin_root / "task_config" / f"{TASK_CONFIG}.yml"
    task_args = _load_yaml_mapping(task_config_path)
    if not bool(task_args.get("eval_video_log")):
        raise ValueError(f"{task_config_path} must set eval_video_log: true.")
    official_instruction = task_args.get("eval_instruction")
    plugin_instruction = _load_yaml_mapping(plugin_config).get("instruction_type")
    if official_instruction is not None and official_instruction != plugin_instruction:
        raise ValueError(
            "Policy instruction type disagrees with official task config: "
            f"{plugin_instruction!r} != {official_instruction!r}."
        )
    return task_config_path


def _official_task_step_limit(robotwin_root: Path, task_name: str) -> int:
    limits = _load_yaml_mapping(
        robotwin_root / "task_config" / "_eval_step_limit.yml"
    )
    value = limits.get(task_name, 1000)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid official step limit for {task_name}: {value!r}.")
    return value


def plan_task_episodes(
    *,
    robotwin_root: Path,
    plugin_config: Path,
    task_name: str,
    output_dir: Path,
    run_id: str,
    episodes_per_task: int,
    max_seed_attempts: int,
    resume: bool,
) -> dict[str, Any]:
    """Reproduce official sequential seed acceptance before parallel rollout."""
    root = robotwin_root.expanduser().resolve()
    plugin = plugin_config.expanduser().resolve()
    task_dir = output_dir.expanduser().resolve() / task_name
    plan_path = task_dir / "episode_plan.json"
    records_path = task_dir / "episode_plan_records.jsonl"
    attempts_path = task_dir / "seed_attempts.jsonl"
    if plan_path.is_file():
        plan = _read_json(plan_path)
        if (
            resume
            and plan.get("status") == "completed"
            and plan.get("run_id") == run_id
            and len(plan.get("episodes", [])) == episodes_per_task
        ):
            print(f"ROBOTWIN_PLAN_RESUME_SKIP task={task_name}", flush=True)
            return plan
        raise FileExistsError(f"Episode plan already exists: {plan_path}.")
    task_dir.mkdir(parents=True, exist_ok=resume)

    attempts = _read_jsonl(attempts_path) if resume else []
    planned = [dict(item["episode"]) for item in attempts if item.get("accepted")]
    if resume and records_path.exists():
        _atomic_text(
            records_path,
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in planned),
        )
    if [int(item["episode_index"]) for item in planned] != list(range(len(planned))):
        raise ValueError(f"Non-contiguous episode plan records in {records_path}.")
    if len(planned) > episodes_per_task:
        raise ValueError(f"Too many episode plan records in {records_path}.")

    instruction_rng = np.random.default_rng(100000)
    for item in planned:
        replayed_index = int(
            instruction_rng.integers(0, int(item["instruction_candidate_count"]))
        )
        if replayed_index != int(item["instruction_index"]):
            raise ValueError(f"Instruction RNG history disagrees in {records_path}.")

    now_seed = max(
        (int(item["seed"]) for item in attempts),
        default=100000 - 1,
    ) + 1
    seed_attempts = len(attempts)
    _load_policy_plugin(plugin)
    official = _load_official_module(root)
    task_config_path = _validate_task_config(root, plugin)
    config = _runtime_config(
        plugin_config=plugin,
        task_name=task_name,
        run_id=run_id,
    )

    previous_cwd = Path.cwd()
    task_env: Any | None = None
    records_mode = "a" if resume and records_path.exists() else "x"
    attempts_mode = "a" if resume and attempts_path.exists() else "x"
    print(
        f"ROBOTWIN_PLAN_START task={task_name} planned={len(planned)}/{episodes_per_task} "
        f"next_seed={now_seed}",
        flush=True,
    )
    try:
        os.chdir(root)
        task_env, task_args, video_size, start_seed = _prepare_official_task(
            official=official,
            root=root,
            task_config_path=task_config_path,
            config=config,
        )
        task_args["eval_mode"] = True
        step_limit = _official_task_step_limit(root, task_name)
        if start_seed != 100000:
            raise RuntimeError(f"Unexpected seed-group start seed: {start_seed}.")
        with (
            records_path.open(records_mode, encoding="utf-8") as records_handle,
            attempts_path.open(attempts_mode, encoding="utf-8") as attempts_handle,
        ):
            while len(planned) < episodes_per_task:
                if seed_attempts >= max_seed_attempts:
                    raise RuntimeError(
                        f"Expert precheck exhausted {max_seed_attempts} attempts for "
                        f"{task_name} after {len(planned)}/{episodes_per_task} episodes."
                    )
                episode_index = len(planned)
                attempted_seed = now_seed
                seed_attempts += 1
                render_freq = task_args["render_freq"]
                task_args["render_freq"] = 0
                try:
                    task_env.setup_demo(
                        now_ep_num=episode_index,
                        seed=attempted_seed,
                        is_test=True,
                        **task_args,
                    )
                    episode_info = task_env.play_once()
                except official.UnStableError as exc:
                    attempt = {
                        "attempt_index": seed_attempts - 1,
                        "seed": attempted_seed,
                        "accepted": False,
                        "reason": "unstable_error",
                        "detail": str(exc),
                    }
                    attempts_handle.write(json.dumps(attempt, sort_keys=True) + "\n")
                    attempts_handle.flush()
                    os.fsync(attempts_handle.fileno())
                    print(
                        f"ROBOTWIN_PLAN_REJECT task={task_name} seed={attempted_seed} "
                        "reason=unstable_error",
                        flush=True,
                    )
                    now_seed += 1
                    continue
                except Exception as exc:
                    attempt = {
                        "attempt_index": seed_attempts - 1,
                        "seed": attempted_seed,
                        "accepted": False,
                        "reason": "expert_exception",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                    attempts_handle.write(json.dumps(attempt, sort_keys=True) + "\n")
                    attempts_handle.flush()
                    os.fsync(attempts_handle.fileno())
                    print(
                        f"ROBOTWIN_PLAN_REJECT task={task_name} seed={attempted_seed} "
                        f"reason=expert_exception error={type(exc).__name__}",
                        flush=True,
                    )
                    now_seed += 1
                    continue
                finally:
                    task_env.close_env()
                    task_args["render_freq"] = render_freq

                accepted = bool(task_env.plan_success and task_env.check_success())
                if not accepted:
                    attempt = {
                        "attempt_index": seed_attempts - 1,
                        "seed": attempted_seed,
                        "accepted": False,
                        "reason": "expert_plan_failed",
                    }
                    attempts_handle.write(json.dumps(attempt, sort_keys=True) + "\n")
                    attempts_handle.flush()
                    os.fsync(attempts_handle.fileno())
                    print(
                        f"ROBOTWIN_PLAN_REJECT task={task_name} seed={attempted_seed} "
                        "reason=expert_plan_failed",
                        flush=True,
                    )
                    now_seed += 1
                    continue

                descriptions = official.generate_episode_descriptions(
                    task_name, [episode_info["info"]], episodes_per_task
                )
                candidates = descriptions[0][INSTRUCTION_TYPE]
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError(
                        f"No {INSTRUCTION_TYPE!r} instruction candidates for {task_name}."
                    )
                instruction_index = int(
                    instruction_rng.integers(0, len(candidates))
                )
                plan_record = {
                    "task": task_name,
                    "episode_index": episode_index,
                    "seed": attempted_seed,
                    "instruction": str(candidates[instruction_index]),
                    "instruction_index": instruction_index,
                    "instruction_candidate_count": len(candidates),
                    "instruction_rng_seed": 100000,
                }
                attempt = {
                    "attempt_index": seed_attempts - 1,
                    "seed": attempted_seed,
                    "accepted": True,
                    "reason": "expert_plan_succeeded",
                    "episode": plan_record,
                }
                attempts_handle.write(json.dumps(attempt, sort_keys=True) + "\n")
                attempts_handle.flush()
                os.fsync(attempts_handle.fileno())
                records_handle.write(json.dumps(plan_record, sort_keys=True) + "\n")
                records_handle.flush()
                os.fsync(records_handle.fileno())
                planned.append(plan_record)
                print(
                    f"ROBOTWIN_PLAN_PROGRESS task={task_name} "
                    f"planned={len(planned)}/{episodes_per_task} seed={attempted_seed}",
                    flush=True,
                )
                now_seed += 1

        plan = {
            "status": "completed",
            "run_id": run_id,
            "task": task_name,
            "task_config": TASK_CONFIG,
            "instruction_type": INSTRUCTION_TYPE,
            "seed_group": SEED_GROUP,
            "start_seed": start_seed,
            "episodes_per_task": episodes_per_task,
            "max_seed_attempts": max_seed_attempts,
            "seed_attempts": seed_attempts,
            "video_size": video_size,
            "task_step_limit": step_limit,
            "episodes": planned,
        }
        _atomic_json(plan_path, plan)
        print(
            f"ROBOTWIN_PLAN_COMPLETE task={task_name} episodes={episodes_per_task} "
            f"attempts={seed_attempts}",
            flush=True,
        )
        return plan
    finally:
        os.chdir(previous_cwd)


def plan_worker(
    *,
    robotwin_root: Path,
    plugin_config: Path,
    tasks_path: Path,
    worker_index: int,
    worker_count: int,
    output_dir: Path,
    run_id: str,
    episodes_per_task: int,
    max_seed_attempts: int,
    resume: bool,
) -> None:
    tasks = assigned_tasks(
        load_official_tasks(tasks_path),
        worker_index=worker_index,
        worker_count=worker_count,
    )
    for task_name in tasks:
        plan_task_episodes(
            robotwin_root=robotwin_root,
            plugin_config=plugin_config,
            task_name=task_name,
            output_dir=output_dir,
            run_id=run_id,
            episodes_per_task=episodes_per_task,
            max_seed_attempts=max_seed_attempts,
            resume=resume,
        )


def planned_episode_pairs(
    *,
    tasks: list[str],
    output_dir: Path,
    episodes_per_task: int,
) -> list[dict[str, Any]]:
    plans = {
        task: _read_json(output_dir.expanduser().resolve() / task / "episode_plan.json")
        for task in tasks
    }
    pairs: list[dict[str, Any]] = []
    for episode_index in range(episodes_per_task):
        for task in tasks:
            plan = plans[task]
            episodes = plan.get("episodes")
            if (
                plan.get("status") != "completed"
                or not isinstance(episodes, list)
                or len(episodes) != episodes_per_task
            ):
                raise RuntimeError(f"Incomplete episode plan for task {task}.")
            episode = dict(episodes[episode_index])
            if episode.get("task") != task or episode.get("episode_index") != episode_index:
                raise RuntimeError(f"Mismatched episode plan for {task}/{episode_index}.")
            episode["video_size"] = plan["video_size"]
            episode["task_step_limit"] = plan["task_step_limit"]
            pairs.append(episode)
    return pairs


def _episode_record_path(output_dir: Path, task_name: str, episode_index: int) -> Path:
    return output_dir / task_name / "_episode_records" / f"episode_{episode_index:04d}.json"


def run_planned_episode(
    *,
    robotwin_root: Path,
    plugin_config: Path,
    episode: dict[str, Any],
    policy_host: str,
    policy_port: int,
    output_dir: Path,
    run_id: str,
    expected_checkpoint: Path,
    worker_index: int,
    heartbeat_steps: int,
    resume: bool,
) -> dict[str, Any]:
    root = robotwin_root.expanduser().resolve()
    plugin = plugin_config.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    task_name = str(episode["task"])
    episode_index = int(episode["episode_index"])
    task_dir = output / task_name
    record_path = _episode_record_path(output, task_name, episode_index)
    outcome_videos = list(task_dir.glob(f"episode_{episode_index:04d}_*.mp4"))
    if resume and record_path.is_file():
        record = _read_json(record_path)
        final_video = Path(str(record["video_path"]))
        if final_video.is_file() and final_video.stat().st_size > 0:
            print(
                f"ROBOTWIN_EPISODE_RESUME_SKIP worker={worker_index} task={task_name} "
                f"episode={episode_index}",
                flush=True,
            )
            return record
    for path in outcome_videos:
        path.unlink()
    record_path.parent.mkdir(parents=True, exist_ok=True)

    _load_policy_plugin(plugin)
    official = _load_official_module(root)
    task_config_path = _validate_task_config(root, plugin)
    config = _runtime_config(
        plugin_config=plugin,
        task_name=task_name,
        run_id=run_id,
        policy_host=policy_host,
        policy_port=policy_port,
        episode_namespace=f"{run_id}:{task_name}:{episode_index:04d}",
    )
    previous_cwd = Path.cwd()
    task_env: Any | None = None
    model: Any | None = None
    ffmpeg: Any | None = None
    pending_video = task_dir / f"episode_{episode_index:04d}_pending.mp4"
    print(
        f"ROBOTWIN_EPISODE_START worker={worker_index} task={task_name} "
        f"episode={episode_index} seed={episode['seed']} policy={policy_host}:{policy_port}",
        flush=True,
    )
    try:
        os.chdir(root)
        task_env, task_args, model, video_size, start_seed = _prepare_official_runtime(
            official=official,
            root=root,
            task_config_path=task_config_path,
            config=config,
        )
        if start_seed != 100000 or video_size != episode["video_size"]:
            raise RuntimeError(f"Runtime disagrees with episode plan for {task_name}.")
        _validate_server_model_info(
            getattr(model, "server_model_info", None),
            expected_checkpoint=expected_checkpoint.expanduser().resolve(),
        )
        task_args["eval_mode"] = True
        task_args["eval_video_save_dir"] = task_dir
        task_env.setup_demo(
            now_ep_num=episode_index,
            seed=int(episode["seed"]),
            is_test=True,
            **task_args,
        )
        task_env.set_instruction(instruction=str(episode["instruction"]))
        ffmpeg = _start_video(
            path=pending_video,
            video_size=video_size,
            popen_factory=lambda command: subprocess.Popen(
                command, stdin=subprocess.PIPE
            ),
        )
        task_env._set_eval_video_ffmpeg(ffmpeg)
        reset_func = official.eval_function_decorator("praxis_ar", "reset_model")
        eval_func = official.eval_function_decorator("praxis_ar", "eval")
        reset_func(model)
        success = False
        last_heartbeat = -heartbeat_steps
        while task_env.take_action_cnt < task_env.step_lim:
            if int(task_env.take_action_cnt) - last_heartbeat >= heartbeat_steps:
                last_heartbeat = int(task_env.take_action_cnt)
                print(
                    f"ROBOTWIN_EPISODE_HEARTBEAT worker={worker_index} task={task_name} "
                    f"episode={episode_index} step={task_env.take_action_cnt}/{task_env.step_lim}",
                    flush=True,
                )
            observation = task_env.get_obs()
            eval_func(task_env, model, observation)
            if task_env.eval_success:
                success = True
                break
        if not success:
            terminal_observation = task_env.get_obs()
            terminal_rgb = terminal_observation["observation"]["head_camera"]["rgb"]
            if ffmpeg.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable for terminal frame.")
            ffmpeg.stdin.write(np.asarray(terminal_rgb, dtype=np.uint8).tobytes())
        _close_video_process(ffmpeg)
        ffmpeg = None
        if hasattr(task_env, "eval_video_ffmpeg"):
            del task_env.eval_video_ffmpeg
        outcome = "success" if success else "failure"
        final_video = task_dir / f"episode_{episode_index:04d}_{outcome}.mp4"
        pending_video.rename(final_video)
        width, height = (int(value) for value in video_size.split("x", 1))
        record = {
            "task": task_name,
            "episode_index": episode_index,
            "seed": int(episode["seed"]),
            "instruction": str(episode["instruction"]),
            "instruction_index": int(episode["instruction_index"]),
            "instruction_rng_seed": int(episode["instruction_rng_seed"]),
            "success": success,
            "terminal_reason": "success" if success else "step_limit",
            "policy_steps": int(task_env.take_action_cnt),
            "task_step_limit": int(task_env.step_lim),
            "video_width": width,
            "video_height": height,
            "video_path": str(final_video),
            "policy_host": policy_host,
            "policy_port": int(policy_port),
            "worker_index": worker_index,
        }
        _atomic_json(record_path, record)
        print(
            f"ROBOTWIN_EPISODE_RESULT worker={worker_index} task={task_name} "
            f"episode={episode_index} seed={episode['seed']} Success={str(success).lower()} "
            f"policy_steps={record['policy_steps']}",
            flush=True,
        )
        return record
    except BaseException:
        if ffmpeg is not None:
            _close_video_process(ffmpeg)
        if hasattr(task_env, "eval_video_ffmpeg"):
            del task_env.eval_video_ffmpeg
        if pending_video.is_file():
            incomplete = task_dir / f"episode_{episode_index:04d}_incomplete.mp4"
            if incomplete.exists():
                incomplete.unlink()
            pending_video.rename(incomplete)
        raise
    finally:
        if task_env is not None:
            task_env.close_env()
            if getattr(task_env, "render_freq", 0):
                task_env.viewer.close()
        if model is not None:
            close_model = official.eval_function_decorator("praxis_ar", "close_model")
            close_model(model)
        os.chdir(previous_cwd)


def record_global_progress(output_dir: Path, record: dict[str, Any]) -> dict[str, int]:
    root = output_dir.expanduser().resolve()
    path = root / "progress.jsonl"
    lock_path = root / ".progress.lock"
    event = {
        "task": record["task"],
        "episode_index": int(record["episode_index"]),
        "success": bool(record["success"]),
    }
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _read_jsonl(path)
        keys = {(item["task"], int(item["episode_index"])) for item in existing}
        key = (event["task"], event["episode_index"])
        if key not in keys:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            existing.append(event)
        completed = len(existing)
        successes = sum(bool(item["success"]) for item in existing)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(
        f"ROBOTWIN_PROGRESS completed={completed}/1000 successes={successes} "
        f"Success={successes / completed:.6f}",
        flush=True,
    )
    return {"completed": completed, "successes": successes}


def episode_worker(
    *,
    robotwin_root: Path,
    plugin_config: Path,
    tasks_path: Path,
    worker_index: int,
    worker_count: int,
    policy_host: str,
    policy_port: int,
    output_dir: Path,
    run_id: str,
    expected_checkpoint: Path,
    episodes_per_task: int,
    heartbeat_steps: int,
    resume: bool,
) -> None:
    tasks = load_official_tasks(tasks_path)
    episodes = planned_episode_pairs(
        tasks=tasks,
        output_dir=output_dir,
        episodes_per_task=episodes_per_task,
    )
    for episode in episodes[worker_index::worker_count]:
        record = run_planned_episode(
            robotwin_root=robotwin_root,
            plugin_config=plugin_config,
            episode=episode,
            policy_host=policy_host,
            policy_port=policy_port,
            output_dir=output_dir,
            run_id=run_id,
            expected_checkpoint=expected_checkpoint,
            worker_index=worker_index,
            heartbeat_steps=heartbeat_steps,
            resume=resume,
        )
        record_global_progress(output_dir, record)


def aggregate_episode_suite(
    *,
    tasks_path: Path,
    output_dir: Path,
    expected_checkpoint: Path,
    run_id: str,
    episodes_per_task: int,
) -> dict[str, Any]:
    tasks = load_official_tasks(tasks_path)
    output = output_dir.expanduser().resolve()
    checkpoint = expected_checkpoint.expanduser().resolve()
    for task_name in tasks:
        task_dir = output / task_name
        plan = _read_json(task_dir / "episode_plan.json")
        records = [
            _read_json(_episode_record_path(output, task_name, episode_index))
            for episode_index in range(episodes_per_task)
        ]
        records_path = task_dir / "episode_records.jsonl"
        _atomic_text(
            records_path,
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        )
        manifest = {
            "status": "rollout_completed",
            "failure": None,
            "task": task_name,
            "task_config": TASK_CONFIG,
            "instruction_type": INSTRUCTION_TYPE,
            "seed": SEED_GROUP,
            "run_id": run_id,
            "episodes_per_task": episodes_per_task,
            "record_episodes_per_task": episodes_per_task,
            "max_seed_attempts": plan["max_seed_attempts"],
            "task_step_limit": plan["task_step_limit"],
            "episode_parallel": True,
            "expected_checkpoint": {"checkpoint": {"path": str(checkpoint)}},
        }
        _atomic_json(task_dir / "run_manifest.json", manifest)
        verification = verify_task_eval(
            output_dir=output,
            task_name=task_name,
            episodes_per_task=episodes_per_task,
            episode_records_path=records_path,
        )
        manifest.update(
            {"status": "completed", "failure": None, "verification": verification}
        )
        _atomic_json(task_dir / "run_manifest.json", manifest)
        print(
            f"ROBOTWIN_VERIFY_TASK_COMPLETE task={task_name} "
            f"successes={verification['successes']}/{episodes_per_task}",
            flush=True,
        )
    return aggregate_suite(
        tasks_path=tasks_path,
        output_dir=output,
        expected_checkpoint=checkpoint,
        run_id=run_id,
        episodes_per_task=episodes_per_task,
    )


def _common_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--plugin-config", default=DEFAULT_PLUGIN_SOURCE / "deploy_policy.yml", type=Path)
    parser.add_argument("--tasks", default=DEFAULT_TASKS_PATH, type=Path)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--resume", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    planner = subparsers.add_parser("plan-worker")
    _common_worker_arguments(planner)
    planner.add_argument("--max-seed-attempts", type=int, default=1000)

    worker = subparsers.add_parser("episode-worker")
    _common_worker_arguments(worker)
    worker.add_argument("--policy-host", required=True)
    worker.add_argument("--policy-port", required=True, type=int)
    worker.add_argument("--expected-checkpoint", required=True, type=Path)
    worker.add_argument("--heartbeat-steps", type=int, default=100)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--tasks", default=DEFAULT_TASKS_PATH, type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--expected-checkpoint", required=True, type=Path)
    aggregate.add_argument("--episodes-per-task", type=int, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan-worker":
        result = plan_worker(
            robotwin_root=args.robotwin_root,
            plugin_config=args.plugin_config,
            tasks_path=args.tasks,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            output_dir=args.output_dir,
            run_id=args.run_id,
            episodes_per_task=args.episodes_per_task,
            max_seed_attempts=args.max_seed_attempts,
            resume=args.resume,
        )
    elif args.command == "episode-worker":
        result = episode_worker(
            robotwin_root=args.robotwin_root,
            plugin_config=args.plugin_config,
            tasks_path=args.tasks,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            output_dir=args.output_dir,
            run_id=args.run_id,
            expected_checkpoint=args.expected_checkpoint,
            episodes_per_task=args.episodes_per_task,
            heartbeat_steps=args.heartbeat_steps,
            resume=args.resume,
        )
    else:
        result = aggregate_episode_suite(
            tasks_path=args.tasks,
            output_dir=args.output_dir,
            expected_checkpoint=args.expected_checkpoint,
            run_id=args.run_id,
            episodes_per_task=args.episodes_per_task,
        )
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
