from __future__ import annotations

import argparse
import contextlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_WEIGHTED_METRICS = {
    "elapsed_steps",
    "moved_correct_obj",
    "moved_wrong_obj",
    "is_src_obj_grasped",
    "consecutive_grasp",
    "src_on_target",
    "success",
    "ever_success",
    "success_rate",
    "ever_success_rate",
    "final_success_rate",
    "avg_first_success_step",
    "avg_episode_length",
    "avg_reward",
    "avg_sum_reward",
    "avg_max_reward",
}
_SUM_METRICS = {"eval_s", "time/env.step+inference", "time/env.step", "time/inference", "time/total"}


def _replace_option(command: list[str], option: str, value: str) -> list[str]:
    updated = list(command)
    try:
        index = updated.index(option)
    except ValueError as exc:
        raise ValueError(f"Missing required command option {option!r}.") from exc
    if index + 1 >= len(updated):
        raise ValueError(f"Missing value after command option {option!r}.")
    updated[index + 1] = value
    return updated


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5.0)
    except ProcessLookupError:
        proc.wait(timeout=5.0)


def _run_batch(
    *,
    command: list[str],
    batch_dir: Path,
    inactivity_timeout_sec: float,
) -> tuple[str, dict[str, Any] | None]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = batch_dir / "metrics.json"
    log_path = batch_dir / "stdout.log"
    batch_command = _replace_option(
        command, "--metrics-output-path", str(metrics_path.resolve())
    )
    started = time.monotonic()
    status = "error"
    return_code: int | None = None

    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            batch_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        last_output = time.monotonic()
        try:
            while proc.poll() is None:
                events = selector.select(timeout=min(1.0, inactivity_timeout_sec))
                if events:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if chunk:
                        log_file.write(chunk)
                        log_file.flush()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                        last_output = time.monotonic()
                        continue
                if time.monotonic() - last_output >= inactivity_timeout_sec:
                    status = "timeout"
                    _terminate_process_group(proc)
                    break
            if status != "timeout":
                remainder = proc.stdout.read()
                if remainder:
                    log_file.write(remainder)
                    sys.stdout.buffer.write(remainder)
                    sys.stdout.buffer.flush()
                return_code = proc.wait()
                status = "ok" if return_code == 0 and metrics_path.is_file() else "error"
        finally:
            selector.close()
            proc.stdout.close()
            _terminate_process_group(proc)

    record = {
        "status": status,
        "return_code": return_code,
        "elapsed_sec": time.monotonic() - started,
        "command": batch_command,
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
    }
    (batch_dir / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "ok":
        return status, None
    return status, json.loads(metrics_path.read_text(encoding="utf-8"))


def _aggregate_metrics(
    completed: list[tuple[int, dict[str, Any]]], timed_out_seeds: list[int]
) -> dict[str, Any]:
    total_episodes = sum(int(metrics["n_episodes"]) for _, metrics in completed)
    total_episodes += len(timed_out_seeds)
    if total_episodes <= 0:
        raise RuntimeError("Cannot aggregate zero SIMPLER episodes.")

    result: dict[str, Any] = {
        "n_episodes": total_episodes,
        "total_episodes": total_episodes,
        "timed_out_episodes": sorted(timed_out_seeds),
        "completed_episode_seeds": sorted(
            seed + offset
            for seed, metrics in completed
            for offset in range(int(metrics["n_episodes"]))
        ),
    }
    for key in _WEIGHTED_METRICS:
        numerator = sum(
            float(metrics.get(key, 0.0)) * int(metrics["n_episodes"])
            for _, metrics in completed
        )
        result[key] = numerator / total_episodes
    for key in _SUM_METRICS:
        result[key] = sum(float(metrics.get(key, 0.0)) for _, metrics in completed)

    for _, metrics in completed:
        for key in ("task_description", "task_descriptions"):
            if key in metrics:
                result[key] = metrics[key]
    result["time/episodes_per_second"] = total_episodes / max(
        float(result.get("eval_s", 0.0)), 1e-9
    )
    return result


def evaluate_resilient_batches(
    *,
    command: list[str],
    output_path: Path,
    work_dir: Path,
    inactivity_timeout_sec: float,
) -> None:
    seed_index = command.index("--seed") + 1
    count_index = command.index("--num-episodes") + 1
    initial_seed = int(command[seed_index])
    initial_count = int(command[count_index])
    pending = [(initial_seed, initial_count)]
    completed: list[tuple[int, dict[str, Any]]] = []
    timed_out: list[int] = []
    attempt = 0

    while pending:
        seed, count = pending.pop(0)
        attempt += 1
        batch_command = _replace_option(command, "--seed", str(seed))
        batch_command = _replace_option(batch_command, "--num-episodes", str(count))
        batch_dir = work_dir / f"attempt_{attempt:04d}_seed_{seed}_count_{count}"
        status, metrics = _run_batch(
            command=batch_command,
            batch_dir=batch_dir,
            inactivity_timeout_sec=inactivity_timeout_sec,
        )
        if status == "ok" and metrics is not None:
            completed.append((seed, metrics))
            continue
        if status != "timeout":
            raise RuntimeError(
                "SIMPLER subprocess failed without an inactivity timeout: "
                f"seed={seed} count={count} details={batch_dir / 'result.json'}"
            )
        if count == 1:
            timed_out.append(seed)
            continue
        left_count = count // 2
        pending.insert(0, (seed + left_count, count - left_count))
        pending.insert(0, (seed, left_count))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_aggregate_metrics(completed, timed_out), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


@contextlib.contextmanager
def resilient_simpler_commands(*, inactivity_timeout_sec: float):
    from praxis_eval.envs.simpler import eval as simpler_eval

    original_builder = simpler_eval.build_simpler_eval_command

    def resilient_builder(**kwargs) -> list[str]:
        command = original_builder(**kwargs)
        metrics_index = command.index("--metrics-output-path") + 1
        output_path = Path(command[metrics_index])
        return [
            sys.executable,
            "-m",
            "praxis.scripts.resilient_simpler",
            "--output-path",
            str(output_path),
            "--work-dir",
            str(output_path.parent / "resilient_batches"),
            "--inactivity-timeout-sec",
            str(float(inactivity_timeout_sec)),
            "--",
            *command,
        ]

    simpler_eval.build_simpler_eval_command = resilient_builder
    try:
        yield
    finally:
        simpler_eval.build_simpler_eval_command = original_builder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--inactivity-timeout-sec", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("missing SIMPLER command after --")
    evaluate_resilient_batches(
        command=command,
        output_path=args.output_path,
        work_dir=args.work_dir,
        inactivity_timeout_sec=float(args.inactivity_timeout_sec),
    )


if __name__ == "__main__":
    main()
