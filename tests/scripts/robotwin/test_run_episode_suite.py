# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.robotwin.run_episode_suite as module
from scripts.robotwin.run_episode_suite import (
    aggregate_episode_suite,
    plan_task_episodes,
    planned_episode_pairs,
    record_global_progress,
)


class _UnstableError(Exception):
    pass


class _PlanningTask:
    def __init__(self) -> None:
        self.plan_success = False
        self.step_lim = 700
        self.setup_seeds: list[int] = []

    def setup_demo(self, *, seed: int, **_kwargs: object) -> None:
        self.setup_seeds.append(seed)
        self.plan_success = seed >= 100002

    def play_once(self) -> dict[str, object]:
        seed = self.setup_seeds[-1]
        if seed == 100000:
            raise _UnstableError("unstable")
        return {"info": {"seed": seed}}

    def check_success(self) -> bool:
        return self.plan_success

    def close_env(self) -> None:
        pass


def _write_plugin(tmp_path: Path) -> Path:
    path = tmp_path / "deploy_policy.yml"
    path.write_text(yaml.safe_dump({"instruction_type": "seen"}))
    return path


def test_planner_preserves_sequential_seed_acceptance_and_instruction_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task_env = _PlanningTask()
    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda _task, info, _count: [
            {"seen": [f"instruction-{info[0]['seed']}-0", "second"]}
        ],
    )
    monkeypatch.setattr(module, "_load_policy_plugin", lambda _path: None)
    monkeypatch.setattr(module, "_load_official_module", lambda _root: official)
    monkeypatch.setattr(
        module, "_validate_task_config", lambda root, _plugin: root / "demo_clean.yml"
    )
    monkeypatch.setattr(
        module,
        "_prepare_official_task",
        lambda **_kwargs: (task_env, {"render_freq": 0}, "640x480", 100000),
    )

    output = tmp_path / "output"
    result = plan_task_episodes(
        robotwin_root=tmp_path,
        plugin_config=_write_plugin(tmp_path),
        task_name="pick_apple",
        output_dir=output,
        run_id="formal",
        episodes_per_task=2,
        max_seed_attempts=20,
        resume=False,
    )

    assert [item["seed"] for item in result["episodes"]] == [100002, 100003]
    assert task_env.setup_seeds == [100000, 100001, 100002, 100003]
    attempts = [
        json.loads(line)
        for line in (output / "pick_apple" / "seed_attempts.jsonl").read_text().splitlines()
    ]
    assert [(item["seed"], item["reason"]) for item in attempts] == [
        (100000, "unstable_error"),
        (100001, "expert_plan_failed"),
        (100002, "expert_plan_succeeded"),
        (100003, "expert_plan_succeeded"),
    ]
    stdout = capsys.readouterr().out
    assert "ROBOTWIN_PLAN_START" in stdout
    assert "ROBOTWIN_PLAN_PROGRESS task=pick_apple planned=2/2" in stdout
    assert "ROBOTWIN_PLAN_COMPLETE" in stdout


def test_episode_major_assignment_covers_1000_pairs_once_across_24_workers(
    tmp_path: Path,
) -> None:
    tasks = [f"task_{index:02d}" for index in range(50)]
    for task in tasks:
        task_dir = tmp_path / task
        task_dir.mkdir()
        (task_dir / "episode_plan.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "video_size": "640x480",
                    "task_step_limit": 700,
                    "episodes": [
                        {"task": task, "episode_index": index, "seed": 100000 + index}
                        for index in range(20)
                    ],
                }
            )
        )

    pairs = planned_episode_pairs(
        tasks=tasks,
        output_dir=tmp_path,
        episodes_per_task=20,
    )
    assignments = [pairs[index::24] for index in range(24)]
    identities = [
        (item["task"], item["episode_index"])
        for assignment in assignments
        for item in assignment
    ]

    assert len(pairs) == 1000
    assert len(identities) == len(set(identities)) == 1000
    assert sorted(len(assignment) for assignment in assignments) == [41] * 8 + [42] * 16
    assert [item["task"] for item in pairs[:50]] == tasks
    assert {item["episode_index"] for item in pairs[:50]} == {0}


def test_planner_resume_rebuilds_plan_from_atomic_accepted_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    task_dir = output / "pick_apple"
    task_dir.mkdir(parents=True)
    accepted_episode = {
        "task": "pick_apple",
        "episode_index": 0,
        "seed": 100000,
        "instruction": "pick the apple",
        "instruction_index": 0,
        "instruction_candidate_count": 1,
        "instruction_rng_seed": 100000,
    }
    (task_dir / "seed_attempts.jsonl").write_text(
        json.dumps(
            {
                "attempt_index": 0,
                "seed": 100000,
                "accepted": True,
                "reason": "expert_plan_succeeded",
                "episode": accepted_episode,
            }
        )
        + "\n"
    )
    (task_dir / "episode_plan_records.jsonl").write_text("")
    task_env = _PlanningTask()
    official = SimpleNamespace(
        UnStableError=_UnstableError,
        generate_episode_descriptions=lambda _task, info, _count: [
            {"seen": [f"instruction-{info[0]['seed']}"]}
        ],
    )
    monkeypatch.setattr(module, "_load_policy_plugin", lambda _path: None)
    monkeypatch.setattr(module, "_load_official_module", lambda _root: official)
    monkeypatch.setattr(
        module, "_validate_task_config", lambda root, _plugin: root / "demo_clean.yml"
    )
    monkeypatch.setattr(
        module,
        "_prepare_official_task",
        lambda **_kwargs: (task_env, {"render_freq": 0}, "640x480", 100000),
    )

    result = plan_task_episodes(
        robotwin_root=tmp_path,
        plugin_config=_write_plugin(tmp_path),
        task_name="pick_apple",
        output_dir=output,
        run_id="formal",
        episodes_per_task=2,
        max_seed_attempts=20,
        resume=True,
    )

    assert [item["seed"] for item in result["episodes"]] == [100000, 100002]
    assert task_env.setup_seeds == [100001, 100002]
    rebuilt = [
        json.loads(line)
        for line in (task_dir / "episode_plan_records.jsonl").read_text().splitlines()
    ]
    assert [item["seed"] for item in rebuilt] == [100000, 100002]


def test_progress_is_deduplicated_and_printed_live(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    success = {"task": "task_a", "episode_index": 0, "success": True}
    failure = {"task": "task_a", "episode_index": 1, "success": False}

    assert record_global_progress(tmp_path, success) == {"completed": 1, "successes": 1}
    assert record_global_progress(tmp_path, success) == {"completed": 1, "successes": 1}
    assert record_global_progress(tmp_path, failure) == {"completed": 2, "successes": 1}

    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 2
    assert "ROBOTWIN_PROGRESS completed=2/1000 successes=1 Success=0.500000" in capsys.readouterr().out


def test_aggregate_builds_canonical_task_records_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_path = tmp_path / "tasks.yml"
    tasks_path.write_text(yaml.safe_dump({"tasks": ["task_a"]}))
    output = tmp_path / "output"
    task_dir = output / "task_a"
    records_dir = task_dir / "_episode_records"
    records_dir.mkdir(parents=True)
    (task_dir / "episode_plan.json").write_text(
        json.dumps({"max_seed_attempts": 1000, "task_step_limit": 700})
    )
    for episode_index, success in enumerate((True, False)):
        (records_dir / f"episode_{episode_index:04d}.json").write_text(
            json.dumps(
                {
                    "task": "task_a",
                    "episode_index": episode_index,
                    "seed": 100000 + episode_index,
                    "success": success,
                }
            )
        )
    monkeypatch.setattr(module, "load_official_tasks", lambda _path: ["task_a"])
    monkeypatch.setattr(
        module,
        "verify_task_eval",
        lambda **_kwargs: {"successes": 1, "episodes": 2},
    )
    monkeypatch.setattr(
        module,
        "aggregate_suite",
        lambda **_kwargs: {"status": "completed", "total_successes": 1},
    )

    result = aggregate_episode_suite(
        tasks_path=tasks_path,
        output_dir=output,
        expected_checkpoint=tmp_path / "checkpoint",
        run_id="formal",
        episodes_per_task=2,
    )

    canonical = [
        json.loads(line)
        for line in (task_dir / "episode_records.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((task_dir / "run_manifest.json").read_text())
    assert [item["episode_index"] for item in canonical] == [0, 1]
    assert manifest["status"] == "completed"
    assert manifest["episode_parallel"] is True
    assert manifest["task_config"] == "demo_clean"
    assert manifest["instruction_type"] == "seen"
    assert result == {"status": "completed", "total_successes": 1}
