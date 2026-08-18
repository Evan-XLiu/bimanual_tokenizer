# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.robotwin.run_suite import (
    aggregate_suite,
    assigned_tasks,
    load_official_tasks,
    supervise_suite,
    supervisor_exit_code,
)


def test_official_clean_task_set_has_exactly_50_unique_tasks() -> None:
    tasks = load_official_tasks()

    assert len(tasks) == 50
    assert len(set(tasks)) == 50
    assert tasks[0] == "adjust_bottle"
    assert tasks[-1] == "turn_switch"


def test_twenty_four_way_assignment_covers_each_task_once_without_seed_sharding() -> (
    None
):
    tasks = load_official_tasks()
    assignments = [
        assigned_tasks(tasks, worker_index=index, worker_count=24)
        for index in range(24)
    ]

    assert sorted(task for assignment in assignments for task in assignment) == sorted(
        tasks
    )
    assert [len(assignment) for assignment in assignments] == [
        *([3] * 2),
        *([2] * 22),
    ]


def test_plugin_matches_official_demo_clean_seen_protocol() -> None:
    root = Path("scripts/robotwin")
    plugin = yaml.safe_load(
        (root / "policy" / "praxis_ar" / "deploy_policy.yml").read_text()
    )

    assert plugin["instruction_type"] == "seen"


def test_aggregate_suite_requires_ema_seen_clean_and_all_media(tmp_path: Path) -> None:
    tasks = ["task_a", "task_b"]
    tasks_path = tmp_path / "tasks.yml"
    tasks_path.write_text(yaml.safe_dump({"tasks": tasks}))
    checkpoint = tmp_path / "step_10095" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    output = tmp_path / "eval"
    for task, successes in zip(tasks, (3, 1), strict=True):
        task_dir = output / task
        task_dir.mkdir(parents=True)
        (task_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "episodes": 4,
                    "successes": successes,
                    "success_rate": successes / 4,
                }
            )
        )
        (task_dir / "media_manifest.json").write_text(
            json.dumps({"complete": True, "episodes": [{}, {}, {}, {}]})
        )
        (task_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "task_config": "demo_clean",
                    "instruction_type": "seen",
                    "seed": 0,
                    "record_episodes_per_task": 4,
                    "expected_checkpoint": {
                        "checkpoint": {"path": str(checkpoint.resolve())}
                    },
                }
            )
        )

    import scripts.robotwin.run_suite as module

    original_count = module.EXPECTED_TASK_COUNT
    module.EXPECTED_TASK_COUNT = 2
    try:
        result = aggregate_suite(
            tasks_path=tasks_path,
            output_dir=output,
            expected_checkpoint=checkpoint,
            run_id="formal-ema",
            episodes_per_task=4,
        )
    finally:
        module.EXPECTED_TASK_COUNT = original_count

    assert result["checkpoint_label"] == "final EMA step10095"
    assert result["total_successes"] == 4
    assert result["total_episodes"] == 8
    assert result["macro_success_rate"] == 0.5
    assert result["aggregate_success_rate"] == 0.5
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.csv").is_file()


def test_dynamic_supervisor_skips_completed_tasks_and_refills_failed_slot(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tasks = ["task_a", "task_b", "task_c", "task_d"]
    tasks_path = tmp_path / "tasks.yml"
    tasks_path.write_text(yaml.safe_dump({"tasks": tasks}))
    output = tmp_path / "eval"
    completed_dir = output / "task_a"
    completed_dir.mkdir(parents=True)
    (completed_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed"})
    )
    (completed_dir / "comparison.json").write_text(json.dumps({"episodes": 2}))
    (completed_dir / "episode_records.jsonl").write_text("{}\n{}\n")
    calls: list[tuple[str, str, int]] = []

    def fake_runner(**kwargs):
        task = kwargs["task_name"]
        calls.append(
            (
                task,
                kwargs["subprocess_env"]["CUDA_VISIBLE_DEVICES"],
                kwargs["policy_port"],
            )
        )
        kwargs["line_callback"](
            f"ROBOTWIN_EPISODE_RESULT task={task} task_completed=1/2\n"
        )
        if task == "task_b":
            raise RuntimeError("simulator failed")
        kwargs["line_callback"](
            f"ROBOTWIN_EPISODE_RESULT task={task} task_completed=2/2\n"
        )
        return {"episodes": 2}

    import scripts.robotwin.run_suite as module

    monkeypatch.setattr(module, "EXPECTED_TASK_COUNT", len(tasks))
    summary = supervise_suite(
        robotwin_root=tmp_path / "RoboTwin",
        tasks_path=tasks_path,
        policy_host="policy",
        base_policy_port=50100,
        output_dir=output,
        run_id="resume",
        expected_checkpoint=tmp_path / "checkpoint",
        gpu_count=1,
        simulators_per_gpu=1,
        server_count=1,
        episodes_per_task=2,
        resume=True,
        task_runner=fake_runner,
    )

    assert [call[0] for call in calls] == ["task_b", "task_c", "task_d"]
    assert all(call[1:] == ("0", 50100) for call in calls)
    assert summary["status"] == "incomplete"
    assert summary["completed_tasks"] == 3
    assert summary["failed_tasks"] == {
        "task_b": "builtins.RuntimeError: simulator failed"
    }
    assert summary["remaining_tasks"] == ["task_b"]
    assert summary["completed_episodes"] == 7
    assert supervisor_exit_code(summary) == 1
    output_text = capsys.readouterr().out
    failed_position = output_text.index("event=task_failed")
    refill_position = output_text.index("event=task_start", failed_position + 1)
    assert refill_position > failed_position
    assert "active_slots=1 total_slots=1" in output_text
    assert "completed_episodes=7/8" in output_text
