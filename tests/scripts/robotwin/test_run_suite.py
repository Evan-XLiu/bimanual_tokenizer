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
)


def test_official_clean_task_set_has_exactly_50_unique_tasks() -> None:
    tasks = load_official_tasks()

    assert len(tasks) == 50
    assert len(set(tasks)) == 50
    assert tasks[0] == "adjust_bottle"
    assert tasks[-1] == "turn_switch"


def test_twenty_four_way_assignment_covers_each_task_once_without_seed_sharding() -> None:
    tasks = load_official_tasks()
    assignments = [
        assigned_tasks(tasks, worker_index=index, worker_count=24) for index in range(24)
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
