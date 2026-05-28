from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

from praxis_eval.evaluation.artifacts import resolve_eval_step_dir

from praxis.scripts.eval_monitor import (
    WANDB_RUN_INFO_FILENAME,
    _cleanup_eval_dir,
    _cleanup_staging_dir,
    _cleanup_unneeded_eval_dirs,
    _collect_finished_eval,
    _launch_eval,
    _parse_args,
    _read_candidate_manifest,
    _read_evaluated_steps,
    _should_preempt_current_eval,
    _snapshot_candidate,
    _staging_dir_for_step,
    _WandBState,
    _write_evaluated_steps,
)
from praxis.scripts.eval_monitor_modules.main import (
    MonitorState,
    maybe_timeout_current_eval,
    record_eval_failure,
)


def _write_manifest(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _write_candidate_checkpoint(path: Path, *, marker: str) -> None:
    model_dir = path / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "marker.txt").write_text(marker + "\n")
    (model_dir / "config.json").write_text("{}\n")
    (model_dir / "model.safetensors").write_text("weights\n")
    (model_dir / "policy_io.json").write_text("{}\n")
    (model_dir / "norm_stats.pt").write_text("dummy\n")
    (model_dir / "train_config.yaml").write_text("policy:\n  name: dummy\n")


def _write_sharded_candidate_checkpoint(path: Path, *, marker: str) -> None:
    model_dir = path / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "marker.txt").write_text(marker + "\n")
    (model_dir / "config.json").write_text("{}\n")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "left.weight": "model-00001-of-00002.safetensors",
                    "right.weight": "model-00002-of-00002.safetensors",
                },
            }
        )
        + "\n"
    )
    (model_dir / "model-00001-of-00002.safetensors").write_text("left\n")
    (model_dir / "model-00002-of-00002.safetensors").write_text("right\n")
    (model_dir / "policy_io.json").write_text("{}\n")
    (model_dir / "norm_stats.pt").write_text("dummy\n")
    (model_dir / "train_config.yaml").write_text("policy:\n  name: dummy\n")


def test_parse_args_defaults_record_episodes_to_zero(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "praxis-eval-monitor",
            "--train-dir",
            str(tmp_path),
            "--num-eval-per-task",
            "10",
            "--num-parallel-env",
            "4",
        ],
    )

    args = _parse_args()

    assert args.record_episodes_per_task == 0


def test_parse_args_flattens_repeated_eval_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "praxis-eval-monitor",
            "--train-dir",
            str(tmp_path),
            "--num-eval-per-task",
            "10",
            "--num-parallel-env",
            "4",
            "--eval-override",
            "eval.policy_kwargs.decode_keep_k=4",
            "--eval-override",
            "env.python_bin=/tmp/python",
            "eval.rollout_failure_retries=2",
        ],
    )

    args = _parse_args()

    assert args.eval_override == [
        "eval.policy_kwargs.decode_keep_k=4",
        "env.python_bin=/tmp/python",
        "eval.rollout_failure_retries=2",
    ]


def test_eval_monitor_timeout_counts_as_retry_and_then_skips(tmp_path: Path) -> None:
    class _Logger:
        def warning(self, *_args, **_kwargs) -> None:
            return

    cleanup_calls: list[int] = []
    state = MonitorState(
        evaluated=set(),
        fail_counts={},
        failed=set(),
        current_proc=object(),
        current_step=100,
        current_launch_ts=0.0,
    )

    def cleanup_current_eval(
        state, *, eval_base, keep_results, kill_process, mark_progress
    ):
        del eval_base, keep_results, kill_process, mark_progress
        cleanup_calls.append(cast(int, state.current_step))
        state.current_proc = None
        state.current_step = None

    def record_failure(state, *, step, rc):
        record_eval_failure(
            state,
            step=step,
            rc=rc,
            max_eval_retries=2,
            logger=_Logger(),
        )

    maybe_timeout_current_eval(
        state=state,
        eval_base=tmp_path,
        eval_timeout_sec=1,
        require_step_fn=lambda step: cast(int, step),
        record_eval_failure_fn=record_failure,
        cleanup_current_eval_fn=cleanup_current_eval,
        logger=_Logger(),
    )

    assert state.fail_counts == {100: 1}
    assert state.failed == set()
    assert cleanup_calls == [100]

    state.current_proc = object()
    state.current_step = 100
    state.current_launch_ts = 0.0
    maybe_timeout_current_eval(
        state=state,
        eval_base=tmp_path,
        eval_timeout_sec=1,
        require_step_fn=lambda step: cast(int, step),
        record_eval_failure_fn=record_failure,
        cleanup_current_eval_fn=cleanup_current_eval,
        logger=_Logger(),
    )

    assert state.fail_counts == {100: 2}
    assert state.failed == {100}
    assert cleanup_calls == [100, 100]


def test_read_candidate_manifest_orders_best_first_and_assigns_overlap_to_best(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _write_candidate_checkpoint(checkpoint_dir / "best" / "step_100", marker="best-100")
    _write_candidate_checkpoint(checkpoint_dir / "best" / "step_90", marker="best-90")
    _write_candidate_checkpoint(
        checkpoint_dir / "recent" / "step_100", marker="recent-100"
    )
    _write_candidate_checkpoint(
        checkpoint_dir / "recent" / "step_110", marker="recent-110"
    )
    _write_candidate_checkpoint(
        checkpoint_dir / "recent" / "step_105", marker="recent-105"
    )

    _write_manifest(
        checkpoint_dir / "best" / "topk_manifest.json",
        [
            {
                "step": 100,
                "metric_key": "val_reconst_loss",
                "value": 0.1,
                "file": "step_100",
            },
            {
                "step": 90,
                "metric_key": "val_reconst_loss",
                "value": 0.2,
                "file": "step_90",
            },
        ],
    )
    _write_manifest(
        checkpoint_dir / "recent" / "recent_manifest.json",
        [
            {"step": 110, "file": "step_110"},
            {"step": 105, "file": "step_105"},
            {"step": 100, "file": "step_100"},
        ],
    )

    manifest = _read_candidate_manifest(checkpoint_dir)

    assert manifest is not None
    assert [entry["step"] for entry in manifest] == [100, 90, 110, 105]
    step_100 = next(entry for entry in manifest if entry["step"] == 100)
    assert step_100["path"] == checkpoint_dir / "best" / "step_100"
    assert step_100["tier"] == "best"
    step_110 = next(entry for entry in manifest if entry["step"] == 110)
    assert step_110["path"] == checkpoint_dir / "recent" / "step_110"
    assert step_110["tier"] == "recent"


def test_should_preempt_current_eval_only_for_recent_with_unevaluated_best() -> None:
    manifest = [
        {"step": 100, "path": Path("/tmp/best_100"), "tier": "best"},
        {"step": 95, "path": Path("/tmp/recent_95"), "tier": "recent"},
    ]

    assert _should_preempt_current_eval(
        current_tier="recent",
        manifest=manifest,
        evaluated=set(),
        failed=set(),
    )
    assert not _should_preempt_current_eval(
        current_tier="best",
        manifest=manifest,
        evaluated=set(),
        failed=set(),
    )
    assert not _should_preempt_current_eval(
        current_tier="recent",
        manifest=manifest,
        evaluated={100},
        failed=set(),
    )
    assert not _should_preempt_current_eval(
        current_tier="recent",
        manifest=manifest,
        evaluated=set(),
        failed={100},
    )


def test_evaluated_steps_roundtrip(tmp_path: Path) -> None:
    eval_base = tmp_path / "eval"

    _write_evaluated_steps(eval_base, {300, 100, 200})

    assert _read_evaluated_steps(eval_base) == {100, 200, 300}


def test_cleanup_unneeded_eval_dirs_keeps_results_and_drops_media_for_evicted_steps(
    tmp_path: Path,
) -> None:
    eval_base = tmp_path / "eval"
    for step in (100, 110, 120):
        step_dir = resolve_eval_step_dir(eval_base, step)
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "results.json").write_text("{}\n")
        media = step_dir / "media"
        media.mkdir(parents=True, exist_ok=True)
        (media / "clip.mp4").write_text("x\n")

    _cleanup_unneeded_eval_dirs(
        eval_base,
        retained_steps={110},
        active_step=120,
    )

    assert resolve_eval_step_dir(eval_base, 100).exists()
    assert not (resolve_eval_step_dir(eval_base, 100) / "media").exists()
    assert resolve_eval_step_dir(eval_base, 110).exists()
    assert (resolve_eval_step_dir(eval_base, 110) / "media").exists()
    assert resolve_eval_step_dir(eval_base, 120).exists()
    assert (resolve_eval_step_dir(eval_base, 120) / "media").exists()


def test_snapshot_candidate_copies_pretrained_model_and_cleanup_removes_staging(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "checkpoints" / "recent" / "step_200"
    _write_candidate_checkpoint(candidate_dir, marker="recent-200")
    eval_base = tmp_path / "eval"

    staged_model_dir = _snapshot_candidate(
        {"step": 200, "path": candidate_dir, "tier": "recent"},
        eval_base,
    )

    assert (
        staged_model_dir == _staging_dir_for_step(eval_base, 200) / "pretrained_model"
    )
    assert (staged_model_dir / "marker.txt").read_text().strip() == "recent-200"
    assert (staged_model_dir / "train_config.yaml").exists()

    _cleanup_staging_dir(eval_base, 200)
    assert not _staging_dir_for_step(eval_base, 200).exists()


def test_snapshot_candidate_recopies_incomplete_staging_dir(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "checkpoints" / "best" / "step_300"
    _write_candidate_checkpoint(candidate_dir, marker="best-300")
    eval_base = tmp_path / "eval"
    staged_model_dir = _staging_dir_for_step(eval_base, 300) / "pretrained_model"
    staged_model_dir.mkdir(parents=True, exist_ok=True)
    (staged_model_dir / "train_config.yaml").write_text("policy:\n  name: stale\n")

    out = _snapshot_candidate(
        {"step": 300, "path": candidate_dir, "tier": "best"},
        eval_base,
    )

    assert out == staged_model_dir
    assert (staged_model_dir / "train_config.yaml").exists()
    assert (staged_model_dir / "config.json").exists()
    assert (staged_model_dir / "model.safetensors").read_text().strip() == "weights"
    assert (staged_model_dir / "marker.txt").read_text().strip() == "best-300"


def test_snapshot_candidate_recopies_sharded_staging_with_missing_shard(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "checkpoints" / "best" / "step_320"
    _write_sharded_candidate_checkpoint(candidate_dir, marker="best-320")
    eval_base = tmp_path / "eval"
    staged_model_dir = _staging_dir_for_step(eval_base, 320) / "pretrained_model"
    staged_model_dir.mkdir(parents=True, exist_ok=True)
    (staged_model_dir / "train_config.yaml").write_text("policy:\n  name: stale\n")
    (staged_model_dir / "config.json").write_text("{}\n")
    (staged_model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "left.weight": "model-00001-of-00002.safetensors",
                    "right.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
        + "\n"
    )
    (staged_model_dir / "model-00001-of-00002.safetensors").write_text("stale\n")

    out = _snapshot_candidate(
        {"step": 320, "path": candidate_dir, "tier": "best"},
        eval_base,
    )

    assert out == staged_model_dir
    assert (staged_model_dir / "marker.txt").read_text().strip() == "best-320"
    assert (
        staged_model_dir / "model-00002-of-00002.safetensors"
    ).read_text().strip() == "right"


def test_launch_eval_appends_extra_eval_overrides(tmp_path: Path, monkeypatch) -> None:
    pretrained = tmp_path / "ckpt" / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=True)
    (pretrained / "config.json").write_text("{}\n")
    (pretrained / "model.safetensors").write_text("weights\n")
    (pretrained / "train_config.yaml").write_text(
        "\n".join(
            [
                "policy:",
                "  name: bar_fast_pali",
                "env:",
                "  type: libero",
                "  task: libero_10",
                "device: cuda",
            ]
        )
        + "\n"
    )
    eval_output_dir = tmp_path / "eval_out"

    captured: dict[str, object] = {}

    class _DummyProc:
        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return _DummyProc()

    monkeypatch.setattr("praxis.scripts.eval_monitor.subprocess.Popen", _fake_popen)

    _launch_eval(
        eval_command=["praxis-eval"],
        pretrained_path=pretrained,
        eval_output_dir=eval_output_dir,
        num_eval_per_task=10,
        num_parallel_env=4,
        record_episodes_per_task=2,
        extra_eval_overrides=[
            "eval.policy_kwargs.decode_keep_k=4",
            "eval.policy_kwargs.decode_temperature=0.7",
            "eval.policy_kwargs.decode_top_k=8",
            "eval.policy_kwargs.decode_strict=true",
        ],
    )

    cmd = cast(list[str], captured["cmd"])
    assert "policy.pretrained_path=" + str(pretrained) in cmd
    assert "eval.output_dir=" + str(eval_output_dir) in cmd
    assert "eval.policy_kwargs.decode_keep_k=4" in cmd
    assert "+eval.policy_kwargs.decode_keep_k=4" not in cmd
    assert "eval.policy_kwargs.decode_temperature=0.7" in cmd
    assert "+eval.policy_kwargs.decode_temperature=0.7" not in cmd
    assert "eval.policy_kwargs.decode_top_k=8" in cmd
    assert "+eval.policy_kwargs.decode_top_k=8" not in cmd
    assert "+eval.policy_kwargs.decode_strict=true" in cmd


def test_cleanup_eval_dir_removes_whole_dir_for_incomplete_eval(tmp_path: Path) -> None:
    eval_base = tmp_path / "eval"
    step_dir = resolve_eval_step_dir(eval_base, 200)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "media").mkdir(parents=True, exist_ok=True)
    (step_dir / "media" / "clip.mp4").write_text("x\n")
    (step_dir / "partial.txt").write_text("incomplete\n")

    _cleanup_eval_dir(eval_base, 200, keep_results=False)
    assert not step_dir.exists()


def test_wandb_state_uses_temp_dir_and_cleans_up_on_init_failure(
    tmp_path: Path, monkeypatch
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / WANDB_RUN_INFO_FILENAME).write_text(
        json.dumps({"id": "run123", "project": "proj", "entity": "ent"}) + "\n"
    )

    captured: dict[str, str] = {}

    def _fake_init(**kwargs):
        captured["dir"] = str(kwargs["dir"])
        raise RuntimeError("network down")

    fake_wandb = type("FakeWandb", (), {"init": staticmethod(_fake_init)})
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wb = _WandBState(train_dir)
    assert wb._ensure() is None
    assert "dir" in captured
    assert not Path(captured["dir"]).exists()
    assert not (tmp_path / "wandb").exists()


def test_wandb_state_finish_cleans_temp_run_dir(tmp_path: Path, monkeypatch) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / WANDB_RUN_INFO_FILENAME).write_text(
        json.dumps({"id": "run123", "project": "proj", "entity": "ent"}) + "\n"
    )

    captured: dict[str, str] = {}

    class _DummyRun:
        pass

    def _fake_init(**kwargs):
        run_dir = Path(str(kwargs["dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifact.txt").write_text("x\n")
        captured["dir"] = str(run_dir)
        return _DummyRun()

    finished = {"called": False}

    def _fake_finish(*, exit_code=None, quiet=None) -> None:
        assert exit_code == 0
        assert quiet is True
        finished["called"] = True

    fake_wandb = type(
        "FakeWandb",
        (),
        {
            "init": staticmethod(_fake_init),
            "finish": staticmethod(_fake_finish),
            "define_metric": staticmethod(lambda *args, **kwargs: None),
        },
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wb = _WandBState(train_dir)
    run = wb._ensure()
    assert run is not None
    assert Path(captured["dir"]).exists()

    wb.finish()
    assert finished["called"] is True
    assert not Path(captured["dir"]).exists()


def test_wandb_state_disables_after_log_failure(tmp_path: Path, monkeypatch) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / WANDB_RUN_INFO_FILENAME).write_text(
        json.dumps({"id": "run123", "project": "proj", "entity": "ent"}) + "\n"
    )

    calls = {"log": 0, "finish": 0}

    class _DummyRun:
        pass

    def _fake_init(**kwargs):
        run_dir = Path(str(kwargs["dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        return _DummyRun()

    def _fake_log(_payload) -> None:
        calls["log"] += 1
        raise RuntimeError("wandb transient failure")

    def _fake_finish(*, exit_code=None, quiet=None) -> None:
        assert exit_code == 0
        assert quiet is True
        calls["finish"] += 1

    fake_wandb = type(
        "FakeWandb",
        (),
        {
            "init": staticmethod(_fake_init),
            "log": staticmethod(_fake_log),
            "finish": staticmethod(_fake_finish),
            "define_metric": staticmethod(lambda *args, **kwargs: None),
        },
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wb = _WandBState(train_dir)
    wb.log(100, {"sim_eval/success_rate": 0.5})
    assert wb._disabled is True
    assert wb._run is None
    assert calls["log"] == 1
    assert calls["finish"] == 1

    wb.log(200, {"sim_eval/success_rate": 0.6})
    assert calls["log"] == 1
    assert calls["finish"] == 1


def test_eval_monitor_keeps_policy_diagnostics_out_of_wandb(
    tmp_path: Path,
) -> None:
    eval_base = tmp_path / "eval"
    step_dir = resolve_eval_step_dir(eval_base, 100)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "results.json").write_text(
        json.dumps(
            {
                "overall": {
                    "success_rate": 0.5,
                    "avg_episode_length": 123.0,
                },
                "policy_diagnostics": {
                    "flow_sampling_steps": {
                        "count": 2,
                        "mean": 10.0,
                    },
                    "decode_keep_k": {
                        "count": 2,
                        "mean": 16.0,
                    },
                },
            }
        )
        + "\n"
    )

    class _DoneProc:
        def poll(self):
            return 0

    class _TopK:
        def consider(self, *args, **kwargs):
            return None

    class _WB:
        def __init__(self) -> None:
            self.calls: list[tuple[int, dict[str, float]]] = []

        def log(self, step: int, metrics: dict[str, float]) -> None:
            self.calls.append((step, dict(metrics)))

    state = MonitorState(
        evaluated=set(),
        fail_counts={},
        failed=set(),
        current_proc=_DoneProc(),
        current_step=100,
    )
    wb = _WB()

    _collect_finished_eval(
        state=state,
        eval_base=eval_base,
        eval_topk=_TopK(),
        wb=wb,
    )

    assert wb.calls == [
        (
            100,
            {
                "sim_eval/success_rate": 0.5,
                "sim_eval/avg_episode_length": 123.0,
            },
        )
    ]
