# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

from scripts.infra.check_eval_jobs import (
    JobHealth,
    _print_table,
    classify_log,
    classify_missing_log,
    resolve_log_err,
)


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_classify_unhealthy_when_crash_marker_exists(tmp_path: Path) -> None:
    log_path = (
        tmp_path / "outputs/slurm/eval_monitor_robocasa_bar_boat_pali/123_0_log.err"
    )
    _write_log(
        log_path,
        "Eval:   0%|          | 0/2000 [00:01<?, ?ep/s]\n"
        "Traceback (most recent call last):\n"
        "CalledProcessError\n",
    )

    result = classify_log("123", log_path)

    assert result.job_name == "robocasa/bar_boat_pali"
    assert result.status == "unhealthy"
    assert result.pbar_seen is True
    assert result.progressed is False


def test_classify_pending_when_new_attempt_started_after_crash(tmp_path: Path) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_x/126_0_log.err"
    _write_log(
        log_path,
        "Eval:   0%|          | 0/250 [06:00<?, ?ep/s]\n"
        "Traceback (most recent call last):\n"
        "EOFError\n"
        "Eval:   0%|          | 0/250 [00:00<?, ?ep/s]\n",
    )

    result = classify_log("126", log_path)

    assert result.status == "pending"
    assert result.pbar_seen is True
    assert result.progressed is False
    assert result.last_progress == "0/250"
    assert result.last_success_rate_pct is None
    assert result.last_sec_per_ep is None
    assert "latest eval attempt started after earlier crash marker" in result.reason


def test_classify_healthy_when_new_attempt_progressed_after_crash(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_x/127_0_log.err"
    _write_log(
        log_path,
        "Eval:   0%|          | 0/250 [06:00<?, ?ep/s]\n"
        "Traceback (most recent call last):\n"
        "EOFError\n"
        "Eval:   0%|          | 0/250 [00:00<?, ?ep/s, succ_rate=0.0%]\n"
        "Eval:   4%|▍         | 10/250 [02:22<00:00, 14.23s/ep, avg_len=308.0, ep_s=0.07, succ_rate=40.0%]\n",
    )

    result = classify_log("127", log_path)

    assert result.status == "healthy"
    assert result.pbar_seen is True
    assert result.progressed is True
    assert result.last_progress == "10/250"
    assert result.last_success_rate_pct == 40.0
    assert result.last_sec_per_ep == 14.23
    assert "latest eval attempt progressed after earlier crash marker" in result.reason


def test_classify_pending_when_pbar_seen_but_no_progress(tmp_path: Path) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_x/124_0_log.err"
    _write_log(
        log_path,
        "\rEval:   0%|          | 0/2000 [09:06<?, ?ep/s, succ_rate=0.0%]\r\n",
    )

    result = classify_log("124", log_path)

    assert result.status == "pending"
    assert result.pbar_seen is True
    assert result.progressed is False
    assert result.last_progress == "0/2000"
    assert result.last_success_rate_pct == 0.0
    assert result.last_sec_per_ep is None


def test_classify_healthy_when_pbar_progressed(tmp_path: Path) -> None:
    log_path = tmp_path / "outputs/slurm/eval_robocasa_bar_boat_pali/125_0_log.err"
    _write_log(
        log_path,
        "Eval:   0%|          | 0/2000 [00:01<?, ?ep/s, succ_rate=0.0%]\n"
        "Eval:   1%|          | 20/2000 [05:00<00:00, 15.00s/ep, avg_len=380.0, ep_s=0.07, succ_rate=35.0%]\n",
    )

    result = classify_log("125", log_path)

    assert result.status == "healthy"
    assert result.job_name == "robocasa/bar_boat_pali"
    assert result.pbar_seen is True
    assert result.progressed is True
    assert result.last_progress == "20/2000"
    assert result.last_success_rate_pct == 35.0
    assert result.last_sec_per_ep == 15.0


def test_classify_ignores_mshab_task_progress_without_global_pbar(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_mshab_bar_oat_pali/130_0_log.err"
    _write_log(
        log_path,
        "MSHAB_EVAL task_alias=set_table_pick done=20/50 succ_rate=35.0% "
        "succ_end=30.0% elapsed_s=300.0 15.00s/ep\n",
    )

    result = classify_log("130", log_path)

    assert result.job_name == "mshab/bar_oat_pali"
    assert result.status == "pending"
    assert result.last_progress is None
    assert result.last_success_rate_pct is None
    assert result.last_sec_per_ep is None


def test_classify_pending_when_only_simpler_start_line_seen(tmp_path: Path) -> None:
    log_path = (
        tmp_path / "outputs/slurm/eval_monitor_simpler_bar_oat_pali/134_0_log.err"
    )
    _write_log(
        log_path,
        "SIMPLER_EVAL env_id=PutCarrotOnPlateInScene-v1 done=0/50 "
        "succ_rate=0.0% elapsed_s=0.0\n",
    )

    result = classify_log("134", log_path)

    assert result.job_name == "simpler/bar_oat_pali"
    assert result.status == "pending"
    assert result.last_progress is None
    assert result.last_success_rate_pct is None
    assert result.last_sec_per_ep is None


def test_classify_ignores_simpler_task_progress_without_global_pbar(
    tmp_path: Path,
) -> None:
    log_path = (
        tmp_path / "outputs/slurm/eval_monitor_simpler_bar_oat_pali/135_0_log.err"
    )
    _write_log(
        log_path,
        "SIMPLER_EVAL env_id=PutCarrotOnPlateInScene-v1 done=0/50 "
        "succ_rate=0.0% elapsed_s=0.0\n"
        "SIMPLER_EVAL env_id=PutCarrotOnPlateInScene-v1 done=16/50 "
        "succ_rate=12.5% elapsed_s=64.0 4.00s/ep\n",
    )

    result = classify_log("135", log_path)

    assert result.job_name == "simpler/bar_oat_pali"
    assert result.status == "pending"
    assert result.last_progress is None
    assert result.last_success_rate_pct is None
    assert result.last_sec_per_ep is None


def test_classify_healthy_when_new_simpler_global_pbar_progressed_after_crash(
    tmp_path: Path,
) -> None:
    log_path = (
        tmp_path / "outputs/slurm/eval_monitor_simpler_bar_oat_pali/136_0_log.err"
    )
    _write_log(
        log_path,
        "SIMPLER_EVAL env_id=PutCarrotOnPlateInScene-v1 done=0/50 "
        "succ_rate=0.0% elapsed_s=0.0\n"
        "Traceback (most recent call last):\n"
        "EOFError\n"
        "SIMPLER_EVAL env_id=PutCarrotOnPlateInScene-v1 done=0/50 "
        "succ_rate=0.0% elapsed_s=0.0\n"
        "Eval:  20%|██        | 50/250 [03:37<14:28, 4.34s/ep, "
        "avg_len=301.0, ep_s=0.23, succ_rate=4.0%]\n",
    )

    result = classify_log("136", log_path)

    assert result.status == "healthy"
    assert result.last_progress == "50/250"
    assert result.last_success_rate_pct == 4.0
    assert result.last_sec_per_ep == 4.34
    assert "latest eval attempt progressed after earlier crash marker" in result.reason


def test_classify_uses_global_pbar_not_simpler_task_lines(tmp_path: Path) -> None:
    log_path = (
        tmp_path / "outputs/slurm/eval_monitor_simpler_bar_oat_pali/137_0_log.err"
    )
    _write_log(
        log_path,
        "SIMPLER_EVAL env_id=TaskA-v1 done=50/50 succ_rate=0.0% "
        "elapsed_s=60.0 1.20s/ep\n"
        "Eval:  20%|██        | 50/250 [01:00<04:00, 1.20s/ep, "
        "avg_len=100.0, ep_s=0.83, succ_rate=0.0%]\n"
        "SIMPLER_EVAL env_id=TaskB-v1 done=50/50 succ_rate=100.0% "
        "elapsed_s=60.0 1.20s/ep\n"
        "Eval:  40%|████      | 100/250 [02:00<03:00, 1.20s/ep, "
        "avg_len=110.0, ep_s=0.83, succ_rate=50.0%]\n",
    )

    result = classify_log("137", log_path)

    assert result.job_name == "simpler/bar_oat_pali"
    assert result.status == "healthy"
    assert result.last_progress == "100/250"
    assert result.last_success_rate_pct == 50.0
    assert result.last_sec_per_ep == 1.2


def test_classify_log_derives_simpler_job_name_from_overrides(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_bridge_bar_oat_pali/132_0_log.err"
    _write_log(
        log_path,
        "Error executing job with overrides: ['policy=bar_oat_pali', "
        "'env.type=simpler', 'env.task=bridge']\n"
        "Traceback (most recent call last):\n",
    )

    result = classify_log("132", log_path)

    assert result.job_name == "simpler/bar_oat_pali"
    assert result.status == "unhealthy"


def test_classify_log_derives_simpler_job_name_from_stdout_config(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_bridge_bar_oat_pali/133_0_log.err"
    _write_log(log_path, "Running Real2Sim Evaluation\n")
    _write_log(
        log_path.with_name("133_0_log.out"),
        "policy:\n  name: bar_oat_pali\nenv:\n  type: simpler\n  task: bridge\n",
    )

    result = classify_log("133", log_path)

    assert result.job_name == "simpler/bar_oat_pali"
    assert result.status == "pending"


def test_classify_pending_when_only_mshab_start_line_seen(tmp_path: Path) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_mshab_bar_fast_pali/131_0_log.err"
    _write_log(
        log_path,
        "MSHAB_EVAL task_alias=set_table_pick done=0/50 succ_rate=0.0% "
        "succ_end=0.0% elapsed_s=12.0\n",
    )

    result = classify_log("131", log_path)

    assert result.job_name == "mshab/bar_fast_pali"
    assert result.status == "pending"
    assert result.last_progress is None
    assert result.last_success_rate_pct is None
    assert result.last_sec_per_ep is None


def test_classify_healthy_derives_sec_per_ep_from_ep_s_when_needed(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_x/128_0_log.err"
    _write_log(
        log_path,
        "Eval:   8%|▊         | 20/250 [05:00<?, ?ep/s, avg_len=380.0, ep_s=0.08, succ_rate=35.0%]\n",
    )

    result = classify_log("128", log_path)

    assert result.status == "healthy"
    assert result.last_success_rate_pct == 35.0
    assert result.last_sec_per_ep == 12.5


def test_resolve_log_err_picks_newest_match(tmp_path: Path) -> None:
    older = tmp_path / "outputs/slurm/a/200_0_log.err"
    newer = tmp_path / "outputs/slurm/b/200_0_log.err"
    _write_log(older, "x")
    _write_log(newer, "y")

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = resolve_log_err(tmp_path / "outputs/slurm", "200")

    assert resolved == newer


def test_classify_missing_log_is_unhealthy() -> None:
    result = classify_missing_log("999")
    assert result.job_name == "N/A"
    assert result.status == "unhealthy"
    assert result.reason == "log.err not found"


def test_print_table_shows_compact_job_name(capsys) -> None:
    _print_table(
        [
            JobHealth(
                job_id="125",
                job_name="libero/bar_oat_pali",
                status="healthy",
                log_err="outputs/slurm/eval_monitor_x/125_0_log.err",
                pbar_seen=True,
                progressed=True,
                last_progress="20/2000",
                last_success_rate_pct=35.0,
                last_sec_per_ep=15.04,
                reason="eval progress marker advanced",
            ),
            JobHealth(
                job_id="456",
                job_name="robocasa/bar_boat_pali",
                status="unhealthy",
                log_err="outputs/slurm/eval_monitor_x/456_0_log.err",
                pbar_seen=False,
                progressed=False,
                last_progress=None,
                last_success_rate_pct=None,
                last_sec_per_ep=None,
                reason="crash marker: CalledProcessError",
            ),
        ]
    )

    lines = capsys.readouterr().out.splitlines()
    header = lines[0]
    healthy_row = lines[2]
    unhealthy_row = lines[3]
    assert "job" in header
    assert healthy_row.startswith("libero/bar_oat_pali   |125")
    assert unhealthy_row.startswith("robocasa/bar_boat_pali|456")
    assert "|job_id|" in header
    assert "last_prog" in header
    assert "sr(%)" in header
    assert "succ_rate" not in header
    assert "s/ep" in header
    assert "eta" in header
    assert "sec/ep" not in header
    assert " pbar " not in header
    assert "progressed" not in header
    assert "| 35.0|15.0|00:08:16|progress" in healthy_row
    assert "35.0%" not in healthy_row
    assert unhealthy_row.endswith("crash: called-process")


def test_classify_log_uses_na_when_job_name_is_not_derivable(tmp_path: Path) -> None:
    log_path = tmp_path / "outputs/slurm/eval_monitor_weird_slug/129_0_log.err"
    _write_log(
        log_path,
        "Eval:   1%|          | 20/2000 [05:00<00:00, 15.00s/ep, avg_len=380.0, ep_s=0.07, succ_rate=35.0%]\n",
    )

    result = classify_log("129", log_path)

    assert result.job_name == "N/A"
