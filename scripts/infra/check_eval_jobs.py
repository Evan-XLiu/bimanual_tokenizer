#!/usr/bin/env python3
"""Check eval job health from Slurm stderr logs.

Classification rules:
- unhealthy: crash markers found in log.err
- pending: Eval tqdm appeared but no progress yet
- healthy: Eval tqdm appeared and progressed

Only the shared global ``Eval: ... | done/total`` progress bar is treated as
authoritative. Env-family task logs such as ``SIMPLER_EVAL`` and ``MSHAB_EVAL``
are per-task debug output and are intentionally ignored here.

The script resolves each job id to:
  outputs/slurm/**/<job_id>_*_log.err
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_INFRA_DIR = Path(__file__).resolve().parent
if str(_INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(_INFRA_DIR))

from launch.source_checkout import install_source_checkout_imports  # noqa: E402

install_source_checkout_imports(_INFRA_DIR)

from launch.config import supported_policy_config_names  # noqa: E402
from praxis_eval.registry import available_drivers  # noqa: E402

EVAL_PBAR_RE = re.compile(r"Eval:\s+.*?\|\s*(\d+)\s*/\s*(\d+)")
SUCC_RATE_RE = re.compile(r"\bsucc_rate=([0-9]+(?:\.[0-9]+)?)%")
SEC_PER_EP_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)s/ep")
EP_S_RE = re.compile(r"\bep_s=([0-9]+(?:\.[0-9]+)?)")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
ENV_TYPE_OVERRIDE_RE = re.compile(r"(?:^|['\"\s,+])env\.type=([A-Za-z0-9_-]+)")
ENV_TYPE_CONFIG_RE = re.compile(
    r"(?ms)^env:\n(?:^[ \t].*\n)*?^[ \t]+type:\s*([A-Za-z0-9_-]+)\s*$"
)
POLICY_OVERRIDE_RE = re.compile(r"(?:^|['\"\s,+])policy=([A-Za-z0-9_-]+)\b")
CONFIG_NAME_RE = re.compile(r"(?m)^[ \t]+name:\s*([A-Za-z0-9_-]+)\s*$")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
JOB_ID_PREFIX_RE = re.compile(r"^\s*(\d+)(?:\s|$)")

CRASH_MARKERS = (
    "CalledProcessError",
    "returned non-zero exit status",
    "Submitted job triggered an exception",
    "EOFError",
    "error: *** JOB",
    "error: *** STEP",
)

REGISTERED_EVAL_ENVS = frozenset(available_drivers())
REGISTERED_POLICY_NAMES = frozenset(supported_policy_config_names())


@dataclass(frozen=True)
class EvalProgress:
    done: int
    total: int
    success_rate_pct: float | None
    sec_per_ep: float | None


@dataclass(frozen=True)
class JobHealth:
    job_id: str
    job_name: str
    status: str
    log_err: str | None
    pbar_seen: bool
    progressed: bool
    last_progress: str | None
    last_success_rate_pct: float | None
    last_sec_per_ep: float | None
    reason: str


def _match_float(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    return float(match.group(1))


def _normalize_job_id_arg(raw_job_id: str) -> str:
    match = JOB_ID_PREFIX_RE.match(str(raw_job_id))
    if match is not None:
        return match.group(1)
    return str(raw_job_id).strip()


def _derive_job_name(log_err_path: Path | None) -> str:
    if log_err_path is None:
        return "N/A"

    env_name = _derive_env_name_from_parent(log_err_path)
    if env_name is None:
        env_name = _derive_env_name_from_logs(log_err_path)

    variant = _derive_job_variant_from_parent(log_err_path)
    if variant is None:
        variant = _derive_job_variant_from_logs(log_err_path)

    if env_name is None and variant is None:
        return "N/A"
    return f"{env_name or 'NA'}/{variant or 'NA'}"


def _match_unique_registered_token(text: str, registered: frozenset[str]) -> str | None:
    matches = {token for token in _tokens(text) if token in registered}
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _trim_eval_parent_name(log_err_path: Path) -> str | None:
    parent_name = log_err_path.parent.name
    if parent_name.startswith("eval_monitor_"):
        return parent_name[len("eval_monitor_") :]
    if parent_name.startswith("eval_"):
        return parent_name[len("eval_") :]
    return None


def _derive_env_name_from_parent(log_err_path: Path) -> str | None:
    parent_name = _trim_eval_parent_name(log_err_path)
    if parent_name is None:
        return None

    return _match_unique_registered_token(parent_name, REGISTERED_EVAL_ENVS)


def _derive_job_variant_from_parent(log_err_path: Path) -> str | None:
    parent_name = _trim_eval_parent_name(log_err_path)
    if parent_name is None:
        return None

    return _match_unique_registered_slug(parent_name, REGISTERED_POLICY_NAMES)


def _derive_env_name_from_logs(log_err_path: Path) -> str | None:
    for text in _read_job_log_texts(log_err_path):
        env_name = _match_env_name_in_text(text)
        if env_name is not None:
            return env_name
    return None


def _derive_job_variant_from_logs(log_err_path: Path) -> str | None:
    for text in _read_job_log_texts(log_err_path):
        policy_variant = _match_unique_registered_name_values(
            POLICY_OVERRIDE_RE.findall(text), REGISTERED_POLICY_NAMES
        )
        if policy_variant is not None:
            return policy_variant

        config_variant = _match_unique_registered_name_values(
            CONFIG_NAME_RE.findall(text), REGISTERED_POLICY_NAMES
        )
        if config_variant is not None:
            return config_variant

        text_variant = _match_unique_registered_slug(text, REGISTERED_POLICY_NAMES)
        if text_variant is not None:
            return text_variant
    return None


def _match_env_name_in_text(text: str) -> str | None:
    override_match = ENV_TYPE_OVERRIDE_RE.search(text)
    if override_match is not None:
        env_name = override_match.group(1)
        if env_name in REGISTERED_EVAL_ENVS:
            return env_name

    config_match = ENV_TYPE_CONFIG_RE.search(text)
    if config_match is not None:
        env_name = config_match.group(1)
        if env_name in REGISTERED_EVAL_ENVS:
            return env_name
    env_name = _match_unique_registered_token(text, REGISTERED_EVAL_ENVS)
    if env_name is not None:
        return env_name
    return None


def _match_unique_registered_name_values(
    values: list[str], registered: frozenset[str]
) -> str | None:
    matches = set()
    for value in values:
        token = _match_unique_registered_slug(value, registered)
        if token is not None:
            matches.add(token)
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _match_unique_registered_slug(text: str, registered: frozenset[str]) -> str | None:
    normalized = str(text).lower()
    matches = {
        name
        for name in registered
        if re.search(
            rf"(?:^|[^A-Za-z0-9]){re.escape(name)}(?:$|[^A-Za-z0-9])",
            normalized,
        )
    }
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _read_job_log_texts(log_err_path: Path) -> list[str]:
    paths = [log_err_path]
    if log_err_path.name.endswith("_log.err"):
        paths.append(log_err_path.with_name(log_err_path.name[:-8] + "_log.out"))

    return [
        path.read_text(encoding="utf-8", errors="replace")
        for path in paths
        if path.exists()
    ]


def _parse_eval_progress(text: str) -> list[EvalProgress]:
    progress: list[EvalProgress] = []
    for line in text.replace("\r", "\n").splitlines():
        progress_match = _first_progress_match(line)
        if progress_match is None:
            continue

        sec_per_ep = _match_float(SEC_PER_EP_RE.search(line))
        if sec_per_ep is None:
            ep_s = _match_float(EP_S_RE.search(line))
            if ep_s is not None and ep_s > 0:
                sec_per_ep = 1.0 / ep_s

        progress.append(
            EvalProgress(
                done=int(progress_match.group(1)),
                total=int(progress_match.group(2)),
                success_rate_pct=_match_float(SUCC_RATE_RE.search(line)),
                sec_per_ep=sec_per_ep,
            )
        )
    return progress


def _first_progress_match(text: str) -> re.Match[str] | None:
    return EVAL_PBAR_RE.search(text)


def _has_crash_marker(text: str) -> tuple[bool, str | None]:
    if TRACEBACK_RE.search(text):
        return True, "traceback"
    for marker in CRASH_MARKERS:
        if marker in text:
            return True, marker
    return False, None


def _last_crash_marker(text: str) -> tuple[int | None, str | None]:
    last_pos: int | None = None
    last_reason: str | None = None

    for match in TRACEBACK_RE.finditer(text):
        last_pos = match.start()
        last_reason = "traceback"

    for marker in CRASH_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            if last_pos is None or idx >= last_pos:
                last_pos = idx
                last_reason = marker
            start = idx + len(marker)

    return last_pos, last_reason


def _latest_attempt_text_after_crash(text: str) -> tuple[str | None, str | None]:
    """Return the newest eval-attempt slice if the log restarted after a crash.

    Eval-monitor keeps appending retries into the same log.err file. Historical
    tracebacks should not permanently mark the job unhealthy if a newer attempt
    has started afterward.
    """

    crash_pos, crash_reason = _last_crash_marker(text)
    if crash_pos is None:
        return None, None

    tail = text[crash_pos:]
    progress_match = _first_progress_match(tail)
    if progress_match is None:
        return None, crash_reason
    return tail[progress_match.start() :], crash_reason


def classify_log(job_id: str, log_err_path: Path) -> JobHealth:
    text = log_err_path.read_text(encoding="utf-8", errors="replace")
    normalized = text.replace("\r", "\n")
    job_name = _derive_job_name(log_err_path)

    crashed, crash_reason = _has_crash_marker(normalized)
    latest_attempt_text, latest_attempt_crash_reason = _latest_attempt_text_after_crash(
        normalized
    )
    active_text = latest_attempt_text if latest_attempt_text is not None else normalized

    progress = _parse_eval_progress(active_text)

    pbar_seen = len(progress) > 0
    progressed = any(item.done > 0 for item in progress)
    last_progress = None
    last_success_rate_pct = None
    last_sec_per_ep = None
    if progress:
        last = progress[-1]
        last_progress = f"{last.done}/{last.total}"
        last_success_rate_pct = last.success_rate_pct
        last_sec_per_ep = last.sec_per_ep

    if crashed:
        if latest_attempt_text is not None:
            reason_prefix = (
                f"latest eval attempt {'progressed' if progressed else 'started'} "
                f"after earlier crash marker: {latest_attempt_crash_reason}"
            )
            return JobHealth(
                job_id=job_id,
                job_name=job_name,
                status="healthy" if progressed else "pending",
                log_err=str(log_err_path),
                pbar_seen=pbar_seen,
                progressed=progressed,
                last_progress=last_progress,
                last_success_rate_pct=last_success_rate_pct,
                last_sec_per_ep=last_sec_per_ep,
                reason=reason_prefix,
            )
        return JobHealth(
            job_id=job_id,
            job_name=job_name,
            status="unhealthy",
            log_err=str(log_err_path),
            pbar_seen=pbar_seen,
            progressed=progressed,
            last_progress=last_progress,
            last_success_rate_pct=last_success_rate_pct,
            last_sec_per_ep=last_sec_per_ep,
            reason=f"crash marker: {crash_reason}",
        )

    if pbar_seen and progressed:
        return JobHealth(
            job_id=job_id,
            job_name=job_name,
            status="healthy",
            log_err=str(log_err_path),
            pbar_seen=True,
            progressed=True,
            last_progress=last_progress,
            last_success_rate_pct=last_success_rate_pct,
            last_sec_per_ep=last_sec_per_ep,
            reason="eval progress marker advanced",
        )

    if pbar_seen:
        return JobHealth(
            job_id=job_id,
            job_name=job_name,
            status="pending",
            log_err=str(log_err_path),
            pbar_seen=True,
            progressed=False,
            last_progress=last_progress,
            last_success_rate_pct=last_success_rate_pct,
            last_sec_per_ep=last_sec_per_ep,
            reason="eval progress marker appeared but no progress",
        )

    return JobHealth(
        job_id=job_id,
        job_name=job_name,
        status="pending",
        log_err=str(log_err_path),
        pbar_seen=False,
        progressed=False,
        last_progress=None,
        last_success_rate_pct=None,
        last_sec_per_ep=None,
        reason="no eval progress marker found yet",
    )


def classify_missing_log(job_id: str) -> JobHealth:
    return JobHealth(
        job_id=job_id,
        job_name="N/A",
        status="unhealthy",
        log_err=None,
        pbar_seen=False,
        progressed=False,
        last_progress=None,
        last_success_rate_pct=None,
        last_sec_per_ep=None,
        reason="log.err not found",
    )


def resolve_log_err(slurm_dir: Path, job_id: str) -> Path | None:
    matches = sorted(
        slurm_dir.glob(f"**/{job_id}_*_log.err"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return None
    return matches[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify eval job health from log.err files.",
    )
    parser.add_argument(
        "job_ids",
        nargs="+",
        help="Slurm job ids for eval jobs.",
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=Path("outputs/slurm"),
        help="Root directory containing submitit/slurm logs (default: outputs/slurm).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of plain text table.",
    )
    return parser.parse_args()


def _format_success_rate(success_rate_pct: float | None) -> str:
    if success_rate_pct is None:
        return "-"
    return f"{success_rate_pct:.1f}"


def _format_sec_per_ep(sec_per_ep: float | None) -> str:
    if sec_per_ep is None:
        return "-"
    return f"{sec_per_ep:.1f}"


def _format_eta(last_progress: str | None, sec_per_ep: float | None) -> str:
    if last_progress is None or sec_per_ep is None:
        return "-"
    try:
        done_raw, total_raw = last_progress.split("/", 1)
        remaining = max(int(total_raw) - int(done_raw), 0)
    except ValueError:
        return "-"

    total_minutes = round((remaining * sec_per_ep) / 60.0)
    days, day_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(day_minutes, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


def _format_reason(reason: str) -> str:
    if reason == "eval progress marker advanced":
        return "progress"
    if reason == "eval progress marker appeared but no progress":
        return "no progress"
    if reason == "no eval progress marker found yet":
        return "no progress marker"
    if reason == "log.err not found":
        return "log missing"
    if reason.startswith("crash marker: "):
        return f"crash: {_shorten_crash_reason(reason.removeprefix('crash marker: '))}"
    if reason.startswith("latest eval attempt progressed after earlier crash marker: "):
        crash = reason.removeprefix(
            "latest eval attempt progressed after earlier crash marker: "
        )
        return f"retry progressed after crash: {_shorten_crash_reason(crash)}"
    if reason.startswith("latest eval attempt started after earlier crash marker: "):
        crash = reason.removeprefix(
            "latest eval attempt started after earlier crash marker: "
        )
        return f"retry started after crash: {_shorten_crash_reason(crash)}"
    return reason


def _shorten_crash_reason(reason: str) -> str:
    replacements = {
        "CalledProcessError": "called-process",
        "returned non-zero exit status": "nonzero-exit",
        "Submitted job triggered an exception": "submitit-exception",
        "error: *** JOB": "slurm-job-error",
        "error: *** STEP": "slurm-step-error",
    }
    return replacements.get(reason, reason)


def _format_job_name(job_name: str, job_width: int) -> str:
    if "/" not in job_name:
        return f"{job_name:<{job_width}}"
    env_name, method = job_name.split("/", 1)
    return f"{env_name}/{method:<{job_width - len(env_name) - 1}}"


def _job_width(results: list[JobHealth]) -> int:
    env_width = len("job")
    method_width = 0
    plain_width = len("job")
    for item in results:
        if "/" not in item.job_name:
            plain_width = max(plain_width, len(item.job_name))
            continue
        env_name, method = item.job_name.split("/", 1)
        env_width = max(env_width, len(env_name))
        method_width = max(method_width, len(method))
    split_width = env_width + 1 + method_width if method_width > 0 else 0
    return max(len("job"), plain_width, split_width)


def _print_table(results: list[JobHealth]) -> None:
    formatted_rows = [
        {
            "job": item.job_name,
            "job_id": item.job_id,
            "status": item.status,
            "last_prog": item.last_progress or "-",
            "sr(%)": _format_success_rate(item.last_success_rate_pct),
            "s/ep": _format_sec_per_ep(item.last_sec_per_ep),
            "eta": _format_eta(item.last_progress, item.last_sec_per_ep),
            "reason": _format_reason(item.reason),
        }
        for item in results
    ]

    job_width = _job_width(results)
    job_id_width = max(len("job_id"), *(len(row["job_id"]) for row in formatted_rows))
    status_width = max(len("status"), *(len(row["status"]) for row in formatted_rows))
    last_prog_width = max(
        len("last_prog"), *(len(row["last_prog"]) for row in formatted_rows)
    )
    success_rate_width = max(
        len("sr(%)"), *(len(row["sr(%)"]) for row in formatted_rows)
    )
    sec_per_ep_width = max(len("s/ep"), *(len(row["s/ep"]) for row in formatted_rows))
    eta_width = max(len("eta"), *(len(row["eta"]) for row in formatted_rows))

    header = (
        f"{'job':<{job_width}}|{'job_id':<{job_id_width}}|"
        f"{'status':<{status_width}}|{'last_prog':<{last_prog_width}}|"
        f"{'sr(%)':>{success_rate_width}}|{'s/ep':>{sec_per_ep_width}}|"
        f"{'eta':>{eta_width}}|reason"
    )
    print(header)
    print("-" * len(header))
    for item, row in zip(results, formatted_rows, strict=True):
        print(
            f"{_format_job_name(item.job_name, job_width)}|"
            f"{row['job_id']:<{job_id_width}}|"
            f"{row['status']:<{status_width}}|"
            f"{row['last_prog']:<{last_prog_width}}|"
            f"{row['sr(%)']:>{success_rate_width}}|"
            f"{row['s/ep']:>{sec_per_ep_width}}|"
            f"{row['eta']:>{eta_width}}|"
            f"{row['reason']}"
        )


def main() -> None:
    args = _parse_args()
    slurm_dir = args.slurm_dir

    results: list[JobHealth] = []
    for raw_job_id in args.job_ids:
        job_id = _normalize_job_id_arg(raw_job_id)
        log_err = resolve_log_err(slurm_dir, job_id)
        if log_err is None:
            results.append(classify_missing_log(job_id))
            continue
        results.append(classify_log(job_id, log_err))

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True))
        return

    _print_table(results)


if __name__ == "__main__":
    main()
