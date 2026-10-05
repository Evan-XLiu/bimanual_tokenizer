#!/usr/bin/env python3
"""Existing-checkpoint VLA and tokenizer latency on paired real observations."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from praxis.policies.factory import _load_pretrained_policy

ROOT = Path(__file__).resolve().parents[2]
SUITE = [
    (
        "BIN",
        "bar_bin_pali",
        "hopper_train/594020.hopper-m-02",
        "best/step_17000",
        [14] * 32,
    ),
    (
        "FAST",
        "bar_fast_pali",
        "hopper_bar_fast_pali_fft/600669.hopper-m-02",
        "best/step_5200",
        [1] * 129,
    ),
]


def specs():
    return [
        {
            "label": label,
            "policy_name": name,
            "checkpoint": str(
                ROOT / "outputs" / run / "train/checkpoints" / step / "pretrained_model"
            ),
            "schedule": schedule,
        }
        for label, name, run, step, schedule in SUITE
    ]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + chr(10))


def prepare(args):
    source = ROOT / "outputs/vla_latency/20260928_native5"
    out = args.output_dir
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory")
    manifest = json.loads((source / "manifest.json").read_text())
    provenance = []
    for spec in specs():
        p = Path(spec["checkpoint"])
        if json.loads((p / "policy_io.json").read_text()) != manifest["policy_io"]:
            raise ValueError("Policy IO mismatch")
        cfg = json.loads((p / "config.json").read_text())
        if cfg["vlm_model_name"] != "google/paligemma2-3b-mix-224":
            raise ValueError("Backbone mismatch")
        provenance.append(
            {
                **spec,
                "config": cfg,
                "config_sha256": digest(p / "config.json"),
                "weights_size": (p / "model.safetensors").stat().st_size,
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "observations.pt").symlink_to(source / "observations.pt")
    manifest.update(
        models=provenance,
        paired_source=str(source),
        token_count_scope="Action tokens exclude EOS and padding; GT lengths are unpadded and untruncated. FAST VLM calls include the EOS or final budget step.",
        script_sha256=digest(__file__),
    )
    write_json(out / "manifest.json", manifest)
    (out / "benchmark_vla_latency.executed.py").write_text(Path(__file__).read_text())
    print(f"PREPARED {out}", flush=True)


def move_obs(obs, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in obs.items()
    }


def predict(policy, obs, device):
    moved = move_obs(obs, device)
    actions, info = policy.predict_action_chunk(
        moved, decode_temperature=0.0, decode_top_k=1
    )
    return actions.float().cpu(), info


def measured(call, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    result = call()
    torch.cuda.synchronize(device)
    return result, (time.perf_counter_ns() - start) / 1e6


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid latency measurements")
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p50_ms": float(np.quantile(values, 0.5)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "frequency_hz": float(1000 / values.mean()),
    }


def summarize(out, rows, manifest):
    results = {}
    for spec in manifest["models"]:
        label = spec["label"]
        current = [r for r in rows if r["model"] == label]
        if not current:
            continue
        entry = {
            "schedule": spec["schedule"],
            "tokens": 448 if label == "BIN" else None,
            "token_budget": 448 if label == "BIN" else 128,
            "blocks": 32 if label == "BIN" else None,
        }
        for metric in [
            "vla_ms",
            "encode_ms",
            "decode_ms",
            "profile_model_ms",
            "profile_detokenize_ms",
        ]:
            values = [r[metric] for r in current if metric in r]
            if values:
                entry[metric] = stats(values)
        vla = [r for r in current if "vla_ms" in r]
        if vla:
            entry["peak_allocated_gb"] = (
                max(r["peak_allocated_bytes"] for r in vla) / 1e9
            )
            entry["resident_allocated_gb"] = (
                max(r["resident_allocated_bytes"] for r in vla) / 1e9
            )
            entry["round_means_ms"] = {
                str(k): float(np.mean([r["vla_ms"] for r in vla if r["round"] == k]))
                for k in sorted({r["round"] for r in vla})
            }
            per_episode = {}
            for r in vla:
                per_episode.setdefault(r["episode_index"], []).append(r["vla_ms"])
            episode_values = list(per_episode.values())
            sums = np.array([sum(v) for v in episode_values])
            counts = np.array([len(v) for v in episode_values])
            draws = np.random.default_rng(42).integers(
                len(sums), size=(5000, len(sums))
            )
            means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            entry["vla_ms"]["ci95_half_width_ms"] = float(
                1.95996398454 * means.std(ddof=1)
            )
        for field in ["tokens", "vlm_calls", "gt_tokens"]:
            values = np.asarray([r[field] for r in current if field in r], dtype=float)
            if len(values):
                entry[field + "_distribution"] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "p50": float(np.quantile(values, 0.5)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(values.max()),
                }
        if vla:
            entry["eos_stop_rate"] = (
                float(np.mean([bool(r.get("stopped_on_eos")) for r in vla]))
                if label == "FAST"
                else None
            )
            entry["budget_exhausted_count"] = sum(
                bool(r.get("budget_exhausted")) for r in vla
            )
        results[label] = entry
    if "OAT_single" in results:
        baseline = results["OAT_single"]["vla_ms"]["mean_ms"]
        for entry in results.values():
            entry["speedup_vs_oat_single"] = baseline / entry["vla_ms"]["mean_ms"]
    write_json(out / "summary.json", results)
    columns = [
        "model",
        "tokens",
        "blocks",
        "vla_mean_ms",
        "vla_ci95_half_width_ms",
        "vla_p50_ms",
        "vla_p95_ms",
        "chunks_per_second",
        "speedup_vs_oat_single",
        "encode_mean_ms",
        "decode_mean_ms",
        "peak_allocated_gb",
    ]
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for model, s in results.items():
            v = s["vla_ms"]
            writer.writerow(
                [
                    model,
                    s["tokens"],
                    s["blocks"],
                    v["mean_ms"],
                    v.get("ci95_half_width_ms"),
                    v["p50_ms"],
                    v["p95_ms"],
                    v["frequency_hz"],
                    s.get("speedup_vs_oat_single"),
                    s.get("encode_ms", {}).get("mean_ms"),
                    s.get("decode_ms", {}).get("mean_ms"),
                    s["peak_allocated_gb"],
                ]
            )
    return results


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; refusing CPU fallback")
    out = args.output_dir
    if (out / "measurements.jsonl").exists():
        raise FileExistsError("Refusing to overwrite existing timing records")
    manifest = json.loads((out / "manifest.json").read_text())
    if digest(out / "observations.pt") != manifest["observations_sha256"]:
        raise ValueError("Input cache hash mismatch")
    cached = torch.load(out / "observations.pt", map_location="cpu", weights_only=False)
    observations, actions, records = (
        cached["observations"],
        cached["actions"],
        cached["records"],
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    os.environ["ROLLOUT_PROFILE"] = "0"
    machine = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory": torch.cuda.get_device_properties(device).total_memory,
        "threads": args.threads,
        "batch_size": 1,
        "precision": "checkpoint bfloat16 VLM, native codec dtype",
        "attention_backend": "checkpoint code uses SDPA",
        "engine": "PyTorch eager, native KV cache, no vLLM/compile",
        "primary_profile_enabled": False,
        "job_id": os.environ.get("PBS_JOBID"),
        "script_sha256": digest(__file__),
    }
    machine["nvidia_smi"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,memory.used,utilization.gpu",
            "--format=csv",
        ],
        text=True,
    )
    write_json(out / "hardware.json", machine)
    print(json.dumps(machine), flush=True)
    rows = []
    with (
        (out / "measurements.jsonl").open("w", buffering=1) as log,
        torch.inference_mode(),
    ):
        for repetition in range(args.repetitions):
            offset = repetition % len(manifest["models"])
            order = manifest["models"][offset:] + manifest["models"][:offset]
            for spec in order:
                label = spec["label"]
                print(f"LOAD round={repetition} model={label}", flush=True)
                policy = (
                    _load_pretrained_policy(spec["policy_name"], spec["checkpoint"])
                    .to(device)
                    .eval()
                )
                if list(policy._resolve_block_schedule()) != spec["schedule"]:
                    raise ValueError(f"Unexpected generation schedule for {label}")
                print(f"WARMUP {label} {args.warmup}", flush=True)
                for i in range(args.warmup):
                    predict(policy, observations[i % len(observations)], device)
                torch.cuda.synchronize(device)
                for i, obs in enumerate(observations):
                    resident = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    (pred, info), elapsed = measured(
                        partial(predict, policy, obs, device), device
                    )
                    if pred.shape != (1, 32, 14) or not torch.isfinite(pred).all():
                        raise ValueError(f"Bad action output for {label}")
                    tokens = info["pred_action_token_ids"][0]
                    if (label == "BIN" and len(tokens) != 448) or (
                        label == "FAST" and not 0 < len(tokens) <= 128
                    ):
                        raise ValueError(
                            f"Unexpected generated token length for {label}"
                        )
                    row = {
                        "model": label,
                        "round": repetition,
                        "sample": i,
                        "episode_index": records[i]["episode_index"],
                        "frame_index": records[i]["frame_index"],
                        "vla_ms": elapsed,
                        "tokens": len(tokens),
                        "stopped_on_eos": info.get("stopped_on_eos"),
                        "budget_exhausted": label == "FAST"
                        and not bool(info.get("stopped_on_eos")),
                        "vlm_calls": 32
                        if label == "BIN"
                        else (
                            len(tokens) + 1
                            if info.get("stopped_on_eos")
                            else len(spec["schedule"])
                        ),
                        "resident_allocated_bytes": resident,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    }
                    rows.append(row)
                    log.write(json.dumps(row) + chr(10))
                    if (i + 1) % 50 == 0:
                        print(
                            f"TIMED {label} round={repetition} {i + 1}/{len(observations)}",
                            flush=True,
                        )
                if repetition == 0:
                    codec = policy.action_tokenizer
                    for _ in range(args.warmup):
                        a = actions[0].to(device)
                        z = codec.tokenize(a)
                        codec.detokenize(z)
                    for i, action in enumerate(actions):
                        a = action.to(device)
                        z, enc_ms = measured(partial(codec.tokenize, a), device)
                        decoded, dec_ms = measured(partial(codec.detokenize, z), device)
                        if (
                            decoded.shape != a.shape
                            or not torch.isfinite(decoded).all()
                        ):
                            raise ValueError("Invalid codec output")
                        row = {
                            "model": label,
                            "round": repetition,
                            "sample": i,
                            "episode_index": records[i]["episode_index"],
                            "encode_ms": enc_ms,
                            "decode_ms": dec_ms,
                            "gt_tokens": int(
                                codec.tokenize(a, pad=False, return_lengths=True)[1][
                                    0
                                ].item()
                            )
                            if label == "FAST"
                            else int(z.numel()),
                        }
                        rows.append(row)
                        log.write(json.dumps(row) + chr(10))
                    os.environ["ROLLOUT_PROFILE"] = "1"
                    for i, obs in enumerate(observations[: min(50, len(observations))]):
                        predict(policy, obs, device)
                        stages = policy._rollout_profile_chunk_stages
                        row = {
                            "model": label,
                            "round": repetition,
                            "sample": i,
                            "profile_model_ms": stages["model_decode_seconds"] * 1000,
                            "profile_detokenize_ms": stages[
                                "detokenize_unnormalize_seconds"
                            ]
                            * 1000,
                        }
                        rows.append(row)
                        log.write(json.dumps(row) + chr(10))
                    os.environ["ROLLOUT_PROFILE"] = "0"
                    del codec, a, z, decoded
                results = summarize(out, rows, manifest)
                print(f"RESULT {label} {results[label]['vla_ms']}", flush=True)
                del policy, pred, info
                gc.collect()
                torch.cuda.empty_cache()
    expected = len(observations) * args.repetitions * len(manifest["models"])
    if sum("vla_ms" in r for r in rows) != expected:
        raise RuntimeError("Incomplete timing run")
    write_json(
        out / "complete.json",
        {
            "vla_requests": expected,
            "models": len(manifest["models"]),
            "repetitions": args.repetitions,
            "samples": len(observations),
        },
    )
    print(f"BENCHMARK_COMPLETE {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=ROOT
        / "outputs/task1_fk_reconstruction/20260928_validation/shared_samples.npz",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.samples, args.warmup, args.repetitions, args.threads) < 1:
        parser.error("sample/warmup/repetition/thread counts must be positive")
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
