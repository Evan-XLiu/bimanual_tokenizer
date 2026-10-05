#!/usr/bin/env python3
"""Measure BTT's second-block conditioning on fixed handover demonstrations.

For every selected handover episode, one observation is taken just before the
receiver's first gripper closure.  The VLM is evaluated twice:

* ``GT_LEFT``: the second block receives the ground-truth left-arm tokens.
* ``PRED_LEFT``: the second block receives greedy tokens predicted by the first
  block from the observation alone.

The script reports right-block NLL/token accuracy, decoded receiver errors, and
receiver gripper timing.  It intentionally evaluates the two conditions on the
same observation/action chunk and instruction.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from praxis.data.lerobot_adapter import LeRobotDatasetAdapter
from praxis.policies.factory import make_policy
from praxis.policies.core.vlm.training import (
    build_prepared_action_token_attention_mask,
    forward_vlm_prefix_state,
    prepare_action_token_ce_from_token_targets,
)
from praxis.policies.core.vlm.supervision import IGNORE_INDEX

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data/lerobot/local/robotwin_new_demo_clean"
DEFAULT_DONOR = ROOT / "outputs/btt_convention_task_analysis/20260930/mic_training_donor.json"
DEFAULT_BTT = ROOT / (
    "outputs/hopper_bar_btt_pali_robotwin_new_step300000_30k/"
    "630822.hopper-m-02/train/checkpoints/latest/pretrained_model"
)
DEFAULT_WO = ROOT / (
    "outputs/hopper_bar_btt_wo_convention_pali_robotwin_new_step300000_30k/"
    "633464.hopper-m-02/train/checkpoints/latest/pretrained_model"
)


def _load_fk():
    path = ROOT / "scripts/tokenizer/evaluate_fk_reconstruction.py"
    spec = importlib.util.spec_from_file_location("btt_fk", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import FK helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RobotwinFK


def _load_episode_table() -> dict[int, dict[str, Any]]:
    table = pq.read_table(
        DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pydict()
    rows = {}
    for i, ep in enumerate(table["episode_index"]):
        task = table["tasks"][i][0]
        rows[int(ep)] = {
            "length": int(table["length"][i]),
            "task": str(task),
            "start": int(table["dataset_from_index"][i]),
        }
    return rows


def _load_samples(donor_path: Path, limit: int | None) -> list[dict[str, Any]]:
    donor_rows = json.loads(donor_path.read_text())
    episodes = _load_episode_table()
    out = []
    for row in donor_rows:
        ep = int(row["episode"])
        if ep not in episodes:
            raise ValueError(f"Donor episode {ep} is absent from {DATASET_ROOT}")
        donor = str(row["donor"])
        if donor not in ("left", "right"):
            raise ValueError(f"Invalid donor for episode {ep}: {donor!r}")
        receiver = "right" if donor == "left" else "left"
        receiver_close = int(row[f"first_{receiver}_closed_frame"])
        # The 32-step action chunk should include the receiver's approach and
        # closure while keeping the observation before the contact event.
        # Leave one second (16 frames at 15 Hz) before closure and keep the
        # closure event inside the 32-step action chunk.
        frame = max(0, receiver_close - 16)
        out.append(
            {
                "episode": ep,
                "donor": donor,
                "receiver": receiver,
                "receiver_close_frame": receiver_close,
                "frame": frame,
                "task": str(row["instruction"]),
                "episode_length": episodes[ep]["length"],
                "dataset_index": episodes[ep]["start"] + frame,
            }
        )
    if limit is not None:
        out = out[: int(limit)]
    return out


def _batch_from_dataset(dataset, indices: list[int]) -> dict[str, Any]:
    items = [dataset[i] for i in indices]
    tensor_keys = [
        "observation.state",
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "action",
    ]
    batch: dict[str, Any] = {
        key: torch.stack([item[key] for item in items], dim=0) for key in tensor_keys
    }
    batch["task"] = [str(item["task"]) for item in items]
    batch["action_is_pad"] = torch.stack(
        [item["action_is_pad"] for item in items], dim=0
    )
    return batch


def _project_prediction_logits(policy, prepared):
    """Return action-vocabulary logits in [B,16,num_action_tokens]."""
    adapter = policy._require_vlm_ce_prefix_adapter()
    mask = build_prepared_action_token_attention_mask(policy, prepared)
    with torch.inference_mode(), policy._autocast_context():
        state = adapter.prepare_prefix(prepared.vlm_inputs, layer_attention_mask=mask)
        state = forward_vlm_prefix_state(adapter, state, gradient_checkpointing=False)
        prediction_mask = prepared.supervision.prediction_position_mask
        selected = state.hidden[prediction_mask]
        vocab = policy._require_action_token_vocabulary()
        candidate = torch.as_tensor(
            vocab.token_ids, dtype=torch.long, device=selected.device
        )
        logits = adapter.final_hidden_to_logits(
            selected, candidate_token_ids=candidate
        )
    bsz = int(prediction_mask.shape[0])
    horizon = int(prepared.supervision.target_vlm_token_ids.shape[1])
    return logits.float().reshape(bsz, horizon, -1), candidate


def _raw_ids_from_logits(policy, logits, candidate):
    vocab = policy._require_action_token_vocabulary()
    chosen_vlm = candidate[logits.argmax(dim=-1)]
    return vocab.vlm_ids_to_raw_ids(chosen_vlm).long()


def _condition(policy, batch, target_rows):
    actions = batch["action"]
    prepared = prepare_action_token_ce_from_token_targets(
        policy,
        batch,
        token_ids_batch=target_rows,
        token_lengths=None,
        actions=actions,
        ignore_index=IGNORE_INDEX,
    )
    logits, candidate = _project_prediction_logits(policy, prepared)
    return prepared, logits, candidate


def _nll_and_accuracy(policy, logits, target_raw):
    vocab = policy._require_action_token_vocabulary()
    candidate_list = vocab.token_ids
    index = {int(v): i for i, v in enumerate(candidate_list)}
    target_vlm = vocab.raw_ids_to_vlm_ids(target_raw.long())
    target = torch.tensor(
        [[index[int(x)] for x in row] for row in target_vlm],
        dtype=torch.long,
        device=logits.device,
    )
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(target.shape)
    correct = logits.argmax(dim=-1).eq(target)
    return nll, correct


def _first_crossing(values: np.ndarray, threshold: float = 0.5) -> float:
    # RoboTwin's normalized Aloha gripper command is 1=open, 0=closed.
    hit = np.flatnonzero(values <= threshold)
    return float(hit[0]) if hit.size else float("nan")


def _action_metrics(fk, gt, pred, receiver_index: int) -> dict[str, float]:
    gt_np = gt.detach().cpu().numpy().astype(np.float64)
    pred_np = pred.detach().cpu().numpy().astype(np.float64)
    gt_pose = fk(gt_np)
    pred_pose = fk(pred_np)
    pos = np.linalg.norm(
        gt_pose[:, :, receiver_index, :3, 3]
        - pred_pose[:, :, receiver_index, :3, 3],
        axis=-1,
    ) * 100.0
    # Rotation geodesics are reported as an auxiliary contact-quality signal.
    r = gt_pose[:, :, receiver_index, :3, :3]
    rh = pred_pose[:, :, receiver_index, :3, :3]
    rel = np.swapaxes(r, -1, -2) @ rh
    cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    rot = np.rad2deg(np.arccos(cos))
    gripper_dim = 6 if receiver_index == 0 else 13
    gt_g = gt_np[:, :, gripper_dim]
    pred_g = pred_np[:, :, gripper_dim]
    close_errors = []
    gt_close = []
    pred_close = []
    for a, b in zip(gt_g, pred_g, strict=True):
        x, y = _first_crossing(a), _first_crossing(b)
        gt_close.append(x)
        pred_close.append(y)
        if np.isfinite(x) and np.isfinite(y):
            close_errors.append(abs(x - y))
    return {
        "receiver_position_cm_mean": float(pos.mean()),
        "receiver_position_cm_final": float(pos[:, -1].mean()),
        "receiver_rotation_deg_mean": float(rot.mean()),
        "receiver_gripper_mae": float(np.abs(gt_g - pred_g).mean()),
        "receiver_close_time_error_steps": float(
            np.mean(close_errors) if close_errors else float("nan")
        ),
        "receiver_gt_close_offset_mean": float(np.nanmean(gt_close)),
        "receiver_pred_close_offset_mean": float(np.nanmean(pred_close)),
    }


def run_model(label: str, checkpoint: Path, samples, batch_size: int, fk, device: str):
    print(f"LOADING {label} checkpoint={checkpoint}", flush=True)
    policy = make_policy(
        OmegaConf.create({"name": "bar_btt_pali", "pretrained_path": str(checkpoint)})
    )
    policy.to(device)
    policy.eval()
    dataset = LeRobotDatasetAdapter(
        "local/robotwin_new_demo_clean",
        policy.io,
        root=DATASET_ROOT,
        episodes=[int(s["episode"]) for s in samples],
        video_backend="pyav",
    )
    # Filtered LeRobotDataset preserves the requested episode order.
    offsets = {}
    cursor = 0
    for ep in [int(s["episode"]) for s in samples]:
        offsets[ep] = cursor
        cursor += int(_load_episode_table()[ep]["length"])

    output = []
    for start in range(0, len(samples), batch_size):
        group = samples[start : start + batch_size]
        indices = [offsets[int(s["episode"])] + int(s["frame"]) for s in group]
        batch = _batch_from_dataset(dataset, indices)
        batch["action"] = batch["action"].to(device)
        for key in list(batch):
            if key.startswith("observation."):
                batch[key] = batch[key].to(device)

        target_rows = policy.tokenize_actions_to_policy_tokens(batch["action"])
        left_gt = [row[:8] for row in target_rows]
        # First pass: GT left in the second block.  Its first block is still
        # masked, so its argmax is exactly the model's free left prediction.
        prepared_gt, logits_gt, candidate = _condition(policy, batch, target_rows)
        pred_all_first = _raw_ids_from_logits(policy, logits_gt[:, :8], candidate)
        pred_left_rows = [row.tolist() for row in pred_all_first]

        # Second pass: explicitly replace the left block with the predicted
        # block, while retaining the same observation and target right block.
        pred_condition_rows = [
            pred_left_rows[i] + target_rows[i][8:]
            for i in range(len(target_rows))
        ]
        prepared_pred, logits_pred, _ = _condition(
            policy, batch, pred_condition_rows
        )

        left_target = torch.as_tensor([r[:8] for r in target_rows], device=device)
        right_target = torch.as_tensor([r[8:] for r in target_rows], device=device)
        nll_left, acc_left = _nll_and_accuracy(policy, logits_gt[:, :8], left_target)
        nll_gt, acc_gt = _nll_and_accuracy(policy, logits_gt[:, 8:], right_target)
        nll_pred, acc_pred = _nll_and_accuracy(policy, logits_pred[:, 8:], right_target)
        pred_right_gt = _raw_ids_from_logits(policy, logits_gt[:, 8:], candidate)
        pred_right_pred = _raw_ids_from_logits(policy, logits_pred[:, 8:], candidate)

        decode_gt_rows = [
            target_rows[i][:8] + pred_right_gt[i].tolist() for i in range(len(target_rows))
        ]
        # Hold the donor/left action fixed to GT in both decodes.  This makes
        # the FK and gripper comparison isolate the receiver token prediction;
        # donor degradation is reported separately as left-token error.
        decode_pred_rows = [
            target_rows[i][:8] + pred_right_pred[i].tolist()
            for i in range(len(target_rows))
        ]
        decode_gt = policy._decode_generated_action_tokens(
            decode_gt_rows, device=torch.device(device), horizon=16
        )
        decode_pred = policy._decode_generated_action_tokens(
            decode_pred_rows, device=torch.device(device), horizon=16
        )

        for i, sample in enumerate(group):
            receiver_index = 0 if sample["receiver"] == "left" else 1
            m_gt = _action_metrics(
                fk,
                batch["action"][i : i + 1],
                decode_gt[i : i + 1],
                receiver_index,
            )
            m_pred = _action_metrics(
                fk,
                batch["action"][i : i + 1],
                decode_pred[i : i + 1],
                receiver_index,
            )
            row = {
                "model": label,
                "episode": int(sample["episode"]),
                "frame": int(sample["frame"]),
                "donor": sample["donor"],
                "receiver": sample["receiver"],
                "task": sample["task"],
                "left_token_accuracy": float(acc_left[i].float().mean()),
                "right_token_accuracy_gt_left": float(acc_gt[i].float().mean()),
                "right_token_accuracy_pred_left": float(acc_pred[i].float().mean()),
                "right_nll_gt_left": float(nll_gt[i].mean()),
                "right_nll_pred_left": float(nll_pred[i].mean()),
                "left_pred_token_error_rate": float(
                    torch.as_tensor(pred_left_rows[i], device=device)
                    .ne(torch.as_tensor(left_gt[i], device=device))
                    .float()
                    .mean()
                ),
            }
            row.update({f"gt_left_{k}": v for k, v in m_gt.items()})
            row.update({f"pred_left_{k}": v for k, v in m_pred.items()})
            output.append(row)
        print(f"DONE {label} {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    del policy, dataset
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btt", type=Path, default=DEFAULT_BTT)
    parser.add_argument("--wo-convention", type=Path, default=DEFAULT_WO)
    parser.add_argument("--donor-json", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/btt_handover_conditioning")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", choices=("btt", "wo", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    samples = _load_samples(args.donor_json, args.limit)
    fk = _load_fk()(ROOT.parent / "RoboTwin/assets/embodiments/aloha-agilex/config.yml")
    all_rows = []
    if args.model in ("btt", "both"):
        all_rows.extend(run_model("BTT", args.btt, samples, args.batch_size, fk, args.device))
    if args.model in ("wo", "both"):
        all_rows.extend(run_model("BTT_w/o_convention", args.wo_convention, samples, args.batch_size, fk, args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_episode.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {}
    numeric = [k for k, v in all_rows[0].items() if isinstance(v, (int, float)) and k not in ("episode", "frame")]
    for model in sorted({r["model"] for r in all_rows}):
        rows = [r for r in all_rows if r["model"] == model]
        summary[model] = {
            "n": len(rows),
            "by_direction": {},
            "overall": {k: float(np.nanmean([r[k] for r in rows])) for k in numeric},
        }
        for direction in ("left_to_right", "right_to_left"):
            donor = "left" if direction == "left_to_right" else "right"
            subset = [r for r in rows if r["donor"] == donor]
            summary[model]["by_direction"][direction] = {
                "n": len(subset),
                **{k: float(np.nanmean([r[k] for r in subset])) for k in numeric},
            }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
