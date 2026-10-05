#!/usr/bin/env python3
"""Paired, held-out tokenizer reconstruction in RoboTwin end-effector space.

No VLA policy is loaded. The default suite is the four tokenizers used by the
current RoboTwin runs. Native absolute joint targets are evaluated at the 32
sampled waypoints, without controller interpolation or physics simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from omegaconf import OmegaConf

from praxis.codecs.factory import make_codec
from praxis.training.core.episode_splits import split_validation_episodes

ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN = ROOT.parent / "RoboTwin"
SUITE = [
    ("BTT", "btt", "hopper_btt_robotwin_new/621340.hopper-m-02", "best/step_297000"),
    (
        "OAT",
        "oat",
        "hopper_oat_robotwin_new_2500/600693.hopper-m-02",
        "best/step_296000",
    ),
    (
        "QueST",
        "quest",
        "hopper_quest_robotwin_new/613294.hopper-m-02",
        "best/step_299000",
    ),
    (
        "Acodec",
        "acodec",
        "hopper_acodec_robotwin_new/627717.hopper-m-02",
        "recent/step_300000",
    ),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(axis, angles):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    a = np.asarray(angles)[..., None, None]
    return np.eye(3) + np.sin(a) * skew + (1 - np.cos(a)) * (skew @ skew)


def rpy_matrix(rpy):
    r, p, y = rpy
    return rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)


def quaternion_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz) / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def sapien_axis_frame(axis):
    """Match SAPIEN URDFLoader's joint axis frame (joint.global_pose)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    axis1 = np.cross(axis, [0, 0, 1] if abs(axis[0]) > 0.9 else [1, 0, 0])
    axis1 /= np.linalg.norm(axis1)
    return np.stack((axis, axis1, np.cross(axis, axis1)), axis=1)


class RobotwinFK:
    """Vectorized URDF FK with RoboTwin get_arm_pose / _trans_endpose convention."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text())
        if not self.cfg.get("dual_arm"):
            raise ValueError(
                "This evaluator requires the configured dual-arm embodiment."
            )
        self.urdf_path = (self.config_path.parent / self.cfg["urdf_path"]).resolve()
        root = ET.parse(self.urdf_path).getroot()
        self.joints = {j.attrib["name"]: j for j in root.findall("joint")}
        by_child = {j.find("child").attrib["link"]: j for j in self.joints.values()}
        self.chains = []
        self.ee_frames = []
        for i in range(2):
            ee = self.joints[self.cfg["ee_joints"][i]]
            chain = [ee]
            link = ee.find("parent").attrib["link"]
            while link in by_child:
                j = by_child[link]
                chain.append(j)
                link = j.find("parent").attrib["link"]
            chain.reverse()
            moving = [j.attrib["name"] for j in chain if j.attrib["type"] != "fixed"]
            if moving != self.cfg["arm_joints_name"][i]:
                raise ValueError(f"Unexpected arm joint order: {moving}")
            self.chains.append(chain)
            axis = np.fromstring(ee.find("axis").attrib["xyz"], sep=" ")
            self.ee_frames.append(sapien_axis_frame(axis))

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape[-1] != 14 or not np.isfinite(q).all():
            raise ValueError("Expected finite, raw 14-D absolute joint/gripper values.")
        poses = []
        for arm, chain in enumerate(self.chains):
            bases = self.cfg["robot_pose"]
            base = bases[min(arm, len(bases) - 1)]
            t = np.broadcast_to(np.eye(4), (*q.shape[:-1], 4, 4)).copy()
            t[..., :3, :3] = quaternion_matrix(base[3:])
            t[..., :3, 3] = base[:3]
            names = self.cfg["arm_joints_name"][arm]
            for j in chain:
                origin = j.find("origin")
                o = np.eye(4)
                if origin is not None:
                    o[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    o[:3, :3] = rpy_matrix(
                        np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                    )
                t = t @ o
                if j.attrib["type"] != "fixed":
                    if j.attrib["type"] not in ("revolute", "continuous"):
                        raise ValueError(
                            "Arm chain contains an unsupported joint type."
                        )
                    a = q[..., arm * 7 + names.index(j.attrib["name"])]
                    axis = np.fromstring(j.find("axis").attrib["xyz"], sep=" ")
                    t[..., :3, :3] = t[..., :3, :3] @ rotation(axis, a)
            t[..., :3, :3] = (
                t[..., :3, :3]
                @ self.ee_frames[arm]
                @ np.asarray(self.cfg["global_trans_matrix"])
                @ np.asarray(self.cfg["delta_matrix"])
            )
            # RoboTwin get_arm_pose -> get_*_ee_pose -> is_endpose=False.
            offset = float(self.cfg["gripper_bias"]) - 0.12
            t[..., :3, 3] += t[..., :3, :3] @ np.array([offset, 0, 0])
            poses.append(t)
        return np.stack(poses, axis=-3)


def to_joint_trajectory(actions, q0, semantics="absolute"):
    if semantics == "absolute":
        return np.asarray(actions, dtype=np.float64).copy()
    if semantics == "delta":
        result = np.asarray(actions, dtype=np.float64).copy()
        joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        result[..., joints] = q0[:, None, joints] + np.cumsum(
            result[..., joints], axis=1
        )
        return result
    raise ValueError(semantics)


def geodesic_deg(a, b):
    d = np.swapaxes(a, -1, -2) @ b
    cos = np.clip((np.trace(d, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    vee = np.stack(
        [
            d[..., 2, 1] - d[..., 1, 2],
            d[..., 0, 2] - d[..., 2, 0],
            d[..., 1, 0] - d[..., 0, 1],
        ],
        axis=-1,
    )
    sin = np.linalg.norm(vee, axis=-1) / 2
    return np.rad2deg(np.arctan2(sin, cos))


def chunk_metrics(gt, pred, gt_pose, pred_pose):
    p, ph = gt_pose[..., :3, 3], pred_pose[..., :3, 3]
    r, rh = gt_pose[..., :3, :3], pred_pose[..., :3, :3]
    pos = np.linalg.norm(p - ph, axis=-1) * 100
    rot = geodesic_deg(r, rh)
    relative_r = np.swapaxes(r[..., 0, :, :], -1, -2) @ r[..., 1, :, :]
    relative_rh = np.swapaxes(rh[..., 0, :, :], -1, -2) @ rh[..., 1, :, :]
    result = {
        "left_position_cm": pos[..., 0].mean(axis=1),
        "right_position_cm": pos[..., 1].mean(axis=1),
        "left_rotation_deg": rot[..., 0].mean(axis=1),
        "right_rotation_deg": rot[..., 1].mean(axis=1),
        "relative_position_cm": (
            np.linalg.norm(
                (p[..., 1, :] - p[..., 0, :]) - (ph[..., 1, :] - ph[..., 0, :]), axis=-1
            )
            * 100
        ).mean(axis=1),
        "relative_rotation_deg": geodesic_deg(relative_r, relative_rh).mean(axis=1),
        "action_mse": np.square(gt - pred).mean(axis=(1, 2)),
    }
    for label, index in [("left", 6), ("right", 13)]:
        err = gt[..., index] - pred[..., index]
        result[f"{label}_gripper_mae"] = np.abs(err).mean(axis=1)
        result[f"{label}_gripper_mse"] = np.square(err).mean(axis=1)
    return result


def summarize(metrics, episodes, bootstrap_seed=42, repetitions=5000):
    """Chunk-weighted means, episode-cluster bootstrap normal 95% CIs."""
    ids, inv = np.unique(episodes, return_inverse=True)
    if len(ids) < 2:
        raise ValueError("At least two held-out episodes are needed for a CI.")
    draws = np.random.default_rng(bootstrap_seed).integers(
        len(ids), size=(repetitions, len(ids))
    )
    counts = np.bincount(inv)
    out = {}
    for name, values in metrics.items():
        sums = np.bincount(inv, weights=values)
        samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        mean = float(values.mean())
        half = float(1.959963984540054 * samples.std(ddof=1))
        out[name] = {
            "mean": mean,
            "ci95_half_width": half,
            "ci95": [mean - half, mean + half],
        }
    return out


def load_shared_chunks(specs, dataset_root, stride, max_chunks):
    info = json.loads((dataset_root / "meta/info.json").read_text())
    if info["robot_type"] != "aloha_agilex":
        raise ValueError("Unexpected dataset embodiment")
    memberships, provenance = [], []
    for label, name, checkpoint in specs:
        cfg = OmegaConf.load(checkpoint / "train_config.yaml")
        if Path(str(cfg.dataset.root)).resolve() != dataset_root.resolve():
            raise ValueError(f"Dataset mismatch for {label}")
        if OmegaConf.select(cfg, "dataset.exclude_episodes_file"):
            raise ValueError(
                "Explicit exclusion files need a reviewed split implementation."
            )
        pool = list(cfg.dataset.episodes or range(info["total_episodes"]))
        seed = OmegaConf.select(cfg, "trainer.split_seed", default=cfg.seed)
        val = split_validation_episodes(
            pool, val_ratio=float(cfg.trainer.val_ratio), split_seed=seed
        )
        memberships.append(val)
        provenance.append(
            {
                "label": label,
                "codec": name,
                "checkpoint": str(checkpoint),
                "train_config_sha256": sha256(checkpoint / "train_config.yaml"),
                "config_sha256": sha256(checkpoint / "config.json"),
                "split_seed": seed,
                "val_ratio": float(cfg.trainer.val_ratio),
            }
        )
    if not all(v == memberships[0] for v in memberships):
        raise ValueError("Tokenizers do not share the same held-out episode split")
    val = memberships[0]
    files = sorted((dataset_root / "data").glob("*/*.parquet"))
    table = pq.read_table(
        files,
        columns=[
            "action",
            "observation.state",
            "episode_index",
            "frame_index",
            "index",
        ],
        filters=[("episode_index", "in", val)],
    )
    ep = table["episode_index"].to_numpy()
    frame = table["frame_index"].to_numpy()
    index = table["index"].to_numpy()
    actions = np.array(table["action"].to_pylist(), dtype=np.float32)
    states = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
    chunks, initial, records = [], [], []
    for e in val:
        rows = np.flatnonzero(ep == e)
        rows = rows[np.argsort(frame[rows])]
        if not np.all(np.diff(frame[rows]) == 1):
            raise ValueError(f"Missing/nonconsecutive frames in episode {e}")
        for start in range(0, len(rows) - 31, stride):
            rr = rows[start : start + 32]
            chunks.append(actions[rr])
            initial.append(states[rr[0]])
            records.append((int(e), int(frame[rr[0]]), int(index[rr[0]])))
    if max_chunks and max_chunks < len(chunks):
        chosen = np.sort(
            np.random.default_rng(42).choice(len(chunks), max_chunks, replace=False)
        )
        chunks = [chunks[i] for i in chosen]
        initial = [initial[i] for i in chosen]
        records = [records[i] for i in chosen]
    if not chunks:
        raise ValueError("No complete held-out chunks")
    return (
        np.stack(chunks),
        np.stack(initial),
        np.array(records),
        {
            "models": provenance,
            "validation_episodes": val,
            "dataset_root": str(dataset_root),
            "dataset_info_sha256": sha256(dataset_root / "meta/info.json"),
            "sample_stride": stride,
            "chunk_horizon": 32,
            "selection": "complete non-padded chunks; deterministic random subset only if capped",
            "data_files": [{"path": str(p), "sha256": sha256(p)} for p in files],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/lerobot/local/robotwin_new_demo_clean",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=ROBOTWIN / "assets/embodiments/aloha-agilex/config.yml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stride < 1 or args.batch_size < 1 or args.max_chunks < 0:
        parser.error("stride/batch size must be positive and max-chunks nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    specs = [
        (
            label,
            name,
            ROOT / "outputs" / run / "train/checkpoints" / step / "pretrained_model",
        )
        for label, name, run, step in SUITE
    ]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    actions, q0, records, manifest = load_shared_chunks(
        specs, args.dataset_root, args.stride, args.max_chunks
    )
    fk = RobotwinFK(args.robot_config)
    gt_q = to_joint_trajectory(actions, q0, "absolute")
    gt_pose = fk(gt_q)
    manifest.update(
        {
            "n_chunks": len(actions),
            "n_episodes": len(np.unique(records[:, 0])),
            "action_semantics": "absolute joint targets, grippers absolute normalized commands",
            "q0": "observation.state at chunk start; retained but no integration for absolute targets",
            "fk_definition": "RoboTwin get_arm_pose -> Robot.get_*_ee_pose; world frame; joint.global_pose with global_trans_matrix and delta_matrix; gripper_bias minus 0.12 m",
            "relative_rotation": "R_left.T @ R_right, compared by SO(3) geodesic",
            "relative_position": "world-frame (p_right - p_left) difference norm",
            "ci_method": "mean of per-chunk temporal means; 5000 episode-cluster bootstrap replicates; mean +/- 1.9599639845 * bootstrap std; seed 42",
            "split_note": "validation used during checkpoint selection, not untouched test set",
            "robot_config": str(args.robot_config),
            "robot_config_sha256": sha256(args.robot_config),
            "urdf": str(fk.urdf_path),
            "urdf_sha256": sha256(fk.urdf_path),
            "evaluator_sha256": sha256(__file__),
            "device": args.device,
            "torch_version": torch.__version__,
            "gripper_units": "normalized command; not cm; no clipping",
        }
    )
    np.savez_compressed(
        out / "shared_samples.npz",
        actions=actions,
        q0=q0,
        episode_index=records[:, 0],
        frame_index=records[:, 1],
        global_index=records[:, 2],
        gt_joint_trajectory=gt_q,
        gt_ee_pose=gt_pose,
    )
    manifest["samples_sha256"] = sha256(out / "shared_samples.npz")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"SAMPLES {len(actions)} chunks / {manifest['n_episodes']} held-out episodes",
        flush=True,
    )
    summaries = {}
    for label, name, checkpoint in specs:
        print(f"START {label}", flush=True)
        model = (
            make_codec(
                {"kind": "tokens", "name": name, "pretrained_path": str(checkpoint)}
            )
            .to(args.device)
            .eval()
        )
        reconstructed, token_batches = [], []
        with torch.inference_mode():
            for start in range(0, len(actions), args.batch_size):
                batch = torch.from_numpy(actions[start : start + args.batch_size]).to(
                    args.device
                )
                tokens = model.tokenize(batch)
                pred = model.detokenize(tokens)
                if pred.shape != batch.shape or not torch.isfinite(pred).all():
                    raise ValueError(f"Invalid reconstruction from {label}")
                reconstructed.append(pred.cpu().numpy())
                token_batches.append(tokens.cpu().numpy())
        pred = np.concatenate(reconstructed).astype(np.float64)
        pred_q = to_joint_trajectory(pred, q0, "absolute")
        pred_pose = fk(pred_q)
        metrics = chunk_metrics(actions.astype(np.float64), pred, gt_pose, pred_pose)
        summaries[label] = summarize(metrics, records[:, 0])
        np.savez_compressed(
            out / f"{label}_reconstruction.npz",
            reconstructed_actions=pred,
            reconstructed_joint_trajectory=pred_q,
            reconstructed_ee_pose=pred_pose,
            token_ids=np.concatenate(token_batches),
        )
        with (out / f"{label}_per_chunk.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode_index", "frame_index", "global_index", *metrics])
            writer.writerows(
                [*records[i], *(v[i] for v in metrics.values())]
                for i in range(len(actions))
            )
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(
            f"DONE {label}: relative position {summaries[label]['relative_position_cm']}",
            flush=True,
        )
        del model
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "metric", "mean", "ci95_half_width", "ci95_lower", "ci95_upper"]
        )
        for model, metrics in summaries.items():
            for name, stat in metrics.items():
                writer.writerow(
                    [model, name, stat["mean"], stat["ci95_half_width"], *stat["ci95"]]
                )
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
