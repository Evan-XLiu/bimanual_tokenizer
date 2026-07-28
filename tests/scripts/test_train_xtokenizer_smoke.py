# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from praxis.codecs.latents.xtokenizer.inference import load_exported_xtokenizer
from praxis.data.bridge_xtokenizer import build_bridge_xtokenizer_episode
from praxis.data.lerobot_adapter import write_lerobot_cache_provenance

DATASET_REPO_ID = "smoke/bridge"
DATASET_REVISION = "a" * 40
MODEL_ID = "Qwen/synthetic-smoke"
MODEL_REVISION = "synthetic-model-v1"
EXTRACTOR_REVISION = "praxis-smoke-v1"
VL_FEATURE_DIM = 8
ACTION_HORIZON = 4
COMPRESSION_RATIO = 4
DOF_DIMS = {
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
    "follow_left_gripper": 1,
}
DOF_ORDER = tuple(DOF_DIMS)


def _write_bridge_fixture(root: Path) -> None:
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=10,
        root=root,
        robot_type="widowx",
        use_videos=False,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
        },
    )
    for episode in range(2):
        for frame in range(8):
            scale = float(frame + 1 + episode)
            dataset.add_frame(
                {
                    "observation.state": np.asarray(
                        [
                            0.02 * scale,
                            -0.01 * scale,
                            0.03 * scale,
                            0.01 * scale,
                            -0.015 * scale,
                            0.02 * scale,
                            (frame + episode) % 2,
                        ],
                        dtype=np.float32,
                    ),
                    "action": np.asarray(
                        [
                            0.01 * scale,
                            -0.008 * (frame + 1),
                            0.006 * (episode + 1),
                            0.004 * scale,
                            -0.003 * (frame + 1),
                            0.005 * (episode + frame + 1),
                            (frame + episode + 1) % 2,
                        ],
                        dtype=np.float32,
                    ),
                    "task": "synthetic Bridge smoke",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    write_lerobot_cache_provenance(
        root=root,
        repo_id=DATASET_REPO_ID,
        revision=DATASET_REVISION,
    )


def _write_stats_config(path: Path, *, dataset_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"repo_id: {DATASET_REPO_ID}",
                f"root: {dataset_root}",
                f'dataset_revision: "{DATASET_REVISION}"',
                "robot_type_name: WidowX",
                "episodes: null",
                f"action_horizon: {ACTION_HORIZON}",
                "action_layout:",
                "  convention: absolute",
                "  dof_dims:",
                "    follow_left_ee_cartesian_pos: 3",
                "    follow_left_ee_rotation_6D: 6",
                "    follow_left_gripper: 1",
                "  dof_order:",
                "    - follow_left_ee_cartesian_pos",
                "    - follow_left_ee_rotation_6D",
                "    - follow_left_gripper",
            ]
        )
        + "\n"
    )


def _offline_env() -> dict[str, str]:
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _run(command: list[str], *, cwd: Path, expect_success: bool = True):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if expect_success:
        assert result.returncode == 0, (
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _prepare_sidecar(
    repo_root: Path,
    output_root: Path,
    *,
    seed: int,
) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/tokenizer/prepare_vl_features.py",
            "--output-root",
            str(output_root),
            "--dataset-repo-id",
            DATASET_REPO_ID,
            "--dataset-revision",
            DATASET_REVISION,
            "--extractor-revision",
            EXTRACTOR_REVISION,
            "--model-id",
            MODEL_ID,
            "--model-revision",
            MODEL_REVISION,
            "--stride",
            str(COMPRESSION_RATIO),
            "--feature-dim",
            str(VL_FEATURE_DIM),
            "--episodes",
            "2",
            "--frames",
            "8",
            "--seed",
            str(seed),
            "--synthetic",
            "--global-instruction",
            "synthetic global",
            "--local-subtask",
            "synthetic local",
        ],
        cwd=repo_root,
    )


def _train_command(
    *,
    dataset_root: Path,
    stats_path: Path,
    sidecar_root: Path,
    output_root: Path,
    max_steps: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "praxis-train",
        "--config-name",
        "train_codecs",
        "codec=latents/xtokenizer",
        "dataset=xtokenizer_bridge",
        "train_recipe/codecs=xtokenizer",
        "device=cpu",
        f"output_dir={output_root}",
        f"dataset.root={dataset_root}",
        f"dataset.cache_dir={output_root / 'cache'}",
        f"dataset.repo_id={DATASET_REPO_ID}",
        f"dataset.dataset_revision={DATASET_REVISION}",
        f"dataset.xtokenizer_quantile_stats={stats_path}",
        f"dataset.action_horizon={ACTION_HORIZON}",
        f"dataset.vl_feature_sidecar={sidecar_root}",
        f"dataset.vl_model_id={MODEL_ID}",
        f"dataset.vl_model_revision={MODEL_REVISION}",
        f"dataset.vl_extractor_revision={EXTRACTOR_REVISION}",
        "dataset.vl_image_keys=[observation.images.face,observation.images.left_wrist,observation.images.right_wrist]",
        "dataset.vl_global_instruction=synthetic global",
        "dataset.vl_local_subtask=synthetic local",
        f"codec.action_horizon={ACTION_HORIZON}",
        f"codec.compression_ratio={COMPRESSION_RATIO}",
        "codec.hidden_dim=8",
        "codec.latent_dim=8",
        "codec.encoder_num_heads=2",
        "codec.encoder_depth=1",
        "codec.decoder_num_heads=2",
        "codec.decoder_depth=1",
        "codec.mam_num_heads=2",
        "codec.mam_depth=1",
        "codec.prediction_num_heads=2",
        "codec.prediction_depth=1",
        "codec.dropout=0.0",
        "codec.codebook_size=8",
        "codec.num_quantizers=4",
        "codec.dead_code_threshold=0",
        "codec.embodiment_vocab_size=2",
        f"codec.vl_feature_dim={VL_FEATURE_DIM}",
        "codec.mam_warmup_epochs=0",
        "codec.mam_warmup_steps=0",
        f"trainer.max_steps={max_steps}",
        "trainer.batch_size=2",
        "trainer.num_workers=0",
        "trainer.prefetch_factor=null",
        "trainer.use_wandb=false",
        "trainer.use_ema=false",
        "trainer.mixed_precision=no",
        "trainer.lr=0.001",
        "trainer.warmup_steps=0",
        "trainer.log_every=1",
        "trainer.val_ratio=0.0",
        "trainer.val_every=100",
        "trainer.reconst_every=100",
        "trainer.checkpoint_every=1",
        "trainer.checkpoint_candidates.best.k=0",
        "trainer.checkpoint_candidates.recent.k=0",
        "hydra.job.chdir=false",
    ]
    if resume_checkpoint is not None:
        command.extend(
            [
                "run_mode=resume",
                f"resume_checkpoint={resume_checkpoint}",
            ]
        )
    return command


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_xtokenizer_train_resume_reject_changed_sidecar_and_export(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = tmp_path / "dataset"
    stats_config = tmp_path / "stats_dataset.yaml"
    stats_path = tmp_path / "stats_v3_h4.json"
    sidecar_root = tmp_path / "sidecar"
    output_root = tmp_path / "run"
    _write_bridge_fixture(dataset_root)
    _write_stats_config(stats_config, dataset_root=dataset_root)

    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/tokenizer/prepare_bridge_xtokenizer_stats.py",
            "--dataset-config",
            str(stats_config),
            "--stats-horizon",
            str(ACTION_HORIZON),
            "--output",
            str(stats_path),
        ],
        cwd=repo_root,
    )
    stats_payload = json.loads(stats_path.read_text())
    assert stats_payload["format_version"] == 3
    normalization = stats_payload["normalization"]
    assert normalization["stats_horizon"] == ACTION_HORIZON
    assert normalization["predict_action"]["space"] == "observation_relative"
    for block in (
        normalization["predict_action"],
        normalization["observation_state"],
    ):
        assert all(
            minimum < maximum
            for minimum, maximum in zip(block["q001"], block["q999"], strict=True)
        )
    _prepare_sidecar(repo_root, sidecar_root, seed=7)

    first = _run(
        _train_command(
            dataset_root=dataset_root,
            stats_path=stats_path,
            sidecar_root=sidecar_root,
            output_root=output_root,
            max_steps=2,
        ),
        cwd=repo_root,
    )
    assert "Training complete. 2 steps." in first.stdout
    sentinel = output_root / "TRAINING_COMPLETE"
    checkpoint = output_root / "checkpoints" / "latest"
    assert sentinel.is_file()
    assert checkpoint.is_dir()

    records = [
        json.loads(line)
        for line in (output_root / "train_metrics.jsonl").read_text().splitlines()
    ]
    required_metrics = {
        "train/loss",
        "train/action_l1_loss",
        "train/rotation_geodesic_loss",
        "train/commitment_loss",
        "train/mam_loss",
        "train/alignment_loss",
        "train/alignment_global_loss",
        "train/alignment_local_loss",
        "train/prediction_loss",
        "train/total_loss",
        *(f"train/codebook_perplexity_q{level}" for level in range(4)),
    }
    assert records[-1]["step"] == 2
    assert required_metrics <= records[-1].keys()
    assert all(
        math.isfinite(float(value))
        for record in records
        for key, value in record.items()
        if key != "step"
    )

    resumed = _run(
        _train_command(
            dataset_root=dataset_root,
            stats_path=stats_path,
            sidecar_root=sidecar_root,
            output_root=output_root,
            max_steps=3,
            resume_checkpoint=checkpoint,
        ),
        cwd=repo_root,
    )
    assert "Training complete. 3 steps." in resumed.stdout
    assert "Resumed from checkpoint" in resumed.stdout
    resumed_records = [
        json.loads(line)
        for line in (output_root / "train_metrics.jsonl").read_text().splitlines()
    ]
    assert resumed_records[-1]["step"] == 3

    changed_sidecar = tmp_path / "changed_sidecar"
    _prepare_sidecar(repo_root, changed_sidecar, seed=11)
    checkpoint_digest = _tree_digest(checkpoint)
    rejected = _run(
        _train_command(
            dataset_root=dataset_root,
            stats_path=stats_path,
            sidecar_root=changed_sidecar,
            output_root=output_root,
            max_steps=4,
            resume_checkpoint=checkpoint,
        ),
        cwd=repo_root,
        expect_success=False,
    )
    assert rejected.returncode != 0
    assert "sidecar identity mismatch" in (rejected.stdout + rejected.stderr)
    assert _tree_digest(checkpoint) == checkpoint_digest

    artifact_root = tmp_path / "artifact"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/tokenizer/export_xtokenizer.py",
            str(checkpoint),
            str(artifact_root),
        ],
        cwd=repo_root,
    )
    assert (artifact_root / "manifest.json").is_file()
    exported = load_exported_xtokenizer(artifact_root)
    raw_anchor = np.asarray(
        [0.1, -0.2, 0.3, 0.02, -0.03, 0.04, 0.0],
        dtype=np.float32,
    )
    raw_actions = np.asarray(
        [
            [0.01, -0.02, 0.03, 0.01, 0.02, -0.01, 1.0],
            [0.02, -0.01, 0.01, -0.02, 0.01, 0.02, 0.0],
            [-0.01, 0.02, 0.02, 0.01, -0.01, 0.03, 1.0],
            [0.03, 0.01, -0.02, -0.01, 0.03, 0.01, 0.0],
        ],
        dtype=np.float32,
    )
    episode = build_bridge_xtokenizer_episode(
        np.repeat(raw_anchor[None], len(raw_actions), axis=0),
        raw_actions,
        DOF_DIMS,
        dof_order=DOF_ORDER,
    )
    tokens = exported.encode(
        torch.from_numpy(episode.absolute_actions).unsqueeze(0),
        anchor=torch.from_numpy(episode.observation_states_absolute[0]).unsqueeze(0),
        robot_type=torch.zeros(1, dtype=torch.long),
    )
    assert tokens[0].shape == (ACTION_HORIZON // COMPRESSION_RATIO, 4)
    assert tokens.dtype == torch.long
    assert bool((tokens >= 0).all())
    assert bool((tokens < 8).all())
