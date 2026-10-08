# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from praxis.policies.catalog import (
    DEFAULT_POLICY_BACKBONE,
    DEFAULT_POLICY_CODEC,
    DEFAULT_POLICY_METHOD,
    POLICY_BACKBONES,
    POLICY_CODECS,
    POLICY_CODECS_BY_METHOD,
    POLICY_METHODS,
    policy_spec,
    supported_policy_names,
)

CODECS = POLICY_CODECS
# The catalog's default matrix intentionally excludes Molmo2-ER because it is
# currently available only for the explicit KI OAT/FAST/BTT policy specs.  The
# launcher still needs to expose that backbone so those extra specs can be
# submitted through the normal Slurm path.
BACKBONES = (*POLICY_BACKBONES, "molmo")
METHODS = POLICY_METHODS
DEFAULT_METHOD = DEFAULT_POLICY_METHOD
DEFAULT_CODEC = DEFAULT_POLICY_CODEC
DEFAULT_BACKBONE = DEFAULT_POLICY_BACKBONE
SKIP_FIT_CODECS = frozenset({"bin", "fast"})

DATASET_CONFIG_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "praxis" / "configs" / "dataset"
)

# Dataset name -> HuggingFace repo_id. All launcher-supported datasets must be
# registered here explicitly; there is no implicit ``lerobot/<name>`` fallback.
DATASET_REPO_IDS: dict[str, str] = {
    "libero": "lerobot/libero",
    "libero_10": "lerobot/libero_10",
    "libero_spatial": "lerobot/libero_spatial",
    "libero_goal": "lerobot/libero_goal",
    "libero_object": "lerobot/libero_object",
    "libero_long": "lerobot/libero_long",
    "metaworld_mt50": "reece-omahoney/metaworld_mt50_video",
    "bridge_single_view": "chaoqi-liu/bridge_single_view",
    "mshab_settable": "chaoqi-liu/mshab_settable_clean",
    "robocasa_mt5": "chaoqi-liu/robocasa_mt5_PH100_TH500",
    "robomimic_mt4": "chaoqi-liu/robomimic_mt4_ph",
    "robotwin2_clean": "local/robotwin2_depth_merged_clean",
    "robotwin_new_demo_clean": "local/robotwin_new_demo_clean",
    "bicat_box": "local/bicat_box",
    "bicat_bottles": "local/bicat_bottles",
}
DATASETS = tuple(DATASET_REPO_IDS)

DATASET_OVERRIDES: dict[str, list[str]] = {}


def resolve_dataset_repo_id(name: str) -> str:
    """Return the HF repo_id for a launcher dataset name.

    Raises ``KeyError`` if ``name`` is not a registered launcher dataset.
    """
    try:
        return DATASET_REPO_IDS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown launcher dataset {name!r}. "
            f"Register it in DATASET_REPO_IDS. "
            f"Known: {', '.join(sorted(DATASET_REPO_IDS))}."
        ) from exc


def resolve_dataset_overrides(name: str) -> list[str]:
    """Return extra Hydra overrides needed for a launcher dataset."""
    return list(DATASET_OVERRIDES.get(name, ()))


def has_dataset_config(name: str) -> bool:
    """Return whether a launcher dataset has a dedicated Hydra config file."""
    return (DATASET_CONFIG_ROOT / f"{name}.yaml").exists()


def resolve_dataset_train_overrides(name: str) -> list[str]:
    """Return the canonical training-time Hydra overrides for a dataset.

    Datasets with dedicated Hydra config files should be selected by config group
    so dataset-owned fields come from the config itself. Datasets without a
    config file fall back to explicit scalar overrides.
    """
    if has_dataset_config(name):
        return [f"dataset={name}"]

    return [
        f"dataset.name={name}",
        f"dataset.repo_id={resolve_dataset_repo_id(name)}",
        *resolve_dataset_overrides(name),
    ]


SUBMITIT_LOG_BASE = Path("outputs/slurm")
POLICY_CONFIG_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "praxis" / "configs" / "policy"
)
SLURM_MACHINE_RANK_PLACEHOLDER = "__SLURM_PROCID__"
MASTER_ADDR_PLACEHOLDER = "__MASTER_ADDR__"

RESOURCE_DEFAULTS: dict[str, tuple[int, str]] = {
    "tokenizer": (2, "64G"),
    "policy_qwen": (4, "64G"),
    "policy_pali": (4, "64G"),
    "policy_smol": (4, "64G"),
    "policy_molmo": (4, "64G"),
    "eval": (1, "64G"),
    "eval_monitor": (1, "64G"),
}

CODEC_OVERRIDES: dict[str, list[str]] = {
    "oat": [
        "codecs@codec=tokens/oat",
    ],
    "bimanualoat": [
        "codecs@codec=tokens/bimanualoat",
    ],
    "boat": [
        "codecs@codec=tokens/boat",
    ],
    "acodec": [
        "codecs@codec=tokens/acodec",
    ],
    "fast": [
        "codecs@codec=tokens/fast",
    ],
    "bin": [
        "codecs@codec=tokens/bin",
    ],
    "quest": [
        "codecs@codec=tokens/quest",
    ],
    "bquest": [
        "codecs@codec=tokens/bquest",
    ],
}

CODEC_TRAINER_OVERRIDES: list[str] = [
    "trainer.num_workers=8",
]

POLICY_OVERRIDES: dict[str, list[str]] = {
    "qwen": [
        "policy.gradient_checkpointing=true",
    ],
    "pali": [
        "policy.gradient_checkpointing=true",
    ],
    "smol": [],
    "molmo": [
        "policy.gradient_checkpointing=true",
    ],
}

POLICY_TRAINER_OVERRIDES: dict[str, list[str]] = {
    "qwen": [
        "trainer.batch_size=8",
        "trainer.num_workers=2",
        "trainer.ema_decay=0.99",
        "trainer.log_every=100",
        "trainer.val_every=500",
        "trainer.reconst_every=500",
        "trainer.checkpoint_every=500",
        "trainer.sim_eval_every=null",
        "trainer.wandb_upload_eval_videos=false",
    ],
    "pali": [
        "trainer.batch_size=8",
        "trainer.num_workers=2",
        "trainer.ema_decay=0.99",
        "trainer.log_every=100",
        "trainer.val_every=500",
        "trainer.reconst_every=500",
        "trainer.checkpoint_every=500",
        "trainer.sim_eval_every=null",
        "trainer.wandb_upload_eval_videos=false",
    ],
    "smol": [
        "trainer.batch_size=16",
        "trainer.num_workers=2",
        "trainer.ema_decay=0.99",
        "trainer.log_every=100",
        "trainer.val_every=500",
        "trainer.reconst_every=500",
        "trainer.checkpoint_every=500",
        "trainer.sim_eval_every=null",
        "trainer.wandb_upload_eval_videos=false",
    ],
    "molmo": [
        "trainer.batch_size=4",
        "trainer.num_workers=2",
        "trainer.ema_decay=0.99",
        "trainer.log_every=100",
        "trainer.val_every=500",
        "trainer.checkpoint_every=500",
        "trainer.sim_eval_every=null",
        "trainer.wandb_upload_eval_videos=false",
    ],
}

POLICY_OVERRIDES_BY_METHOD: dict[str, dict[str, list[str]]] = {
    "bar": POLICY_OVERRIDES,
    "ki": {backbone: [] for backbone in BACKBONES},
    "fm": {backbone: [] for backbone in BACKBONES},
}


class StageResources(NamedTuple):
    gpus: int
    mem: str
    partition: str
    account: str
    num_nodes: int


class LaunchContext(NamedTuple):
    dry_run: bool
    project_root: str
    ds_short: str
    date_str: str
    gpus_override: int | None
    mem_override: str | None
    num_nodes: int
    main_process_ip: str | None
    main_process_port: int
    override: list[str]
    num_eval_per_task: int
    num_parallel_env: int
    record_episodes_per_task: int
    stale_timeout_sec: int
    eval_timeout_sec: int
    eval_override: list[str]
    python_bin: str | None
    git_ref: str | None
    git_worktree_dir: str


def dataset_short(name: str) -> str:
    """libero_10 -> libero10 (matches exec.sh naming convention)."""
    return name.replace("_", "")


def _resolve_axis(
    value: str | None,
    *,
    default: str,
    choices: tuple[str, ...],
    label: str,
) -> str:
    resolved = value or default
    if resolved not in choices:
        raise ValueError(
            f"Unsupported {label} {resolved!r}. Expected one of: {', '.join(choices)}."
        )
    return resolved


def resolve_method(method: str | None) -> str:
    """Resolve the policy generation method."""
    return _resolve_axis(
        method,
        default=DEFAULT_METHOD,
        choices=POLICY_METHODS,
        label="policy method",
    )


def resolve_codec(codec: str | None) -> str:
    """Resolve the action-token codec."""
    return _resolve_axis(
        codec,
        default=DEFAULT_CODEC,
        choices=CODECS,
        label="codec",
    )


def method_uses_codec(method: str) -> bool:
    """Return whether policy ``method`` requires an action-token codec axis."""
    return bool(POLICY_CODECS_BY_METHOD[resolve_method(method)])


def resolve_policy_codec(method: str, codec: str | None) -> str | None:
    """Resolve the policy codec only for methods that use one."""
    resolved_method = resolve_method(method)
    if method_uses_codec(resolved_method):
        return resolve_codec(codec)
    if codec is not None:
        raise ValueError(f"Policy method {resolved_method!r} does not use a codec.")
    return None


def resolve_backbone(backbone: str | None) -> str:
    """Resolve the policy backbone."""
    return _resolve_axis(
        backbone,
        default=DEFAULT_BACKBONE,
        choices=BACKBONES,
        label="backbone",
    )


def supported_policy_config_names() -> tuple[str, ...]:
    """Return launcher-supported policy config names."""
    return supported_policy_names()


def policy_config_name(method: str, codec: str | None, backbone: str) -> str:
    """Return the canonical Hydra policy config group name."""
    resolved_method = resolve_method(method)
    resolved_codec = resolve_policy_codec(resolved_method, codec)
    resolved_backbone = resolve_backbone(backbone)
    return policy_spec(resolved_method, resolved_codec, resolved_backbone).name


def policy_overrides(method: str, backbone: str) -> list[str]:
    """Return launcher Hydra overrides for one policy family/backbone."""
    resolved_method = resolve_method(method)
    resolved_backbone = resolve_backbone(backbone)
    try:
        return list(POLICY_OVERRIDES_BY_METHOD[resolved_method][resolved_backbone])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported policy override matrix entry: "
            f"method={method!r}, backbone={backbone!r}."
        ) from exc
