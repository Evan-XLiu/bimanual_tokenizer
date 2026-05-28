"""Shared checkpoint artifact constants/helpers for policy/tokenizer artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import torch
from huggingface_hub import HfApi, constants, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

POLICY_IO_FILENAME = "policy_io.json"
NORM_STATS_FILENAME = "norm_stats.pt"


@dataclass(frozen=True)
class HubDownloadOptions:
    """Download options forwarded to ``hf_hub_download``."""

    revision: str | None = None
    cache_dir: str | Path | None = None
    force_download: bool = False
    proxies: dict | None = None
    resume_download: bool | None = None
    local_files_only: bool = False
    token: str | bool | None = None

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "force_download": self.force_download,
            "proxies": self.proxies,
            "resume_download": self.resume_download,
            "token": self.token,
            "local_files_only": self.local_files_only,
        }


@dataclass(frozen=True)
class ResolvedWeightFile:
    """Resolved checkpoint weight file and loader format."""

    path: Path
    format: Literal["safetensors", "pickle"]


def normalize_push_to_hub_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate and default HF upload kwargs used by save_pretrained()."""
    push_kwargs = dict(kwargs)
    normalized = {
        "commit_message": push_kwargs.pop(
            "commit_message",
            "Push model using huggingface_hub.",
        ),
        "private": push_kwargs.pop("private", None),
        "token": push_kwargs.pop("token", None),
        "branch": push_kwargs.pop("branch", None),
        "create_pr": push_kwargs.pop("create_pr", None),
        "allow_patterns": push_kwargs.pop("allow_patterns", None),
        "ignore_patterns": push_kwargs.pop("ignore_patterns", None),
        "delete_patterns": push_kwargs.pop("delete_patterns", None),
    }
    if push_kwargs:
        unexpected = ", ".join(sorted(push_kwargs))
        raise TypeError(f"Unexpected push_to_hub kwargs: {unexpected}")
    return normalized


def push_saved_pretrained_dir(
    *,
    save_dir: Path,
    repo_id: str | None,
    push_kwargs: dict[str, Any],
) -> str:
    """Upload an already-written pretrained artifact directory to the Hub."""
    if repo_id is None:
        repo_id = save_dir.name
    api = HfApi(token=push_kwargs["token"])
    created_repo = api.create_repo(
        repo_id=repo_id,
        private=push_kwargs["private"],
        exist_ok=True,
    )
    return api.upload_folder(
        repo_id=created_repo.repo_id,
        repo_type="model",
        folder_path=save_dir,
        commit_message=push_kwargs["commit_message"],
        revision=push_kwargs["branch"],
        create_pr=push_kwargs["create_pr"],
        allow_patterns=push_kwargs["allow_patterns"],
        ignore_patterns=push_kwargs["ignore_patterns"],
        delete_patterns=push_kwargs["delete_patterns"],
    )


def safetensors_shard_filenames(model_dir: str | Path) -> tuple[str, ...]:
    """Return shard filenames referenced by a HF safetensors index."""
    root = Path(model_dir)
    index_path = root / constants.SAFETENSORS_INDEX_FILE
    try:
        index = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid safetensors index at {index_path}.") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Invalid safetensors index at {index_path}: missing weight_map."
        )
    shard_names = set(weight_map.values())
    if not all(isinstance(name, str) for name in shard_names):
        raise ValueError(
            f"Invalid safetensors index at {index_path}: shard names must be strings."
        )
    return tuple(sorted(shard_names))


def is_complete_safetensors_dir(model_dir: str | Path) -> bool:
    """Return whether a directory has single-file or complete sharded weights."""
    root = Path(model_dir)
    if not (root / constants.SAFETENSORS_INDEX_FILE).exists():
        return (root / constants.SAFETENSORS_SINGLE_FILE).exists()
    try:
        return all((root / name).exists() for name in safetensors_shard_filenames(root))
    except ValueError:
        return False


def _is_missing_local_artifact_path(model_id: str) -> bool:
    """Return whether a missing artifact id is clearly a local filesystem path."""
    local_root = Path(model_id)
    if local_root.is_absolute() or model_id.startswith((".", "~")):
        return True
    # Hugging Face repo ids have at most one slash (`namespace/name`). More path
    # segments are local checkpoint paths such as outputs/.../pretrained_model.
    return len(PurePosixPath(model_id).parts) > 2


def resolve_sidecar_path(
    model_id_or_path: str | Path,
    filename: str,
    *,
    hub_options: HubDownloadOptions | None = None,
) -> Path | None:
    """Resolve sidecar filename from either a local directory or HF repo id."""
    model_id = str(model_id_or_path)
    local_path = Path(model_id) / filename
    if local_path.exists():
        return local_path

    local_root = Path(model_id)
    if local_root.exists():
        return None
    if _is_missing_local_artifact_path(model_id):
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            **(hub_options or HubDownloadOptions()).to_kwargs(),
        )
    except EntryNotFoundError:
        return None
    return Path(downloaded)


def resolve_pretrained_weight_file(
    model_id_or_path: str | Path,
    *,
    hub_options: HubDownloadOptions | None = None,
) -> ResolvedWeightFile:
    """Resolve a pretrained model weight file from local disk or the HF Hub."""
    model_id = str(model_id_or_path)
    local_root = Path(model_id)
    if local_root.is_dir():
        safetensors_index_path = local_root / constants.SAFETENSORS_INDEX_FILE
        if safetensors_index_path.exists():
            if not is_complete_safetensors_dir(local_root):
                raise FileNotFoundError(
                    f"Sharded safetensors checkpoint under {local_root} is incomplete."
                )
            return ResolvedWeightFile(local_root, "safetensors")
        safetensors_path = local_root / constants.SAFETENSORS_SINGLE_FILE
        if safetensors_path.exists():
            return ResolvedWeightFile(safetensors_path, "safetensors")
        pickle_path = local_root / constants.PYTORCH_WEIGHTS_NAME
        if pickle_path.exists():
            return ResolvedWeightFile(pickle_path, "pickle")
        raise FileNotFoundError(
            f"No pretrained weights found under {local_root}. Expected "
            f"{constants.SAFETENSORS_SINGLE_FILE}, "
            f"{constants.SAFETENSORS_INDEX_FILE}, or {constants.PYTORCH_WEIGHTS_NAME}."
        )
    if local_root.exists():
        raise FileNotFoundError(
            f"Pretrained model path must be a directory, got {local_root}."
        )
    if _is_missing_local_artifact_path(model_id):
        raise FileNotFoundError(f"Pretrained model path does not exist: {local_root}.")

    options = hub_options or HubDownloadOptions()
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=constants.SAFETENSORS_SINGLE_FILE,
            **options.to_kwargs(),
        )
        return ResolvedWeightFile(Path(path), "safetensors")
    except EntryNotFoundError:
        try:
            index_path = hf_hub_download(
                repo_id=model_id,
                filename=constants.SAFETENSORS_INDEX_FILE,
                **options.to_kwargs(),
            )
        except EntryNotFoundError:
            path = hf_hub_download(
                repo_id=model_id,
                filename=constants.PYTORCH_WEIGHTS_NAME,
                **options.to_kwargs(),
            )
            return ResolvedWeightFile(Path(path), "pickle")

        for shard_filename in safetensors_shard_filenames(Path(index_path).parent):
            hf_hub_download(
                repo_id=model_id,
                filename=shard_filename,
                **options.to_kwargs(),
            )
        return ResolvedWeightFile(Path(index_path).parent, "safetensors")


def collect_normalizer_stats(
    normalizer: Any,
    stats: dict[str, dict[str, torch.Tensor]],
    modes: dict[str, str],
) -> None:
    """Extract CPU stats/modes from Normalize/Unnormalize-like runtime modules."""
    if normalizer is None:
        return
    for key in getattr(normalizer, "keys", []):
        if key not in stats and key in normalizer.stats:
            stats[key] = {
                k: v.detach().cpu()
                if isinstance(v, torch.Tensor)
                else torch.as_tensor(v)
                for k, v in normalizer.stats[key].items()
            }
        if key not in modes and key in normalizer.modes:
            modes[key] = normalizer.modes[key].value
