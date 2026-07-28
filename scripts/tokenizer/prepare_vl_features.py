#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare resumable offline Qwen vision-language features for X-Tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import numpy as np

from praxis.data.xtokenizer_vl import (
    VLFeatureManifest,
    file_sha256,
    vl_sidecar_generation_fingerprint,
)

DEFAULT_EXTRACTOR_REVISION = "praxis-qwen2.5-vl-paper-a3-v2"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGE_KEYS = (
    "observation.images.face",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


@dataclass(frozen=True)
class _Options:
    output_root: Path
    dataset_repo_id: str
    dataset_revision: str
    extractor_revision: str
    model_id: str
    model_revision: str
    layer: int
    stride: int
    feature_dim: int
    seed: int
    batch_size: int
    synthetic: bool
    image_keys: tuple[str, ...]
    global_instruction: str
    local_subtask: str


class _FeatureSource(Protocol):
    episode_indices: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    image_keys: tuple[str, ...]

    def extract_episode(
        self,
        episode_index: int,
        length: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _SyntheticFeatureSource:
    def __init__(
        self,
        *,
        episodes: int,
        frames: int,
        feature_dim: int,
        stride: int,
        seed: int,
        image_keys: tuple[str, ...],
    ) -> None:
        self.episode_indices = tuple(range(episodes))
        self.episode_lengths = (frames,) * episodes
        self.feature_dim = feature_dim
        self.stride = stride
        self.seed = seed
        self.image_keys = image_keys

    def extract_episode(
        self,
        episode_index: int,
        length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        seed = np.random.SeedSequence([self.seed, episode_index])
        generator = np.random.default_rng(seed)
        feature_length = (length + self.stride - 1) // self.stride
        features = generator.standard_normal(
            (feature_length, len(self.image_keys), self.feature_dim),
            dtype=np.float32,
        )
        valid = np.ones(
            (feature_length, len(self.image_keys)),
            dtype=np.bool_,
        )
        return features, valid


class _QwenFeatureSource:
    """LeRobot-backed Qwen extractor; heavyweight imports remain offline-only."""

    def __init__(self, args: argparse.Namespace, options: _Options) -> None:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from praxis.data.lerobot_adapter import (
            _episode_bounds_from_metadata,
            write_lerobot_cache_provenance,
        )

        if not args.dataset_repo_id:
            raise ValueError("--dataset-repo-id is required without --synthetic.")
        self._torch = torch
        self._dataset = LeRobotDataset(
            repo_id=options.dataset_repo_id,
            root=args.dataset_root,
            revision=options.dataset_revision,
            force_cache_sync=True,
            download_videos=True,
        )
        self._dataset._ensure_hf_dataset_loaded()
        write_lerobot_cache_provenance(
            root=args.dataset_root,
            repo_id=options.dataset_repo_id,
            revision=options.dataset_revision,
        )
        episode_meta = self._dataset.meta.episodes
        if episode_meta is None:
            raise ValueError("LeRobot dataset episode metadata is unavailable.")
        episode_bounds = _episode_bounds_from_metadata(episode_meta)
        available = tuple(episode_bounds)
        if args.episodes is not None:
            if args.episodes <= 0:
                raise ValueError("--episodes must be positive.")
            available = available[: args.episodes]
        self.episode_indices = available
        self._episode_starts = {index: episode_bounds[index][0] for index in available}
        self.episode_lengths = tuple(
            min(
                episode_bounds[index][1] - episode_bounds[index][0],
                args.frames
                if args.frames is not None
                else episode_bounds[index][1] - episode_bounds[index][0],
            )
            for index in available
        )
        if any(length <= 0 for length in self.episode_lengths):
            raise ValueError(
                "Selected LeRobot episodes must contain at least one frame."
            )

        self._processor = cast(
            Any,
            AutoProcessor.from_pretrained(
                options.model_id,
                revision=options.model_revision,
                max_pixels=200_704,
            ),
        )
        self._model = cast(
            Any,
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                options.model_id,
                revision=options.model_revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ),
        )
        self._device = torch.device(args.device)
        self._model.to(self._device)
        self._model.eval()
        self._layer = options.layer
        self._feature_dim = options.feature_dim
        self._batch_size = options.batch_size
        self._stride = options.stride
        metadata_features = getattr(self._dataset.meta, "features", {})
        available_image_keys = (
            tuple(
                sorted(
                    (
                        str(key)
                        for key in metadata_features
                        if str(key).startswith("observation.images.")
                    ),
                    key=_view_sort_key,
                )
            )
            if isinstance(metadata_features, Mapping)
            else ()
        )
        self.image_keys = options.image_keys or available_image_keys
        if not self.image_keys:
            raise ValueError(
                "LeRobot metadata contains no observation image fields. "
                "Pass one or more --image-key values."
            )
        self._image_keys = self.image_keys
        self._global_instruction = options.global_instruction
        self._local_subtask = options.local_subtask

    def extract_episode(
        self,
        episode_index: int,
        length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = self._episode_starts[episode_index]
        feature_batches: list[np.ndarray] = []
        valid_batches: list[np.ndarray] = []
        frame_offsets = range(0, length, self._stride)
        sampled_frames = list(frame_offsets)
        for offset in range(0, len(sampled_frames), self._batch_size):
            batch_frames = sampled_frames[offset : offset + self._batch_size]
            samples = [self._dataset[start + frame] for frame in batch_frames]
            features, valid = self._extract_samples(samples)
            feature_batches.append(features)
            valid_batches.append(valid)
        return (
            np.concatenate(feature_batches).astype(np.float32, copy=False),
            np.concatenate(valid_batches),
        )

    def _extract_samples(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        feature_array = np.zeros(
            (len(samples), len(self._image_keys), self._feature_dim),
            dtype=np.float32,
        )
        valid = np.zeros(
            (len(samples), len(self._image_keys)),
            dtype=np.bool_,
        )
        texts: list[str] = []
        images: list[Any] = []
        locations: list[tuple[int, int]] = []
        for sample_index, sample in enumerate(samples):
            task = sample.get("task", self._global_instruction)
            instruction = str(task) if task is not None else self._global_instruction
            prompt = f"Task: {instruction}\nCurrent step: {self._local_subtask}"
            for view_index, key in enumerate(self._image_keys):
                image_value = sample.get(key)
                if image_value is None:
                    continue
                image = _as_pil_image(image_value, torch_module=torch)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                texts.append(
                    self._processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                images.append(image)
                locations.append((sample_index, view_index))
        if not locations:
            return feature_array, valid
        inputs = self._processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output = self._model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states.")
        hidden = hidden_states[self._layer].float()
        attention_mask = inputs.get("attention_mask")
        input_ids = inputs.get("input_ids")
        image_token_id = getattr(
            getattr(self._model, "config", None),
            "image_token_id",
            None,
        )
        if (
            attention_mask is None
            or input_ids is None
            or not isinstance(image_token_id, int)
        ):
            raise RuntimeError(
                "Qwen extraction requires input_ids, attention_mask, and image_token_id."
            )
        sequence_valid = attention_mask.to(hidden.device, dtype=torch.bool)
        visual_mask = sequence_valid & input_ids.to(hidden.device).eq(image_token_id)
        text_mask = sequence_valid & ~visual_mask
        if bool((~visual_mask.any(dim=1)).any()) or bool((~text_mask.any(dim=1)).any()):
            raise RuntimeError(
                "Qwen extraction found an empty visual or text token set."
            )
        visual = _masked_mean(hidden, visual_mask)
        text = _masked_mean(hidden, text_mask)
        extracted = (0.5 * (visual + text)).detach().cpu().numpy().astype(np.float32)
        expected_shape = (len(locations), self._feature_dim)
        if extracted.shape != expected_shape:
            raise ValueError(
                "Extracted Qwen feature dimension mismatch: "
                f"got {extracted.shape}, expected {expected_shape}."
            )
        for feature, (sample_index, view_index) in zip(
            extracted,
            locations,
            strict=True,
        ):
            feature_array[sample_index, view_index] = feature
            valid[sample_index, view_index] = True
        return feature_array, valid


def _masked_mean(hidden: Any, mask: Any) -> Any:
    weights = mask.to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _view_sort_key(key: str) -> tuple[int, str]:
    normalized = key.lower()
    if any(name in normalized for name in ("face", "front", "head", "main")):
        return (0, normalized)
    if "left" in normalized and "wrist" in normalized:
        return (1, normalized)
    if "right" in normalized and "wrist" in normalized:
        return (2, normalized)
    return (3, normalized)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--extractor-revision",
        default=DEFAULT_EXTRACTOR_REVISION,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--layer", type=int, default=-3)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=3584)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset-repo-id")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-key", action="append")
    parser.add_argument(
        "--global-instruction",
        default="Follow the robot manipulation instruction.",
    )
    parser.add_argument(
        "--local-subtask",
        default="Describe the current manipulation-relevant visual state.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or resume sidecar shards and publish one validated manifest."""
    args = _build_parser().parse_args(argv)
    options = _validate_options(args)
    if not options.synthetic:
        options = _resolve_hub_revisions(options)
    if options.synthetic:
        if args.episodes is None or args.frames is None:
            raise ValueError("--synthetic requires --episodes and --frames.")
        if args.episodes <= 0 or args.frames <= 0:
            raise ValueError("--episodes and --frames must be positive.")

    options.output_root.mkdir(parents=True, exist_ok=True)
    shards_root = options.output_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    with _exclusive_writer_lock(options.output_root):
        _cleanup_stale_temporary_files(options.output_root)
        _cleanup_unreferenced_generations(
            options.output_root,
            _manifest_referenced_generation_files(options.output_root),
        )
        source: _FeatureSource
        if options.synthetic:
            assert args.episodes is not None
            assert args.frames is not None
            source = _SyntheticFeatureSource(
                episodes=args.episodes,
                frames=args.frames,
                feature_dim=options.feature_dim,
                stride=options.stride,
                seed=options.seed,
                image_keys=options.image_keys,
            )
        else:
            source = _QwenFeatureSource(args, options)
            if source.image_keys != options.image_keys:
                options = replace(options, image_keys=source.image_keys)

        for episode_index, length in zip(
            source.episode_indices,
            source.episode_lengths,
            strict=True,
        ):
            shard_path = shards_root / f"episode_{episode_index:06d}.npz"
            fingerprint = _shard_fingerprint(
                options,
                episode_index=episode_index,
                length=length,
            )
            if _shard_matches(
                shard_path,
                fingerprint=fingerprint,
                length=length,
                stride=options.stride,
                num_views=len(options.image_keys),
                feature_dim=options.feature_dim,
            ):
                continue
            features, valid = source.extract_episode(episode_index, length)
            _validate_extracted_features(
                features,
                valid,
                length=length,
                stride=options.stride,
                num_views=len(options.image_keys),
                feature_dim=options.feature_dim,
            )
            next_features = np.zeros_like(features)
            if len(features) > 1:
                next_features[:-1] = features[1:]
            _atomic_save_npz(
                shard_path,
                fingerprint=np.asarray(fingerprint),
                features=features,
                next_features=next_features,
                valid=valid,
            )

        _publish_sidecar(options, source)
    print(
        json.dumps(
            {
                "output_root": str(options.output_root),
                "episodes": len(source.episode_indices),
                "frames": sum(source.episode_lengths),
                "feature_dim": options.feature_dim,
                "resumable_shards": len(source.episode_indices),
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_options(args: argparse.Namespace) -> _Options:
    for name in (
        "dataset_revision",
        "extractor_revision",
        "model_id",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--{name.replace('_', '-')} must be non-empty.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.feature_dim <= 0:
        raise ValueError("--feature-dim must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    model_revision = args.model_revision
    if model_revision is None:
        if not args.synthetic:
            raise ValueError(
                "--model-revision is required for real extraction so the "
                "sidecar cannot silently track a mutable model revision."
            )
        model_revision = "synthetic-v1"
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("--model-revision must be non-empty.")
    dataset_repo_id = args.dataset_repo_id
    if not args.synthetic and (
        not isinstance(dataset_repo_id, str) or not dataset_repo_id.strip()
    ):
        raise ValueError("--dataset-repo-id is required without --synthetic.")
    if dataset_repo_id is None:
        dataset_repo_id = "synthetic"
    image_keys = tuple(args.image_key or ())
    if args.synthetic and not image_keys:
        image_keys = DEFAULT_IMAGE_KEYS
    if any(not isinstance(key, str) or not key.strip() for key in image_keys):
        raise ValueError("--image-key values must be non-empty.")
    if len(set(image_keys)) != len(image_keys):
        raise ValueError("--image-key values must be unique.")
    image_keys = tuple(sorted(image_keys, key=_view_sort_key))
    for name in ("global_instruction", "local_subtask"):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--{name.replace('_', '-')} must be non-empty.")
    return _Options(
        output_root=args.output_root,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=args.dataset_revision,
        extractor_revision=args.extractor_revision,
        model_id=args.model_id,
        model_revision=model_revision,
        layer=args.layer,
        stride=args.stride,
        feature_dim=args.feature_dim,
        seed=args.seed,
        batch_size=args.batch_size,
        synthetic=args.synthetic,
        image_keys=image_keys,
        global_instruction=args.global_instruction,
        local_subtask=args.local_subtask,
    )


def _resolve_hub_revisions(options: _Options) -> _Options:
    """Resolve real extraction inputs to immutable Hub commit identifiers."""
    from huggingface_hub import HfApi

    api = HfApi()
    dataset_info = api.dataset_info(
        options.dataset_repo_id,
        revision=options.dataset_revision,
    )
    model_info = api.model_info(
        options.model_id,
        revision=options.model_revision,
    )
    return replace(
        options,
        dataset_revision=_validated_hub_sha(
            getattr(dataset_info, "sha", None),
            label="dataset",
        ),
        model_revision=_validated_hub_sha(
            getattr(model_info, "sha", None),
            label="model",
        ),
    )


def _validated_hub_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(
            f"Unable to resolve {label} revision to a 40-character Hub commit SHA."
        )
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(
            f"Resolved {label} revision is not a hexadecimal Hub commit SHA."
        ) from None
    return value.lower()


@contextmanager
def _exclusive_writer_lock(output_root: Path) -> Iterator[None]:
    """Serialize shard replacement, publication, and generation cleanup."""
    import fcntl

    lock_path = output_root / ".prepare_vl_features.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cleanup_stale_temporary_files(output_root: Path) -> None:
    """Reclaim temporary files left by a terminated prior writer."""
    for root in (output_root, output_root / "shards"):
        for path in root.glob(".*.tmp"):
            path.unlink(missing_ok=True)


def _manifest_referenced_generation_files(output_root: Path) -> set[str]:
    try:
        raw = json.loads((output_root / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()
    if not isinstance(raw, dict):
        return set()
    referenced: set[str] = set()
    for key in ("features_file", "next_features_file", "valid_file"):
        value = raw.get(key)
        if not isinstance(value, str):
            continue
        try:
            _published_array_path(output_root, value)
        except ValueError:
            continue
        referenced.add(value)
    return referenced


def _publish_sidecar(options: _Options, source: _FeatureSource) -> None:
    episode_count = len(source.episode_indices)
    max_feature_frames = max(
        (length + options.stride - 1) // options.stride
        for length in source.episode_lengths
    )
    feature_shape = (
        episode_count,
        max_feature_frames,
        len(options.image_keys),
        options.feature_dim,
    )
    generation_spec_fingerprint = _generation_spec_fingerprint(options, source)
    existing_files = _matching_published_generation(
        options,
        source,
        generation_spec_fingerprint=generation_spec_fingerprint,
    )
    if existing_files is not None:
        _cleanup_unreferenced_generations(options.output_root, existing_files)
        return

    generation = f"{generation_spec_fingerprint[:16]}-{uuid4().hex}"
    features_path = options.output_root / f"features-{generation}.npy"
    next_features_path = options.output_root / f"next_features-{generation}.npy"
    valid_path = options.output_root / f"valid-{generation}.npy"
    temporary_paths: list[Path] = []
    open_memmaps: list[np.memmap] = []
    try:
        feature_temp, features = _temporary_memmap(
            features_path,
            dtype=np.float32,
            shape=feature_shape,
        )
        temporary_paths.append(feature_temp)
        open_memmaps.append(features)
        next_temp, next_features = _temporary_memmap(
            next_features_path,
            dtype=np.float32,
            shape=feature_shape,
        )
        temporary_paths.append(next_temp)
        open_memmaps.append(next_features)
        valid_temp, valid = _temporary_memmap(
            valid_path,
            dtype=np.bool_,
            shape=feature_shape[:-1],
        )
        temporary_paths.append(valid_temp)
        open_memmaps.append(valid)
        features[:] = 0
        next_features[:] = 0
        valid[:] = False
        for row, (episode_index, length) in enumerate(
            zip(source.episode_indices, source.episode_lengths, strict=True)
        ):
            shard_path = (
                options.output_root / "shards" / f"episode_{episode_index:06d}.npz"
            )
            feature_length = (length + options.stride - 1) // options.stride
            with np.load(shard_path, allow_pickle=False) as shard:
                features[row, :feature_length] = shard["features"]
                next_features[row, :feature_length] = shard["next_features"]
                valid[row, :feature_length] = shard["valid"]
        features.flush()
        next_features.flush()
        valid.flush()
        _close_memmaps(open_memmaps)
        open_memmaps.clear()
        os.replace(feature_temp, features_path)
        os.replace(next_temp, next_features_path)
        os.replace(valid_temp, valid_path)
    except BaseException:
        _close_memmaps(open_memmaps)
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        **_manifest_semantics(options, source),
        "features_sha256": file_sha256(features_path),
        "next_features_sha256": file_sha256(next_features_path),
        "valid_sha256": file_sha256(valid_path),
        "features_file": features_path.name,
        "next_features_file": next_features_path.name,
        "valid_file": valid_path.name,
    }
    manifest["generation_fingerprint"] = vl_sidecar_generation_fingerprint(manifest)
    _atomic_write_text(
        options.output_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _cleanup_unreferenced_generations(
        options.output_root,
        {
            features_path.name,
            next_features_path.name,
            valid_path.name,
        },
    )


def _manifest_semantics(
    options: _Options,
    source: _FeatureSource,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "dataset_repo_id": options.dataset_repo_id,
        "dataset_revision": options.dataset_revision,
        "extractor_revision": options.extractor_revision,
        "model_id": options.model_id,
        "model_revision": options.model_revision,
        "layer": options.layer,
        "stride": options.stride,
        "feature_dim": options.feature_dim,
        "episode_indices": list(source.episode_indices),
        "episode_lengths": list(source.episode_lengths),
        "image_keys": list(options.image_keys),
        "global_instruction": options.global_instruction,
        "local_subtask": options.local_subtask,
        "synthetic": options.synthetic,
    }


def _generation_spec_fingerprint(
    options: _Options,
    source: _FeatureSource,
) -> str:
    payload = {
        "dataset_repo_id": options.dataset_repo_id,
        "dataset_revision": options.dataset_revision,
        "extractor_revision": options.extractor_revision,
        "model_id": options.model_id,
        "model_revision": options.model_revision,
        "layer": options.layer,
        "stride": options.stride,
        "feature_dim": options.feature_dim,
        "seed": options.seed if options.synthetic else None,
        "synthetic": options.synthetic,
        "image_keys": options.image_keys,
        "global_instruction": options.global_instruction,
        "local_subtask": options.local_subtask,
        "episodes": list(
            zip(source.episode_indices, source.episode_lengths, strict=True)
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _matching_published_generation(
    options: _Options,
    source: _FeatureSource,
    *,
    generation_spec_fingerprint: str,
) -> set[str] | None:
    manifest_path = options.output_root / "manifest.json"
    try:
        manifest = VLFeatureManifest.from_path(manifest_path)
        expected_semantics = _manifest_semantics(options, source)
        for key, expected in expected_semantics.items():
            actual = getattr(manifest, key)
            if isinstance(actual, tuple):
                actual = list(actual)
            if actual != expected:
                return None
        generation_prefix = generation_spec_fingerprint[:16]
        if not (
            manifest.features_file.startswith(f"features-{generation_prefix}-")
            and manifest.next_features_file.startswith(
                f"next_features-{generation_prefix}-"
            )
            and manifest.valid_file.startswith(f"valid-{generation_prefix}-")
        ):
            return None
        filenames = {
            manifest.features_file,
            manifest.next_features_file,
            manifest.valid_file,
        }
        features_path = _published_array_path(
            options.output_root,
            manifest.features_file,
        )
        next_features_path = _published_array_path(
            options.output_root,
            manifest.next_features_file,
        )
        valid_path = _published_array_path(
            options.output_root,
            manifest.valid_file,
        )
        if (
            file_sha256(features_path) != manifest.features_sha256
            or file_sha256(next_features_path) != manifest.next_features_sha256
            or file_sha256(valid_path) != manifest.valid_sha256
        ):
            return None
        feature_shape = (
            len(source.episode_indices),
            max(
                (length + options.stride - 1) // options.stride
                for length in source.episode_lengths
            ),
            len(options.image_keys),
            options.feature_dim,
        )
        features = np.load(
            features_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        next_features = np.load(
            next_features_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        valid = np.load(
            valid_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            features.shape != feature_shape
            or next_features.shape != feature_shape
            or valid.shape != feature_shape[:-1]
            or features.dtype != np.float32
            or next_features.dtype != np.float32
            or valid.dtype != np.bool_
        ):
            return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, ValueError):
        return None
    return filenames


def _published_array_path(output_root: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError(
            f"Published generation filename must stay in the output root: {filename!r}."
        )
    return output_root / relative


def _cleanup_unreferenced_generations(
    output_root: Path,
    referenced_files: set[str],
) -> None:
    for pattern in ("features-*.npy", "next_features-*.npy", "valid-*.npy"):
        for path in output_root.glob(pattern):
            if path.name not in referenced_files:
                path.unlink(missing_ok=True)


def _temporary_memmap(
    destination: Path,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int, ...],
) -> tuple[Path, np.memmap]:
    temporary_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        array = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path, array


def _close_memmaps(arrays: list[np.memmap]) -> None:
    for array in arrays:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _shard_fingerprint(
    options: _Options,
    *,
    episode_index: int,
    length: int,
) -> str:
    payload = {
        "dataset_repo_id": options.dataset_repo_id,
        "dataset_revision": options.dataset_revision,
        "extractor_revision": options.extractor_revision,
        "model_id": options.model_id,
        "model_revision": options.model_revision,
        "layer": options.layer,
        "stride": options.stride,
        "feature_dim": options.feature_dim,
        "seed": options.seed if options.synthetic else None,
        "synthetic": options.synthetic,
        "image_keys": options.image_keys,
        "global_instruction": options.global_instruction,
        "local_subtask": options.local_subtask,
        "episode_index": episode_index,
        "length": length,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _shard_matches(
    path: Path,
    *,
    fingerprint: str,
    length: int,
    stride: int,
    num_views: int,
    feature_dim: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            features = shard["features"]
            next_features = shard["next_features"]
            return bool(
                str(shard["fingerprint"].item()) == fingerprint
                and features.shape
                == ((length + stride - 1) // stride, num_views, feature_dim)
                and next_features.shape == features.shape
                and shard["valid"].shape == features.shape[:-1]
                and features.dtype == np.float32
                and next_features.dtype == np.float32
                and shard["valid"].dtype == np.bool_
                and np.isfinite(features).all()
                and np.isfinite(next_features).all()
            )
    except (OSError, ValueError, KeyError):
        return False


def _validate_extracted_features(
    features: np.ndarray,
    valid: np.ndarray,
    *,
    length: int,
    stride: int,
    num_views: int,
    feature_dim: int,
) -> None:
    expected_shape = (
        (length + stride - 1) // stride,
        num_views,
        feature_dim,
    )
    if features.shape != expected_shape:
        raise ValueError(
            "Extracted feature shape mismatch: "
            f"got {features.shape}, expected {expected_shape}."
        )
    if features.dtype != np.float32:
        raise ValueError(f"Extracted features must use float32, got {features.dtype}.")
    if not np.isfinite(features).all():
        raise ValueError("Extracted features contain non-finite values.")
    if valid.shape != expected_shape[:-1] or valid.dtype != np.bool_:
        raise ValueError(
            "Extracted view-valid mask must have shape "
            f"{expected_shape[:-1]} and boolean dtype."
        )
    if bool(np.any(features[~valid] != 0)):
        raise ValueError("Missing-view feature entries must be zero-filled.")


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            cast(Any, np.savez)(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _as_pil_image(value: Any, *, torch_module: Any) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch_module.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got {tensor.shape}.")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
    else:
        array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


if __name__ == "__main__":
    raise SystemExit(main())
