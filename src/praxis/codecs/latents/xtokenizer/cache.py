# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Cache-generation adapter for exported X-Tokenizer artifacts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from praxis.data.bridge_xtokenizer import resolve_bridge_xtokenizer_layout


class XTokenizerCacheAdapter:
    """Present a native inference codec through the legacy cache API."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        device: str | torch.device,
        action_horizon: int,
        robot_type: str,
    ) -> None:
        from praxis.codecs.latents.xtokenizer.inference import (
            load_exported_xtokenizer_snapshot,
        )

        codec, self.manifest = load_exported_xtokenizer_snapshot(artifact_dir)
        if self.manifest["action_horizon"] != action_horizon:
            raise ValueError(
                "Exported X-Tokenizer action layout is incompatible with cache "
                f"horizon {action_horizon}."
            )
        layout = self.manifest["dof_layout"]
        self.dof_order = tuple(layout["dof_order"])
        self.dof_dims = {name: int(layout["dof_dims"][name]) for name in self.dof_order}
        required_dofs = {
            "follow_left_ee_cartesian_pos": 3,
            "follow_left_ee_rotation_6D": 6,
            "follow_left_gripper": 1,
        }
        for key, dim in required_dofs.items():
            if self.dof_dims.get(key) != dim:
                raise ValueError(
                    "Exported X-Tokenizer action layout is incompatible with "
                    f"Bridge: expected {key!r} with dimension {dim}."
                )
        self.bridge_layout = resolve_bridge_xtokenizer_layout(
            self.dof_dims,
            dof_order=self.dof_order,
        )
        self.embodiment_registry = dict(self.manifest["embodiment_registry"])
        if robot_type not in self.embodiment_registry:
            raise ValueError(
                f"Exported X-Tokenizer has no embodiment registry entry for {robot_type!r}."
            )
        if self.manifest["num_quantizers"] != 4:
            raise ValueError(
                "Bridge cache requires exactly four X-Tokenizer quantizers."
            )
        if self.manifest["codebook_size"] > torch.iinfo(torch.uint16).max + 1:
            raise ValueError(
                "Bridge cache uint16 storage cannot represent the codebook."
            )

        self.codec = codec.to(device)
        self.codec.eval()
        self.device = torch.device(device)
        self.artifact_sha256 = self.manifest["artifact_sha256"]
        self.source_checkpoint_sha256 = self.manifest["source_checkpoint_sha256"]
        self.codebook_size = self.manifest["codebook_size"]
        self.num_quantizers = self.manifest["num_quantizers"]
        self.compression_ratio = self.manifest["compression_ratio"]
        self.action_horizon = self.manifest["action_horizon"]

    @torch.no_grad()
    def encode(
        self,
        absolute_actions: Tensor,
        *,
        absolute_anchor: Tensor,
        dof_mask: Tensor,
        robot_type: str,
        token_order: str,
    ) -> dict[str, Tensor]:
        if token_order != "time_major":
            raise ValueError(
                "Native X-Tokenizer cache adapter requires time_major order."
            )
        try:
            robot_type_id = self.embodiment_registry[robot_type]
        except KeyError as exc:
            raise ValueError(f"Unknown X-Tokenizer embodiment {robot_type!r}.") from exc
        if absolute_actions.ndim != 3 or absolute_actions.shape[1:] != (
            self.codec.action_horizon,
            self.codec.action_dim,
        ):
            raise ValueError(
                "Cache actions must match the exported action layout, got "
                f"{tuple(absolute_actions.shape)}."
            )
        if dof_mask.shape != absolute_actions.shape or dof_mask.dtype != torch.bool:
            raise ValueError("dof_mask must be boolean and match cache actions.")
        if absolute_anchor.shape != (
            absolute_actions.shape[0],
            self.codec.action_dim,
        ):
            raise ValueError(
                "absolute_anchor must have shape "
                f"[B, {self.codec.action_dim}], got {tuple(absolute_anchor.shape)}."
            )
        action_valid = dof_mask.any(dim=-1)
        robot_ids = torch.full(
            (absolute_actions.shape[0],),
            robot_type_id,
            dtype=torch.long,
            device=absolute_actions.device,
        )
        indices = self.codec.encode(
            absolute_actions,
            anchor=absolute_anchor,
            robot_type=robot_ids,
            action_mask=action_valid,
            action_dof_mask=dof_mask,
        )
        return {"indices": indices}


def load_xtokenizer_cache_adapter(
    path: Path,
    *,
    device: str | torch.device,
    action_horizon: int,
    robot_type: str,
) -> XTokenizerCacheAdapter:
    return XTokenizerCacheAdapter(
        path,
        device=device,
        action_horizon=action_horizon,
        robot_type=robot_type,
    )
