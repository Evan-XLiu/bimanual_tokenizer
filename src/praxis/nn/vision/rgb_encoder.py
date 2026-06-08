# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared RGB encoder: ResNet backbone + SpatialSoftmax pooling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SpatialSoftmax(nn.Module):
    """Spatial softmax pooling that returns 2D keypoint coordinates."""

    def __init__(self, input_shape: tuple[int, int, int], num_kp: int = 32) -> None:
        super().__init__()
        in_c, in_h, in_w = input_shape
        self._in_h = in_h
        self._in_w = in_w
        self._out_c = num_kp

        self.nets = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x = (
            np.linspace(-1.0, 1.0, in_w).reshape(1, -1).repeat(in_h, axis=0).flatten()
        )
        pos_y = (
            np.linspace(-1.0, 1.0, in_h).reshape(-1, 1).repeat(in_w, axis=1).flatten()
        )
        pos_grid = np.stack([pos_x, pos_y], axis=1).astype(np.float32)
        self.register_buffer("pos_grid", torch.from_numpy(pos_grid))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.nets(features)  # (B, num_kp, H, W)
        features = features.reshape(-1, self._in_h * self._in_w)  # (B*num_kp, H*W)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ torch.as_tensor(
            self.pos_grid, device=attention.device
        )
        return expected_xy.view(-1, self._out_c, 2)


def replace_bn_with_gn(module: nn.Module) -> nn.Module:
    """Recursively replace all BatchNorm2d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = max(1, num_channels // 16)
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            )
        else:
            replace_bn_with_gn(child)
    return module


def _get_output_shape(
    backbone: nn.Module, input_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    dummy = torch.zeros(1, *input_shape)
    was_training = backbone.training
    try:
        backbone.eval()
        with torch.inference_mode():
            out = backbone(dummy)
    finally:
        backbone.train(was_training)
    return tuple(out.shape[1:])


class ResNetSpatialSoftmaxEncoder(nn.Module):
    """Encode images into a compact feature vector with SpatialSoftmax pooling."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        vision_backbone: str = "resnet18",
        pretrained_weights: str | None = None,
        use_group_norm: bool = True,
        num_keypoints: int = 32,
        resize_shape: tuple[int, int] | None = (96, 96),
        crop_shape: tuple[int, int] | None = (84, 84),
        crop_is_random: bool = True,
    ) -> None:
        super().__init__()

        self.resize = (
            torchvision.transforms.Resize(resize_shape, antialias=True)
            if resize_shape
            else None
        )
        self.do_crop = crop_shape is not None
        if self.do_crop:
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.random_crop = (
                torchvision.transforms.RandomCrop(crop_shape)
                if crop_is_random
                else self.center_crop
            )

        weights = pretrained_weights if pretrained_weights else None
        backbone_model = getattr(torchvision.models, vision_backbone)(weights=weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if use_group_norm:
            replace_bn_with_gn(self.backbone)

        dummy_c = input_shape[0]
        if resize_shape and crop_shape:
            dummy_shape = (dummy_c, crop_shape[0], crop_shape[1])
        elif resize_shape:
            dummy_shape = (dummy_c, resize_shape[0], resize_shape[1])
        elif crop_shape:
            dummy_shape = (dummy_c, crop_shape[0], crop_shape[1])
        else:
            dummy_shape = input_shape

        feat_shape = _get_output_shape(self.backbone, dummy_shape)

        self.pool = SpatialSoftmax(feat_shape, num_kp=num_keypoints)
        self.feature_dim = num_keypoints * 2

        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            x = self.random_crop(x) if self.training else self.center_crop(x)

        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.out(x))
        return x
