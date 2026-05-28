"""Tests for shared RGB encoder in praxis.nn.vision."""

from __future__ import annotations

import torch

from praxis.nn.vision.rgb_encoder import (
    ResNetSpatialSoftmaxEncoder,
    SpatialSoftmax,
    replace_bn_with_gn,
)


class TestSpatialSoftmax:
    def test_output_shape(self):
        ssm = SpatialSoftmax(input_shape=(512, 3, 3), num_kp=32)
        x = torch.randn(2, 512, 3, 3)
        out = ssm(x)
        assert out.shape == (2, 32, 2)

    def test_output_range(self):
        ssm = SpatialSoftmax(input_shape=(64, 7, 7), num_kp=16)
        x = torch.randn(4, 64, 7, 7)
        out = ssm(x)
        assert out.min() >= -1.1
        assert out.max() <= 1.1


class TestReplaceBnWithGn:
    def test_replaces_batchnorm(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        replace_bn_with_gn(model)
        assert isinstance(model[1], torch.nn.GroupNorm)
        assert model[1].num_channels == 16


class TestResNetSpatialSoftmaxEncoder:
    def test_output_shape(self):
        enc = ResNetSpatialSoftmaxEncoder(
            input_shape=(3, 96, 96),
            num_keypoints=32,
            resize_shape=(96, 96),
            crop_shape=(84, 84),
        )
        x = torch.randn(2, 3, 96, 96)
        out = enc(x)
        assert out.shape == (2, 64)

    def test_feature_dim_property(self):
        enc = ResNetSpatialSoftmaxEncoder(
            input_shape=(3, 96, 96),
            num_keypoints=16,
            resize_shape=(96, 96),
            crop_shape=(84, 84),
        )
        assert enc.feature_dim == 32

    def test_eval_uses_center_crop(self):
        enc = ResNetSpatialSoftmaxEncoder(
            input_shape=(3, 96, 96),
            crop_shape=(84, 84),
            crop_is_random=True,
        )
        enc.eval()
        x = torch.randn(1, 3, 96, 96)
        out1 = enc(x)
        out2 = enc(x)
        assert torch.allclose(out1, out2)

    def test_no_resize_no_crop(self):
        enc = ResNetSpatialSoftmaxEncoder(
            input_shape=(3, 64, 64),
            resize_shape=None,
            crop_shape=None,
        )
        x = torch.randn(1, 3, 64, 64)
        out = enc(x)
        assert out.shape == (1, 64)

    def test_shape_probe_preserves_batchnorm_stats_without_group_norm(self):
        enc = ResNetSpatialSoftmaxEncoder(
            input_shape=(3, 64, 64),
            use_group_norm=False,
            resize_shape=None,
            crop_shape=None,
        )

        batch_norms = [
            module
            for module in enc.backbone.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        ]
        assert batch_norms
        assert enc.training

        for batch_norm in batch_norms:
            assert batch_norm.training
            assert torch.equal(
                batch_norm.running_mean,
                torch.zeros_like(batch_norm.running_mean),
            )
            assert torch.equal(
                batch_norm.running_var,
                torch.ones_like(batch_norm.running_var),
            )
            assert batch_norm.num_batches_tracked.item() == 0
