# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from praxis.codecs.latents.quest.modeling import ResidualTemporalBlock


class TestResidualTemporalBlock:
    def test_residual_stride_uses_product_of_block_strides(self):
        block = ResidualTemporalBlock(
            inp_channels=4,
            out_channels=8,
            kernel_size=[3, 3],
            stride=[2, 1],
            causal=False,
            residual=True,
        )
        x = torch.randn(2, 8, 4)

        out = block(x)

        assert block.residual_conv.stride == (2,)
        assert out.shape == (2, 4, 8)
