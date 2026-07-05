# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from praxis.nn.quantization.fsq import FiniteScalarQuantization


class TestFiniteScalarQuantization:
    def test_explicit_zero_min_corrupt_tokens_is_preserved(self):
        quantizer = FiniteScalarQuantization(
            levels=[8, 8], corrupt_tokens_p=0.5, min_corrupt_tokens_p=0.0
        )

        assert quantizer.min_corrupt_tokens_p == 0.0

    @pytest.mark.parametrize("levels", [[], [1, 8]])
    def test_invalid_levels_raise(self, levels):
        with pytest.raises(ValueError, match="levels"):
            FiniteScalarQuantization(levels=levels)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"drop_quant_p": -0.1}, "drop_quant_p"),
            ({"drop_quant_p": 1.1}, "drop_quant_p"),
            ({"corrupt_tokens_p": -0.1}, "corrupt_tokens_p"),
            ({"corrupt_tokens_p": 1.1}, "corrupt_tokens_p"),
            ({"min_corrupt_tokens_p": -0.1}, "min_corrupt_tokens_p"),
            ({"min_corrupt_tokens_p": 1.1}, "min_corrupt_tokens_p"),
            ({"apply_corrupt_tokens_p": -0.1}, "apply_corrupt_tokens_p"),
            ({"apply_corrupt_tokens_p": 1.1}, "apply_corrupt_tokens_p"),
            (
                {"corrupt_tokens_p": 0.1, "min_corrupt_tokens_p": 0.2},
                "min_corrupt_tokens_p must be <= corrupt_tokens_p",
            ),
        ],
    )
    def test_invalid_probabilities_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FiniteScalarQuantization(levels=[8, 8], **kwargs)

    def test_indices_and_embeddings_roundtrip(self):
        quantizer = FiniteScalarQuantization(levels=[3, 4])
        indices = torch.arange(quantizer.codebook_size)

        codes = quantizer.indices_to_embedding(indices)
        recovered = quantizer.codes_to_indices(codes)

        assert codes.shape == (12, 2)
        assert torch.equal(recovered, indices.to(torch.int32))

    def test_bfloat16_codes_roundtrip_default_mixed_radix_levels(self):
        quantizer = FiniteScalarQuantization(levels=[8, 8, 6, 5])
        indices = torch.arange(quantizer.codebook_size)

        codes = quantizer.indices_to_embedding(indices).to(torch.bfloat16)
        recovered = quantizer.codes_to_indices(codes)

        assert torch.equal(recovered, indices.to(torch.int32))

    def test_forward_returns_quantized_latents_and_long_tokens(self):
        quantizer = FiniteScalarQuantization(levels=[8, 8])
        latents = torch.randn(2, 4, 2, dtype=torch.bfloat16)

        quantized, tokens = quantizer(latents)

        assert quantized.shape == latents.shape
        assert quantized.dtype == torch.float32
        assert tokens.shape == (2, 4)
        assert tokens.dtype == torch.long
        assert torch.all(tokens >= 0)
        assert torch.all(tokens < quantizer.codebook_size)

    def test_forward_rejects_list_input(self):
        quantizer = FiniteScalarQuantization(levels=[8, 8])

        with pytest.raises(TypeError, match="expects a tensor"):
            quantizer([torch.randn(1, 2)])  # type: ignore[arg-type]

    def test_quantization_dropout_is_disabled_in_eval(self):
        quantizer = FiniteScalarQuantization(levels=[8, 8], drop_quant_p=1.0)
        latents = torch.randn(2, 4, 2)

        quantizer.train()
        train_quantized, _ = quantizer(latents)
        quantizer.eval()
        eval_quantized, _ = quantizer(latents)

        assert not torch.equal(train_quantized, eval_quantized)
        assert torch.all(eval_quantized <= 1.0)
        assert torch.all(eval_quantized >= -1.0)

    def test_corruption_can_replace_all_tokens_when_forced(self, monkeypatch):
        quantizer = FiniteScalarQuantization(
            levels=[8, 8],
            corrupt_tokens_p=1.0,
            min_corrupt_tokens_p=1.0,
            apply_corrupt_tokens_p=1.0,
        )
        quantizer.train()
        latents = torch.zeros(2, 4, 2)

        monkeypatch.setattr("praxis.nn.quantization.fsq.random.random", lambda: 0.0)
        monkeypatch.setattr("praxis.nn.quantization.fsq.random.uniform", lambda a, b: b)
        monkeypatch.setattr(
            "praxis.nn.quantization.fsq.torch.randint",
            lambda *args, **kwargs: torch.full(
                kwargs["size"], 1, device=kwargs["device"]
            ),
        )
        quantized, tokens = quantizer(latents)

        assert quantized.shape == latents.shape
        assert tokens.shape == (2, 4)
        assert torch.all(tokens == 1)
