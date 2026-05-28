from __future__ import annotations

import pytest

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
