from __future__ import annotations

import pytest
import torch

from praxis.codecs.tokens.offline_targets.tokenizer import OfflineTargetTokenizer


def test_offline_target_codec_is_metadata_only() -> None:
    codec = OfflineTargetTokenizer(
        config={"codebook_size": 2048, "block_schedule": [4] * 8}
    )

    assert codec.codebook_size == 2048
    assert codec.block_schedule == (4,) * 8
    assert codec.max_code_seq_len == 32
    assert not list(codec.parameters())

    with pytest.raises(RuntimeError, match="provide cached action_token_ids"):
        codec.tokenize(torch.zeros(1, 32, 7))


def test_offline_target_codec_cannot_become_an_inference_dependency() -> None:
    codec = OfflineTargetTokenizer()

    with pytest.raises(RuntimeError, match="KI inference uses its flow branch"):
        codec.detokenize(torch.zeros(1, 32, dtype=torch.long))
