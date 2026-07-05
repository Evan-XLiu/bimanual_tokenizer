# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from praxis.policies.bar.variants import (
    BinQwenPolicy,
    BOATQwenPolicy,
    OATQwenPolicy,
    QueSTQwenPolicy,
)

from ...core._bounded_test_utils import make_basic_bounded_policy
from ._test_utils import (
    DummyProcessor,
    DummyVLM,
    make_io,
    mock_build_vlm_inputs,
    mock_build_vlm_inputs_all_ignore,
    stats_for_io,
)

__all__ = [
    "DummyFixedTokenizer",
    "apply_monkeypatches",
    "make_bounded_policy",
    "make_io",
    "mock_build_vlm_inputs_all_ignore",
    "stats_for_io",
]


class DummyFixedTokenizer(nn.Module):
    def __init__(self, fixed_len: int = 4, codebook: int = 16) -> None:
        super().__init__()
        self._fixed_code_seq_len = fixed_len
        self._codebook_size = codebook
        self.stats_set_calls = 0
        self.normalize_action = None
        self.unnormalize_action = None
        self.io = None

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def max_code_seq_len(self) -> int | None:
        return self._fixed_code_seq_len

    @property
    def supports_prefix_decode(self) -> bool:
        return True

    def tokenize(self, actions, **kwargs):
        del kwargs
        batch_size = int(actions.shape[0])
        return torch.randint(
            0, self._codebook_size, (batch_size, self._fixed_code_seq_len)
        )

    def detokenize(self, tokens, **kwargs):
        del kwargs
        batch_size = tokens.shape[0]
        return torch.zeros(
            (batch_size, 2, 2), dtype=torch.float32, device=tokens.device
        )

    def set_io(self, io):
        self.io = io

    def set_normalization_stats(self, stats, modes=None):
        del stats, modes
        self.stats_set_calls += 1
        self.normalize_action = object()
        self.unnormalize_action = object()


PolicyClass = (
    type[BinQwenPolicy]
    | type[BOATQwenPolicy]
    | type[QueSTQwenPolicy]
    | type[OATQwenPolicy]
)


def apply_monkeypatches(
    monkeypatch,
    *,
    tokenizer: DummyFixedTokenizer | None = None,
    processor: DummyProcessor | None = None,
    vlm: DummyVLM | None = None,
    use_mock_inputs: bool = True,
) -> tuple[DummyVLM, DummyProcessor, DummyFixedTokenizer]:
    dummy_vlm = DummyVLM(vocab_size=200, hidden_dim=16) if vlm is None else vlm
    dummy_processor = DummyProcessor() if processor is None else processor
    dummy_tokenizer = (
        DummyFixedTokenizer(fixed_len=4, codebook=16)
        if tokenizer is None
        else tokenizer
    )

    def mock_load_vlm(self, model_name, dtype_str):
        del model_name, dtype_str
        self.vlm = dummy_vlm
        self.processor = dummy_processor

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.qwen.qwen_base.QwenPolicyBase._load_vlm",
        mock_load_vlm,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.base.make_codec",
        lambda _cfg: dummy_tokenizer,
    )
    if use_mock_inputs:
        monkeypatch.setattr(
            "praxis.policies.bar.backbones.qwen.qwen_base.QwenPolicyBase._build_vlm_inputs",
            mock_build_vlm_inputs,
        )
    return dummy_vlm, dummy_processor, dummy_tokenizer


def make_bounded_policy(monkeypatch, policy_cls: PolicyClass, **extra_kwargs):
    return make_basic_bounded_policy(
        monkeypatch,
        policy_cls,
        make_io_fn=make_io,
        fixed_tokenizer_cls=DummyFixedTokenizer,
        processor_cls=DummyProcessor,
        vlm_cls=DummyVLM,
        patch_family_fn=apply_monkeypatches,
        base_kwargs={
            "n_obs_steps": 1,
            "action_horizon": 2,
            "n_action_steps": 1,
        },
        extra_kwargs=extra_kwargs,
    )
