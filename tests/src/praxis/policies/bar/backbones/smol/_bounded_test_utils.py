from __future__ import annotations

from praxis.policies.bar.variants import (
    BinSmolPolicy,
    BOATSmolPolicy,
    OATSmolPolicy,
    QueSTSmolPolicy,
)

from ...core._bounded_test_utils import (
    make_basic_bounded_policy,
    patch_bounded_vlm_inputs,
)
from ._test_utils import (
    _DummyFixedTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _mock_build_vlm_inputs,
    _patch_smol,
    _stats_for_io,
)

__all__ = [
    "_DummyFixedTokenizer",
    "_DummyProcessor",
    "_DummyVLM",
    "_make_bounded_policy",
    "_make_io",
    "_patch_bounded_vlm_inputs",
    "_patch_smol",
    "_stats_for_io",
]

_PolicyClass = (
    type[BinSmolPolicy]
    | type[BOATSmolPolicy]
    | type[OATSmolPolicy]
    | type[QueSTSmolPolicy]
)


def _patch_bounded_vlm_inputs(monkeypatch) -> None:
    patch_bounded_vlm_inputs(
        monkeypatch,
        target="praxis.policies.bar.backbones.smol.smol_base.SmolPolicyBase._build_vlm_inputs",
        build_vlm_inputs_fn=_mock_build_vlm_inputs,
    )


def _make_bounded_policy(monkeypatch, policy_cls: _PolicyClass, **extra_kwargs):
    return make_basic_bounded_policy(
        monkeypatch,
        policy_cls,
        make_io_fn=_make_io,
        fixed_tokenizer_cls=_DummyFixedTokenizer,
        processor_cls=_DummyProcessor,
        vlm_cls=_DummyVLM,
        patch_family_fn=_patch_smol,
        extra_kwargs=extra_kwargs,
    )
