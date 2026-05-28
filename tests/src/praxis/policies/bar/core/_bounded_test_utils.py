from __future__ import annotations

from collections.abc import Callable
from typing import Any


def make_basic_bounded_policy(
    monkeypatch,
    policy_cls,
    *,
    make_io_fn: Callable[[], Any],
    fixed_tokenizer_cls,
    processor_cls,
    vlm_cls,
    patch_family_fn: Callable[..., None],
    extra_kwargs: dict[str, Any] | None = None,
    base_kwargs: dict[str, Any] | None = None,
):
    io = make_io_fn()
    tok = fixed_tokenizer_cls(fixed_len=4)
    processor = processor_cls()
    vlm = vlm_cls()
    patch_family_fn(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    kwargs = {"freeze_action_tokenizer": True}
    if base_kwargs is not None:
        kwargs.update(base_kwargs)
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    return policy_cls.from_io(io, **kwargs)


def patch_bounded_vlm_inputs(monkeypatch, *, target: str, build_vlm_inputs_fn) -> None:
    monkeypatch.setattr(target, build_vlm_inputs_fn)
