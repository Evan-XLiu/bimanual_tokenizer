# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Supported policy-family matrix.

The catalog is the source of truth for generated policy names. Runtime class
resolution lives in ``praxis.policies.factory``; Hydra aliases remain under
``src/praxis/configs/policy``.
"""

from __future__ import annotations

from dataclasses import dataclass

POLICY_CODECS: tuple[str, ...] = ("oat", "acodec", "boat", "fast", "bin", "quest")
POLICY_BACKBONES: tuple[str, ...] = ("pali", "qwen", "smol")

DEFAULT_POLICY_METHOD = "bar"
DEFAULT_POLICY_CODEC = "oat"
DEFAULT_POLICY_BACKBONE = "pali"


@dataclass(frozen=True)
class PolicyFamily:
    """Supported codec/backbone matrix for one policy family."""

    method: str
    codecs: tuple[str, ...] | None
    backbones: tuple[str, ...] = POLICY_BACKBONES

    @property
    def uses_codec(self) -> bool:
        return self.codecs is not None


@dataclass(frozen=True)
class PolicySpec:
    """One concrete policy variant in the supported matrix."""

    method: str
    codec: str | None
    backbone: str

    @property
    def name(self) -> str:
        if self.codec is None:
            return f"{self.method}_{self.backbone}"
        return f"{self.method}_{self.codec}_{self.backbone}"

    @property
    def module_path(self) -> str:
        return f"praxis.policies.{self.method}.variants"

    @property
    def class_name(self) -> str:
        backbone_name = _BACKBONE_CLASS_NAMES[self.backbone]
        if self.codec is None:
            return f"{backbone_name}Policy"
        codec_name = _CODEC_CLASS_PREFIXES[self.codec]
        return f"{codec_name}{backbone_name}Policy"

    @property
    def import_path(self) -> str:
        return f"{self.module_path}:{self.class_name}"


_CODEC_CLASS_PREFIXES: dict[str, str] = {
    "acodec": "ACodec",
    "bin": "Bin",
    "boat": "BOAT",
    "fast": "FAST",
    "oat": "OAT",
    "quest": "QueST",
}
_BACKBONE_CLASS_NAMES: dict[str, str] = {
    "pali": "Pali",
    "qwen": "Qwen",
    "smol": "Smol",
}


POLICY_FAMILIES: tuple[PolicyFamily, ...] = (
    PolicyFamily(method="bar", codecs=POLICY_CODECS),
    PolicyFamily(method="ki", codecs=("oat", "acodec", "fast", "bin", "quest")),
    PolicyFamily(method="fm", codecs=None),
)
POLICY_FAMILIES_BY_METHOD: dict[str, PolicyFamily] = {
    family.method: family for family in POLICY_FAMILIES
}
POLICY_METHODS: tuple[str, ...] = tuple(POLICY_FAMILIES_BY_METHOD)
POLICY_CODECS_BY_METHOD: dict[str, tuple[str, ...]] = {
    method: family.codecs or () for method, family in POLICY_FAMILIES_BY_METHOD.items()
}


def policy_family(method: str) -> PolicyFamily:
    """Return the policy-family spec for ``method``."""
    key = method.strip().lower()
    try:
        return POLICY_FAMILIES_BY_METHOD[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported policy method {method!r}. "
            f"Expected one of: {', '.join(POLICY_METHODS)}."
        ) from exc


def policy_name(method: str, codec: str | None, backbone: str) -> str:
    """Return the canonical concrete policy name after matrix validation."""
    return policy_spec(method, codec, backbone).name


def policy_spec(method: str, codec: str | None, backbone: str) -> PolicySpec:
    """Return a concrete policy spec after matrix validation."""
    family = policy_family(method)
    resolved_codec: str | None
    if family.uses_codec:
        if codec is None:
            raise ValueError(
                f"Policy method {family.method!r} requires a codec. "
                f"Expected one of: {', '.join(family.codecs or ())}."
            )
        resolved_codec = codec.strip().lower()
    elif codec is not None:
        raise ValueError(f"Policy method {family.method!r} does not use a codec.")
    else:
        resolved_codec = None
    if family.codecs is not None and resolved_codec not in family.codecs:
        raise ValueError(
            f"Unsupported policy codec {codec!r} for method {family.method!r}. "
            f"Expected one of: {', '.join(family.codecs)}."
        )
    resolved_backbone = backbone.strip().lower()
    if resolved_backbone not in family.backbones:
        raise ValueError(
            f"Unsupported policy backbone {backbone!r} for method {family.method!r}. "
            f"Expected one of: {', '.join(family.backbones)}."
        )
    return PolicySpec(
        method=family.method,
        codec=resolved_codec,
        backbone=resolved_backbone,
    )


def iter_policy_specs() -> tuple[PolicySpec, ...]:
    """Return all supported concrete policy specs."""
    specs: list[PolicySpec] = []
    for family in POLICY_FAMILIES:
        for backbone in family.backbones:
            if family.codecs is None:
                specs.append(
                    PolicySpec(method=family.method, codec=None, backbone=backbone)
                )
                continue
            specs.extend(
                PolicySpec(method=family.method, codec=codec, backbone=backbone)
                for codec in family.codecs
            )
    return tuple(specs)


def supported_policy_names() -> tuple[str, ...]:
    """Return all supported concrete policy names."""
    return tuple(spec.name for spec in iter_policy_specs())


BAR_POLICY_CODECS = POLICY_FAMILIES_BY_METHOD["bar"].codecs or ()
BAR_POLICY_BACKBONES = POLICY_FAMILIES_BY_METHOD["bar"].backbones
KI_POLICY_CODECS = POLICY_FAMILIES_BY_METHOD["ki"].codecs or ()
KI_POLICY_BACKBONES = POLICY_FAMILIES_BY_METHOD["ki"].backbones
FM_POLICY_BACKBONES = POLICY_FAMILIES_BY_METHOD["fm"].backbones
