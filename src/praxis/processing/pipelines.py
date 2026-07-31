# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LeRobot-aligned processor pipeline wrappers for Praxis."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from lerobot.envs.factory import (
    make_env_pre_post_processors as lerobot_make_env_pre_post_processors,
)
from lerobot.processor import (
    IdentityProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    TransitionKey,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

IDENTITY_PROCESSOR_FACTORY = "identity"
EnvProcessor = PolicyProcessorPipeline[Any, Any]
EnvProcessorPair = tuple[EnvProcessor, EnvProcessor]


def _batch_to_transition_preserving_extra_fields(batch: dict[str, Any]):
    """Keep dataset-owned supervision fields through LeRobot conversion."""
    transition = batch_to_transition(batch)
    converted_keys = transition_to_batch(transition).keys()
    extra_fields = {
        key: value for key, value in batch.items() if key not in converted_keys
    }
    if extra_fields:
        complementary_data = dict(
            transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        )
        complementary_data.update(extra_fields)
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
    return transition


def make_policy_pre_post_processors() -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create default identity policy pre/post processors in LeRobot format."""
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[IdentityProcessorStep()],
        to_transition=_batch_to_transition_preserving_extra_fields,
        to_output=transition_to_batch,
        name="PolicyPreprocessor",
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[IdentityProcessorStep()],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        name="PolicyPostprocessor",
    )
    return preprocessor, postprocessor


def make_identity_env_pre_post_processors(
    env_cfg: Any | None = None,
    policy_cfg: Any | None = None,
) -> EnvProcessorPair:
    """Create identity env processors."""
    _ = env_cfg, policy_cfg
    return (
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
        PolicyProcessorPipeline(steps=[IdentityProcessorStep()]),
    )


def make_env_pre_post_processors(env_cfg: Any, policy_cfg: Any) -> EnvProcessorPair:
    """Create environment processors from env-owned processor declarations."""
    processor_factory = getattr(env_cfg, "processor_factory", None)
    if processor_factory is not None:
        return _call_processor_factory(
            processor_factory,
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )

    return cast(
        EnvProcessorPair,
        lerobot_make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg),
    )


def _call_processor_factory(
    processor_factory: Any,
    *,
    env_cfg: Any,
    policy_cfg: Any,
) -> EnvProcessorPair:
    if processor_factory == IDENTITY_PROCESSOR_FACTORY:
        return make_identity_env_pre_post_processors(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )
    if callable(processor_factory):
        return cast(
            EnvProcessorPair,
            processor_factory(env_cfg=env_cfg, policy_cfg=policy_cfg),
        )
    if isinstance(processor_factory, str):
        module_name, separator, attr_name = processor_factory.partition(":")
        if separator != ":" or not module_name or not attr_name:
            raise ValueError(
                "processor_factory strings must use 'module:function' format; "
                f"got {processor_factory!r}."
            )
        factory = getattr(import_module(module_name), attr_name)
        return cast(EnvProcessorPair, factory(env_cfg=env_cfg, policy_cfg=policy_cfg))
    raise TypeError(
        "processor_factory must be 'identity', a callable, or a 'module:function' "
        f"string; got {type(processor_factory).__name__}."
    )
