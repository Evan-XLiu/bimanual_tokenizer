# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Trainer-owned runtime context shared with opt-in trainable modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingContext:
    """Small immutable context for model behavior that depends on trainer state."""

    global_step: int
