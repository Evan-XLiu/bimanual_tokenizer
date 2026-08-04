# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Random-state helpers shared by evaluation entry points."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_evaluation(seed: int) -> None:
    """Seed host and torch RNGs used by stochastic policy inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
