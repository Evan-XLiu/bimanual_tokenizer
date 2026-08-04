# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def test_seed_evaluation_reproduces_policy_sampling_rng() -> None:
    from praxis.core.randomness import seed_evaluation

    seed_evaluation(42)
    first = (random.random(), np.random.random(), torch.randn(4))

    seed_evaluation(42)
    second = (random.random(), np.random.random(), torch.randn(4))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


def test_eval_seeds_policy_rng_before_loading_checkpoint() -> None:
    eval_path = Path(__file__).resolve().parents[4] / "src/praxis/scripts/eval.py"
    source = eval_path.read_text()

    assert source.index("seed_evaluation(int(cfg.seed))") < source.index(
        "policy = make_policy(cfg.policy)"
    )


def test_eval_forwards_optional_inference_trace_path() -> None:
    eval_path = Path(__file__).resolve().parents[4] / "src/praxis/scripts/eval.py"
    source = eval_path.read_text()

    assert 'os.environ.get("PRAXIS_INFERENCE_TRACE_PATH")' in source


def test_eval_forwards_sampling_seed_to_policy_adapter() -> None:
    eval_path = Path(__file__).resolve().parents[4] / "src/praxis/scripts/eval.py"
    source = eval_path.read_text()

    assert "sampling_seed=int(cfg.seed)" in source
