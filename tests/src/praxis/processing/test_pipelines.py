# SPDX-License-Identifier: Apache-2.0

"""Tests for policy processor pipeline field preservation."""

from __future__ import annotations

import torch

from praxis.processing.pipelines import make_policy_pre_post_processors


def test_policy_preprocessor_preserves_offline_action_token_targets() -> None:
    preprocessor, _ = make_policy_pre_post_processors()
    batch = {
        "observation.state": torch.zeros((2, 1, 7), dtype=torch.float32),
        "action": torch.zeros((2, 32, 7), dtype=torch.float32),
        "action_token_ids": torch.tensor(
            [[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long
        ),
        "action_token_length": torch.tensor([4, 4], dtype=torch.long),
    }

    processed = preprocessor(batch)

    torch.testing.assert_close(
        processed["action_token_ids"], batch["action_token_ids"]
    )
    torch.testing.assert_close(
        processed["action_token_length"], batch["action_token_length"]
    )
