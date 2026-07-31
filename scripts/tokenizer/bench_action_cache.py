# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Equivalence check + microbenchmark for the action-column in-memory cache.

Compares LeRobotActionOnlyDatasetAdapter.__getitem__ (cached tensor path)
against the original per-row Arrow path (_getitem_arrow) on the locally
cached bridge_single_view dataset, then benchmarks both.
"""

from __future__ import annotations

import random
import time

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from praxis.core.types import PolicyIO
from praxis.data.lerobot_adapter import LeRobotActionOnlyDatasetAdapter
from praxis.data.utils import resolve_action_feature_from_metadata

REPO_ID = "chaoqi-liu/bridge_single_view"
ACTION_HORIZON = 32


def build_adapter(episodes: list[int] | None) -> LeRobotActionOnlyDatasetAdapter:
    meta_ds = LeRobotDataset(repo_id=REPO_ID, episodes=episodes)
    action_feature = resolve_action_feature_from_metadata(
        meta_ds.meta.features, context="bench"
    )
    io = PolicyIO(
        obs_features=[],
        action_feature=action_feature,
        n_obs_steps=0,
        n_action_steps=ACTION_HORIZON,
        fps=meta_ds.fps,
    )
    del meta_ds
    return LeRobotActionOnlyDatasetAdapter(repo_id=REPO_ID, io=io, episodes=episodes)


def check_equivalence(
    adapter: LeRobotActionOnlyDatasetAdapter, indices: list[int], label: str
) -> None:
    assert adapter._action_cache is not None, "action cache failed to materialize"
    for idx in indices:
        new = adapter[idx]
        old = adapter._getitem_arrow(idx)
        assert set(new.keys()) == set(old.keys()), (
            f"[{label}] key mismatch at {idx}: {sorted(new)} vs {sorted(old)}"
        )
        for key in old:
            if not torch.equal(new[key], old[key]):
                raise AssertionError(
                    f"[{label}] value mismatch at idx={idx} key={key}:\n"
                    f"new={new[key]}\nold={old[key]}"
                )
    print(f"[{label}] equivalence OK on {len(indices)} indices")


def boundary_indices(adapter: LeRobotActionOnlyDatasetAdapter, n_eps: int) -> list[int]:
    """First and last local rows of the first n_eps episodes in local order."""
    cache = adapter._action_cache
    assert cache is not None
    ep_ids = cache.episode_index
    out: list[int] = []
    seen: set[int] = set()
    for row in range(len(adapter)):
        ep = int(ep_ids[row])
        if ep not in seen:
            seen.add(ep)
            out.append(row)  # first frame of episode
            if row > 0:
                out.append(row - 1)  # last frame of previous episode
            if len(seen) >= n_eps:
                break
    out.append(len(adapter) - 1)
    return out


def bench(adapter: LeRobotActionOnlyDatasetAdapter, indices: list[int]) -> None:
    t0 = time.perf_counter()
    for idx in indices:
        adapter[idx]
    dt_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    for idx in indices:
        adapter._getitem_arrow(idx)
    dt_old = time.perf_counter() - t0

    n = len(indices)
    print(
        f"cached: {n / dt_new:,.0f} samples/s ({1e3 * dt_new / n:.3f} ms/sample) | "
        f"arrow: {n / dt_old:,.0f} samples/s ({1e3 * dt_old / n:.3f} ms/sample) | "
        f"speedup x{dt_old / dt_new:,.1f}"
    )


def main() -> None:
    random.seed(0)

    print("== subset (episodes=0..119, exercises abs->rel remapping) ==")
    t0 = time.perf_counter()
    sub = build_adapter(episodes=list(range(120)))
    print(f"adapter built in {time.perf_counter() - t0:.1f}s, len={len(sub)}")
    idxs = boundary_indices(sub, 20) + random.sample(range(len(sub)), 300)
    check_equivalence(sub, idxs, "subset")

    print("== full dataset (episodes=None) ==")
    t0 = time.perf_counter()
    full = build_adapter(episodes=None)
    print(f"adapter built in {time.perf_counter() - t0:.1f}s, len={len(full)}")
    idxs = boundary_indices(full, 20) + random.sample(range(len(full)), 300)
    check_equivalence(full, idxs, "full")

    bench_idxs = random.sample(range(len(full)), 3000)
    bench(full, bench_idxs)


if __name__ == "__main__":
    main()
