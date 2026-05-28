"""Tests for episode-aware sampling helpers."""

from __future__ import annotations

from praxis.data.lerobot_adapter import _episode_bounds_from_indices


class TestEpisodeBoundsFromIndices:
    def test_empty(self):
        starts, ends = _episode_bounds_from_indices([])
        assert starts == []
        assert ends == []

    def test_single_episode(self):
        starts, ends = _episode_bounds_from_indices([0, 0, 0, 0])
        assert starts == [0]
        assert ends == [4]

    def test_multiple_episodes(self):
        starts, ends = _episode_bounds_from_indices([0, 0, 1, 1, 1, 2, 2])
        assert starts == [0, 2, 5]
        assert ends == [2, 5, 7]
