"""Tests for scripts/tokenizer/boat/independence_diagnostic.py."""

from __future__ import annotations

import importlib.util
import sys

import torch

from tests._paths import repo_root


def _load_module():
    root = repo_root()
    script_path = root / "scripts" / "tokenizer" / "boat" / "independence_diagnostic.py"
    spec = importlib.util.spec_from_file_location(
        "boat_independence_diagnostic", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_target_probe_specs_matches_pow2_boat_schedule() -> None:
    mod = _load_module()

    specs = mod.build_target_probe_specs((1, 1, 2, 4, 8))

    assert len(specs) == 14

    first_multi = specs[0]
    assert first_multi.block_index == 2
    assert first_multi.target_position == 2
    assert first_multi.baseline_positions == (0, 1)
    assert first_multi.other_block_positions == (3,)
    assert first_multi.full_positions == (0, 1, 3)

    last = specs[-1]
    assert last.block_index == 4
    assert last.target_position == 15
    assert last.baseline_positions == tuple(range(8))
    assert last.other_block_positions == tuple(range(8, 15))


def test_split_train_val_is_deterministic_and_preserves_rows() -> None:
    mod = _load_module()
    tokens = torch.arange(60).view(12, 5)

    train_a, val_a = mod.split_train_val(tokens, val_fraction=0.25, seed=7)
    train_b, val_b = mod.split_train_val(tokens, val_fraction=0.25, seed=7)

    assert torch.equal(train_a, train_b)
    assert torch.equal(val_a, val_b)
    assert train_a.shape == (9, 5)
    assert val_a.shape == (3, 5)

    restored = torch.cat([train_a, val_a], dim=0)
    assert sorted(restored[:, 0].tolist()) == sorted(tokens[:, 0].tolist())


def test_independence_probe_supports_zero_context_inputs() -> None:
    mod = _load_module()
    probe = mod.IndependenceProbe(
        num_input_tokens=0,
        codebook_size=11,
        emb_dim=8,
        hidden_dim=16,
    )

    logits = probe(torch.empty((4, 0), dtype=torch.long))

    assert logits.shape == (4, 11)


def test_format_results_table_includes_expected_columns() -> None:
    mod = _load_module()
    result = mod.TargetProbeResult(
        block_index=3,
        block_size=4,
        target_position=5,
        earlier_positions=(0, 1, 2, 3),
        other_block_positions=(4, 6, 7),
        baseline_accuracy=0.25,
        full_accuracy=0.5,
        delta_accuracy=0.25,
        baseline_loss=1.2,
        full_loss=0.8,
        verdict="DEPENDENT",
    )

    table = mod.format_results_table([result])

    assert "Baseline" in table
    assert "Verdict" in table
    assert "DEPENDENT" in table


def test_build_probe_step_plan_reports_real_optimization_budget() -> None:
    mod = _load_module()

    plan = mod.build_probe_step_plan(
        train_examples=40_000,
        val_examples=10_000,
        probe_batch_size=512,
        probe_epochs=100,
        total_probes=28,
    )

    assert plan.steps_per_epoch == 79
    assert plan.steps_per_probe == 7900
    assert plan.total_optimizer_steps == 221_200
