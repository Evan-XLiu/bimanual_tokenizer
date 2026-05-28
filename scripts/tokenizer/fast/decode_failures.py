#!/usr/bin/env python3
"""Evaluate FAST policy strict-decode failure ratio.

Loads a trained FAST policy, runs autoregressive inference on a dataset,
and reports how often the generated token sequences fail strict BPE decoding
(i.e. BPE-decoded length != action_horizon * action_dim).

Usage:
    python scripts/tokenizer/fast/decode_failures.py \
        --policy bar_fast_qwen \
        --checkpoint /path/to/policy/checkpoint \
        --dataset-repo-id lerobot/libero_10 \
        [--dataset-root /local/path] \
        [--dataset-episodes 0 1 2] \
        [--batch-size 32] \
        [--num-workers 4] \
        [--max-batches 100] \
        [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from functools import wraps

import torch

from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer
from praxis.data.lerobot_adapter import LeRobotDatasetAdapter
from praxis.policies.factory import _load_pretrained_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAST policy strict-decode failure ratio."
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy registry name (bar_fast_qwen, bar_fast_smol, bar_fast_pali).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Policy checkpoint: local path or HF Hub repo id.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repo_id (e.g. lerobot/libero_10).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Local dataset root path. None = default HF cache.",
    )
    parser.add_argument(
        "--dataset-episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episode subset to evaluate on. None = all episodes.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap number of batches (useful for slow VLM inference). None = full dataset.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def count_strict_failures(tokenizer: FASTTokenizer, tokens: torch.Tensor) -> int:
    """Count per-sample strict decode failures in a batch of tokens."""
    bpe = tokenizer._require_bpe()
    not_pad = tokens != tokenizer._pad_token_id
    valid_lengths = not_pad.sum(dim=1)
    masked = tokens * not_pad
    masked_list = masked.tolist()
    valid_id_lists = [
        row[: int(vl)]
        for row, vl in zip(masked_list, valid_lengths.tolist(), strict=False)
    ]
    decoded_strs = bpe.batch_decode(valid_id_lists)
    expected_len = tokenizer.action_horizon * tokenizer.action_dim
    return sum(1 for s in decoded_strs if len(s) != expected_len)


def install_detokenize_hook(tokenizer: FASTTokenizer) -> dict[str, int]:
    """Monkey-patch tokenizer.detokenize to count strict failures transparently.

    Returns a stats dict {"failures": int, "total": int} that is updated
    on every detokenize call.
    """
    stats: dict[str, int] = {"failures": 0, "total": 0}
    original_detokenize = tokenizer.detokenize

    @wraps(original_detokenize)
    def hooked_detokenize(tokens, *, strict_decode=None, **kwargs):
        del strict_decode
        stats["failures"] += count_strict_failures(tokenizer, tokens)
        stats["total"] += tokens.shape[0]
        # Force lenient decode so policy evaluation can continue after counting
        # the strict failures this diagnostic is measuring.
        return original_detokenize(tokens, strict_decode=False, **kwargs)

    tokenizer.detokenize = hooked_detokenize  # type: ignore[method-assign]
    return stats


def main() -> None:
    args = parse_args()

    # --- Load policy ---
    print(f"Loading {args.policy} policy from {args.checkpoint} ...")
    policy = _load_pretrained_policy(args.policy, args.checkpoint)
    policy.to(args.device)
    policy.eval()

    # --- Validate it's a FAST policy ---
    tokenizer = getattr(policy, "action_tokenizer", None)
    if not isinstance(tokenizer, FASTTokenizer):
        print(
            f"Error: policy's action_tokenizer is {type(tokenizer).__name__}, expected FASTTokenizer.",
            file=sys.stderr,
        )
        sys.exit(1)

    io = policy.io
    if io is None:
        print("Error: policy checkpoint has no PolicyIO sidecar.", file=sys.stderr)
        sys.exit(1)

    print(
        f"  action_horizon={tokenizer.action_horizon}, action_dim={tokenizer.action_dim}"
    )
    print(
        f"  expected BPE decoded length = {tokenizer.action_horizon * tokenizer.action_dim}"
    )

    # --- Install detokenize hook ---
    stats = install_detokenize_hook(tokenizer)

    # --- Build dataset + dataloader ---
    print(f"Loading dataset {args.dataset_repo_id} ...")
    dataset = LeRobotDatasetAdapter(
        repo_id=args.dataset_repo_id,
        io=io,
        root=args.dataset_root,
        episodes=args.dataset_episodes,
    )
    dataloader = dataset.make_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_episode_aware_sampler=False,
    )
    num_samples = len(dataset)
    print(f"  {num_samples} samples")

    # --- Run inference ---
    total_batches = len(dataloader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    print(f"\nRunning inference ({total_batches} batches) ...")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            # Move observation tensors to device; policy.predict_action_chunk
            # expects a dict of tensors.
            obs = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            policy.predict_action_chunk(obs)

            if (batch_idx + 1) % 10 == 0:
                ratio = (
                    stats["failures"] / stats["total"] if stats["total"] > 0 else 0.0
                )
                print(
                    f"  batch {batch_idx + 1}/{total_batches} | "
                    f"failures={stats['failures']}/{stats['total']} ({ratio:.4%})"
                )

    # --- Report ---
    total = stats["total"]
    failures = stats["failures"]
    ratio = failures / total if total > 0 else float("nan")
    print("\n" + "=" * 40)
    print(f"  Total samples:  {total}")
    print(f"  Failures:       {failures}")
    print(f"  Failure ratio:  {ratio:.4%}")
    print("=" * 40)


if __name__ == "__main__":
    main()
