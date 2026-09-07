#!/usr/bin/env python3
"""Inspect FAST reconstruction error on contiguous LeRobot action chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import PreTrainedTokenizerFast

from praxis.codecs.factory import make_codec
from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--bpe-tokenizer-path", default=None)
    parser.add_argument("--max-chunks", type=int, default=4096)
    args = parser.parse_args()

    codec = make_codec(
        {"kind": "tokens", "name": "fast", "pretrained_path": args.checkpoint}
    )
    if not isinstance(codec, FASTTokenizer):
        raise TypeError(type(codec))
    if args.bpe_tokenizer_path:
        codec._set_bpe(PreTrainedTokenizerFast.from_pretrained(args.bpe_tokenizer_path))
        codec._maybe_load_processor_metadata(args.bpe_tokenizer_path)
    horizon = codec.action_horizon
    chunks: list[torch.Tensor] = []
    for path in sorted(Path(args.dataset_root).glob("data/chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["action", "episode_index"])
        actions = torch.tensor(table["action"].to_pylist(), dtype=torch.float32)
        episodes = torch.tensor(table["episode_index"].to_pylist())
        for start in range(len(actions) - horizon + 1):
            if episodes[start] == episodes[start + horizon - 1]:
                chunks.append(actions[start : start + horizon])
                if len(chunks) >= args.max_chunks:
                    break
        if len(chunks) >= args.max_chunks:
            break

    truth = torch.stack(chunks)
    tokens, lengths = codec.tokenize(truth, return_lengths=True)
    recon = codec.detokenize(tokens)
    error = recon - truth
    names = [*(f"left_{i}" for i in range(7)), *(f"right_{i}" for i in range(7))]
    print(f"chunks={len(truth)} token_length_mean={lengths.float().mean():.2f}")
    print("dimension mae rmse max_abs truth_step_mae recon_step_mae")
    for dim, name in enumerate(names):
        dim_error = error[..., dim]
        truth_step = truth[:, 1:, dim] - truth[:, :-1, dim]
        recon_step = recon[:, 1:, dim] - recon[:, :-1, dim]
        print(
            f"{name:>8} {dim_error.abs().mean():.6f} "
            f"{dim_error.square().mean().sqrt():.6f} {dim_error.abs().max():.6f} "
            f"{truth_step.abs().mean():.6f} {recon_step.abs().mean():.6f}"
        )
    worst = error.abs().flatten(1).mean(1).topk(min(3, len(error))).indices
    for index in worst.tolist():
        print(f"worst_chunk={index} mean_abs={error[index].abs().mean():.6f}")
        for step in range(horizon):
            print(
                f"  t={step:02d} left_truth={truth[index, step, :7].tolist()} "
                f"left_recon={recon[index, step, :7].tolist()}"
            )


if __name__ == "__main__":
    main()
