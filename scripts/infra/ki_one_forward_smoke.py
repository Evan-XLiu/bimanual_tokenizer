"""Real-backbone smoke checks for KI one-forward CE/FM training."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.core.action_expert import sample_flow_training_point
from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.policy import BasePolicy
from praxis.policies.core.vlm.supervision import IGNORE_INDEX
from praxis.policies.core.vlm.training import (
    build_prepared_action_token_attention_mask,
    prepare_action_token_ce,
)
from praxis.policies.factory import get_policy_class

_CODECS = ("acodec", "bin", "fast", "oat", "quest")
_BACKBONES = ("pali", "qwen", "smol")
_POLICIES = tuple(
    f"ki_{codec}_{backbone}" for codec in _CODECS for backbone in _BACKBONES
)
_DEFAULT_TOKENIZER_PATHS = {
    "acodec": Path(
        "outputs/train_codecs/20260411/libero_acodec_codec/"
        "checkpoints/best/step_326000/pretrained_model"
    ),
    "oat": Path(
        "outputs/train_codecs/20260411/libero_oat_codec/"
        "checkpoints/best/step_688000/pretrained_model"
    ),
}


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested, but CUDA is unavailable.")
    for policy_name in args.policies:
        _run_one(_args_for_policy(args, str(policy_name)))


def _run_one(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    print(f"Starting {args.policy} ({_tokenizer_label(args)})", flush=True)

    policy = _make_policy(args).to(device)
    batch = _make_batch(args, device=device)

    parity_diff = _check_ce_parity(policy, batch)
    fm_loss_diff, fm_pred_diff = _check_fm_parity(policy, batch)
    gradients = _check_gradients(policy, batch)

    print(
        f"{args.policy} ({_tokenizer_label(args)}): ce_parity_diff={parity_diff:.6g} "
        f"fm_loss_diff={fm_loss_diff:.6g} fm_pred_diff={fm_pred_diff:.6g} "
        f"combined=({gradients['combined_vlm']:.6g}, {gradients['combined_expert']:.6g}) "
        f"ce_only=({gradients['ce_only_vlm']:.6g}, {gradients['ce_only_expert']:.6g}) "
        f"flow_only=({gradients['flow_only_vlm']:.6g}, {gradients['flow_only_expert']:.6g})"
    )
    del batch
    del policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="+", choices=_POLICIES)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help=(
            "Optional action-tokenizer checkpoint. Defaults to the LIBERO checkpoint "
            "for codecs with one available in outputs/train_codecs/20260411."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--expert-hidden-size", type=int, default=32)
    parser.add_argument("--expert-num-layers", type=int, default=1)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing for debugging.",
    )
    return parser.parse_args()


def _args_for_policy(args: argparse.Namespace, policy: str) -> argparse.Namespace:
    values = vars(args).copy()
    values["policy"] = policy
    return argparse.Namespace(**values)


def _policy_codec(policy: str) -> str:
    parts = policy.split("_")
    if len(parts) != 3 or parts[0] != "ki" or parts[1] not in _CODECS:
        raise ValueError(f"Could not infer codec from KI policy name {policy!r}.")
    return parts[1]


def _resolved_tokenizer_path(args: argparse.Namespace) -> Path | None:
    if args.tokenizer_path is not None:
        return cast(Path, args.tokenizer_path)
    codec = _policy_codec(str(args.policy))
    path = _DEFAULT_TOKENIZER_PATHS.get(codec)
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"Default {codec!r} tokenizer checkpoint does not exist: {path}. "
            "Pass --tokenizer-path to use a different checkpoint."
        )
    return path


def _tokenizer_config(args: argparse.Namespace) -> dict[str, object]:
    codec = _policy_codec(str(args.policy))
    path = _resolved_tokenizer_path(args)
    config: dict[str, object] = {
        "name": codec,
        "kind": "tokens",
    }
    if path is not None:
        config["pretrained_path"] = str(path)
    else:
        config["action_dim"] = int(args.action_dim)
        config["action_horizon"] = int(args.action_horizon)
    return config


def _tokenizer_label(args: argparse.Namespace) -> str:
    path = _resolved_tokenizer_path(args)
    if path is None:
        return f"tokenizer={_policy_codec(str(args.policy))}:from-config"
    return f"tokenizer={path}"


def _make_policy(args: argparse.Namespace) -> BasePolicy:
    io = PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.image",
                type=FeatureType.IMAGE,
                shape=(3, int(args.image_size), int(args.image_size)),
                norm_mode=NormMode.MEAN_STD,
            )
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(int(args.action_dim),),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=1,
        n_action_steps=int(args.n_action_steps),
        language_key="task",
    )
    policy_cls = get_policy_class(str(args.policy))
    config_overrides: dict[str, Any] = {
        "action_dim": int(args.action_dim),
        "action_horizon": int(args.action_horizon),
        "n_action_steps": int(args.n_action_steps),
        "action_tokenizer": _tokenizer_config(args),
        "expert_hidden_size": int(args.expert_hidden_size),
        "expert_num_layers": int(args.expert_num_layers),
        "gradient_checkpointing": not bool(args.no_gradient_checkpointing),
    }
    if (
        _policy_codec(str(args.policy)) == "quest"
        and _resolved_tokenizer_path(args) is None
    ):
        config_overrides["freeze_action_tokenizer"] = False
    return cast(
        Any,
        policy_cls,
    ).from_io(io, **config_overrides)


def _make_batch(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    return {
        "observation.image": torch.rand(
            int(args.batch_size),
            1,
            3,
            int(args.image_size),
            int(args.image_size),
            device=device,
        ),
        "action": torch.randn(
            int(args.batch_size),
            int(args.action_horizon),
            int(args.action_dim),
            device=device,
        ).clamp(-1.0, 1.0),
        "task": _instructions(int(args.batch_size)),
    }


def _instructions(batch_size: int) -> list[str]:
    base = [
        "pick up the cube",
        "open the drawer after moving near the cabinet",
        "place the object in the bowl using the visible gripper",
    ]
    return [base[i % len(base)] for i in range(batch_size)]


def _check_ce_parity(policy: BasePolicy, batch: dict[str, Any]) -> float:
    policy_any = cast(Any, policy)
    policy.eval()
    policy_any.config.flow_loss_weight = 0.0
    with torch.inference_mode():
        standalone_ce = policy_any.compute_ce_loss(
            batch,
            actions=cast(torch.Tensor, batch["action"]),
        )
        one_forward_loss, _info = policy.forward(batch)
    torch.testing.assert_close(one_forward_loss, standalone_ce, atol=1e-4, rtol=1e-4)
    return float((one_forward_loss - standalone_ce).abs().item())


def _check_fm_parity(policy: BasePolicy, batch: dict[str, Any]) -> tuple[float, float]:
    """Compare KI's one-forward FM lane against the shared FM training path."""
    policy_any = cast(Any, policy)
    policy.eval()
    raw_batch = dict(batch)
    norm_batch = policy_any._prepare_normalized_batch(raw_batch)
    raw_actions = policy_any._actions_from_batch(raw_batch)
    norm_actions = policy_any._actions_from_batch(norm_batch).to(
        device=raw_actions.device,
        dtype=raw_actions.dtype,
    )
    prefix_batch = policy_any._observations_for_state_prefix(
        raw_batch,
        norm_batch=norm_batch,
    )
    prepared = prepare_action_token_ce(
        policy_any,
        prefix_batch,
        actions=raw_actions,
        ignore_index=IGNORE_INDEX,
    )
    sample = sample_flow_training_point(
        norm_actions,
        sampling=policy_any.config.flow_time_sampling,
        beta_alpha=policy_any.config.flow_beta_alpha,
        beta_beta=policy_any.config.flow_beta_beta,
        min_time=policy_any.config.flow_min_time,
    )
    instructions = extract_instructions(
        prefix_batch,
        language_key=None if policy.io is None else policy.io.language_key,
    )
    fm_vlm_inputs = policy_any._build_vlm_inputs_for_action_expert(
        prefix_batch,
        instructions=instructions,
    )
    expert_context_mask = policy_any._build_flow_expert_context_mask(
        prefix_batch,
        prepared,
    )
    schedule_attention_mask = build_prepared_action_token_attention_mask(
        policy_any,
        prepared,
    )
    model = policy_any._require_vlm_action_expert()
    action_is_pad = policy_any._action_pad_mask_from_batch(
        norm_batch,
        actions=norm_actions,
    )
    with torch.inference_mode(), policy_any._autocast_context():
        fm_loss, fm_pred = model.loss(
            vlm_inputs=fm_vlm_inputs,
            sample=sample,
            action_is_pad=action_is_pad,
            detach_prefix=False,
            gradient_checkpointing=False,
        )
        ki_loss, ki_output = model.loss_with_context(
            vlm_inputs=prepared.vlm_inputs,
            layer_attention_mask=schedule_attention_mask,
            expert_context_mask=expert_context_mask,
            sample=sample,
            action_is_pad=action_is_pad,
            detach_expert_context=False,
            gradient_checkpointing=False,
        )
    mismatch_summary = _fm_context_summary(prepared.vlm_inputs, fm_vlm_inputs, prepared)
    try:
        torch.testing.assert_close(ki_loss, fm_loss, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            ki_output.pred_velocity,
            fm_pred,
            atol=1e-2,
            rtol=5e-2,
        )
    except AssertionError as exc:
        raise AssertionError(f"{exc}\n{mismatch_summary}") from exc
    return (
        float((ki_loss - fm_loss).abs().item()),
        float((ki_output.pred_velocity - fm_pred).abs().max().item()),
    )


def _fm_context_summary(
    ki_vlm_inputs: dict[str, torch.Tensor],
    fm_vlm_inputs: dict[str, torch.Tensor],
    prepared: Any,
) -> str:
    rows: list[str] = []
    ki_ids = ki_vlm_inputs["input_ids"].detach().cpu()
    fm_ids = fm_vlm_inputs["input_ids"].detach().cpu()
    ki_prefix_mask = prepared.supervision.prefix_mask.detach().cpu().bool()
    fm_mask = fm_vlm_inputs.get("attention_mask")
    if fm_mask is None:
        fm_valid_mask = torch.ones_like(fm_ids, dtype=torch.bool)
    else:
        fm_valid_mask = fm_mask.detach().cpu().bool()
    for row_idx in range(int(ki_ids.shape[0])):
        ki_prefix = ki_ids[row_idx][ki_prefix_mask[row_idx]]
        fm_prefix = fm_ids[row_idx][fm_valid_mask[row_idx]]
        min_len = min(int(ki_prefix.numel()), int(fm_prefix.numel()))
        equal = bool(
            ki_prefix.shape == fm_prefix.shape and torch.equal(ki_prefix, fm_prefix)
        )
        first_diff = "none"
        if not equal:
            if int(ki_prefix.numel()) != int(fm_prefix.numel()):
                first_diff = "length"
            elif min_len > 0:
                diff = (ki_prefix[:min_len] != fm_prefix[:min_len]).nonzero(
                    as_tuple=False
                )
                if int(diff.numel()) > 0:
                    pos = int(diff[0].item())
                    first_diff = (
                        f"{pos}: ki={int(ki_prefix[pos].item())} "
                        f"fm={int(fm_prefix[pos].item())}"
                    )
        rows.append(
            f"row={row_idx} ki_prefix_len={int(ki_prefix.numel())} "
            f"fm_prefix_len={int(fm_prefix.numel())} prefix_ids_equal={equal} "
            f"first_diff={first_diff}"
        )
    return "FM context summary: " + "; ".join(rows)


def _check_gradients(
    policy: BasePolicy,
    batch: dict[str, Any],
) -> dict[str, float]:
    checks: dict[str, tuple[float, float]] = {
        "combined": (1.0, 1.0),
        "ce_only": (1.0, 0.0),
        "flow_only": (0.0, 1.0),
    }
    out: dict[str, float] = {}
    policy_any = cast(Any, policy)
    for name, (ce_weight, flow_weight) in checks.items():
        policy.train()
        policy_any.config.ce_loss_weight = ce_weight
        policy_any.config.flow_loss_weight = flow_weight
        policy.zero_grad(set_to_none=True)
        loss, _info = policy.forward(batch)
        loss.backward()
        vlm_grad = _grad_l1(cast(nn.Module, policy.vlm))
        expert = cast(Any, policy).action_expert
        if expert is None:
            raise RuntimeError("Policy did not build an action expert.")
        expert_grad = _grad_l1(cast(nn.Module, expert))
        _assert_gradient_pattern(
            name,
            vlm_grad=vlm_grad,
            expert_grad=expert_grad,
            expect_vlm=ce_weight > 0,
            expect_expert=flow_weight > 0,
        )
        out[f"{name}_vlm"] = vlm_grad
        out[f"{name}_expert"] = expert_grad
    return out


def _grad_l1(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().item())
    return total


def _assert_gradient_pattern(
    name: str,
    *,
    vlm_grad: float,
    expert_grad: float,
    expect_vlm: bool,
    expect_expert: bool,
) -> None:
    if expect_vlm and vlm_grad <= 0:
        raise AssertionError(f"{name}: expected nonzero VLM grad.")
    if not expect_vlm and vlm_grad != 0:
        raise AssertionError(f"{name}: expected zero VLM grad, got {vlm_grad}.")
    if expect_expert and expert_grad <= 0:
        raise AssertionError(f"{name}: expected nonzero expert grad.")
    if not expect_expert and expert_grad != 0:
        raise AssertionError(f"{name}: expected zero expert grad, got {expert_grad}.")


if __name__ == "__main__":
    main()
