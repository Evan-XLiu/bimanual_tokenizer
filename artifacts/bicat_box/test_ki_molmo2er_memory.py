"""Exercise the real KI trainer before submitting the full BiCAT run."""

import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter

import torch
from omegaconf import OmegaConf

import praxis.policies.ki.core.policy as ki_policy_module
from praxis.scripts.train import configure_local_dataset_environment
from praxis.training.policy_trainer import PolicyTrainer


def main():
    cfg = OmegaConf.load(sys.argv[1])
    if os.environ.get("TEST_BATCH_SIZE"):
        cfg.trainer.batch_size = int(os.environ["TEST_BATCH_SIZE"])
        cfg.trainer.global_batch_size = None
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out / "train")
    cfg.run_name = "BiCAT_box_molmo2ER_memory_test"
    cfg.trainer.use_wandb = False
    if "--single-resize" in sys.argv[3:]:
        cfg.trainer.val_ratio = 0.0
        cfg.trainer.val_every = 0
        cfg.trainer.reconst_every = 0
        cfg.trainer.checkpoint_candidates.best.k = 0
    result = {"status": "running", "batch_size": int(cfg.trainer.batch_size)}
    stage = "initialization"
    stage_test = os.environ.get("TEST_KI_STAGES") == "1"
    try:
        configure_local_dataset_environment(cfg)
        torch.cuda.reset_peak_memory_stats()
        trainer = PolicyTrainer(cfg)
        policy = trainer.runtime.unwrap_model(trainer.model)
        assert not any(p.requires_grad for p in policy.action_tokenizer.parameters())
        assert not policy.action_tokenizer.training
        assert all(
            p.dtype == torch.bfloat16
            for p in policy.parameters()
            if p.is_floating_point()
        )
        assert policy.io.action_dim == 14
        assert policy.config.use_lora is False
        assert policy.config.gradient_checkpointing is True
        vision = policy.vlm.model.vision_backbone
        assert vision.image_vit.transformer.gradient_checkpointing is True
        assert all(
            not module.float32_attention
            for module in vision.modules()
            if hasattr(module, "float32_attention")
        )
        result["vision_gradient_checkpointing"] = True
        result["vision_dtype"] = str(next(vision.parameters()).dtype)
        if "--single-resize" in sys.argv[3:]:
            original_preprocess = policy.processor.image_processor.preprocess

            def checked_preprocess(*args, **kwargs):
                output = original_preprocess(*args, **kwargs)
                crops = output["image_num_crops"]
                assert (crops == 1).all(), "Expected exactly one crop per camera"
                result["image_preprocessing"] = {
                    "mode": "single_resize",
                    "num_images": len(crops),
                    "num_crops": int(output["pixel_values"].shape[0]),
                    "pooled_visual_tokens": int(output["image_token_pooling"].shape[0]),
                }
                return output

            policy.processor.image_processor.preprocess = checked_preprocess
        trainer.model.train()
        OmegaConf.save(cfg, out / "effective_config.yaml")
        batches = iter(trainer.dataloader)
        result["steps"] = []
        # Smoke mode runs ten optimizer steps and records the configured CE/FM
        # losses without assuming a particular warmstart policy.
        steps = tuple(range(10)) if "--smoke" in sys.argv[3:] else (0, 6000)
        if stage_test:
            boundary = int(policy.config.ce_warmstart_steps)
            assert boundary >= 3
            assert policy.config.action_expert_detach_vlm is True
            steps = tuple(range(boundary - 3, boundary + 3))
            result["gradient_checks"] = []
            result["scheduler_checks"] = []
            for group, base_lr, schedule in zip(
                trainer._get_optim_params(),
                trainer.scheduler.base_lrs,
                trainer.scheduler.lr_lambdas,
            ):
                family = group["family"]
                warmup = 1000 if family == "vision" else 500
                assert schedule(0) == 0
                assert abs(schedule(warmup // 2) - 0.5) < 1e-8
                assert abs(schedule(warmup) - 1.0) < 1e-8
                assert abs(schedule(int(cfg.trainer.max_steps)) - 0.1) < 1e-8
                result["scheduler_checks"].append({
                    "family": family, "warmup_steps": warmup,
                    "peak_lr": base_lr, "final_lr": base_lr * schedule(int(cfg.trainer.max_steps)),
                })

            def align_scheduler(step):
                # Boundary testing skips updates; align LR to the same counter
                # without pretending to have trained the skipped batches.
                trainer.scheduler.last_epoch = step
                for group, base_lr, schedule in zip(
                    trainer.optimizer.param_groups,
                    trainer.scheduler.base_lrs,
                    trainer.scheduler.lr_lambdas,
                ):
                    group["lr"] = base_lr * schedule(step)

            def grad_norm(module):
                norms = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
                return torch.stack(norms).norm().item() if norms else 0.0

            def check_gradients(optimizer, args, kwargs):
                del optimizer, args, kwargs
                check = {
                    "step": trainer.global_step,
                    "vlm_grad_norm": grad_norm(policy.vlm),
                    "expert_grad_norm": grad_norm(policy.action_expert),
                    "state_grad_norm": grad_norm(policy.state_prefix_projector),
                }
                if trainer.global_step < boundary:
                    assert check["vlm_grad_norm"] > 0 and check["expert_grad_norm"] == 0
                elif trainer.global_step == boundary + 3:
                    assert check["vlm_grad_norm"] == check["state_grad_norm"] == 0
                    assert check["expert_grad_norm"] > 0
                    check["probe"] = "joint_forward_with_zero_ce"
                else:
                    assert check["vlm_grad_norm"] > 0 and check["expert_grad_norm"] > 0
                result["gradient_checks"].append(check)
                print(json.dumps({"gradient_check": check}), flush=True)

            grad_handle = trainer.optimizer.register_step_pre_hook(check_gradients)
        for step in steps:
            stage = f"train_step_{step}"
            trainer.global_step = step
            if stage_test:
                align_scheduler(step)
            torch.cuda.synchronize()
            start = perf_counter()
            batch = next(batches)
            data_seconds = perf_counter() - start
            assert trainer._batch_size_from_batch(batch) == int(cfg.trainer.batch_size)
            loss, info, did_step = trainer.train_step(batch)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            assert did_step and torch.isfinite(loss).all()
            if stage_test:
                assert info["ce_loss"] > 0
                assert (info["flow_loss"] > 0) == (step >= boundary)
            record = {
                "global_step": step,
                "loss": loss.item(),
                "data_wait_s": data_seconds,
                "wall_s": elapsed,
                "samples_per_second": int(cfg.trainer.batch_size) / elapsed,
                "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                **info,
            }
            result["steps"].append(record)
            print(json.dumps(record), flush=True)
            (out / "memory_result.json").write_text(json.dumps(result, indent=2) + "\n")
        if stage_test:
            stage = "flow_gradient_isolation_probe"
            original_ce = ki_policy_module.compute_prepared_action_token_ce_loss_from_hidden
            try:
                ki_policy_module.compute_prepared_action_token_ce_loss_from_hidden = (
                    lambda *args, **kwargs: kwargs["hidden"].sum() * 0
                )
                trainer.global_step = boundary + 3
                align_scheduler(boundary + 3)
                loss, info, did_step = trainer.train_step(next(batches))
                torch.cuda.synchronize()
                assert did_step and torch.isfinite(loss).all() and info["flow_loss"] > 0
                result["flow_gradient_isolation"] = "passed"
            finally:
                ki_policy_module.compute_prepared_action_token_ce_loss_from_hidden = original_ce
                grad_handle.remove()
            for phase, records in (
                ("ce_only", result["steps"][1:3]),
                ("ce_and_flow", result["steps"][4:6]),
            ):
                seconds = sum(r["wall_s"] for r in records) / len(records)
                result[phase] = {
                    "seconds_per_step": seconds,
                    "samples_per_second": int(cfg.trainer.batch_size) / seconds,
                    "peak_allocated_gib": max(r["peak_allocated_gib"] for r in records),
                }
        result["status"] = "passed"
        if "--smoke" in sys.argv[3:] and not stage_test:
            steady = result["steps"][2:]
            elapsed = sum(record["wall_s"] for record in steady)
            result["steady_steps"] = len(steady)
            result["steady_seconds_per_step"] = elapsed / len(steady)
            result["steady_samples_per_second"] = (
                int(cfg.trainer.batch_size) * len(steady) / elapsed
            )
        trainer.logger.finish()
    except torch.cuda.OutOfMemoryError:
        result.update(status="out_of_memory", stage=stage)
        traceback.print_exc()
        raise
    except Exception as exc:
        result.update(status="failed", stage=stage, error=str(exc))
        traceback.print_exc()
        raise
    finally:
        if torch.cuda.is_available():
            result.update(
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                total_memory_gib=torch.cuda.get_device_properties(0).total_memory
                / 2**30,
            )
        (out / "memory_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
