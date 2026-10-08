"""Exercise the real KI trainer before submitting the full BiCAT run."""

import json
import sys
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf

from praxis.scripts.train import configure_local_dataset_environment
from praxis.training.policy_trainer import PolicyTrainer


def main():
    cfg = OmegaConf.load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out / "train")
    cfg.run_name = "BiCAT_box_molmo2ER_memory_test"
    cfg.trainer.use_wandb = False
    result = {"status": "running", "batch_size": int(cfg.trainer.batch_size)}
    stage = "initialization"
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
        trainer.model.train()
        if "--smoke" in sys.argv[3:]:
            trainer.tc.max_steps = 10
        batches = iter(trainer.dataloader)
        result["steps"] = []
        # Smoke mode runs ten optimizer steps and records the configured CE/FM
        # losses without assuming a particular warmstart policy.
        steps = tuple(range(10)) if "--smoke" in sys.argv[3:] else (0, 6000)
        for step in steps:
            stage = f"train_step_{step}"
            trainer.global_step = step
            batch = next(batches)
            assert trainer._batch_size_from_batch(batch) == int(cfg.trainer.batch_size)
            loss, info, did_step = trainer.train_step(batch)
            torch.cuda.synchronize()
            assert did_step and torch.isfinite(loss).all()
            record = {"global_step": step, "loss": loss.item(), **info}
            result["steps"].append(record)
            print(json.dumps(record), flush=True)
        result["status"] = "passed"
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
