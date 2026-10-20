"""Run the reference BAR pipeline with four durable, resumable checkpoints."""
from pathlib import Path
import json
import os
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).resolve().parent
STEPS = (12000, 18000, 24000, 30000)


def validate(name):
    cfg = OmegaConf.load(CONFIGS / f"{name}.yaml")
    assert cfg.trainer.batch_size == 128
    assert cfg.trainer.global_batch_size is None
    assert cfg.trainer.max_steps == 30000
    assert cfg.trainer.checkpoint_start_step == 12000
    assert cfg.trainer.checkpoint_every == 6000
    assert cfg.trainer.checkpoint_candidates.best.k == 0
    assert cfg.trainer.checkpoint_candidates.recent.k == 0
    assert cfg.policy.freeze_action_tokenizer
    assert cfg.dataset.action_key == "action_joint_angles"
    assert cfg.dataset.episodes is None and cfg.trainer.val_ratio == 0
    count = 0
    for source in cfg.dataset.sources:
        info = json.loads((Path(source.root) / "meta/info.json").read_text())
        assert info["features"][cfg.dataset.action_key]["shape"] == [14]
        count += info["total_episodes"]
    assert count == 405
    from praxis.codecs.factory import make_codec
    from praxis.policies.factory import get_policy_class, validate_tokenizer_cfg
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    validate_tokenizer_cfg(policy_name=cfg.policy.name, cfg_dict=policy_cfg)
    policy_cls = get_policy_class(cfg.policy.name)
    policy_cls.CONFIG_CLASS.from_dict(policy_cfg)
    codec = make_codec(cfg.policy.action_tokenizer)
    assert codec.config.action_dim == 14
    assert codec.config.action_horizon == 32
    print(f"Validated {name}: 405 episodes, 14D, policy={policy_cls.__name__}, codec={type(codec).__name__}", flush=True)
    return cfg


def main():
    name = sys.argv[1]
    assert name in {"btt", "oat", "fast"}
    cfg = validate(name)
    if "--check" in sys.argv:
        return
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert "H200" in torch.cuda.get_device_name(0)
    output = Path(cfg.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite populated output: {output}")
    print(f"GPU={torch.cuda.get_device_name(0)}, output={output}, checkpoints={STEPS}", flush=True)
    from praxis.training import policy_trainer

    class FourCheckpointPolicyTrainer(policy_trainer.PolicyTrainer):
        def save_latest_checkpoint(self):
            step = int(self.global_step)
            if step not in STEPS:
                raise RuntimeError(f"Unexpected checkpoint step {step}")
            if getattr(self, "_last_durable_step", None) == step:
                return
            root = Path(self.tc.checkpoint_dir)
            target = root / f"step_{step}"
            # Standard checkpoint writer retains optimizer, scheduler and EMA state.
            self.save_checkpoint(target)
            if self.runtime.is_main_process:
                latest = root / "latest"
                if latest.exists() and not latest.is_symlink():
                    raise RuntimeError(f"Expected a symlink, found {latest}")
                temporary = root / f".latest-link-{os.getpid()}"
                temporary.symlink_to(target.name, target_is_directory=True)
                os.replace(temporary, latest)
                print(f"Saved checkpoint {target}; latest is a symlink only.", flush=True)
            self.runtime.wait_for_everyone()
            self._last_durable_step = step

    policy_trainer.PolicyTrainer = FourCheckpointPolicyTrainer
    from praxis.scripts.train import main as train
    sys.argv = [sys.argv[0], "--config-path", str(CONFIGS), "--config-name", name]
    train()


if __name__ == "__main__":
    main()
