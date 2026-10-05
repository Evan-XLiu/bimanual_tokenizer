"""Reuse the evaluated BTT configuration and verify the exact frozen codec."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "outputs/hopper_bar_btt_pali_robotwin_new_30k/622038.hopper-m-02/resolved_config.yaml"
MANIFEST = ROOT / "outputs/robotwin_eval/rollout8task_100/BTT/handover_mic/run_manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.finalize:
        latest = out / "train/checkpoints/latest"
        assert json.loads((latest / "training_state/training_step.json").read_text())["step"] == 30000
        config = json.loads((latest / "pretrained_model/config.json").read_text())
        assert config["btt_arm_order"] == "right_then_left"
        assert config["action_tokenizer"]["name"] == "btt"
        wandb_info = json.loads((latest / "training_state/wandb_run_info.json").read_text())
        assert wandb_info["id"] and wandb_info["project"] == "praxis-vla"
        target = out / "train/checkpoints/step_30000/pretrained_model"
        shutil.copytree(latest / "pretrained_model", target)
        (out / "training_complete.json").write_text(json.dumps({
            "step": 30000, "checkpoint": str(target), "wandb": wandb_info,
        }, indent=2) + "\n")
        print(f"BTT_RIGHT_FIRST_30K_EXPORT checkpoint={target}")
        return

    reference = OmegaConf.load(REFERENCE)
    artifact = json.loads(MANIFEST.read_text())["server_model_info"]["serving_artifact"]
    codec = artifact["dependencies"]["action_tokenizer"]
    assert reference.policy.action_tokenizer.pretrained_path == codec["path"]
    for filename, info in codec["files"].items():
        assert hashlib.sha256((Path(codec["path"]) / filename).read_bytes()).hexdigest() == info["sha256"], filename
    candidate = OmegaConf.create(OmegaConf.to_container(reference, resolve=False))
    candidate.policy.btt_arm_order = "right_then_left"
    candidate.run_name = "robotwin_new_bar_btt_right_then_left_pali_30k"
    candidate.output_dir = str(out / "train")
    assert candidate.trainer.use_wandb and candidate.trainer.wandb_project == "praxis-vla"
    assert candidate.trainer.max_steps == 30000
    assert candidate.policy.freeze_action_tokenizer and candidate.resume_checkpoint is None
    # All original numerical, dataset, optimizer and validation settings match.
    audit = OmegaConf.to_container(candidate, resolve=False)
    audit["policy"].pop("btt_arm_order")
    audit["run_name"] = reference.run_name
    audit["output_dir"] = reference.output_dir
    assert audit == OmegaConf.to_container(reference, resolve=False)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "resolved_config.yaml"
    if config_path.exists():
        raise FileExistsError(config_path)
    OmegaConf.save(candidate, config_path)
    shutil.copy2(REFERENCE, out / "reference_config.yaml")
    (out / "config_audit.json").write_text(json.dumps({
        "reference_config": str(REFERENCE), "reference_manifest": str(MANIFEST),
        "tokenizer": codec, "changed_fields": ["policy.btt_arm_order", "run_name", "output_dir"],
        "hyperparameter_parity": True,
    }, indent=2) + "\n")
    print(f"BTT_RIGHT_FIRST_CONFIG_AUDIT_OK output={out} tokenizer={codec['path']}")


if __name__ == "__main__":
    main()
