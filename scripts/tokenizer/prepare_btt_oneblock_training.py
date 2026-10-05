"""Clone an executed BTT training configuration with identity-only changes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = {
    "tokenizer": ROOT
    / "outputs/hopper_btt_robotwin_new/621340.hopper-m-02/resolved_config.yaml",
    "vla": ROOT
    / "outputs/hopper_bar_btt_pali_robotwin_new_step300000_30k/630822.hopper-m-02/resolved_config.yaml",
}


def prepare_config(stage, reference, output_dir, run_name, tokenizer_checkpoint=None):
    """Return snapshot plus a precise whitelist of experiment-identity changes."""
    source = OmegaConf.load(reference)
    cfg = copy.deepcopy(source)
    if source.resume_checkpoint is not None:
        raise ValueError("Reference must start from scratch, not resume a run.")
    changes = {"run_name": run_name, "output_dir": str(Path(output_dir).resolve())}
    if stage == "tokenizer":
        if source.codec.name != "btt" or source.trainer._trainer_type != "codecs":
            raise ValueError("Expected a standard BTT tokenizer training reference.")
        if source.codec.pretrained_path is not None:
            raise ValueError("Tokenizer reference must train from scratch.")
        changes["codec.name"] = "btt_oneblock"
        expected = {
            "trainer.max_steps": 300000,
            "trainer.batch_size": 512,
            "trainer.lr": 5e-5,
            "trainer.wandb_project": "tokenizer_pretrain",
        }
    elif stage == "vla":
        if (
            source.policy.name != "bar_btt_pali"
            or source.trainer._trainer_type != "policy"
        ):
            raise ValueError("Expected the standard BAR BTT Pali VLA reference.")
        if source.policy.pretrained_path is not None:
            raise ValueError("VLA must use the original pretrained VLM initialization.")
        if tokenizer_checkpoint is None:
            raise ValueError("VLA stage requires a btt_oneblock tokenizer checkpoint.")
        checkpoint = Path(tokenizer_checkpoint).resolve()
        if not (checkpoint / "config.json").is_file():
            checkpoint /= "pretrained_model"
        for name in (
            "config.json",
            "model.safetensors",
            "norm_stats.pt",
            "policy_io.json",
        ):
            if not (checkpoint / name).is_file():
                raise ValueError(f"Missing tokenizer artifact: {checkpoint / name}")
        if (
            json.loads((checkpoint / "config.json").read_text())["name"]
            != "btt_oneblock"
        ):
            raise ValueError(
                "VLA requires an explicitly saved btt_oneblock checkpoint."
            )
        changes.update(
            {
                "policy.name": "bar_btt_oneblock_pali",
                "policy.action_tokenizer.name": "btt_oneblock",
                "policy.action_tokenizer.pretrained_path": str(checkpoint),
            }
        )
        expected = {
            "trainer.max_steps": 30000,
            "trainer.batch_size": 64,
            "trainer.wandb_project": "praxis-vla",
            "policy.freeze_action_tokenizer": True,
        }
    else:
        raise ValueError(f"Unknown stage: {stage}")
    expected.update(
        {"seed": 42, "trainer.use_wandb": True, "trainer.mixed_precision": "bf16"}
    )
    for key, value in expected.items():
        if OmegaConf.select(source, key) != value:
            raise ValueError(f"Reference mismatch for {key}: expected {value!r}")
    for key, value in changes.items():
        OmegaConf.update(cfg, key, value, merge=False)
    # Check unresolved values so run-dependent interpolations remain intact.
    restored = copy.deepcopy(cfg)
    for key in changes:
        OmegaConf.update(restored, key, OmegaConf.select(source, key), merge=False)
    if OmegaConf.to_container(restored, resolve=False) != OmegaConf.to_container(
        source, resolve=False
    ):
        raise AssertionError("Unexpected change outside experiment identity fields.")
    return cfg, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(REFERENCES), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path)
    parser.add_argument("--save-config", type=Path, required=True)
    args = parser.parse_args()
    reference = (args.reference or REFERENCES[args.stage]).resolve()
    cfg, changes = prepare_config(
        args.stage, reference, args.output_dir, args.run_name, args.tokenizer_checkpoint
    )
    args.save_config.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a configuration for an existing run.
    with args.save_config.open("x") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False))
    audit = {
        "stage": args.stage,
        "reference": str(reference),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "changed_identity_fields": changes,
        "all_other_fields_identical": True,
    }
    args.save_config.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
