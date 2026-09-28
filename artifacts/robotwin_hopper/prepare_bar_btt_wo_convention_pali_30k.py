#!/usr/bin/env python3
"""Prepare and verify the exact BTT w/o convention VLA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / (
    "outputs/hopper_bar_btt_pali_robotwin_new_30k/"
    "622038.hopper-m-02/train/train_config.yaml"
)
TOKENIZER = ROOT / (
    "outputs/hopper_btt_wo_convention_robotwin_new/628094.hopper-m-02/"
    "train/checkpoints/best/step_300000/pretrained_model"
)
RUN_NAME = "robotwin_new_bar_btt_wo_convention_pali_step300000_30k"
REFERENCE_SHA256 = "9c7c839e82b1bead9539f61ff97f2d00e80c9fd043dfdc455c3b17a19909cd61"
EXPECTED_HASHES = {
    "config.json": "69526a1f388f57b577e6262ad5b724b01803376918727a3eb619dde123f893d4",
    "model.safetensors": "5aa04849a9743b331f3b68bb3fa340db41705feaff4856509d8a7d75f9361851",
    "train_config.yaml": "74a77296d88e230143444ee36b2e72f18e5360f61a4077672ee659f52461c183",
    "norm_stats.pt": "4b9e7e69a4cb1bf67acc655cba23f329c1a6125d8c74ee1b37c19bd2c54a56e5",
    "policy_io.json": "72fa701aa2ad4e46dc1b7777959eac43ffce47c6ef2ccbb33b1131e823e77038",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ABLATION_AUDIT_FAILED: {message}")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def differences(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(name)
            else:
                result.extend(differences(left[key], right[key], name))
        return result
    return [] if left == right else [prefix]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()

    require(TOKENIZER.resolve() == TOKENIZER, "tokenizer path was redirected")
    hashes = {name: digest(TOKENIZER / name) for name in EXPECTED_HASHES}
    require(hashes == EXPECTED_HASHES, "tokenizer checkpoint SHA-256 mismatch")
    tokenizer_config = json.loads((TOKENIZER / "config.json").read_text())
    tokenizer_train = OmegaConf.load(TOKENIZER / "train_config.yaml")
    require(tokenizer_config["name"] == "btt", "expected BTT tokenizer")
    require(tokenizer_config["right_arm_sign"] == [1.0] * 7, "config enables convention")
    require(list(tokenizer_train.codec.right_arm_sign) == [1] * 7, "tokenizer training used convention")
    manifest = json.loads((TOKENIZER.parent.parent / "topk_manifest.json").read_text())
    require(any(item["step"] == 300000 and item["file"] == "step_300000" for item in manifest),
            "step 300000 is missing from the tokenizer checkpoint manifest")
    with safe_open(str(TOKENIZER / "model.safetensors"), framework="np") as weights:
        signs = weights.get_tensor("encoder.right_arm_sign").reshape(-1)
        require(np.array_equal(signs, np.ones(7)), "saved encoder weights enable convention")
        for key in ("shared_action_min", "shared_action_max"):
            actual = weights.get_tensor(f"encoder.{key}").reshape(-1)
            expected = np.asarray(tokenizer_config[key], dtype=actual.dtype)
            require(np.array_equal(actual, expected), f"saved encoder {key} differs from config")

    require(digest(REFERENCE) == REFERENCE_SHA256, "reference training configuration changed")
    reference = OmegaConf.load(REFERENCE)
    require(reference.policy.name == "bar_btt_pali", "wrong reference policy")
    require(reference.policy.vlm_model_name == "google/paligemma2-3b-mix-224", "wrong VLM")
    require(reference.policy.freeze_action_tokenizer is True, "tokenizer must remain frozen")
    require(reference.trainer.max_steps == 30000, "expected 30000 VLA training steps")
    require(reference.trainer.use_wandb is True and reference.trainer.wandb_project == "praxis-vla",
            "reference W&B logging changed")
    require(reference.policy.pretrained_path is None and reference.resume_checkpoint is None,
            "expected a fresh policy training run")
    require((Path(reference.dataset.root) / "meta/info.json").is_file(), "dataset is missing")
    require((Path(reference.dataset.root) / "meta/stats.json").is_file(), "dataset statistics are missing")
    original = OmegaConf.to_container(reference, resolve=False)
    reference.policy.action_tokenizer.pretrained_path = str(TOKENIZER)
    reference.run_name = RUN_NAME
    reference.output_dir = str(output / "train")
    expected = OmegaConf.to_container(reference, resolve=False)
    changes = differences(original, expected)
    require(set(changes) == {"policy.action_tokenizer.pretrained_path", "run_name", "output_dir"},
            f"unexpected differences from reference: {changes}")

    if args.candidate:
        candidate = OmegaConf.to_container(OmegaConf.load(args.candidate), resolve=False)
        require(candidate == expected, f"resolved training config mismatch: {differences(expected, candidate)}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        require(not (output / "train").exists(), "refusing to overwrite an existing training run")
        (output / "reference_config.yaml").write_bytes(REFERENCE.read_bytes())
        OmegaConf.save(reference, output / "resolved_config.yaml")

    audit = {
        "status": "passed",
        "reference_config": str(REFERENCE),
        "reference_config_sha256": digest(REFERENCE),
        "tokenizer_pretrained_path": str(TOKENIZER),
        "tokenizer_step": 300000,
        "right_arm_sign_config": tokenizer_config["right_arm_sign"],
        "right_arm_sign_saved_weights": signs.tolist(),
        "tokenizer_file_sha256": hashes,
        "reference_config_differences": changes,
        "training_steps": reference.trainer.max_steps,
        "wandb_project": reference.trainer.wandb_project,
        "wandb_entity": reference.trainer.wandb_entity,
        "run_name": RUN_NAME,
    }
    if args.candidate:
        audit["verified_candidate"] = str(args.candidate.resolve())
    (output / "ablation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"TOKENIZER_AUDIT_OK path={TOKENIZER} step=300000 right_arm_sign={signs.tolist()}", flush=True)
    print(f"TOKENIZER_WEIGHTS_SHA256 {hashes['model.safetensors']}", flush=True)
    print("CONFIG_AUDIT_OK exact reference parity except tokenizer checkpoint, run_name, output_dir", flush=True)


if __name__ == "__main__":
    main()
