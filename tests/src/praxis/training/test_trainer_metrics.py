from __future__ import annotations

import praxis.training.core.trainer as trainer_module
from praxis.training.core.trainer import DefaultTrainer


def test_trainer_module_does_not_reexport_component_helpers() -> None:
    assert not hasattr(trainer_module, "accumulate_train_info_metrics")
    assert not hasattr(trainer_module, "reduce_train_info_metrics")
    assert not hasattr(trainer_module, "filter_checkpoint_incompatible_keys")


def test_format_logged_metrics_namespaces_train_eval_and_sim_eval() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    payload = trainer._format_logged_metrics(
        {
            "train_loss": 1.0,
            "val_loss": 2.0,
            "val_reconst_loss": 3.0,
            "success_rate": 0.5,
            "avg_episode_length": 42.0,
        },
        default_group="eval",
    )
    assert payload == {
        "train/loss": 1.0,
        "eval/loss": 2.0,
        "eval/reconst_loss": 3.0,
        "sim_eval/success_rate": 0.5,
        "sim_eval/avg_episode_length": 42.0,
    }
