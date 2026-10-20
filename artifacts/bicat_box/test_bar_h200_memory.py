"""Measure full BAR forward/backward/AdamW memory on real training batches."""
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from praxis.scripts.train import configure_local_dataset_environment
from praxis.training.policy_trainer import PolicyTrainer

cfg = OmegaConf.load(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cfg.output_dir = str(out / 'train')
cfg.trainer.use_wandb = False
cfg.trainer.wandb_required = False
result = {'batch_size': int(cfg.trainer.batch_size), 'status': 'failed'}
try:
    configure_local_dataset_environment(cfg)
    trainer = PolicyTrainer(cfg)
    trainer.model.train()
    batches = iter(trainer.dataloader)
    result['steps'] = []
    for step in range(3):
        batch = next(batches)
        assert trainer._batch_size_from_batch(batch) == cfg.trainer.batch_size
        loss, info, updated = trainer.train_step(batch)
        torch.cuda.synchronize()
        assert updated and torch.isfinite(loss).all()
        record = {'step': step, 'loss': loss.item(), 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
        result['steps'].append(record)
        print(json.dumps(record), flush=True)
    trainer.logger.finish()
    result['status'] = 'passed'
except torch.cuda.OutOfMemoryError:
    result['status'] = 'out_of_memory'
    raise
except Exception as exc:
    result['error'] = str(exc)
    raise
finally:
    (out / 'memory_result.json').write_text(json.dumps(result, indent=2) + '\n')
