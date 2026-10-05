"""Compare BTT vocabulary usage using tokenizers referenced by saved VLA configs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from praxis.codecs.factory import make_codec

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / 'outputs/task1_fk_reconstruction/20260928_validation'
OUT = ROOT / 'outputs/tokenizer_code_usage/20260929_btt_ablation'
POLICIES = {
    'BTT_300k': 'hopper_bar_btt_pali_robotwin_new_step300000_30k/630822.hopper-m-02',
    'BTT_wo_convention': 'hopper_bar_btt_wo_convention_pali_robotwin_new_step300000_30k/633464.hopper-m-02',
    'BTT_independent': 'hopper_bar_btt_independent_pali_robotwin_new_step300000_30k/634432.hopper-m-02',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def usage(tokens, size):
    counts = np.bincount(tokens.ravel(), minlength=size)
    left = np.bincount(tokens[:, :8].ravel(), minlength=size)
    right = np.bincount(tokens[:, 8:].ravel(), minlength=size)
    lset, rset = left > 0, right > 0
    union = int(np.count_nonzero(lset | rset))
    intersection = int(np.count_nonzero(lset & rset))
    p, q = left / left.sum(), right / right.sum()
    mid = (p + q) / 2
    js = sum(float((x[x > 0] * np.log2(x[x > 0] / mid[x > 0])).sum()) for x in (p, q)) / 2
    return {
        'codebook_size': size, 'used_codes': union, 'unused_codes': size - union,
        'coverage': union / size, 'left_used': int(lset.sum()),
        'right_used': int(rset.sum()), 'intersection': intersection,
        'jaccard': intersection / union, 'arm_js_divergence_bits': js,
        'entropy_bits': entropy(counts), 'perplexity': 2 ** entropy(counts),
        'unused_ids': np.flatnonzero(counts == 0).tolist(),
        'position_used_codes': [int(np.unique(tokens[:, i]).size) for i in range(16)],
        'token_count': int(counts.sum()),
    }, counts, left, right


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLES / 'shared_samples.npz'
    manifest = json.loads((SAMPLES / 'manifest.json').read_text())
    assert sha256(sample_path) == manifest['samples_sha256']
    saved = np.load(sample_path)
    actions = torch.from_numpy(saved['actions'].copy())
    episodes = saved['episode_index']
    assert tuple(actions.shape) == (815, 32, 14)
    models, arrays = {}, {}
    for label, suffix in POLICIES.items():
        policy = ROOT / 'outputs' / suffix / 'train/checkpoints/latest/pretrained_model'
        policy_config = json.loads((policy / 'config.json').read_text())
        step = json.loads((policy.parent / 'training_state/training_step.json').read_text())['step']
        assert step == 30000 and policy_config['freeze_action_tokenizer']
        codec_spec = policy_config['action_tokenizer']
        checkpoint = Path(codec_spec['pretrained_path'])
        assert 'step_300000' in str(checkpoint)
        cfg = json.loads((checkpoint / 'config.json').read_text())
        train = OmegaConf.load(checkpoint / 'train_config.yaml')
        assert train.seed == 42 and train.trainer.val_ratio == 0.05
        assert Path(train.dataset.root).resolve() == Path(manifest['dataset_root']).resolve()
        codec = make_codec(codec_spec).cpu().eval()
        assert codec.codebook_size == 1920
        all_tokens, all_recon, swap_matches = [], [], []
        with torch.inference_mode():
            for batch in actions.split(32):
                tokens = codec.tokenize(batch)
                recon = codec.detokenize(tokens)
                assert tokens.shape == (len(batch), 16) and recon.shape == batch.shape
                assert torch.isfinite(recon).all() and tokens.min() >= 0 and tokens.max() < 1920
                # Swap normalized inputs: isolates encoder/projection symmetry from normalization.
                normalized = codec._normalize_action_tensor(batch)
                swapped = torch.cat((normalized[:, :, 7:], normalized[:, :, :7]), dim=-1)
                actual = codec._tokenize_norm_actions(swapped).token_ids
                expected = torch.cat((tokens[:, 8:], tokens[:, :8]), dim=1)
                swap_matches.append(actual.eq(expected).cpu().numpy())
                all_tokens.append(tokens.cpu().numpy())
                all_recon.append(recon.cpu().numpy())
        tokens = np.concatenate(all_tokens)
        recon = np.concatenate(all_recon)
        matches = np.concatenate(swap_matches)
        stats, counts, left, right = usage(tokens, 1920)
        error = recon.astype(np.float64) - actions.numpy().astype(np.float64)
        mse = (error ** 2).mean(axis=(1, 2))
        mae = np.abs(error).mean(axis=(1, 2))
        left_mse = (error[:, :, :7] ** 2).mean(axis=(1, 2))
        right_mse = (error[:, :, 7:] ** 2).mean(axis=(1, 2))
        stats.update({
            'raw_action_mse': float(mse.mean()), 'raw_action_mae': float(mae.mean()),
            'left_raw_mse': float(left_mse.mean()), 'right_raw_mse': float(right_mse.mean()),
            'normalized_swap_token_match_rate': float(matches.mean()),
            'normalized_swap_exact_sequence_rate': float(matches.all(axis=1).mean()),
            'parameters': sum(p.numel() for p in codec.parameters()),
            'vla_checkpoint': str(policy), 'vla_step': step,
            'vla_config_sha256': sha256(policy / 'config.json'),
            'tokenizer_checkpoint': str(checkpoint), 'tokenizer_step': 300000,
            'tokenizer_config': cfg, 'tokenizer_weights_sha256': sha256(checkpoint / 'model.safetensors'),
            'train_seed': int(train.seed), 'val_ratio': float(train.trainer.val_ratio),
        })
        models[label] = stats
        arrays[label] = {'token_ids': tokens, 'reconstructed_actions': recon, 'per_chunk_mse': mse}
        np.savez_compressed(OUT / (label + '.npz'), **arrays[label])
        with (OUT / (label + '_counts.csv')).open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['token_id', 'total_count', 'left_count', 'right_count'])
            writer.writerows(zip(range(1920), counts.tolist(), left.tolist(), right.tolist()))
        with (OUT / (label + '_per_chunk.csv')).open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'frame', 'raw_mse', 'raw_mae', 'left_mse', 'right_mse'])
            writer.writerows(zip(episodes, saved['frame_index'], mse, mae, left_mse, right_mse))
        print(label, json.dumps({k: v for k, v in stats.items() if k not in ['tokenizer_config', 'unused_ids']}), flush=True)
        del codec
    # Paired episode-cluster bootstrap: chunk-weighted raw MSE difference.
    unique = np.unique(episodes)
    sizes = np.array([(episodes == e).sum() for e in unique])
    draws = np.random.default_rng(42).integers(len(unique), size=(5000, len(unique)))
    paired = {}
    for label in ['BTT_wo_convention', 'BTT_independent']:
        diff = arrays[label]['per_chunk_mse'] - arrays['BTT_300k']['per_chunk_mse']
        sums = np.array([diff[episodes == e].sum() for e in unique])
        samples = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        paired[label + '_minus_BTT_300k'] = {'mean_difference': float(diff.mean()), 'ci95_percentile': np.quantile(samples, [0.025, 0.975]).tolist()}
    result = {
        'scope': 'Pure tokenizer GT encode/decode; VLA configs used for exact tokenizer provenance; no VLA forward or rollout',
        'samples': 815, 'episodes': len(unique), 'action_shape': [32, 14],
        'samples_sha256': sha256(sample_path), 'device': 'cpu', 'dtype': 'float32',
        'batch_size': 32, 'threads': 4, 'models': models, 'paired_raw_mse': paired,
        'bootstrap': '5000 episode-cluster resamples, seed 42, percentile 95% CI; sample uncertainty only',
    }
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2))
    # Verify saved tokens/histograms and reconstruction aggregates.
    for label, stats in models.items():
        disk = np.load(OUT / (label + '.npz'))
        recomputed, counts, _, _ = usage(disk['token_ids'], 1920)
        assert recomputed['used_codes'] == stats['used_codes'] and int(counts.sum()) == 13040
        assert np.isclose(disk['per_chunk_mse'].mean(), stats['raw_action_mse'])
        with (OUT / (label + '_counts.csv')).open() as f:
            csv_counts = np.array([int(row['total_count']) for row in csv.DictReader(f)])
        assert np.array_equal(counts, csv_counts)
    (OUT / 'verification.json').write_text(json.dumps({'passed': True, 'models': list(models), 'checks': ['checkpoint steps and split', 'sample hash', 'token range and shape', 'finite reconstructions', 'saved histogram agreement', 'saved mean MSE agreement']}, indent=2))
    print('COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
