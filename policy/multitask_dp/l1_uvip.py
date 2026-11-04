"""Six-task E300 L1 EMA labels and the corresponding S0 Student."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset

from .data import FullSensorDataset
from .l1 import FINAL_TASKS, L1Policy, L1SensorEncoder, SixTaskDataset
from .model import DenoisingSample
from .schema import CONDITION_NAMES, fingerprint
from .stage2 import Stage2Policy
from .train_stage2 import Stage2Trainer
from .train import move_batch
from .generate_utility import finalize
from .utility import (atomic_json, commit_shard, complete_window_index, file_hash,
                      read_shard, shard_ranges, window_draws, aggregate_losses)


def load_teacher(path):
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (saved['epoch'], saved['cursor'], saved['global_step'], saved['ema_step']) != (300, 0, 314400, 314400):
        raise ValueError('Expected selected E300 EMA / 314400 updates')
    if fingerprint(saved['protocol']) != saved['protocol_id'] or tuple(saved['config']['tasks']) != FINAL_TASKS:
        raise ValueError('Wrong six-task Teacher protocol')
    model = L1Policy(saved['config'])
    model.load_state_dict(saved['ema'], strict=True)
    for key, value in model.normalizer.state_dict().items():
        torch.testing.assert_close(value, torch.tensor(saved['protocol']['normalizer'][key], dtype=value.dtype), rtol=0, atol=0)
    st = Path(path).stat()
    return model.eval().requires_grad_(False), dict(path=str(Path(path).resolve()), bytes=st.st_size,
        mtime_ns=st.st_mtime_ns, epoch=300, weights='ema', global_step=314400,
        protocol_id=saved['protocol_id'], cache_id=saved['protocol']['cache_id'], architecture='l1_late_language_v1')


def selected_index(source):
    index = complete_window_index(source.manifest)
    task_ids = [source.manifest['tasks'].index(task) for task in FINAL_TASKS]
    index = index[np.isin(index['task_index'], task_ids)].copy()
    index['window_id'] = np.arange(len(index))
    return index


@torch.inference_mode()
def evaluate_draws(policy, batch, steps, noise, microbatch):
    if policy.training or any(p.requires_grad for p in policy.parameters()):
        raise ValueError('Labels require the frozen eval L1 Teacher')
    count = len(batch['action'])
    if steps.shape != (count, 64) or noise.shape != (count, 64, policy.horizon, 8):
        raise ValueError('Invalid K64 draw shapes')
    tokens = policy.obs_encoder.encode_tokens(batch['obs'], policy.normalizer)
    conditions = torch.stack([policy.obs_encoder.condition(tokens, batch['obs']['language'][:, 0],
                  flags.expand(count, -1)) for flags in policy.condition_presence], 1)
    device = tokens.device
    repeated = {key: batch[key].repeat_interleave(64, 0) for key in ('action', 'action_loss_mask')}
    sample = policy.prepare_denoising_sample(repeated, torch.as_tensor(noise.reshape(-1, policy.horizon, 8), device=device),
                                             torch.as_tensor(steps.reshape(-1), device=device))
    result = torch.empty(count * 64 * 9, device=device)
    for start in range(0, result.numel(), microbatch):
        flat = torch.arange(start, min(start + microbatch, result.numel()), device=device)
        draw, condition = flat // 9, flat % 9
        selected = DenoisingSample(*(getattr(sample, k)[draw] for k in ('noisy_action', 'noise', 'timesteps', 'target', 'loss_mask')))
        result[start:start + len(flat)] = policy.denoising_loss_per_sample(selected, conditions[draw // 64, condition])
    return result.reshape(count, 64, 9).cpu().numpy().astype(np.float64)


class L1Stage2Policy(Stage2Policy):
    def __init__(self, config, utility_rms=None, encoder=None):
        super().__init__(config, utility_rms, encoder or L1SensorEncoder())
        self.language_token_projection = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU())

    def encode_stage2(self, observations, utility_override=None):
        sensors = self.obs_encoder.encode_tokens(observations, self.normalizer)
        language = observations['language'][:, 0]
        z = self.obs_encoder.condition(sensors, language, self.condition_presence[:1].expand(len(sensors), -1))
        tokens = torch.cat((sensors, self.language_token_projection(language)[:, None]), 1)
        utility = self.utility_predictor(z.detach())
        detached = utility.detach() if utility_override is None else utility_override.detach()
        context = torch.cat((self.z_projection(z), self.utility_projection(detached)), -1)
        modulated = torch.stack([net(tokens[:, i], context) for i, net in enumerate(self.film_hypernets)]
                                + [tokens[:, 4], tokens[:, 5]], 1)
        query = z + self.query_projection(detached)
        attended, _ = self.cross_attention(query[:, None], modulated, modulated, need_weights=False)
        fused = self.fusion_norm(attended[:, 0])
        condition = self.fusion_norm(fused + self.fusion_ff(fused))
        return dict(tokens=tokens, z_t=z, u_hat=utility, modulated_tokens=modulated, conditioning=condition)

    def training_losses(self, batch, stage, denoising_sample=None):
        self.language_token_projection.requires_grad_(stage != 'A')
        return super().training_losses(batch, stage, denoising_sample)


class LabeledSixTaskDataset(Dataset):
    def __init__(self, source, labels, teacher):
        self.source = source
        document = json.loads((labels / 'contract.json').read_text())
        self.contract, self.contract_id = document['contract'], document['contract_id']
        if fingerprint(self.contract) != self.contract_id or self.contract['teacher'] != teacher:
            raise ValueError('Label/Teacher contract mismatch')
        if self.contract['dataset_manifest'] != source.manifest or self.contract['tasks'] != list(FINAL_TASKS):
            raise ValueError('Wrong dataset or six-task selection')
        complete = json.loads((labels / 'complete.json').read_text())
        if complete['contract_id'] != self.contract_id:
            raise ValueError('Incomplete matching labels')
        for name, digest in complete['artifacts'].items():
            if file_hash(labels / name) != digest:
                raise ValueError('Corrupt label array: ' + name)
        self.index = np.load(labels / 'window_index.npy')
        np.testing.assert_array_equal(self.index, selected_index(source))
        self.utility = np.load(labels / 'utility.npy')
        if self.utility.shape != (len(self.index), 8) or not np.isfinite(self.utility).all():
            raise ValueError('Invalid utility labels')
        self.utility_rms = np.maximum(np.sqrt(np.square(self.utility).mean(0)), 1e-6).astype(np.float32)
        np.testing.assert_allclose(self.utility_rms, complete['utility_rms'], rtol=1e-6)
        counts = np.bincount(self.index['task_index'])
        self.sample_weights = torch.from_numpy(1. / counts[self.index['task_index']])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index[index]
        item = self.source[int(row['source_index'])]
        if not item['observation_valid_mask'].all() or not item['action_loss_mask'].all() or item['task_index'] != row['task_index']:
            raise ValueError('Incomplete or misaligned selected window')
        item['utility'] = torch.from_numpy(self.utility[index].astype(np.float32))
        return item

    def epoch_indices(self, seed, epoch):
        return torch.multinomial(self.sample_weights, len(self), True, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def stopped():
    return Path(os.environ['UNIVTAC_STAGE2_STOP_FILE']).exists()


def generate(teacher, identity, source, config, output):
    (output / 'shards').mkdir(parents=True, exist_ok=True)
    if SixTaskDataset(source).manifest['cache_id'] != identity['cache_id']:
        raise ValueError('Teacher six-task data identity differs')
    index = selected_index(source)
    contract = dict(version='univtac-l1-e300-ema-six-task-k64-v1', teacher=identity,
        dataset_manifest=source.manifest, tasks=list(FINAL_TASKS), conditions=list(CONDITION_NAMES),
        horizon=16, n_obs_steps=2, k=64, timestep_seed=config['timestep_seed'], noise_seed=config['noise_seed'],
        shard_size=config['shard_size'], precision='strict FP32; TF32 disabled',
        window_index_sha256=hashlib.sha256(index.tobytes()).hexdigest())
    cid = fingerprint(contract)
    doc = dict(contract=contract, contract_id=cid)
    path = output / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != doc:
        raise ValueError('Label continuation identity changed')
    atomic_json(path, doc)
    if (output / 'complete.json').exists():
        return
    # Actual native full/removal forward equivalence, across all six tasks.
    ids = [int(index['source_index'][np.flatnonzero(index['task_index'] == t)[0]]) for t in np.unique(index['task_index'])]
    probe = move_batch(torch.utils.data.default_collate([source[i] for i in ids]), torch.device('cuda:0'))
    draws = [window_draws(i, config['timestep_seed'], config['noise_seed']) for i in range(len(ids))]
    steps, noise = np.stack([x[0] for x in draws]), np.stack([x[1] for x in draws])
    actual = evaluate_draws(teacher, probe, steps, noise, config['microbatch'])
    native = teacher.paired_denoising_losses(probe, torch.as_tensor(noise[:, 0], device='cuda:0'),
                                           torch.as_tensor(steps[:, 0], device='cuda:0'))['losses']
    np.testing.assert_allclose(actual[:, 0], native.cpu().numpy(), rtol=5e-5, atol=1e-7)
    atomic_json(output / 'preflight.json', dict(native_paired_loss_equal=True, tasks=list(FINAL_TASKS), conditions=9))
    del probe, actual, native
    started = time.monotonic()
    for start, stop in shard_ranges(len(index), config['shard_size']):
        if read_shard(output / 'shards', start, stop, cid, index[start:stop]) is not None:
            continue
        if stopped(): return
        rows = index[start:stop]
        draws = [window_draws(int(row['window_id']), config['timestep_seed'], config['noise_seed']) for row in rows]
        steps, noise = np.stack([x[0] for x in draws]), np.stack([x[1] for x in draws])
        loader = DataLoader(Subset(source, rows['source_index'].tolist()), batch_size=config['label_batch'],
                            num_workers=0, generator=torch.Generator().manual_seed(700042))
        parts, offset = [], 0
        for batch in loader:
            count = len(batch['action'])
            parts.append(evaluate_draws(teacher, move_batch(batch, torch.device('cuda:0')),
                         steps[offset:offset + count], noise[offset:offset + count], config['microbatch']))
            offset += count
        losses = np.concatenate(parts)
        commit_shard(output / 'shards', start, stop, cid, dict(window_index=rows, timesteps=steps,
                     noise=noise, losses=losses, **aggregate_losses(losses, steps)))
        event = dict(event='labels', done=stop, total=len(index), elapsed_seconds=time.monotonic() - started)
        print(json.dumps(event), flush=True)
        with (output / 'progress.jsonl').open('a') as f: f.write(json.dumps(event) + '\n')
    report = finalize(output, cid, index, config['shard_size'])
    atomic_json(output / 'COMPLETE.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, lambda *_: Path(os.environ['UNIVTAC_STAGE2_STOP_FILE']).touch())
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    teacher, identity = load_teacher(config['teacher'])
    source = FullSensorDataset(config['cache'], preload_ram=True)
    teacher.to('cuda:0')
    generate(teacher, identity, source, config, args.output / 'labels')
    if stopped(): return
    dataset = LabeledSixTaskDataset(source, args.output / 'labels', identity)
    teacher.cpu()
    torch.manual_seed(config['seed'])
    policy = L1Stage2Policy(teacher.config, dataset.utility_rms, copy.deepcopy(teacher.obs_encoder).requires_grad_(True))
    policy.initialize_from_teacher(teacher)
    del teacher
    gc.collect();torch.cuda.empty_cache()
    trainer = Stage2Trainer(policy, dataset, config, args.output / 'train')
    latest = args.output / 'train/checkpoints/latest.ckpt'
    if latest.exists(): trainer.restore(latest)
    report = args.output / 'train/preflight_report.json'
    if trainer.global_step == 0 and (not report.exists() or not json.loads(report.read_text())['passed']):
        trainer.preflight(100)
    elif trainer.global_step > 0 and (not report.exists() or not json.loads(report.read_text())['passed']):
        raise ValueError('Resume requires successful original GPU preflight')
    trainer.fit()
    complete = args.output / 'train/complete.json'
    if complete.exists():
        atomic_json(args.output / 'train/COMPLETE.json', json.loads(complete.read_text()))


def load_student(path, weights='ema', device='cpu'):
    """Restore an E300 L1 Student without its original data or Teacher file."""
    if weights not in ('raw', 'ema'): raise ValueError(weights)
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = saved['ema' if weights == 'ema' else 'model']
    model = L1Stage2Policy(saved['policy_config'], state['utility_rms'])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


if __name__ == '__main__': main()
