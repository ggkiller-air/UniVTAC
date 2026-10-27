#!/usr/bin/env python3
"""CPU coverage of the UniVTAC Transformer port, label alignment and optimizer resume."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from check_multitask_dp import assert_nested_equal
from check_utility_e400 import make_batch, tiny_policy
from policy.multitask_dp.data import FullSensorDataset
from policy.multitask_dp.generate_utility import finalize, load_teacher
from policy.multitask_dp.model import FullSensorPolicy, SensorEncoder
from policy.multitask_dp.schema import CACHE_VERSION, CONDITION_NAMES, IMAGE_KEYS, fingerprint
from policy.multitask_dp.stage2 import Stage2Policy, make_optimizer_scheduler, set_stage_trainability, stage_for_epoch
from policy.multitask_dp.stage2_data import Stage2Dataset
from policy.multitask_dp.train_stage2 import Stage2Trainer, build_policy
from policy.multitask_dp.train import capture_rng
from policy.multitask_dp.utility import (aggregate_losses, atomic_json, commit_shard,
    complete_window_index, shard_ranges, window_draws)


def make_teacher() -> FullSensorPolicy:
    """Keep 256D tokens with the production interfaces and a tiny action backbone."""
    base = tiny_policy()
    config = dict(base.config, token_dim=256)
    encoder = SensorEncoder(2, 256, copy.deepcopy(base.obs_encoder.image_encoders["head"]), 8)
    return FullSensorPolicy(config, encoder).eval().requires_grad_(False)


def seed_everything(seed: int) -> None:
    """Reset all CPU streams used by the actual trainer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_labeled_fixture(root: Path) -> tuple[Path, Path]:
    """Create a genuine source cache and committed label shards for two short episodes."""
    cache, labels = root / "cache", root / "labels"
    cache.mkdir()
    (labels / "shards").mkdir(parents=True)
    rng = np.random.default_rng(45)
    for key in IMAGE_KEYS:
        np.save(cache / f"{key}.npy", rng.integers(0, 256, size=(50, 3, 8, 8), dtype=np.uint8))
    for key in ("state", "action"):
        np.save(cache / f"{key}.npy", rng.standard_normal((50, 8), dtype=np.float32))
    np.save(cache / "language.npy", rng.standard_normal((2, 512), dtype=np.float32))
    manifest = {"version": CACHE_VERSION, "image_size": 8, "total_steps": 50, "tasks": ["a", "b"],
                "episodes": [{"start": 0, "end": 20, "task_index": 0}, {"start": 20, "end": 50, "task_index": 1}]}
    manifest["cache_id"] = fingerprint(manifest)
    atomic_json(cache / "manifest.json", manifest)
    index = complete_window_index(manifest)
    contract = {"teacher": {"epoch": 400, "weights": "ema", "sha256": "fixture"},
                "conditions": list(CONDITION_NAMES), "dataset_manifest": manifest,
                "horizon": 16, "n_obs_steps": 2, "shard_size": 7,
                "window_index_sha256": hashlib.sha256(index.tobytes()).hexdigest()}
    contract_id = fingerprint(contract)
    atomic_json(labels / "contract.json", {"contract": contract, "contract_id": contract_id})
    for start, stop in shard_ranges(len(index), 7):
        draws = [window_draws(i, 11, 22) for i in range(start, stop)]
        times = np.stack([draw[0] for draw in draws])
        noise = np.stack([draw[1] for draw in draws])
        losses = rng.uniform(0, 1, (stop - start, 64, 9))
        commit_shard(labels / "shards", start, stop, contract_id,
                     {"window_index": index[start:stop], "timesteps": times, "noise": noise,
                      "losses": losses, **aggregate_losses(losses, times)})
    finalize(labels, contract_id, index, 7)
    return cache, labels


def check_model() -> None:
    seed_everything(12)
    teacher = make_teacher()
    policy = build_policy(teacher, np.arange(1, 9, dtype=np.float32))
    for name, value in teacher.obs_encoder.state_dict().items():
        torch.testing.assert_close(policy.obs_encoder.state_dict()[name], value, rtol=0, atol=0)
    assert policy.obs_encoder.sensor_identity.data_ptr() != teacher.obs_encoder.sensor_identity.data_ptr()
    assert not torch.equal(next(policy.model.parameters()), next(teacher.model.parameters()))
    batch = make_batch()
    batch["utility"] = torch.randn(2, 8)
    encoded = policy.encode_stage2(batch["obs"])
    assert encoded["tokens"].shape == (2, 6, 256)
    assert encoded["u_hat"].shape == (2, 8)
    assert encoded["conditioning"].shape == (2, 256)
    assert len(policy.film_hypernets) == 4
    assert len({hyper.network[0].weight.data_ptr() for hyper in policy.film_hypernets}) == 4
    torch.testing.assert_close(encoded["modulated_tokens"], encoded["tokens"], rtol=0, atol=0)
    assert torch.count_nonzero(policy.query_projection[-1].weight) == 0
    # Changed FiLM really affects only its own image token, never state/language.
    with torch.no_grad():
        policy.film_hypernets[0].network[-1].bias[256:].fill_(0.5)
    modulated = policy.encode_stage2(batch["obs"])
    torch.testing.assert_close(modulated["modulated_tokens"][:, 0], modulated["tokens"][:, 0] + 0.5)
    torch.testing.assert_close(modulated["modulated_tokens"][:, 1:], modulated["tokens"][:, 1:], rtol=0, atol=0)
    result = policy.training_losses(batch, "C")
    torch.testing.assert_close(result["utility_target"], batch["utility"] / torch.arange(1, 9))
    result["utility_loss"].backward(retain_graph=True)
    assert policy.utility_predictor[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.obs_encoder.parameters())
    assert all(p.grad is None for p in policy.model.parameters())
    policy.zero_grad(set_to_none=True)
    result["action_loss"].backward()
    assert all(p.grad is None for p in policy.utility_predictor.parameters())
    assert policy.obs_encoder.fusion[0].weight.grad.abs().sum() > 0
    for key in IMAGE_KEYS:
        assert policy.obs_encoder.image_encoders[key][-1].weight.grad.abs().sum() > 0
    for hyper in policy.film_hypernets:
        assert hyper.network[-1].weight.grad.abs().sum() > 0
    sample = policy.prepare_denoising_sample(batch)
    expected = policy.training_losses(batch, "C", sample)["action_loss"]
    torch.testing.assert_close(policy.compute_loss(batch, denoising_sample=sample), expected, rtol=0, atol=0)
    assert [stage_for_epoch(i) for i in (1, 2, 6, 7, 600)] == ["A", "B", "B", "C", "C"]
    for stage in ("A", "B", "C"):
        set_stage_trainability(policy, stage)
        assert all(p.requires_grad == (stage == "C") for p in policy.obs_encoder.parameters())
        assert all(p.requires_grad == (stage != "A") for p in policy.model.parameters())
        assert all(p.requires_grad for p in policy.utility_predictor.parameters())
    optimizer, scheduler = make_optimizer_scheduler(policy, 10)
    assert optimizer.param_groups[0]["lr"] == 0
    encoder_rate = scheduler.lr_lambdas[0]
    assert encoder_rate(59) == 0 and encoder_rate(60) > 0
    assert encoder_rate(559) > encoder_rate(60)
    policy.eval()
    pred = policy.predict_action(batch["obs"])
    assert pred["action"].shape == (2, 8, 8) and torch.isfinite(pred["action"]).all()
    clone = build_policy(teacher, np.ones(8, np.float32))
    clone.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(clone.utility_rms, policy.utility_rms, rtol=0, atol=0)
    torch.testing.assert_close(clone.encode_stage2(batch["obs"])["conditioning"],
                               policy.encode_stage2(batch["obs"])["conditioning"], rtol=0, atol=0)
    for rms in (torch.ones(9), torch.zeros(8), torch.full((8,), float("nan"))):
        try:
            Stage2Policy(teacher.config, rms, copy.deepcopy(teacher.obs_encoder))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid RMS accepted")
    print("PASS teacher inheritance/fresh DP, 6 tokens/8 utilities/4 independent FiLMs, identity and gradient isolation", flush=True)
    print("PASS A/B/C gates, actual Transformer denoising/inference paths, checkpoint conditioning parity", flush=True)


def check_data_and_resume(output: Path) -> None:
    cache, labels = make_labeled_fixture(output)
    dataset = Stage2Dataset(cache, labels, preload_ram=False)
    assert len(dataset) == 20
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["source_index"] == dataset.index[i]["source_index"]
        np.testing.assert_allclose(sample["utility"], dataset.utility[i], rtol=1e-6)
    task_indices = dataset.index["task_index"]
    assert abs(float(dataset.sample_weights[task_indices == 0].sum()) - float(dataset.sample_weights[task_indices == 1].sum())) < 1e-14
    assert dataset.epoch_indices(42, 1) == dataset.epoch_indices(42, 1)
    torch.testing.assert_close(torch.from_numpy(dataset.utility_rms),
        torch.tensor(np.maximum(np.sqrt((dataset.utility**2).mean(axis=0)), 1e-6), dtype=torch.float32))
    # Compare interrupted versus uninterrupted *actual* optimizer updates in all phases.
    for workers, epoch in ((0, 1), (0, 2), (0, 7), (2, 7)):
        config = {"batch_size": 3, "num_epochs": 8, "num_workers": workers, "seed": 42}
        tag = f"workers{workers}_phase{stage_for_epoch(epoch)}"
        seed_everything(77)
        teacher = make_teacher()
        initial = build_policy(teacher, dataset.utility_rms)
        full = Stage2Trainer(copy.deepcopy(initial), dataset, config, output / f"full_{tag}", "cpu")
        full.epoch = epoch
        seed_everything(99)
        full.fit(max_steps=3)
        split = Stage2Trainer(copy.deepcopy(initial), dataset, config, output / f"split_{tag}", "cpu")
        split.epoch = epoch
        seed_everything(99)
        split.fit(max_steps=1)
        restored = Stage2Trainer(copy.deepcopy(initial), dataset, config, output / f"split_{tag}", "cpu")
        restored.restore(output / f"split_{tag}" / "checkpoints/latest.ckpt")
        restored.fit(max_steps=2)
        assert restored.global_step == full.global_step == 3
        assert_nested_equal(restored.policy.state_dict(), full.policy.state_dict())
        assert_nested_equal(restored.ema.state_dict(), full.ema.state_dict())
        assert_nested_equal(restored.optimizer.state_dict(), full.optimizer.state_dict())
        assert_nested_equal(restored.scheduler.state_dict(), full.scheduler.state_dict())
    print("PASS complete source/label join, signed RMS, balanced replay; exact raw/EMA/Adam/scheduler resume in A/B/C, workers=0/2", flush=True)
    # Combined preflight must leave the subsequent formal run identical to fresh.
    config = {"batch_size": 3, "num_epochs": 8, "num_workers": 0, "seed": 42}
    teacher = make_teacher()
    initial = build_policy(teacher, dataset.utility_rms)
    checked = Stage2Trainer(copy.deepcopy(initial), dataset, config, output / "preflight", "cpu")
    seed_everything(101)
    rng = capture_rng()
    report = checked.preflight(steps=7)
    assert report["passed"] and checked.global_step == checked.cursor == 0
    assert not checked.optimizer.state
    assert_nested_equal(checked.policy.state_dict(), initial.state_dict())
    assert_nested_equal(checked.ema.state_dict(), initial.state_dict())
    torch.testing.assert_close(torch.get_rng_state(), rng["torch"], rtol=0, atol=0)
    assert random.getstate() == rng["python"]
    np.testing.assert_array_equal(np.random.get_state()[1], rng["numpy"][1])
    checked.fit(max_steps=3)
    baseline = Stage2Trainer(copy.deepcopy(initial), dataset, config, output / "no_preflight", "cpu")
    seed_everything(101)
    baseline.fit(max_steps=3)
    assert_nested_equal(checked.policy.state_dict(), baseline.policy.state_dict())
    assert_nested_equal(checked.ema.state_dict(), baseline.ema.state_dict())
    assert_nested_equal(checked.optimizer.state_dict(), baseline.optimizer.state_dict())
    assert_nested_equal(checked.scheduler.state_dict(), baseline.scheduler.state_dict())
    print("PASS A/B/C combined preflight reset of model/EMA/Adam/scheduler/RNG; formal updates equal fresh run", flush=True)
    # Resume the retained checkpoint at an already completed epoch boundary.
    config = {"batch_size": 3, "num_epochs": 1, "num_workers": 0, "seed": 42,
              "archive_dir": str((output / "external_archive").resolve())}
    teacher = make_teacher()
    initial = build_policy(teacher, dataset.utility_rms)
    boundary = Stage2Trainer(copy.deepcopy(initial), dataset, config, output / "boundary", "cpu")
    boundary.fit()
    retained = output / "boundary/checkpoints/epoch0001.ckpt"
    retained_hash = hashlib.sha256(retained.read_bytes()).hexdigest()
    archived = output / "external_archive/epoch0001.ckpt"
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == retained_hash
    resumed = Stage2Trainer(copy.deepcopy(initial), dataset, config, output / "boundary", "cpu")
    resumed.restore(retained)
    resumed.fit()
    assert resumed.global_step == boundary.global_step == 7
    assert_nested_equal(resumed.policy.state_dict(), boundary.policy.state_dict())
    assert retained_hash == hashlib.sha256(retained.read_bytes()).hexdigest()
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == retained_hash
    print("PASS completed-epoch boundary resume and immutable snapshot retention", flush=True)
    # Hash rejection must happen before any student optimization.
    with (labels / "utility.npy").open("ab") as stream:
        stream.write(b"corrupt")
    try:
        Stage2Dataset(cache, labels)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt label archive accepted")
    print("PASS corrupted training labels rejected", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    check_model()
    check_data_and_resume(args.output)
    report = {"cpu_checks": "passed", "gpu_training_launched": False}
    if args.teacher:
        teacher, identity = load_teacher(args.teacher)
        policy = build_policy(teacher, np.ones(8, np.float32))
        for name, value in teacher.obs_encoder.state_dict().items():
            torch.testing.assert_close(policy.obs_encoder.state_dict()[name], value, rtol=0, atol=0)
        for name, value in teacher.normalizer.state_dict().items():
            torch.testing.assert_close(policy.normalizer.state_dict()[name], value, rtol=0, atol=0)
        assert not torch.equal(next(policy.model.parameters()), next(teacher.model.parameters()))
        assert policy.obs_encoder.fusion[0].weight.shape == (256, 1542)
        report["teacher"] = identity
        print("PASS actual full-size E400 EMA encoder/normalizer inheritance and fresh 8D DP", flush=True)
    atomic_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
