"""Verify six-task filtering, L1 causality, mask invariance and exact CPU resume."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
import yaml

from policy.multitask_dp.data import FullSensorDataset
from policy.multitask_dp.l1 import FINAL_TASKS, L1Policy, L1SensorEncoder, SixTaskDataset
from policy.multitask_dp.schema import CACHE_VERSION, IMAGE_KEYS, TASK_TEXTS, fingerprint
from policy.multitask_dp.train import Trainer


def fixture(root: Path) -> SixTaskDataset:
    """Give excluded tasks extreme state/action values to detect statistics leakage."""
    root.mkdir()
    tasks = list(TASK_TEXTS)
    episodes = [{"task": task, "task_index": i, "start": i * 4, "end": (i + 1) * 4}
                for i, task in enumerate(tasks)]
    manifest = {"version": CACHE_VERSION, "tasks": tasks, "episodes": episodes,
                "total_steps": 32, "image_size": 8}
    manifest["cache_id"] = fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(42)
    for key in IMAGE_KEYS:
        np.save(root / f"{key}.npy", rng.integers(0, 256, (32, 3, 8, 8), dtype=np.uint8))
    for key in ("state", "action"):
        values = rng.normal(size=(32, 8)).astype(np.float32)
        values[:8] = 100000
        np.save(root / f"{key}.npy", values)
    np.save(root / "language.npy", rng.normal(size=(8, 512)).astype(np.float32))
    return SixTaskDataset(FullSensorDataset(root, preload_ram=False))


def model(config: dict) -> L1Policy:
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
    return L1Policy(config, L1SensorEncoder(backbone=backbone, image_feature_dim=8))


def assert_nested(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested(a, b)
    else:
        assert left == right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/l1_six_task.yaml").read_text())
    config.update(batch_size=4, num_epochs=2, num_workers=0, precision="fp32", down_dims=[16, 32],
                  diffusion_step_embed_dim=16, preflight_steps=4, preflight_window=2,
                  min_samples_per_second=0, max_data_wait_fraction=1, checkpoint_every_epochs=1,
                  checkpoint_first_step=1)
    # Keep diagnostic fixtures: open read-only mmaps can delay NFS unlinking.
    with nullcontext(tempfile.mkdtemp(dir=args.output)) as tmp:
        root = Path(tmp)
        dataset = fixture(root / "cache")
        assert len(dataset) == 24 and len(dataset.manifest["episodes"]) == 6
        assert set(dataset.manifest["tasks"]) == set(FINAL_TASKS)
        assert all(e["task"] in FINAL_TASKS for e in dataset.manifest["episodes"])
        assert dataset.normalization_statistics()["state"]["max"].max() < 100
        assert dataset.epoch_indices(42, 1) == dataset.epoch_indices(42, 1)
        mass = dataset.sample_weights.numpy()
        for task in range(2, 8):
            np.testing.assert_allclose(mass[dataset.task_indices == task].sum(), 1)
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        torch.manual_seed(42)
        policy = model(config)
        policy.normalizer.fit(dataset.normalization_statistics())
        tokens = policy.obs_encoder.encode_tokens(batch["obs"], policy.normalizer)
        assert tokens.shape == (2, 5, 256)
        changed = copy.deepcopy(batch["obs"])
        changed["language"] += 10
        torch.testing.assert_close(tokens, policy.obs_encoder.encode_tokens(changed, policy.normalizer), rtol=0, atol=0)
        flags = policy.condition_presence[0].expand(2, -1)
        condition = policy.obs_encoder.condition(tokens, batch["obs"]["language"][:, 0], flags)
        assert condition.shape == (2, 256)
        assert not torch.equal(condition, policy.obs_encoder.condition(tokens, changed["language"][:, 0], flags))
        policy.compute_loss(batch).backward()
        assert next(policy.model.parameters()).grad is not None
        assert policy.obs_encoder.language_head[0].weight.grad.abs().sum() > 0
        assert all(next(branch.parameters()).grad.abs().sum() > 0 for branch in policy.obs_encoder.image_encoders.values())
        try:
            policy.compute_loss(batch, torch.ones(2, dtype=torch.long))
        except ValueError:
            pass
        else:
            raise AssertionError("Training accepted input dropout")
        policy.eval()
        for sensor, key in enumerate((*IMAGE_KEYS, "state", "language")):
            removed = policy.condition_presence[sensor + 1].expand(2, -1)
            before = policy.obs_encoder.condition(tokens, batch["obs"]["language"][:, 0], removed)
            modified = copy.deepcopy(batch["obs"])
            modified[key] = (255 - modified[key]) if key in IMAGE_KEYS else modified[key] + 100
            after = policy.obs_encoder.condition(policy.obs_encoder.encode_tokens(modified, policy.normalizer),
                                                  modified["language"][:, 0], removed)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert policy.paired_denoising_losses(batch)["utility"].shape == (2, 8)
        assert policy.predict_action(batch["obs"])["action"].shape == (2, 8, 8)
        torch.manual_seed(42)
        continuous = Trainer(config, root / "continuous", "cpu", model(config), dataset)
        continuous.run(stop_after_steps=4)
        torch.manual_seed(42)
        first = Trainer(config, root / "resumed", "cpu", model(config), dataset)
        first.run(stop_after_steps=2)
        restored = Trainer(config, root / "resumed", "cpu", model(config), dataset)
        restored.resume(root / "resumed/checkpoints/latest.ckpt")
        restored.run(stop_after_steps=4)
        assert_nested(continuous.policy.state_dict(), restored.policy.state_dict())
        assert_nested(continuous.ema.model.state_dict(), restored.ema.model.state_dict())
        assert_nested(continuous.optimizer.state_dict(), restored.optimizer.state_dict())
    (args.output / "report.json").write_text(json.dumps({"status": "passed", "checks": [
        "six_task_filter_and_statistics", "balanced_sampler", "language_excluded_from_z",
        "late_language_conditioning", "all_encoder_action_gradients", "no_training_dropout",
        "six_sensor_removal_invariance", "paired_eight_dim_utility", "inference_shape",
        "exact_raw_ema_adam_resume"], "device": "cpu"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
