#!/usr/bin/env python3
"""CPU regressions for paired K64 labels, source boundaries and committed resume."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.multitask_dp.generate_utility import evaluate_draws, load_teacher
from policy.multitask_dp.model import FullSensorPolicy, SensorEncoder
from policy.multitask_dp.schema import CONDITION_NAMES, IMAGE_KEYS, fingerprint
from policy.multitask_dp.utility import (WEIGHTS, aggregate_losses, commit_shard,
    complete_window_index, read_shard, shard_ranges, validate_contract, window_draws)


def tiny_policy() -> FullSensorPolicy:
    """Exercise the production temporal/fusion/U-Net path with small image features."""
    config = yaml.safe_load((ROOT / "configs/multitask_dp.yaml").read_text())
    config.update(token_dim=16, down_dims=[16, 32], diffusion_step_embed_dim=16, num_inference_steps=2)
    encoder = SensorEncoder(2, 16, nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8)), 8)
    return FullSensorPolicy(config, encoder).eval().requires_grad_(False)


def make_batch(count: int = 2) -> dict:
    """Create causal-history shapes and fully valid eight-joint targets."""
    obs = {key: torch.randint(256, (count, 2, 3, 8, 8), dtype=torch.uint8) for key in IMAGE_KEYS}
    obs.update(state=torch.randn(count, 2, 8), language=torch.randn(count, 2, 512))
    return {"obs": obs, "action": torch.randn(count, 16, 8),
            "action_loss_mask": torch.ones(count, 16, 8, dtype=torch.bool)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = {"episodes": [{"start": 0, "end": 20, "task_index": 0},
                              {"start": 20, "end": 50, "task_index": 1}], "total_steps": 50}
    index = complete_window_index(manifest)
    assert len(index) == 20 and index["source_index"].tolist() == list(range(1, 6)) + list(range(21, 36))
    for row in index:
        assert 1 <= row["timestep"] <= row["episode_length"] - 15
    assert shard_ranges(20, 7) == [(0, 7), (7, 14), (14, 20)]
    draws = [window_draws(i, 11, 22) for i in (0, 1)]
    times, noise = np.stack([draw[0] for draw in draws]), np.stack([draw[1] for draw in draws])
    np.testing.assert_array_equal(noise[1], window_draws(1, 11, 22)[1])
    assert not np.array_equal(noise[0], noise[1])
    synthetic = np.broadcast_to(np.arange(9, dtype=np.float64), (2, 64, 9)).copy()
    synthetic[:, :, 0] = 3
    reduced = aggregate_losses(synthetic, times)
    np.testing.assert_allclose(reduced["utility"], np.tile(np.arange(1, 9) - 3, (2, 1)), atol=1e-15)
    assert reduced["utility"][0, 0] < 0
    np.testing.assert_allclose(reduced["utility_variance"], 0, atol=1e-30)
    # This distinguishes weighted strata from the biased arithmetic draw mean.
    linear = np.repeat(times[:, :, None].astype(np.float64), 9, axis=2)
    expected = sum(weight * np.array([row[(row >= low) & (row < high)].mean() for row in times])
                   for (low, high), weight in zip(((0, 12), (12, 25), (25, 50), (50, 75), (75, 100)), WEIGHTS))
    np.testing.assert_allclose(aggregate_losses(linear, times)["weighted_losses"][:, 0], expected)
    try:
        aggregate_losses(synthetic, np.zeros_like(times))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid stratified draws accepted")
    print("PASS complete episode windows, restart-independent K64 draws, signed weighted labels/variance", flush=True)
    policy, batch = tiny_policy(), make_batch()
    actual = evaluate_draws(policy, batch, times, noise, 73)
    alternate = evaluate_draws(policy, batch, times, noise, 128)
    np.testing.assert_allclose(actual, alternate, rtol=3e-5, atol=3e-6)
    # Compare several vectorized draws against the original independent mask API.
    for draw in (0, 17, 63):
        expected_losses = policy.paired_denoising_losses(batch, torch.from_numpy(noise[:, draw]), torch.from_numpy(times[:, draw]))
        np.testing.assert_allclose(actual[:, draw], expected_losses["losses"].numpy(), rtol=3e-5, atol=3e-6)
    changed = copy.deepcopy(batch)
    changed["obs"]["head"].zero_()
    shared = policy.prepare_denoising_sample(batch, torch.from_numpy(noise[:, 0]), torch.from_numpy(times[:, 0]))
    head_removed = torch.ones(2, dtype=torch.long)
    torch.testing.assert_close(policy.compute_loss_per_sample(batch, head_removed, shared),
                               policy.compute_loss_per_sample(changed, head_removed, shared), rtol=0, atol=0)
    print("PASS vectorized-vs-original paired losses, microbatch agreement, removed sensor isolation", flush=True)
    shards = args.output / "shards"
    shards.mkdir()
    arrays = {"window_index": index[:2], "noise": noise, "timesteps": times,
              "losses": actual, **aggregate_losses(actual, times)}
    commit_shard(shards, 0, 2, "test", arrays)
    assert read_shard(shards, 0, 2, "test", index[:2]) is not None
    assert read_shard(shards, 2, 3, "test", index[2:3]) is None
    try:
        read_shard(shards, 0, 2, "different", index[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("Changed contract accepted")
    path = shards / "shard_000000_000002.npz"
    with path.open("ab") as stream:
        stream.write(b"corruption")
    try:
        read_shard(shards, 0, 2, "test", index[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt shard accepted")
    print("PASS committed shard roundtrip, incomplete resume, identity and corruption rejection", flush=True)
    report = {"cpu_checks": "passed"}
    if args.teacher:
        teacher, identity = load_teacher(args.teacher)
        report["teacher"] = identity
        del teacher
        print(f"PASS actual E400 EMA strict restore, normalizer and protocol; sha256={identity['sha256']}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
