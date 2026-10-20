#!/usr/bin/env python3
"""Bounded CPU regression diagnostics; all generated artifacts live under --output."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.multitask_dp.cache import decode_episode
from policy.multitask_dp.data import FullSensorDataset, window_indices
from policy.multitask_dp.model import FullSensorPolicy, Normalizer, SensorEncoder
from policy.multitask_dp.schema import CACHE_VERSION, CONDITION_NAMES, IMAGE_KEYS, condition_probabilities, fingerprint, presence_table
from policy.multitask_dp.train import ModelEMA, Trainer, learning_rate
from policy.multitask_dp.unet import ConditionalUnet1D


def make_fixture(output: Path) -> Path:
    """Use tiny real JPEG/HDF5 data to test next-state actions and cached channel order."""
    directory = output / "fixture_cache"
    directory.mkdir(parents=True)
    count, image_size = 6, 8
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[:] = [24, 80, 160]
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    raw = output / "fixture.hdf5"
    keys = {key: key + "/rgb" for key in IMAGE_KEYS}
    joints = np.arange(count * 9, dtype=np.float32).reshape(count, 9)
    with h5py.File(raw, "w") as store:
        store.create_dataset("embodiment/joint", data=joints)
        for key in keys.values():
            store.create_dataset(key, data=np.asarray([encoded] * count, dtype=f"S{len(encoded)}"))
    for key in (*IMAGE_KEYS, "state", "action"):
        shape = (count - 1, 3, image_size, image_size) if key in IMAGE_KEYS else (count - 1, 8)
        array = np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+", shape=shape,
                                         dtype=np.uint8 if key in IMAGE_KEYS else np.float32)
        del array
    episode = {"task": "fixture", "task_index": 0, "path": str(raw), "keys": keys, "start": 0, "end": count - 1}
    assert decode_episode((episode, str(directory), image_size)) == count - 1
    np.testing.assert_array_equal(np.load(directory / "state.npy"), joints[:-1, :8])
    np.testing.assert_array_equal(np.load(directory / "action.npy"), joints[1:, :8])
    np.testing.assert_array_equal(np.load(directory / "head.npy")[0], np.moveaxis(decoded, -1, 0))
    np.save(directory / "language.npy", np.ones((1, 512), dtype=np.float32) / np.sqrt(512))
    manifest = {"version": CACHE_VERSION, "episodes": [episode], "image_size": image_size,
                "tasks": ["fixture"], "total_steps": count - 1}
    manifest["cache_id"] = fingerprint(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def tiny_policy(config: dict) -> FullSensorPolicy:
    """Retain actual fusion and diffusion interfaces with an inexpensive diagnostic encoder."""
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 8))
    encoder = SensorEncoder(config["n_obs_steps"], config["token_dim"], backbone, 8)
    return FullSensorPolicy(config, encoder)


def assert_nested_equal(actual: object, expected: object) -> None:
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_nested_equal(first, second)
    else:
        assert actual == expected


def check_unet_reference(source: Path | None) -> None:
    """Require state-dict and forward equality to the pinned AdaViP reference when supplied."""
    if source is None:
        return
    sys.path.insert(0, str(source))
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D as ReferenceUnet

    current = ConditionalUnet1D(8, 16, 16, (16, 32, 64)).eval()
    reference = ReferenceUnet(input_dim=8, global_cond_dim=16, diffusion_step_embed_dim=16,
                              down_dims=[16, 32, 64], kernel_size=5, n_groups=8, cond_predict_scale=True).eval()
    reference.load_state_dict(current.state_dict(), strict=True)
    sample = torch.randn(2, 16, 8)
    timesteps, condition = torch.tensor([1, 5]), torch.randn(2, 16)
    assert torch.equal(current(sample, timesteps, condition), reference(sample, timesteps, global_cond=condition))
    print("PASS reference U-Net state-dict and forward bitwise equality", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = yaml.safe_load((ROOT / "configs/multitask_dp.yaml").read_text())
    config.update(batch_size=2, num_epochs=2, num_workers=0, horizon=8, token_dim=16,
                  down_dims=[16, 32, 64], diffusion_step_embed_dim=16, n_action_steps=4,
                  precision="fp32", checkpoint_every_epochs=1, checkpoint_every_steps=100,
                  checkpoint_first_step=100, preflight_steps=100, preflight_window=20)
    cache = make_fixture(arguments.output)
    config["cache_dir"] = str(cache)
    dataset = FullSensorDataset(cache, config["horizon"], config["n_obs_steps"])
    first, last = dataset[0], dataset[len(dataset) - 1]
    assert first["observation_valid_mask"].tolist() == [False, True]
    assert first["action_loss_mask"][:, 0].tolist() == [False, True, True, True, True, True, False, False]
    assert last["action_loss_mask"][:, 0].tolist() == [True, True, False, False, False, False, False, False]
    observed, actions, obs_valid, action_valid = window_indices(5, 5, 8, 2, 8)
    assert observed.tolist() == [5, 5] and actions.tolist() == [5, 5, 6, 7, 7, 7, 7, 7]
    assert not obs_valid[0] and action_valid.sum() == 3
    assert dataset.epoch_indices(42, 0) == dataset.epoch_indices(42, 0)
    balanced = copy.copy(dataset)
    balanced.sample_weights = torch.tensor([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
    probability = balanced.sample_weights / balanced.sample_weights.sum()
    torch.testing.assert_close(probability[:2].sum(), probability[2:].sum())
    print("PASS actual JPEG/BGR decode, eight-joint next-action alignment, causal windows, padding masks, sampling", flush=True)
    normalizer = Normalizer()
    statistics = dataset.normalization_statistics()
    statistics["state"]["max"][0] = statistics["state"]["min"][0]
    normalizer.fit(statistics)
    assert normalizer.state_scale[0] == 1
    values = torch.randn(3, 8)
    torch.testing.assert_close(normalizer.unnormalize("action", normalizer.normalize("action", values)), values, atol=1e-5, rtol=1e-5)
    assert presence_table().shape == (9, 6)
    np.testing.assert_allclose(condition_probabilities(0.8, [1] * 8), [0.8] + [0.025] * 8)
    for invalid in ([0] * 8, [-1] * 8, [1] * 9):
        try:
            condition_probabilities(0.8, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mask weights accepted")
    policy = tiny_policy(config).eval()
    policy.normalizer.fit(dataset.normalization_statistics())
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    noise, timesteps = torch.randn_like(batch["action"]), torch.tensor([3, 7])
    shared = policy.prepare_denoising_sample(batch, noise, timesteps)
    paired = policy.paired_denoising_losses(batch, noise, timesteps)
    assert paired["losses"].shape == (2, len(CONDITION_NAMES))
    for condition_index in range(len(CONDITION_NAMES)):
        independent = policy.compute_loss_per_sample(batch, torch.full((2,), condition_index), shared)
        torch.testing.assert_close(paired["losses"][:, condition_index], independent, atol=0, rtol=0)
    perturbed = copy.deepcopy(batch)
    perturbed["action"][~batch["action_loss_mask"]] = 999999
    alternate = policy.prepare_denoising_sample(perturbed, noise, timesteps)
    assert torch.equal(shared.noisy_action, alternate.noisy_action)
    policy.train()
    policy.compute_loss(batch, torch.zeros(2, dtype=torch.long)).backward()
    for key in IMAGE_KEYS:
        assert policy.obs_encoder.image_encoders[key][-1].weight.grad.abs().sum() > 0
    for key in ("state_projection", "language_projection"):
        assert getattr(policy.obs_encoder, key)[0].weight.grad.abs().sum() > 0
    try:
        policy.compute_loss(batch, torch.tensor([-1, 0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Negative condition index accepted")
    policy.eval()
    policy.config["num_inference_steps"] = 2
    prediction = policy.predict_action(batch["obs"])
    assert prediction["action"].shape == (2, config["n_action_steps"], 8)
    assert torch.isfinite(prediction["action"]).all()
    print("PASS train normalization, all nine conditions, paired per-sample noise and loss masks, all sensor gradients", flush=True)
    ema = ModelEMA(policy)
    ema.step(policy)
    assert ema.optimization_step == 1
    assert_nested_equal(ema.model.state_dict(), policy.state_dict())
    check_unet_reference(arguments.reference_source)
    torch.manual_seed(123)
    initial = tiny_policy(config)
    uninterrupted = Trainer(config, arguments.output / "uninterrupted", "cpu", copy.deepcopy(initial), dataset)
    uninterrupted.run()
    interrupted = Trainer(config, arguments.output / "interrupted", "cpu", copy.deepcopy(initial), dataset)
    interrupted.run(stop_after_steps=1)
    resumed = Trainer(config, arguments.output / "resumed", "cpu", copy.deepcopy(initial), dataset)
    resumed.resume(arguments.output / "interrupted/checkpoints/latest.ckpt")
    resumed.run()
    assert resumed.global_step == uninterrupted.global_step == 6
    assert resumed.ema.optimization_step == 6
    assert_nested_equal(resumed.policy.state_dict(), uninterrupted.policy.state_dict())
    assert_nested_equal(resumed.ema.model.state_dict(), uninterrupted.ema.model.state_dict())
    assert_nested_equal(resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict())
    worker_config = {**config, "num_workers": 2}
    with_workers = Trainer(worker_config, arguments.output / "workers", "cpu", copy.deepcopy(initial), dataset)
    with_workers.run()
    assert_nested_equal(with_workers.policy.state_dict(), uninterrupted.policy.state_dict())
    assert_nested_equal(with_workers.ema.model.state_dict(), uninterrupted.ema.model.state_dict())
    worker_interrupted = Trainer(worker_config, arguments.output / "worker_interrupted", "cpu", copy.deepcopy(initial), dataset)
    worker_interrupted.run(stop_after_steps=1)
    worker_resumed = Trainer(worker_config, arguments.output / "worker_resumed", "cpu", copy.deepcopy(initial), dataset)
    worker_resumed.resume(arguments.output / "worker_interrupted/checkpoints/latest.ckpt")
    worker_resumed.run()
    assert_nested_equal(worker_resumed.policy.state_dict(), with_workers.policy.state_dict())
    assert_nested_equal(worker_resumed.ema.model.state_dict(), with_workers.ema.model.state_dict())
    assert_nested_equal(worker_resumed.optimizer.state_dict(), with_workers.optimizer.state_dict())
    assert len(list((arguments.output / "uninterrupted/checkpoints").glob("epoch*.ckpt"))) == 2
    assert learning_rate(config, 0, 600) == config["learning_rate"] / config["warmup_steps"]
    bad_config = {**config, "batch_size": 3}
    incompatible = Trainer(bad_config, arguments.output / "incompatible", "cpu", copy.deepcopy(initial), dataset)
    try:
        incompatible.resume(arguments.output / "interrupted/checkpoints/latest.ckpt")
    except ValueError:
        pass
    else:
        raise AssertionError("Incompatible resume accepted")
    gate_config = {**config, "preflight_steps": 4, "preflight_window": 2, "min_samples_per_second": 1e12}
    gated = Trainer(gate_config, arguments.output / "gate_failed", "cpu", copy.deepcopy(initial), dataset)
    try:
        gated.run()
    except RuntimeError as error:
        assert "Throughput preflight failed" in str(error)
    else:
        raise AssertionError("Failed throughput gate continued training")
    assert gated.global_step == 4 and (gated.checkpoints / "latest.ckpt").exists()
    print("PASS interrupted/resumed raw+EMA+Adam bitwise equality with 0/2 workers, update budget, retention, incompatible-resume rejection, failed-gate exit", flush=True)
    print("ALL CPU DIAGNOSTICS PASSED", flush=True)


if __name__ == "__main__":
    main()
