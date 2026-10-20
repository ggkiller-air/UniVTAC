"""Causal episode-bounded windows and deterministic task-balanced sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import CACHE_VERSION, IMAGE_KEYS, fingerprint


def window_indices(anchor: int, start: int, end: int, history: int, horizon: int) -> tuple:
    """Keep the execution anchor at history-1 and mask all replicated edge padding."""
    observations = anchor - history + 1 + np.arange(history)
    actions = anchor - history + 1 + np.arange(horizon)
    return (np.clip(observations, start, end - 1), np.clip(actions, start, end - 1),
            (observations >= start) & (observations < end), (actions >= start) & (actions < end))


class FullSensorDataset(Dataset):
    """Share immutable decoded arrays through Linux fork; never decode inside workers."""

    def __init__(self, directory: str | Path, horizon: int = 16, n_obs_steps: int = 2,
                 preload_ram: bool = True) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["version"] != CACHE_VERSION:
            raise ValueError("Unsupported cache protocol")
        content = {key: value for key, value in self.manifest.items() if key != "cache_id"}
        if fingerprint(content) != self.manifest["cache_id"]:
            raise ValueError("Cache manifest integrity check failed")
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.episodes = self.manifest["episodes"]
        self.ends = np.asarray([episode["end"] for episode in self.episodes])
        self.arrays = {}
        for key in (*IMAGE_KEYS, "state", "action", "language"):
            array = np.load(self.directory / f"{key}.npy", mmap_mode=None if preload_ram else "r", allow_pickle=False)
            if key in IMAGE_KEYS:
                expected = (len(self), 3, self.manifest["image_size"], self.manifest["image_size"])
                if array.shape != expected or array.dtype != np.uint8:
                    raise ValueError(f"Invalid cached image array {key}")
            elif key in ("state", "action"):
                if array.shape != (len(self), 8) or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"Invalid cached joint array {key}")
            elif array.shape != (len(self.manifest["tasks"]), 512) or not np.isfinite(array).all():
                raise ValueError("Invalid cached task embeddings")
            array.flags.writeable = False
            self.arrays[key] = array
        self.task_indices = np.empty(len(self), dtype=np.int64)
        for episode in self.episodes:
            self.task_indices[episode["start"]:episode["end"]] = episode["task_index"]
        counts = np.bincount(self.task_indices, minlength=len(self.manifest["tasks"]))
        self.sample_weights = torch.from_numpy(1.0 / counts[self.task_indices])

    def __len__(self) -> int:
        return self.manifest["total_steps"]

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[int(np.searchsorted(self.ends, index, side="right"))]
        observed, actions, observation_valid, action_valid = window_indices(
            index, episode["start"], episode["end"], self.n_obs_steps, self.horizon,
        )
        observations = {key: torch.from_numpy(self.arrays[key][observed]) for key in IMAGE_KEYS}
        observations["state"] = torch.from_numpy(self.arrays["state"][observed])
        observations["language"] = torch.from_numpy(np.repeat(
            self.arrays["language"][episode["task_index"]][None], self.n_obs_steps, axis=0,
        ))
        return {"obs": observations, "action": torch.from_numpy(self.arrays["action"][actions]),
                "action_loss_mask": torch.from_numpy(np.repeat(action_valid[:, None], 8, axis=1)),
                "observation_valid_mask": torch.from_numpy(observation_valid),
                "task_index": episode["task_index"], "source_index": index}

    def epoch_indices(self, seed: int, epoch: int) -> list[int]:
        """Replay an entire epoch independently of worker prefetching and global RNG."""
        generator = torch.Generator().manual_seed(seed + epoch)
        return torch.multinomial(self.sample_weights, len(self), replacement=True, generator=generator).tolist()

    def normalization_statistics(self) -> dict:
        """Fit only the training cache; this run intentionally uses all episodes and no validation."""
        return {name: {"min": torch.from_numpy(values.min(axis=0)), "max": torch.from_numpy(values.max(axis=0))}
                for name in ("state", "action") for values in [self.arrays[name]]}
