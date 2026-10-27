"""Join finalized E400 K64 labels to exact UniVTAC complete-window source rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import FullSensorDataset
from .schema import CONDITION_NAMES
from .utility import complete_window_index, file_hash, validate_contract


class Stage2Dataset(Dataset):
    """Require a committed, hash-checked complete label archive before student training."""

    def __init__(self, cache: str | Path, labels: str | Path, preload_ram: bool = True) -> None:
        root = Path(labels)
        document = json.loads((root / "contract.json").read_text())
        self.contract_id = validate_contract(document)
        self.contract = document["contract"]
        completion = json.loads((root / "complete.json").read_text())
        if completion["contract_id"] != self.contract_id or tuple(completion["dimensions"]) != CONDITION_NAMES[1:]:
            raise ValueError("Completion marker belongs to different labels")
        names = {"utility.npy", "utility_variance.npy", "weighted_losses.npy", "window_index.npy"}
        if set(completion["artifacts"]) != names:
            raise ValueError("Incomplete label artifacts")
        for name in names:
            if file_hash(root / name) != completion["artifacts"][name]:
                raise ValueError(f"Label artifact integrity failure: {name}")
        self.utility = np.load(root / "utility.npy", allow_pickle=False)
        self.index = np.load(root / "window_index.npy", allow_pickle=False)
        if (self.utility.shape != (len(self.index), 8) or not np.isfinite(self.utility).all()
                or len(self.index) != completion["windows"]):
            raise ValueError("Invalid complete label shapes or values")
        self.source = FullSensorDataset(cache, self.contract["horizon"], self.contract["n_obs_steps"], preload_ram)
        if self.source.manifest != self.contract["dataset_manifest"]:
            raise ValueError("Source cache differs from the frozen label source")
        np.testing.assert_array_equal(self.index, complete_window_index(
            self.source.manifest, self.contract["horizon"], self.contract["n_obs_steps"]))
        if hashlib.sha256(self.index.tobytes()).hexdigest() != self.contract["window_index_sha256"]:
            raise ValueError("Source/window index hash mismatch")
        self.utility_rms = np.maximum(np.sqrt(np.square(self.utility).mean(axis=0)), 1e-6).astype(np.float32)
        np.testing.assert_allclose(self.utility_rms, completion["utility_rms"], rtol=1e-6, atol=1e-10)
        task_indices = self.index["task_index"]
        counts = np.bincount(task_indices, minlength=len(self.source.manifest["tasks"]))
        if (counts == 0).any():
            raise ValueError("Every task must have complete labeled windows")
        self.sample_weights = torch.from_numpy(1.0 / counts[task_indices])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        row = self.index[index]
        item = self.source[int(row["source_index"])]
        if (item["task_index"] != row["task_index"] or item["source_index"] != row["source_index"]
                or not item["observation_valid_mask"].all() or not item["action_loss_mask"].all()):
            raise ValueError("Source/label alignment failure or incomplete window")
        item["utility"] = torch.from_numpy(self.utility[index].astype(np.float32, copy=True))
        item["window_id"] = int(row["window_id"])
        return item

    def epoch_indices(self, seed: int, epoch: int) -> list[int]:
        """Use equal expected mass per task and replay independently of worker prefetch."""
        generator = torch.Generator().manual_seed(seed + epoch)
        return torch.multinomial(self.sample_weights, len(self), replacement=True, generator=generator).tolist()
