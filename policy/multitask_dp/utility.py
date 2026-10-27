"""Deterministic stratified K64 utility labels and checked, resumable shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .schema import CONDITION_NAMES, fingerprint

STRATA = ((0, 12), (12, 25), (25, 50), (50, 75), (75, 100))
BASE_COUNTS = (4, 3, 3, 3, 3)
WEIGHTS = np.array([0.12, 0.13, 0.25, 0.25, 0.25], dtype=np.float64)
INDEX_DTYPE = np.dtype([(key, "<i8") for key in (
    "window_id", "task_index", "episode_index", "timestep", "source_index", "episode_length",
)])


def file_hash(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Publish only fully flushed metadata."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def complete_window_index(manifest: dict, horizon: int = 16, history: int = 2) -> np.ndarray:
    """Enumerate anchors whose history and action horizon fit inside one episode."""
    if horizon < history or history < 1:
        raise ValueError("Invalid horizon/history")
    rows = []
    previous = 0
    for episode_index, episode in enumerate(manifest["episodes"]):
        start, end = episode["start"], episode["end"]
        if start != previous or end <= start:
            raise ValueError("Episode ranges must be contiguous and nonempty")
        length = end - start
        for timestep in range(history - 1, length - horizon + history):
            rows.append((len(rows), episode["task_index"], episode_index,
                         timestep, start + timestep, length))
        previous = end
    if previous != manifest["total_steps"] or not rows:
        raise ValueError("Invalid or empty complete-window index")
    return np.asarray(rows, dtype=INDEX_DTYPE)


def window_draws(window_id: int, timestep_seed: int, noise_seed: int,
                 horizon: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Keep each window's CPU Gaussian/timestep streams independent of batch and restart."""
    if window_id < 0:
        raise ValueError("Window id must be nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([timestep_seed, window_id]))
    timesteps = np.concatenate([
        np.concatenate([rng.integers(low, high, count, dtype=np.int64)
                        for (low, high), count in zip(STRATA, BASE_COUNTS)]) for _ in range(4)
    ])
    noise_rng = np.random.default_rng(np.random.SeedSequence([noise_seed, window_id]))
    noise = noise_rng.standard_normal((64, horizon, 8), dtype=np.float32)
    return timesteps, noise


def aggregate_losses(losses: np.ndarray, timesteps: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate a uniform-timestep expectation using stratum weights, preserving sign."""
    losses = np.asarray(losses, dtype=np.float64)
    if losses.shape != (len(losses), 64, len(CONDITION_NAMES)) or timesteps.shape != losses.shape[:2]:
        raise ValueError("Expected [windows,64,9] losses and matching timesteps")
    if not len(losses) or not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("Losses must be finite and nonnegative")
    blocks = timesteps.reshape(len(losses), 4, 16)
    offset = 0
    for (low, high), count in zip(STRATA, BASE_COUNTS):
        draws = blocks[:, :, offset:offset + count]
        if not ((draws >= low) & (draws < high)).all():
            raise ValueError("Invalid K64 stratum draw order")
        offset += count
    weighted = np.zeros((len(losses), len(CONDITION_NAMES)), dtype=np.float64)
    variance = np.zeros((len(losses), len(CONDITION_NAMES) - 1), dtype=np.float64)
    halves = []
    for first, last in ((0, 32), (32, 64), (0, 64)):
        total = np.zeros_like(weighted)
        for (low, high), weight in zip(STRATA, WEIGHTS):
            mask = (timesteps[:, first:last] >= low) & (timesteps[:, first:last] < high)
            selected = losses[:, first:last][mask].reshape(len(losses), -1, len(CONDITION_NAMES))
            total += weight * selected.mean(axis=1)
            if last - first == 64:
                differences = selected[:, :, 1:] - selected[:, :, :1]
                variance += weight**2 * differences.var(axis=1, ddof=1) / selected.shape[1]
        if last - first == 64:
            weighted = total
        else:
            halves.append(total[:, 1:] - total[:, :1])
    return {"utility": weighted[:, 1:] - weighted[:, :1], "weighted_losses": weighted,
            "utility_variance": variance, "k32_utility": np.stack(halves, axis=1)}


def shard_ranges(count: int, size: int) -> list[tuple[int, int]]:
    """Return stable shard identities."""
    if count <= 0 or size <= 0:
        raise ValueError("Positive shard size and count required")
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def commit_shard(root: Path, start: int, stop: int, contract_id: str, arrays: dict) -> None:
    """Commit data first, then a checksum marker; never overwrite committed evidence."""
    path = root / f"shard_{start:06d}_{stop:06d}.npz"
    if path.with_suffix(".json").exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(path.with_suffix(".json"), {"start": start, "stop": stop,
                "contract_id": contract_id, "sha256": file_hash(path), "bytes": path.stat().st_size})


def read_shard(root: Path, start: int, stop: int, contract_id: str,
               expected_index: np.ndarray) -> dict | None:
    """Verify hashes, source rows and every derived label before reuse."""
    path = root / f"shard_{start:06d}_{stop:06d}.npz"
    marker = path.with_suffix(".json")
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text())
    if (metadata["start"], metadata["stop"], metadata["contract_id"]) != (start, stop, contract_id):
        raise ValueError("Shard contract mismatch")
    if path.stat().st_size != metadata["bytes"] or file_hash(path) != metadata["sha256"]:
        raise ValueError("Corrupt committed utility shard")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(arrays["window_index"], expected_index)
    if arrays["noise"].shape != (stop - start, 64, 16, 8) or not np.isfinite(arrays["noise"]).all():
        raise ValueError("Invalid saved Gaussian draws")
    for key, expected in aggregate_losses(arrays["losses"], arrays["timesteps"]).items():
        np.testing.assert_array_equal(arrays[key], expected)
    return arrays


def validate_contract(document: dict) -> str:
    """Reject edited metadata and a teacher or sensor protocol other than E400 EMA."""
    contract = document["contract"]
    identity = fingerprint(contract)
    if identity != document["contract_id"]:
        raise ValueError("Utility contract integrity failure")
    if (contract["teacher"]["epoch"] != 400 or contract["teacher"]["weights"] != "ema"
            or tuple(contract["conditions"]) != CONDITION_NAMES):
        raise ValueError("Expected selected E400 EMA and UniVTAC conditions")
    return identity
