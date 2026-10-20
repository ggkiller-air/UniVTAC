"""Versioned sensor, task and dropout protocol, independent of ACT experiments."""

from __future__ import annotations

import hashlib
import json

import numpy as np

IMAGE_KEYS = ("head", "wrist", "tactile_left", "tactile_right")
SENSOR_KEYS = (*IMAGE_KEYS, "state", "language")
CONDITION_NAMES = ("full", *SENSOR_KEYS, "vision", "tactile")
TASK_TEXTS = {
    "grasp_classify": "Grasp and classify the object.",
    "insert_HDMI": "Insert the HDMI connector.",
    "insert_hole": "Insert the peg into the hole.",
    "insert_tube": "Insert the tube.",
    "lift_bottle": "Lift the bottle.",
    "lift_can": "Lift the can.",
    "pull_out_key": "Pull out the key.",
    "put_bottle_in_shelf": "Put the bottle in the shelf.",
}
CACHE_VERSION = "univtac-full-sensor-joint-next-bgr-u8-v1"


def fingerprint(value: object) -> str:
    """Hash canonical JSON for dataset and resume compatibility checks."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def presence_table() -> np.ndarray:
    """Return full, six single removals, vision removal and tactile removal."""
    table = np.ones((len(CONDITION_NAMES), len(SENSOR_KEYS)), dtype=bool)
    for sensor_index in range(len(SENSOR_KEYS)):
        table[sensor_index + 1, sensor_index] = False
    table[-2, :2] = False
    table[-1, 2:4] = False
    return table


def condition_probabilities(full_probability: float, weights: list[float]) -> np.ndarray:
    """Allocate the remaining probability over configurable dropout conditions."""
    values = np.asarray(weights, dtype=np.float64)
    if not 0 <= full_probability <= 1 or values.shape != (len(CONDITION_NAMES) - 1,):
        raise ValueError("Invalid full probability or dropout weight count")
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Dropout weights must be finite, nonnegative and nonzero")
    return np.concatenate(([full_probability], (1 - full_probability) * values / values.sum()))
