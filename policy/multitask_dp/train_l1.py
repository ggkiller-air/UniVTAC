"""Train the final six-task L1 Teacher with the existing resumable DP loop."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import FullSensorDataset
from .l1 import FINAL_TASKS, L1Policy, SixTaskDataset
from .train import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("architecture") != "l1_late_language_v1" or tuple(config["tasks"]) != FINAL_TASKS:
        raise ValueError("Expected the final six-task L1 architecture/protocol")
    torch.set_num_threads(4)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    source = FullSensorDataset(config["cache_dir"], config["horizon"], config["n_obs_steps"], config["preload_ram"])
    dataset = SixTaskDataset(source)
    trainer = Trainer(config, args.output, policy=L1Policy(config), dataset=dataset)
    (args.output / "architecture.json").write_text(json.dumps({
        "kind": "univtac-l1-teacher-v1", "tasks": list(FINAL_TASKS),
        "input_tokens": ["head", "wrist", "tactile_left", "tactile_right", "state"],
        "sensor_tokens": [5, 256], "fusion_input": 1285,
        "late_language": [768, 256, 256], "condition_dim": 256,
        "action_dim": 8, "horizon": config["horizon"], "history": config["n_obs_steps"],
        "input_dropout": False, "episodes": len(dataset.manifest["episodes"]),
        "anchors": len(dataset), "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }, indent=2) + "\n")
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: setattr(trainer, "stop_reason", signal.Signals(received).name))
    latest = args.output / "checkpoints/latest.ckpt"
    if latest.exists():
        trainer.resume(latest)
    else:
        (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    trainer.run()
    if trainer.epoch == config["num_epochs"]:
        (args.output / "COMPLETE.json").write_text(json.dumps({
            "epoch": trainer.epoch, "global_step": trainer.global_step,
            "protocol_id": trainer.protocol_id,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
