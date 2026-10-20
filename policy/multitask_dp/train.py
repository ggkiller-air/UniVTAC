"""Single-GPU trainer with bounded throughput gating and exact optimizer-step resume."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import FullSensorDataset
from .model import FullSensorPolicy
from .schema import CONDITION_NAMES, fingerprint


def atomic_save(payload: dict, path: Path) -> None:
    """Publish a complete checkpoint without exposing a partially written latest file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def capture_rng() -> dict:
    """Preserve all random streams that affect optimization and diffusion sampling."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def learning_rate(config: dict, completed_steps: int, total_steps: int) -> float:
    """Warm up once, then cosine decay over the fixed optimizer-update budget."""
    if completed_steps < config["warmup_steps"]:
        factor = (completed_steps + 1) / max(1, config["warmup_steps"])
    else:
        progress = (completed_steps - config["warmup_steps"]) / max(1, total_steps - config["warmup_steps"])
        factor = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return config["learning_rate"] * factor


class ModelEMA:
    """Reference EMA warmup, once per successful optimizer update, with copied buffers."""

    def __init__(self, policy: nn.Module) -> None:
        self.model = copy.deepcopy(policy).eval().requires_grad_(False)
        self.optimization_step = 0

    @torch.no_grad()
    def step(self, policy: nn.Module) -> None:
        warmup_step = max(0, self.optimization_step - 1)
        decay = min(0.9999, 1 - (1 + warmup_step) ** -0.75) if warmup_step else 0.0
        source, target = [], []
        for parameter, averaged in zip(policy.parameters(), self.model.parameters()):
            if parameter.requires_grad:
                source.append(parameter)
                target.append(averaged)
            else:
                averaged.copy_(parameter)
        if target:
            torch._foreach_mul_(target, decay)
            torch._foreach_add_(target, source, alpha=1 - decay)
        for buffer, averaged in zip(policy.buffers(), self.model.buffers()):
            averaged.copy_(buffer)
        self.optimization_step += 1


def move_batch(batch: dict, device: torch.device) -> dict:
    """Transfer compact uint8 image tensors asynchronously from pinned host memory."""
    return {key: move_batch(value, device) if isinstance(value, dict) else value.to(device, non_blocking=True)
            if torch.is_tensor(value) else value for key, value in batch.items()}


class Trainer:
    """Persist raw/EMA weights, normalizer, Adam, schedule, data cursor and all RNG states."""

    def __init__(self, config: dict, output: Path, device: str = "cuda:0",
                 policy: FullSensorPolicy | None = None, dataset: FullSensorDataset | None = None) -> None:
        self.config, self.output = copy.deepcopy(config), Path(output)
        if config["precision"] not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if config["preflight_steps"] < 2 * config["preflight_window"] or config["preflight_steps"] % config["preflight_window"]:
            raise ValueError("Preflight must contain at least two complete measurement windows")
        self.device = torch.device(device)
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoints = self.output / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)
        self.seed = config["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU training requires an actual allocated CUDA GPU")
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("Configured bf16 requires a compatible GPU")
        elif config["precision"] != "fp32":
            raise ValueError("CPU diagnostics require fp32")
        self.dataset = dataset if dataset is not None else FullSensorDataset(
            config["cache_dir"], config["horizon"], config["n_obs_steps"], config["preload_ram"],
        )
        self.steps_per_epoch = math.ceil(len(self.dataset) / config["batch_size"])
        self.total_steps = self.steps_per_epoch * config["num_epochs"]
        self.policy = policy if policy is not None else FullSensorPolicy(config)
        self.policy.normalizer.fit(self.dataset.normalization_statistics())
        self.policy.to(self.device).train()
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=config["learning_rate"],
                                           betas=tuple(config["betas"]), eps=1e-8,
                                           weight_decay=config["weight_decay"], fused=self.device.type == "cuda")
        self.ema = ModelEMA(self.policy)
        self.epoch = self.cursor = self.global_step = 0
        self.epoch_loss_sum = 0.0
        self.epoch_samples = 0
        self.stop_reason = ""
        self.protocol = {"version": 1, "config": config, "cache_id": self.dataset.manifest["cache_id"],
                         "dataset_steps": len(self.dataset), "steps_per_epoch": self.steps_per_epoch,
                         "total_optimizer_steps": self.total_steps, "conditions": CONDITION_NAMES,
                         "normalizer": {key: value.tolist() for key, value in self.policy.normalizer.state_dict().items()}}
        self.protocol_id = fingerprint(self.protocol)
        self.log("initialized", parameters=sum(parameter.numel() for parameter in self.policy.parameters()),
                 steps_per_epoch=self.steps_per_epoch, total_steps=self.total_steps, cache_id=self.dataset.manifest["cache_id"])

    def log(self, event: str, **values: object) -> None:
        record = {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "epoch": self.epoch,
                  "cursor": self.cursor, "global_step": self.global_step, **values}
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with open(self.output / "progress.jsonl", "a") as stream:
            stream.write(line + "\n")

    def save(self, snapshot: bool = False) -> None:
        """Keep every periodic epoch snapshot plus rolling latest, without serializing twice."""
        payload = {"protocol": self.protocol, "protocol_id": self.protocol_id, "config": self.config,
                   "policy": self.policy.state_dict(), "ema": self.ema.model.state_dict(),
                   "ema_step": self.ema.optimization_step, "optimizer": self.optimizer.state_dict(),
                   "epoch": self.epoch, "cursor": self.cursor, "global_step": self.global_step,
                   "epoch_loss_sum": self.epoch_loss_sum, "epoch_samples": self.epoch_samples,
                   "rng": capture_rng(), "schedule": {"total_steps": self.total_steps, "completed_steps": self.global_step}}
        started = time.monotonic()
        latest = self.checkpoints / "latest.ckpt"
        atomic_save(payload, latest)
        if snapshot:
            destination = self.checkpoints / f"epoch{self.epoch:04d}.ckpt"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite retained snapshot {destination}")
            os.link(latest, destination)
        self.log("checkpoint", snapshot=snapshot, save_seconds=time.monotonic() - started,
                 size_bytes=latest.stat().st_size)

    def resume(self, path: Path) -> None:
        """Reject silent batch, cache, normalization, architecture or schedule changes."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["protocol_id"] != self.protocol_id or fingerprint(payload["protocol"]) != self.protocol_id:
            raise ValueError("Resume protocol mismatch: use the exact saved config/cache/normalizer")
        epoch, cursor, step = payload["epoch"], payload["cursor"], payload["global_step"]
        if not 0 <= epoch <= self.config["num_epochs"] or not 0 <= cursor < self.steps_per_epoch:
            raise ValueError("Invalid resume epoch/cursor")
        if step != epoch * self.steps_per_epoch + cursor or payload["ema_step"] != step:
            raise ValueError("Optimizer, sampler cursor and EMA steps disagree")
        if payload["schedule"] != {"total_steps": self.total_steps, "completed_steps": step}:
            raise ValueError("Resume update budget changed")
        self.policy.load_state_dict(payload["policy"], strict=True)
        self.ema.model.load_state_dict(payload["ema"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.epoch, self.cursor, self.global_step = epoch, cursor, step
        self.epoch_loss_sum, self.epoch_samples = payload["epoch_loss_sum"], payload["epoch_samples"]
        self.ema.optimization_step = payload["ema_step"]
        restore_rng(payload["rng"])
        self.log("resumed", checkpoint=str(path))

    def loader(self) -> DataLoader:
        indices = self.dataset.epoch_indices(self.seed, self.epoch)[self.cursor * self.config["batch_size"]:]
        workers = self.config["num_workers"]
        options = {"prefetch_factor": 1, "multiprocessing_context": "fork", "timeout": 180} if workers else {}
        return DataLoader(self.dataset, sampler=indices, batch_size=self.config["batch_size"],
                          num_workers=workers, pin_memory=self.device.type == "cuda", drop_last=False,
                          generator=torch.Generator().manual_seed(self.seed + 100000 + self.epoch), **options)

    def run(self, stop_after_steps: int = 0) -> None:
        """Preflight uses real optimization steps and continues in place after a throughput pass."""
        self.log("started", batch_size=self.config["batch_size"], workers=self.config["num_workers"],
                 precision=self.config["precision"], validation=False)
        if not (self.output / "protocol.json").exists():
            (self.output / "protocol.json").write_text(json.dumps(self.protocol, indent=2) + "\n")
        preflight_windows = []
        measured_steps = window_samples = 0
        window_loss = 0.0
        data_wait = 0.0
        window_start = time.monotonic()
        for_epoch = self.epoch
        while for_epoch < self.config["num_epochs"]:
            iterator = iter(self.loader())
            while self.cursor < self.steps_per_epoch:
                started_wait = time.monotonic()
                batch = move_batch(next(iterator), self.device)
                data_wait += time.monotonic() - started_wait
                rate = learning_rate(self.config, self.global_step, self.total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = rate
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.config["precision"] == "bf16"):
                    loss = self.policy.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {self.global_step}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["gradient_clip_norm"],
                                                               error_if_nonfinite=True)
                self.optimizer.step()
                self.ema.step(self.policy)
                batch_size = batch["action"].shape[0]
                loss_value = loss.item()
                self.epoch_loss_sum += loss_value * batch_size
                self.epoch_samples += batch_size
                self.global_step += 1
                self.cursor += 1
                measured_steps += 1
                window_samples += batch_size
                window_loss += loss_value * batch_size
                if measured_steps % self.config["preflight_window"] == 0:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    elapsed = time.monotonic() - window_start
                    throughput = window_samples / elapsed
                    wait_fraction = data_wait / elapsed
                    self.log("train_window", loss=window_loss / window_samples, lr=rate,
                             gradient_norm=float(gradient_norm), samples_per_second=throughput,
                             data_wait_fraction=wait_fraction,
                             estimated_remaining_hours=(self.total_steps - self.global_step) * self.config["batch_size"] / throughput / 3600)
                    if measured_steps <= self.config["preflight_steps"]:
                        preflight_windows.append((throughput, wait_fraction))
                    if measured_steps == self.config["preflight_steps"]:
                        steady = preflight_windows[1:]
                        speed = float(np.median([value[0] for value in steady]))
                        wait_ratio = float(np.median([value[1] for value in steady]))
                        passed = speed >= self.config["min_samples_per_second"] and wait_ratio <= self.config["max_data_wait_fraction"]
                        self.log("throughput_gate", passed=passed, samples_per_second=speed, data_wait_fraction=wait_ratio)
                        if not passed:
                            self.stop_reason = "throughput_gate_failed"
                    window_start = time.monotonic()
                    window_samples = 0
                    window_loss = data_wait = 0.0
                boundary = self.cursor == self.steps_per_epoch
                if boundary:
                    self.epoch += 1
                    self.cursor = 0
                    self.log("epoch_completed", train_loss=self.epoch_loss_sum / self.epoch_samples,
                             samples=self.epoch_samples, ema_step=self.ema.optimization_step)
                    self.epoch_loss_sum = 0.0
                    self.epoch_samples = 0
                stop_file = os.environ.get("UNIVTAC_DP_STOP_FILE")
                if stop_file and Path(stop_file).exists():
                    self.stop_reason = Path(stop_file).read_text().strip() or "stop_file"
                if stop_after_steps and self.global_step >= stop_after_steps:
                    self.stop_reason = "requested_step_limit"
                save_step = (self.global_step % self.config["checkpoint_every_steps"] == 0
                             or self.global_step == self.config["checkpoint_first_step"])
                if boundary or save_step or self.stop_reason:
                    save_start = time.monotonic()
                    self.save(snapshot=boundary and self.epoch % self.config["checkpoint_every_epochs"] == 0)
                    window_start += time.monotonic() - save_start
                if self.stop_reason:
                    self.log("stopped", reason=self.stop_reason)
                    if self.stop_reason == "throughput_gate_failed":
                        raise RuntimeError("Throughput preflight failed; latest saved, long training not continued")
                    return
                if boundary:
                    break
            del iterator
            for_epoch = self.epoch
        self.log("completed", ema_step=self.ema.optimization_step)


def main() -> None:
    """Start or strictly resume a versioned, independent offline training run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(arguments.config.read_text())
    trainer = Trainer(config, arguments.output)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: setattr(trainer, "stop_reason", signal.Signals(received).name))
    resume = arguments.resume
    latest = arguments.output / "checkpoints/latest.ckpt"
    if resume is None and latest.exists():
        resume = latest
    if resume is not None:
        trainer.resume(resume)
    else:
        (arguments.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    trainer.run()


if __name__ == "__main__":
    main()
