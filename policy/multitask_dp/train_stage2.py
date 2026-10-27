"""Prepared Stage 2 trainer; launching training is deliberately a separate user step."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .generate_utility import load_teacher, reserve_memory
from .schema import fingerprint
from .stage2 import Stage2Policy, make_optimizer_scheduler, set_stage_trainability, stage_for_epoch
from .stage2_data import Stage2Dataset
from .train import atomic_save, capture_rng, move_batch, restore_rng
from .utility import atomic_json

SNAPSHOT_EPOCHS = frozenset((1, 6, 10, 20, 50, 100, 200, 300, 400, 500, 600))


def build_policy(teacher: object, utility_rms: np.ndarray) -> Stage2Policy:
    """Build new action modules while strictly copying E400's encoder and normalizer."""
    encoder = copy.deepcopy(teacher.obs_encoder).requires_grad_(True)
    policy = Stage2Policy(teacher.config, torch.from_numpy(utility_rms), encoder)
    policy.initialize_from_teacher(teacher)
    return policy


class Stage2Trainer:
    """Retain main's A/B/C phases and serialize every state needed for step-exact resume."""

    def __init__(self, policy: Stage2Policy, dataset: Stage2Dataset, config: dict,
                 output: Path, device: str = "cuda:0") -> None:
        self.device, self.config = torch.device(device), copy.deepcopy(config)
        if min(config["batch_size"], config["num_epochs"]) <= 0 or config["num_workers"] < 0:
            raise ValueError("Invalid training sizes")
        self.policy = policy.to(self.device)
        self.ema = copy.deepcopy(policy).eval().requires_grad_(False)
        self.dataset, self.output = dataset, Path(output)
        (self.output / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.steps_per_epoch = math.ceil(len(dataset) / config["batch_size"])
        # Create all optimizer groups before phase A freezes any modules.
        self.optimizer, self.scheduler = make_optimizer_scheduler(policy, self.steps_per_epoch, config["num_epochs"])
        self.epoch, self.cursor, self.global_step = 1, 0, 0
        self.stop_requested = False
        self.contract = {"kind": "univtac-stage2-transformer-t2-v1", "config": self.config,
                         "label_contract_id": dataset.contract_id,
                         "teacher": dataset.contract["teacher"],
                         "utility_rms": policy.utility_rms.detach().cpu().tolist()}
        self.contract_id = fingerprint(self.contract)
        existing = self.output / "contract.json"
        if existing.exists() and json.loads(existing.read_text())["contract_id"] != self.contract_id:
            raise ValueError("Output directory already belongs to another Stage 2 contract")
        atomic_json(self.output / "contract.json", {"contract": self.contract, "contract_id": self.contract_id})

    def request_stop(self, signum: int, frame: object) -> None:
        """Stop only after the current optimizer update has been committed."""
        self.stop_requested = True

    def should_stop(self) -> bool:
        """Also accept the batch shell's early-warning signal during RAM loading."""
        stop_file = os.environ.get("UNIVTAC_STAGE2_STOP_FILE")
        return self.stop_requested or bool(stop_file and Path(stop_file).exists())

    def preflight(self, steps: int = 100) -> dict:
        """Check real A/B/C updates then reset weights, Adam, scheduler, EMA and RNG."""
        if steps < 7 or self.global_step != 0 or self.cursor != 0:
            raise ValueError("Preflight requires fresh training state and at least seven steps")
        initial_model = {key: value.detach().cpu().clone() for key, value in self.policy.state_dict().items()}
        initial_ema = {key: value.detach().cpu().clone() for key, value in self.ema.state_dict().items()}
        initial_rng = capture_rng()
        indices = self.dataset.epoch_indices(self.config["seed"], 1)
        required = steps * self.config["batch_size"]
        indices = (indices * math.ceil(required / len(indices)))[:required]
        options = {"batch_size": self.config["batch_size"], "sampler": indices,
                   "num_workers": self.config["num_workers"], "pin_memory": self.device.type == "cuda",
                   "generator": torch.Generator().manual_seed(self.config["seed"] + 100001)}
        if self.config["num_workers"]:
            options.update(multiprocessing_context="fork", prefetch_factor=2)
        loader = DataLoader(self.dataset, **options)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.policy.train()
        started = previous = time.monotonic()
        data_wait, completed, samples = 0.0, 0, 0
        records = []
        try:
            for index, batch in enumerate(loader):
                data_wait += time.monotonic() - previous
                if self.should_stop():
                    break
                stage = "A" if index < 2 else "B" if index < 6 else "C"
                set_stage_trainability(self.policy, stage)
                batch = move_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                result = self.policy.training_losses(batch, stage)
                loss = result["utility_loss"] if stage == "A" else result["action_loss"] + 0.05 * result["utility_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite preflight loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite preflight gradient")
                if stage == "C":
                    gradient = self.policy.obs_encoder.fusion[0].weight.grad
                    if gradient is None or not torch.isfinite(gradient).all() or not bool(gradient.abs().sum() > 0):
                        raise RuntimeError("Action gradient did not reach the Stage 2 encoder")
                self.optimizer.step()
                self.scheduler.step()
                self.update_ema()
                completed += 1
                samples += len(batch["action"])
                if completed in (2, 6, steps) or completed % 20 == 0:
                    record = {"event": "preflight", "step": completed, "stage": stage,
                              "action_loss": float(result["action_loss"].detach()),
                              "utility_loss": float(result["utility_loss"].detach()),
                              "gradient_norm": float(grad_norm)}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                previous = time.monotonic()
            elapsed = time.monotonic() - started
            report = {"steps": completed, "requested_steps": steps,
                      "samples_per_second": samples / elapsed, "data_wait_fraction": data_wait / elapsed,
                      "elapsed_seconds": elapsed, "interrupted": self.should_stop(), "records": records,
                      "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else None}
            report["passed"] = completed == steps and not report["interrupted"]
            if self.device.type == "cuda":
                report["passed"] = report["passed"] and report["samples_per_second"] >= 20 and report["data_wait_fraction"] <= 0.75 and report["gpu_peak_allocated_bytes"] <= 0.9 * torch.cuda.get_device_properties(0).total_memory
            atomic_json(self.output / "preflight_report.json", report)
        finally:
            self.policy.zero_grad(set_to_none=True)
            self.policy.load_state_dict(initial_model, strict=True)
            self.ema.load_state_dict(initial_ema, strict=True)
            self.policy.to(self.device)
            self.ema.to(self.device)
            set_stage_trainability(self.policy, "C")
            self.optimizer, self.scheduler = make_optimizer_scheduler(
                self.policy, self.steps_per_epoch, self.config["num_epochs"])
            restore_rng(initial_rng)
        if not report["passed"] and not report["interrupted"]:
            raise RuntimeError("Stage 2 preflight failed; see preflight_report.json")
        print(json.dumps({"event": "preflight_reset_complete", "global_step": self.global_step,
                          "optimizer_states": len(self.optimizer.state), "passed": report["passed"]}), flush=True)
        return report

    def save(self, snapshot: bool = False) -> None:
        """Keep latest plus every requested epoch snapshot without a top-k cleanup."""
        payload = {"kind": self.contract["kind"], "model": self.policy.state_dict(),
                   "ema": self.ema.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "scheduler": self.scheduler.state_dict(), "rng": capture_rng(),
                   "epoch": self.epoch, "cursor": self.cursor, "global_step": self.global_step,
                   "contract": self.contract, "contract_id": self.contract_id,
                   "policy_config": self.policy.config}
        latest = self.output / "checkpoints/latest.ckpt"
        atomic_save(payload, latest)
        if snapshot:
            destination = latest.with_name(f"epoch{self.epoch:04d}.ckpt")
            if destination.exists():
                # A boundary checkpoint can be resumed before its epoch cursor
                # is advanced. Keep that immutable snapshot and finish the epoch.
                retained = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
                if (retained["contract_id"], retained["epoch"], retained["cursor"], retained["global_step"]) != (
                        self.contract_id, self.epoch, self.cursor, self.global_step):
                    raise FileExistsError(f"Conflicting retained checkpoint: {destination}")
            else:
                os.link(latest, destination)
            archive_value = self.config.get("archive_dir")
            if archive_value:
                archive = Path(archive_value)
                if archive.resolve() == self.output.resolve() or self.output.resolve() in archive.resolve().parents:
                    raise ValueError("Checkpoint archive must be outside the active training directory")
                archive.mkdir(parents=True, exist_ok=True)
                archived = archive / destination.name
                if not archived.exists():
                    temporary = archived.with_name(f".{archived.name}.{os.getpid()}.partial")
                    shutil.copyfile(destination, temporary)
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
                    os.replace(temporary, archived)
                else:
                    retained = torch.load(archived, map_location="cpu", weights_only=False, mmap=True)
                    if (retained["contract_id"], retained["global_step"]) != (self.contract_id, self.global_step):
                        raise FileExistsError(f"Conflicting external archive: {archived}")

    def restore(self, checkpoint: Path) -> None:
        """Reject changed labels/config and restore model, EMA, Adam, scheduler and RNG."""
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["contract_id"] != self.contract_id or fingerprint(saved["contract"]) != self.contract_id:
            raise ValueError("Stage 2 resume contract differs")
        self.policy.load_state_dict(saved["model"], strict=True)
        self.ema.load_state_dict(saved["ema"], strict=True)
        self.policy.to(self.device)
        self.ema.to(self.device)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scheduler.load_state_dict(saved["scheduler"])
        self.epoch, self.cursor, self.global_step = saved["epoch"], saved["cursor"], saved["global_step"]
        if not 1 <= self.epoch <= self.config["num_epochs"] + 1 or not 0 <= self.cursor <= self.steps_per_epoch:
            raise ValueError("Invalid resume cursor")
        restore_rng(saved["rng"])

    @torch.no_grad()
    def update_ema(self) -> None:
        """Match main's fixed 0.9999 EMA and copy normalizer/RMS buffers exactly."""
        for target, source in zip(self.ema.parameters(), self.policy.parameters()):
            target.lerp_(source, 1 - 0.9999)
        for target, source in zip(self.ema.buffers(), self.policy.buffers()):
            target.copy_(source)

    def fit(self, max_steps: int | None = None) -> None:
        """Run complete labeled windows, with an optional bounded CPU diagnostic stop."""
        initial_step = self.global_step
        started, last_end, measured_samples, data_wait = time.monotonic(), time.monotonic(), 0, 0.0
        if self.should_stop():
            self.save()
            atomic_json(self.output / "stopped.json", {"contract_id": self.contract_id, "reason": "scheduler_signal"})
            return
        while self.epoch <= self.config["num_epochs"]:
            stage = stage_for_epoch(self.epoch)
            self.policy.train()
            set_stage_trainability(self.policy, stage)
            indices = self.dataset.epoch_indices(self.config["seed"], self.epoch)
            indices = indices[self.cursor * self.config["batch_size"]:]
            options = {"batch_size": self.config["batch_size"], "sampler": indices,
                       "num_workers": self.config["num_workers"], "pin_memory": self.device.type == "cuda",
                       "generator": torch.Generator().manual_seed(self.config["seed"] + self.epoch + 100000)}
            if self.config["num_workers"]:
                options.update(multiprocessing_context="fork", prefetch_factor=2)
            loader = DataLoader(self.dataset, **options)
            for batch in loader:
                data_wait += time.monotonic() - last_end
                batch = move_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                result = self.policy.training_losses(batch, stage)
                loss = result["utility_loss"] if stage == "A" else result["action_loss"] + 0.05 * result["utility_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite Stage 2 loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite Stage 2 gradient")
                self.optimizer.step()
                self.scheduler.step()
                self.update_ema()
                self.global_step += 1
                self.cursor += 1
                measured_samples += len(batch["action"])
                if self.global_step % 20 == 0 or self.global_step == 1:
                    error = (result["u_hat"].detach() - result["utility_target"]).float()
                    record = {"event": "train", "epoch": self.epoch, "stage": stage,
                              "cursor": self.cursor, "global_step": self.global_step,
                              "action_loss": float(result["action_loss"].detach()),
                              "utility_loss": float(result["utility_loss"].detach()),
                              "utility_mae_8d": error.abs().mean(dim=0).cpu().tolist(),
                              "utility_rmse_8d": error.square().mean(dim=0).sqrt().cpu().tolist(),
                              "gradient_norm": float(grad_norm),
                              "learning_rates": {group["name"]: group["lr"] for group in self.optimizer.param_groups},
                              "samples_per_second": measured_samples / (time.monotonic() - started),
                              "data_wait_seconds": data_wait,
                              "gpu_allocated_bytes": torch.cuda.memory_allocated() if self.device.type == "cuda" else None}
                    with (self.output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    print(json.dumps(record), flush=True)
                if self.should_stop() or (max_steps is not None and self.global_step - initial_step >= max_steps):
                    self.save()
                    if self.should_stop():
                        atomic_json(self.output / "stopped.json", {"contract_id": self.contract_id,
                                    "reason": "scheduler_signal", "epoch": self.epoch, "global_step": self.global_step})
                    return
                if self.global_step % 1000 == 0:
                    self.save()
                last_end = time.monotonic()
            if self.cursor != self.steps_per_epoch:
                raise RuntimeError("Epoch cursor and sampler length differ")
            self.save(snapshot=self.epoch in SNAPSHOT_EPOCHS)
            print(json.dumps({"event": "epoch_complete", "epoch": self.epoch, "global_step": self.global_step}), flush=True)
            self.epoch += 1
            self.cursor = 0
        self.save()
        atomic_json(self.output / "complete.json", {"contract_id": self.contract_id, "global_step": self.global_step})
        (self.output / "stopped.json").unlink(missing_ok=True)


def main() -> None:
    """CLI is prepared for the next user-authorized training step; no automatic submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-steps", type=int, default=100)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["batch_size"] != 32 or config["num_epochs"] != 600:
        raise ValueError("Official port retains main's batch32/600-epoch protocol")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Formal Stage 2 needs one allocated H200")
    if not (Path(config["labels"]) / "complete.json").exists():
        raise RuntimeError("Utility labels are not finalized")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    memory_reservation = reserve_memory(config["memory_floor"])
    dataset = Stage2Dataset(config["cache"], config["labels"])
    teacher, identity = load_teacher(Path(config["teacher"]), dataset.contract["teacher"]["sha256"])
    if identity != dataset.contract["teacher"]:
        raise ValueError("Stage 2 teacher differs from the label teacher")
    policy = build_policy(teacher, dataset.utility_rms)
    del teacher
    trainer = Stage2Trainer(policy, dataset, config, args.output)
    del policy
    if args.resume:
        trainer.restore(args.resume)
    signal.signal(signal.SIGUSR1, trainer.request_stop)
    signal.signal(signal.SIGTERM, trainer.request_stop)
    print(json.dumps({"event": "stage2_ready", "memory_floor_bytes": memory_reservation.numel(),
                      "windows": len(dataset), "steps_per_epoch": trainer.steps_per_epoch}), flush=True)
    preflight_path = args.output / "preflight_report.json"
    if trainer.global_step == 0 and (not preflight_path.exists() or not json.loads(preflight_path.read_text())["passed"]):
        trainer.preflight(args.preflight_steps)
    elif trainer.global_step > 0 and (not preflight_path.exists() or not json.loads(preflight_path.read_text())["passed"]):
        raise RuntimeError("Resume requires the original successful GPU preflight report")
    print(json.dumps({"event": "formal_start", "epoch": trainer.epoch, "global_step": trainer.global_step}), flush=True)
    trainer.fit()


if __name__ == "__main__":
    main()
