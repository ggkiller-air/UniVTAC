"""Frozen E400 EMA label generation, one encoding per window and paired K64 draws."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import FullSensorDataset
from .model import DenoisingSample, FullSensorPolicy
from .schema import CONDITION_NAMES, fingerprint
from .train import move_batch
from .utility import (aggregate_losses, atomic_json, commit_shard, complete_window_index,
                      file_hash, read_shard, shard_ranges, validate_contract, window_draws)

STOP_REQUESTED = False


def request_stop(signum: int, frame: object) -> None:
    """Finish and commit the current small shard on a scheduler signal."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def reserve_memory(fraction: float) -> torch.Tensor:
    """Hold an explicit, logged allocation requested by the administrator via the user."""
    if not 0 <= fraction <= 0.5:
        raise ValueError("Memory floor must be between zero and 50 percent")
    free, total = torch.cuda.mem_get_info()
    size = max(0, int(total * fraction) - torch.cuda.memory_allocated())
    if size + 8 * 1024**3 > free:
        raise MemoryError("Insufficient memory for requested floor plus model working space")
    allocation = torch.zeros(size, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    print(json.dumps({"event": "admin_requested_memory_floor", "fraction": fraction,
                      "reserved_bytes": allocation.numel(), "device_total_bytes": total,
                      "allocated_bytes": torch.cuda.memory_allocated()}), flush=True)
    return allocation


@torch.inference_mode()
def evaluate_draws(policy: FullSensorPolicy, batch: dict, timesteps: np.ndarray,
                   noise: np.ndarray, microbatch: int) -> np.ndarray:
    """Vectorize masks and draws without repeating expensive image encodings."""
    count = len(batch["action"])
    if timesteps.shape != (count, 64) or noise.shape != (count, 64, policy.horizon, 8) or microbatch < 1:
        raise ValueError("Invalid draw shapes or microbatch")
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("Label teacher must be frozen and in eval mode")
    device = batch["action"].device
    tokens = policy.obs_encoder.encode_tokens(batch["obs"], policy.normalizer)
    conditions = torch.stack([policy.obs_encoder.fuse_tokens(tokens, presence.expand(count, -1))
                              for presence in policy.condition_presence], dim=1)
    repeated = {"action": batch["action"].repeat_interleave(64, dim=0),
                "action_loss_mask": batch["action_loss_mask"].repeat_interleave(64, dim=0)}
    sample = policy.prepare_denoising_sample(repeated,
        torch.from_numpy(noise.reshape(-1, policy.horizon, 8)).to(device),
        torch.from_numpy(timesteps.reshape(-1)).to(device))
    condition_count = len(CONDITION_NAMES)
    result = torch.empty(count * 64 * condition_count, dtype=torch.float32, device=device)
    for start in range(0, result.numel(), microbatch):
        flat = torch.arange(start, min(start + microbatch, result.numel()), device=device)
        draw, condition = flat // condition_count, flat % condition_count
        selected = DenoisingSample(*(getattr(sample, key)[draw] for key in (
            "noisy_action", "noise", "timesteps", "target", "loss_mask")))
        result[start:start + len(flat)] = policy.denoising_loss_per_sample(
            selected, conditions[draw // 64, condition])
    return result.reshape(count, 64, condition_count).cpu().numpy().astype(np.float64)


def load_teacher(path: Path, expected_hash: str | None = None) -> tuple[FullSensorPolicy, dict]:
    """Strictly restore E400 EMA, checkpoint identity and its unchanged normalizer."""
    digest = file_hash(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("Teacher file changed")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if (payload["epoch"], payload["cursor"], payload["global_step"]) != (400, 0, 472400):
        raise ValueError("Not the selected complete UniVTAC E400 checkpoint")
    if fingerprint(payload["protocol"]) != payload["protocol_id"]:
        raise ValueError("Teacher protocol identity mismatch")
    if payload["config"] != payload["protocol"]["config"] or tuple(payload["protocol"]["conditions"]) != CONDITION_NAMES:
        raise ValueError("Teacher configuration/conditions mismatch")
    if payload["config"]["num_diffusion_steps"] != 100:
        raise ValueError("K64 stratification requires the original 100 diffusion steps")
    policy = FullSensorPolicy(payload["config"])
    policy.load_state_dict(payload["ema"], strict=True)
    for key, values in policy.normalizer.state_dict().items():
        if not torch.equal(values, torch.tensor(payload["protocol"]["normalizer"][key], dtype=values.dtype)):
            raise ValueError("Teacher normalizer differs from protocol")
    metadata = {"path": str(path.resolve()), "sha256": digest, "epoch": 400,
                "global_step": 472400, "weights": "ema", "protocol_id": payload["protocol_id"],
                "cache_id": payload["protocol"]["cache_id"], "config": payload["config"],
                "normalizer": payload["protocol"]["normalizer"]}
    return policy.eval().requires_grad_(False), metadata


def finalize(root: Path, contract_id: str, index: np.ndarray, shard_size: int) -> dict:
    """Verify all committed shards before publishing the joined training label arrays."""
    utilities, variances, losses = [], [], []
    for start, stop in shard_ranges(len(index), shard_size):
        arrays = read_shard(root / "shards", start, stop, contract_id, index[start:stop])
        if arrays is None:
            raise ValueError("Cannot finalize incomplete labels")
        utilities.append(arrays["utility"])
        variances.append(arrays["utility_variance"])
        losses.append(arrays["weighted_losses"])
    values = {"utility": np.concatenate(utilities), "utility_variance": np.concatenate(variances),
              "weighted_losses": np.concatenate(losses), "window_index": index}
    artifacts = {}
    for key, value in values.items():
        path = root / f"{key}.npy"
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        artifacts[path.name] = file_hash(path)
    utility = values["utility"]
    report = {"contract_id": contract_id, "windows": len(index), "artifacts": artifacts,
              "utility_rms": np.maximum(np.sqrt(np.square(utility).mean(axis=0)), 1e-6).tolist(),
              "utility_mean": utility.mean(axis=0).tolist(),
              "negative_fraction": (utility < 0).mean(axis=0).tolist(),
              "dimensions": list(CONDITION_NAMES[1:])}
    atomic_json(root / "complete.json", report)
    return report


def main() -> None:
    """Generate or resume only labels; this entrypoint never starts student training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timestep-seed", type=int, default=20261026)
    parser.add_argument("--noise-seed", type=int, default=20261027)
    parser.add_argument("--memory-floor", type=float, default=0.16)
    args = parser.parse_args()
    if min(args.batch_size, args.microbatch, args.shard_size) <= 0 or args.workers < 0:
        raise ValueError("Invalid runtime sizes")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Labels require one allocated H200")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Allocate immediately, before checkpoint hashing and host-cache preload.
    memory_reservation = reserve_memory(args.memory_floor)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    root = args.output.resolve()
    (root / "shards").mkdir(parents=True, exist_ok=True)
    contract_path = root / "contract.json"
    previous = json.loads(contract_path.read_text()) if contract_path.exists() else None
    if previous is not None:
        validate_contract(previous)
    policy, teacher = load_teacher(args.teacher, previous["contract"]["teacher"]["sha256"] if previous else None)
    policy.to("cuda:0")
    dataset = FullSensorDataset(args.cache, preload_ram=True)
    if dataset.manifest["cache_id"] != teacher["cache_id"]:
        raise ValueError("Teacher and label cache identity differ")
    index = complete_window_index(dataset.manifest)
    contract = {"version": "univtac-e400-ema-stratified-k64-v1", "teacher": teacher,
                "dataset_manifest": dataset.manifest, "conditions": list(CONDITION_NAMES),
                "horizon": 16, "n_obs_steps": 2, "action_dim": 8, "k": 64,
                "timestep_seed": args.timestep_seed, "noise_seed": args.noise_seed,
                "precision": "fp32; TF32 disabled", "shard_size": args.shard_size,
                "batch_size": args.batch_size, "microbatch": args.microbatch,
                "window_index_sha256": hashlib.sha256(index.tobytes()).hexdigest()}
    contract_id = fingerprint(contract)
    if previous and previous["contract_id"] != contract_id:
        raise ValueError("Resume label generation contract changed")
    atomic_json(contract_path, {"contract": contract, "contract_id": contract_id})
    print(json.dumps({"event": "labels_ready", "windows": len(index), "contract_id": contract_id,
                      "gpu": torch.cuda.get_device_name(0), "memory_floor_bytes": memory_reservation.numel()}), flush=True)
    start_time, generated = time.monotonic(), 0
    for start, stop in shard_ranges(len(index), args.shard_size):
        if read_shard(root / "shards", start, stop, contract_id, index[start:stop]) is not None:
            continue
        if STOP_REQUESTED or (os.environ.get("UNIVTAC_UTILITY_STOP_FILE") and
                              Path(os.environ["UNIVTAC_UTILITY_STOP_FILE"]).exists()):
            request_stop(0, None)
            break
        rows = index[start:stop]
        draws = [window_draws(int(row["window_id"]), args.timestep_seed, args.noise_seed) for row in rows]
        timesteps, noise = np.stack([draw[0] for draw in draws]), np.stack([draw[1] for draw in draws])
        options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True}
        if args.workers:
            options.update(multiprocessing_context="fork", prefetch_factor=1)
        loader = DataLoader(Subset(dataset, rows["source_index"].tolist()), **options)
        parts, offset = [], 0
        for batch in loader:
            if not batch["observation_valid_mask"].all() or not batch["action_loss_mask"].all():
                raise ValueError("Incomplete window unexpectedly selected")
            count = len(batch["action"])
            parts.append(evaluate_draws(policy, move_batch(batch, torch.device("cuda:0")),
                         timesteps[offset:offset + count], noise[offset:offset + count], args.microbatch))
            offset += count
        losses = np.concatenate(parts)
        arrays = {"window_index": rows, "timesteps": timesteps, "noise": noise, "losses": losses,
                  **aggregate_losses(losses, timesteps)}
        commit_shard(root / "shards", start, stop, contract_id, arrays)
        generated += len(rows)
        elapsed = time.monotonic() - start_time
        record = {"event": "shard_complete", "start": start, "stop": stop, "total": len(index),
                  "windows_per_second": generated / elapsed,
                  "estimated_remaining_hours": (len(index) - stop) / (generated / elapsed) / 3600,
                  "gpu_allocated_bytes": torch.cuda.memory_allocated(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        print(json.dumps(record), flush=True)
        with (root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    if STOP_REQUESTED:
        atomic_json(root / "stopped.json", {"contract_id": contract_id, "reason": "scheduler_signal"})
        return
    report = finalize(root, contract_id, index, args.shard_size)
    (root / "stopped.json").unlink(missing_ok=True)
    print(json.dumps({"event": "completed", **report}), flush=True)


if __name__ == "__main__":
    main()
