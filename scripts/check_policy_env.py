import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torchvision
import yaml


def main():
    parser = argparse.ArgumentParser(description="Bounded ACT dependency and optimizer smoke check.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda":
        if os.environ.get("SLURM_JOB_ID") and not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
            raise RuntimeError("The current Slurm allocation has no GPU.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run inside a GPU allocation.")
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.manual_seed(0)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "policy" / "ACT"))
    from act_policy import ACTPolicy
    from imitate_episodes import forward_pass
    from utils import TacArenaDataset, get_norm_stats

    with (repo_root / "policy" / "ACT" / "train_config_cluster.yml").open() as config_file:
        config = yaml.safe_load(config_file)
    config.update(device=args.device, enc_layers=1, dec_layers=1, dim_feedforward=512, chunk_size=2,
                  task_name="sim-smoke", seed=0, num_epochs=1, ckpt_dir="smoke")
    if args.dataset:
        episode_paths = sorted(args.dataset.glob("episode_*.hdf5"))
        if len(episode_paths) < 2:
            raise ValueError("At least two processed episodes are required.")
        norm_stats, _ = get_norm_stats(str(args.dataset), len(episode_paths))
        dataset = TacArenaDataset([0, 1], str(args.dataset), config["camera_names"], config["tactile_names"], norm_stats, config["chunk_size"])
        batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
        config["state_dim"] = batch[2].shape[-1]
    else:
        batch = (
            torch.rand(2, 1, 3, 64, 64),
            torch.rand(2, 2, 3, 64, 64),
            torch.rand(2, config["state_dim"]),
            torch.rand(2, 2, config["state_dim"]),
            torch.zeros(2, 2, dtype=torch.bool),
        )
    policy = ACTPolicy(config)
    policy.train()
    optimizer = policy.configure_optimizers()
    weight_before = policy.model.action_head.weight.detach().clone()
    loss_dict = forward_pass(batch, policy)
    loss = loss_dict["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss.")
    loss.backward()
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Missing or non-finite gradients.")
    optimizer.step()
    if torch.equal(weight_before, policy.model.action_head.weight):
        raise RuntimeError("Optimizer did not update the action head.")
    with tempfile.TemporaryDirectory(prefix="univtac-check-") as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "policy.ckpt"
        torch.save(policy.state_dict(), checkpoint_path)
        policy.load_state_dict(torch.load(checkpoint_path, map_location=args.device, weights_only=True))
    policy.eval()
    device = next(policy.parameters()).device
    camera, tactile, joint_state, _, _ = batch
    with torch.inference_mode():
        prediction = policy(joint_state.to(device), camera.to(device), tactile.to(device))
    if prediction.shape != (2, 2, config["state_dim"]) or not torch.isfinite(prediction).all():
        raise RuntimeError("Invalid action prediction.")
    result = {
        "python": sys.version.split()[0],
        "environment": sys.prefix,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_physical_gpus": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"),
        "opencv": cv2.__version__,
        "h5py": h5py.__version__,
        "numpy": np.__version__,
        "dataset": str(args.dataset) if args.dataset else "synthetic",
        "loss": loss.item(),
        "prediction_shape": list(prediction.shape),
        "checks": ["forward", "backward", "optimizer_update", "checkpoint_roundtrip", "inference"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("policy_environment_check_passed")


if __name__ == "__main__":
    main()
