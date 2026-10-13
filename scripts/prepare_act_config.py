import argparse
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Resolve author ACT settings and task-specific cameras.")
    parser.add_argument("task_name")
    parser.add_argument("--variant", choices=["baseline", "hypernet"], default="hypernet")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tactile-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "policy" / "ACT"))
    from protocol import author_training_config

    checkpoint = args.tactile_checkpoint or Path(os.environ.get(
        "UNIVTAC_TACTILE_CHECKPOINT", repo_root / "data" / "checkpoints" / "encoder.pth",
    ))
    config = author_training_config(args.task_name, checkpoint, args.variant, args.config)
    config["device"] = args.device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Task: {args.task_name}; variant: {args.variant}; cameras: {config['camera_names']}")
    print(f"Author ACT: state/action={config['state_dim']}, batch={config['batch_size']}, "
          f"steps={config['num_steps']}, chunk={config['chunk_size']}")
    print(f"Verified tactile encoder: {config['tactile_ckpt']}")
    print(f"Resolved configuration: {args.output}")


if __name__ == "__main__":
    main()
