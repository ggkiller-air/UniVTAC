#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: bash scripts/train_act.sh TASK EPISODES [SEED=0] [CONFIG=task-specific author config]" >&2
    exit 2
fi
task_name="$1"
episode_count="$2"
seed="${3:-0}"
config_args=()
if [[ $# -eq 4 ]]; then
    config_args=(--config "$(realpath "$4")")
fi
variant="${ACT_VARIANT:-baseline}"
if [[ "$variant" != "baseline" && "$variant" != "hypernet" ]]; then
    echo "ACT_VARIANT must be baseline or hypernet." >&2
    exit 2
fi
if [[ ! "$task_name" =~ ^[A-Za-z0-9_-]+$ || ! "$episode_count" =~ ^[0-9]+$ || "$episode_count" -lt 2 || ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Use a valid task name, at least two episodes, and a nonnegative integer seed." >&2
    exit 2
fi
source "$REPO_ROOT/scripts/env.sh"
if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}" ]]; then
    echo "This Slurm allocation has no GPU. Request a GPU allocation before training." >&2
    exit 1
fi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; request a GPU allocation"; print(torch.cuda.get_device_name(0))'
run_dir="${ACT_RUN_DIR:-$UNIVTAC_RUN_ROOT/ACT-$variant/$task_name/$(date +%Y%m%d-%H%M%S)-seed$seed-$$}"
if [[ -f "$run_dir/config.yml" ]]; then
    echo "Refusing to overwrite an existing training run: $run_dir" >&2
    exit 1
fi
mkdir -p "$run_dir"
python "$REPO_ROOT/scripts/prepare_act_config.py" "$task_name" --variant "$variant" "${config_args[@]}" --output "$run_dir/config.yml"
git -C "$REPO_ROOT" rev-parse HEAD > "$run_dir/git-revision.txt"
git -C "$REPO_ROOT" diff --binary > "$run_dir/code.patch"
python -m pip freeze > "$run_dir/requirements.lock.txt"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$run_dir/source.tar.gz" -C "$REPO_ROOT" \
    policy/ACT policy/_base_data_preprocessor.py policy/task_settings.json envs/utils \
    scripts/train_act.sh scripts/train_act_hypernet.sh scripts/prepare_act_config.py
cd "$REPO_ROOT/policy/ACT"
if ! python - "$task_name" "$episode_count" <<'PY'
import json
import os
import sys
from pathlib import Path
from protocol import validate_processed_task

config_path = Path(os.environ['UNIVTAC_TASK_CONFIGS'])
if not config_path.exists():
    raise SystemExit(1)
with config_path.open() as config_file:
    task_configs = json.load(config_file)
task = task_configs.get(f'sim-{sys.argv[1]}-demo-{sys.argv[2]}')
if not task:
    raise SystemExit(1)
try:
    validate_processed_task(task, sys.argv[1], int(sys.argv[2]))
except (ValueError, FileNotFoundError):
    raise SystemExit(1)
PY
then
    python process_data.py "$task_name" demo "$episode_count" 2>&1 | tee "$run_dir/preprocess.log"
fi
python imitate_episodes.py --task_name "sim-$task_name-demo-$episode_count" --ckpt_dir "$run_dir" --config_path "$run_dir/config.yml" --seed "$seed" 2>&1 | tee "$run_dir/train.log"
echo "Checkpoints, normalization statistics, configuration, and logs: $run_dir"
