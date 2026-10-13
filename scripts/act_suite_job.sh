#!/usr/bin/env bash
set -euo pipefail

repo_root="$1"
suite_dir="$2"
stage="$3"
cd "$repo_root"
source scripts/env.sh
export PYTHONUNBUFFERED=1
if [[ "$stage" == "collect" ]]; then
    exec python -u scripts/train_act_suite.py collect "$suite_dir"
fi
exec python -u scripts/train_act_suite.py execute "$suite_dir" "$stage" --index "$SLURM_ARRAY_TASK_ID"
