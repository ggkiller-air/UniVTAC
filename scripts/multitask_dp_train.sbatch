#!/usr/bin/env bash
#SBATCH --job-name=univtac_fullsensor_dp_batch64_ep600
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=4-04:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/data/wangzihao/experiments/univtac/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/experiments/univtac/slurm/%x_%j.err

set -euo pipefail
: "${UNIVTAC_DP_SOURCE:?Set the immutable source snapshot}"
: "${UNIVTAC_DP_RUN:?Set the persistent run directory}"
python_bin="${UNIVTAC_DP_PYTHON:-/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python}"
cd "${UNIVTAC_DP_SOURCE}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HOME=/public/home/wangzihao/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="/dev/shm/univtac_dp_${SLURM_JOB_ID}"
export UNIVTAC_DP_STOP_FILE="${UNIVTAC_DP_RUN}/stop_${SLURM_JOB_ID}.txt"
mkdir -p "${TMPDIR}"
chmod 700 "${TMPDIR}"
trap 'printf "SIGUSR1\n" > "${UNIVTAC_DP_STOP_FILE}"' USR1
trap 'printf "SIGTERM\n" > "${UNIVTAC_DP_STOP_FILE}"' TERM
echo "date=$(date -Is) node=$(hostname) job=${SLURM_JOB_ID} physical=${SLURM_JOB_GPUS:-unknown} logical=cuda:0"
taskset -pc $$
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
df -h /dev/shm
sha256sum --check "${UNIVTAC_DP_RUN}/source.sha256"
"${python_bin}" - <<'PY'
import os
import torch

if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
    raise RuntimeError("Slurm allocation has no GPU")
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
properties = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print(f"assigned_gpu={properties.name} total_GiB={total / 2**30:.2f} free_GiB={free / 2**30:.2f}", flush=True)
if free < 60 * 2**30:
    raise RuntimeError("Allocated GPU has insufficient free memory; do not disturb unrelated processes")
PY
(
    while true; do
        date -Is
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
        sleep 300
    done
) >> "${UNIVTAC_DP_RUN}/gpu_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; rmdir "${TMPDIR}" 2>/dev/null || true' EXIT
srun --unbuffered "${python_bin}" -m policy.multitask_dp.train \
    --config "${UNIVTAC_DP_RUN}/launch_config.yaml" --output "${UNIVTAC_DP_RUN}" &
trainer_pid=$!
status=0
wait "${trainer_pid}" || status=$?
if kill -0 "${trainer_pid}" 2>/dev/null; then
    status=0
    wait "${trainer_pid}" || status=$?
fi
if (( status != 0 )); then
    exit "${status}"
fi
if "${python_bin}" - "${UNIVTAC_DP_RUN}/progress.jsonl" <<'PY'
import json
import sys
from pathlib import Path

last = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
sys.exit(0 if last.get("event") == "stopped" and last.get("reason") == "SIGUSR1" and last["epoch"] < 600 else 1)
PY
then
    sbatch --account="${SLURM_JOB_ACCOUNT}" --dependency="afterok:${SLURM_JOB_ID}" \
        --export=ALL "${UNIVTAC_DP_SOURCE}/scripts/multitask_dp_train.sbatch" \
        | tee -a "${UNIVTAC_DP_RUN}/continuation_jobs.log"
fi
