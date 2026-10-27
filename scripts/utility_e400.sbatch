#!/usr/bin/env bash
#SBATCH --job-name=univtac_E400_K64_labels
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/experiments/univtac/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/experiments/univtac/slurm/%x_%j.err
set -euo pipefail
: "${UNIVTAC_UTILITY_SOURCE:?}"
: "${UNIVTAC_UTILITY_RUN:?}"
: "${UNIVTAC_UTILITY_QUEUE:?}"
exec 9>"${UNIVTAC_UTILITY_RUN}/worker.lock"
if ! flock -n 9; then
    echo "The sibling allocation owns this label run; exiting."
    exit 0
fi
while read -r sibling; do
    if [[ "${sibling}" != "${SLURM_JOB_ID}" ]]; then scancel "${sibling}" 2>/dev/null || true; fi
done < "${UNIVTAC_UTILITY_QUEUE}"
cd "${UNIVTAC_UTILITY_SOURCE}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export UNIVTAC_UTILITY_STOP_FILE="${UNIVTAC_UTILITY_RUN}/stop_${SLURM_JOB_ID}.txt"
python_bin="${UNIVTAC_DP_PYTHON:-/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python}"
echo "date=$(date -Is) node=$(hostname) job=${SLURM_JOB_ID} physical=${SLURM_JOB_GPUS:-unknown} logical=cuda:0"
scontrol show job "${SLURM_JOB_ID}" -dd
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
sha256sum --check "${UNIVTAC_UTILITY_RUN}/source.sha256"
trap 'printf "scheduler_signal\n" > "${UNIVTAC_UTILITY_STOP_FILE}"' USR1 TERM
(
    while true; do
        date -Is
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
        sleep 300
    done
) >> "${UNIVTAC_UTILITY_RUN}/gpu_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
srun --unbuffered "${python_bin}" -m policy.multitask_dp.generate_utility \
    --teacher /data/wangzihao/experiments/univtac/full_sensor_dp_batch64_ep600_seed42_20261019_093221/checkpoints/epoch0400.ckpt \
    --cache /data/wangzihao/datasets/univtac/cache/full_sensor_joint_next_bgr_u8_v1 \
    --output "${UNIVTAC_UTILITY_RUN}/labels" --batch-size 16 --microbatch 128 \
    --shard-size 64 --workers 4 --memory-floor 0.16 &
worker_pid=$!
status=0
wait "${worker_pid}" || status=$?
if kill -0 "${worker_pid}" 2>/dev/null; then
    status=0
    wait "${worker_pid}" || status=$?
fi
if (( status != 0 )); then exit "${status}"; fi
if [[ -f "${UNIVTAC_UTILITY_RUN}/labels/complete.json" ]]; then
    echo "Label generation complete; student training awaits user instruction."
elif [[ -f "${UNIVTAC_UTILITY_RUN}/labels/stopped.json" ]]; then
    export UNIVTAC_UTILITY_DEPENDENCY="afterok:${SLURM_JOB_ID}"
    bash scripts/submit_utility_e400.sh >> "${UNIVTAC_UTILITY_RUN}/continuation_jobs.log"
else
    echo "Worker exited without completion or resumable stop marker." >&2
    exit 1
fi
