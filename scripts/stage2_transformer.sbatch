#!/usr/bin/env bash
#SBATCH --job-name=univtac_stage2_transformer_t2
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/experiments/univtac/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/experiments/univtac/slurm/%x_%j.err
set -euo pipefail
: "${UNIVTAC_STAGE2_SOURCE:?}"
: "${UNIVTAC_STAGE2_RUN:?}"
: "${UNIVTAC_STAGE2_QUEUE:?}"
exec 9>"${UNIVTAC_STAGE2_RUN}/worker.lock"
if ! flock -n 9; then
    echo "The sibling allocation owns this training run; exiting."
    exit 0
fi
while read -r sibling; do
    if [[ "${sibling}" != "${SLURM_JOB_ID}" ]]; then scancel "${sibling}" 2>/dev/null || true; fi
done < "${UNIVTAC_STAGE2_QUEUE}"
cd "${UNIVTAC_STAGE2_SOURCE}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export UNIVTAC_STAGE2_STOP_FILE="${UNIVTAC_STAGE2_RUN}/stop_${SLURM_JOB_ID}.txt"
python_bin="${UNIVTAC_DP_PYTHON:-/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python}"
echo "date=$(date -Is) node=$(hostname) job=${SLURM_JOB_ID} physical=${SLURM_JOB_GPUS:-unknown} logical=cuda:0"
scontrol show job "${SLURM_JOB_ID}" -dd
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
sha256sum --check "${UNIVTAC_STAGE2_RUN}/source.sha256"
trap 'printf "scheduler_signal\n" > "${UNIVTAC_STAGE2_STOP_FILE}"' USR1 TERM
(
    while true; do
        date -Is
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
        sleep 300
    done
) >> "${UNIVTAC_STAGE2_RUN}/gpu_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
options=(--config "${UNIVTAC_STAGE2_RUN}/launch_config.yaml" --output "${UNIVTAC_STAGE2_RUN}/train" --preflight-steps 100)
if [[ -f "${UNIVTAC_STAGE2_RUN}/train/checkpoints/latest.ckpt" ]]; then
    options+=(--resume "${UNIVTAC_STAGE2_RUN}/train/checkpoints/latest.ckpt")
fi
srun --unbuffered "${python_bin}" -m policy.multitask_dp.train_stage2 "${options[@]}" &
worker_pid=$!
status=0
wait "${worker_pid}" || status=$?
if kill -0 "${worker_pid}" 2>/dev/null; then
    status=0
    wait "${worker_pid}" || status=$?
fi
if (( status != 0 )); then exit "${status}"; fi
if [[ -f "${UNIVTAC_STAGE2_RUN}/train/complete.json" ]]; then
    echo "Stage 2 training complete."
elif [[ -f "${UNIVTAC_STAGE2_RUN}/train/stopped.json" ]]; then
    export UNIVTAC_STAGE2_DEPENDENCY="afterok:${SLURM_JOB_ID}"
    bash scripts/submit_stage2_transformer.sh >> "${UNIVTAC_STAGE2_RUN}/continuation_jobs.log"
else
    echo "Worker exited without completion or a scheduler stop marker." >&2
    exit 1
fi
