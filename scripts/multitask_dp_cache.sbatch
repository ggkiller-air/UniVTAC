#!/usr/bin/env bash
#SBATCH --job-name=univtac_dp_decode
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/wangzihao/experiments/univtac/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/experiments/univtac/slurm/%x_%j.err

set -euo pipefail
: "${UNIVTAC_DP_SOURCE:?Set the immutable source directory}"
: "${UNIVTAC_DP_CACHE:?Set the persistent cache directory}"
python_bin="${UNIVTAC_DP_PYTHON:-/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python}"
cd "${UNIVTAC_DP_SOURCE}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HOME=/public/home/wangzihao/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${python_bin}" -m policy.multitask_dp.cache \
    --root "${UNIVTAC_DP_DATA_ROOT:-/data/wangzihao/datasets/univtac/isaac45}" \
    --output "${UNIVTAC_DP_CACHE}" --workers "${SLURM_CPUS_PER_TASK:-16}"
