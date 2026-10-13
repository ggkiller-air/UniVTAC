#!/usr/bin/env bash

UNIVTAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v conda >/dev/null 2>&1; then
    for conda_root in "$HOME/.local/miniforge3" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -f "$conda_root/etc/profile.d/conda.sh" ]]; then
            source "$conda_root/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found." >&2
    return 1 2>/dev/null || exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-UniVTAC}" || { return 1 2>/dev/null || exit 1; }
unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
if [[ -f "$UNIVTAC_ROOT/.env" ]]; then
    source "$UNIVTAC_ROOT/.env"
fi
export UNIVTAC_ROOT
export UNIVTAC_DATA_ROOT="${UNIVTAC_DATA_ROOT:-$UNIVTAC_ROOT/data}"
export UNIVTAC_RAW_DIR="$UNIVTAC_DATA_ROOT"
export UNIVTAC_PROCESSED_ROOT="${UNIVTAC_PROCESSED_ROOT:-$UNIVTAC_DATA_ROOT/processed/ACT/author}"
export UNIVTAC_TASK_CONFIGS="${UNIVTAC_TASK_CONFIGS:-$UNIVTAC_PROCESSED_ROOT/SIM_TASK_CONFIGS.json}"
export UNIVTAC_RUN_ROOT="${UNIVTAC_RUN_ROOT:-$HOME/experiments/univtac}"
export UNIVTAC_TACTILE_CHECKPOINT="${UNIVTAC_TACTILE_CHECKPOINT:-$UNIVTAC_DATA_ROOT/checkpoints/encoder.pth}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND=Agg
