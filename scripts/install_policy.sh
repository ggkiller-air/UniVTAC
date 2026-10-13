#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-UniVTAC}"

if ! command -v conda >/dev/null 2>&1; then
    for conda_root in "$HOME/.local/miniforge3" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -f "$conda_root/etc/profile.d/conda.sh" ]]; then
            source "$conda_root/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Initialize Conda before running this script." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    conda create -n "$CONDA_ENV_NAME" --override-channels -c https://conda.anaconda.org/conda-forge python=3.10 pip 'setuptools<82' wheel -y
fi
conda activate "$CONDA_ENV_NAME"
unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
if [[ -f "$REPO_ROOT/.env" ]]; then
    source "$REPO_ROOT/.env"
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "UniVTAC requires Python 3.10"'
wheel_args=()
if [[ -n "${POLICY_FIND_LINKS:-}" ]]; then
    wheel_args=(--find-links "$POLICY_FIND_LINKS")
fi
torch_index_args=(--index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}")
policy_index_args=(--index-url "${POLICY_INDEX_URL:-https://pypi.org/simple}")
if [[ "${POLICY_OFFLINE:-0}" == "1" ]]; then
    torch_index_args=(--no-index)
    policy_index_args=(--no-index)
fi
python -m pip install uv "${policy_index_args[@]}" "${wheel_args[@]}"
python -m uv pip install --python "$CONDA_PREFIX/bin/python" torch==2.5.1 torchvision==0.20.1 "${torch_index_args[@]}" "${wheel_args[@]}"
python -m uv pip install --python "$CONDA_PREFIX/bin/python" -r "$REPO_ROOT/scripts/requirements-policy.txt" "${policy_index_args[@]}" "${wheel_args[@]}"
python -m pip check
echo "Offline ACT/Ablation/ViTAL dependencies installed in $CONDA_PREFIX."
echo "Run: source $REPO_ROOT/scripts/env.sh"
echo "Then: python $REPO_ROOT/scripts/check_policy_env.py --device cpu"
