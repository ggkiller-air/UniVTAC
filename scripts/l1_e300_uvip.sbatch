#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${UNIVTAC_L1_SOURCE:?immutable source}"
: "${UNIVTAC_L1_RUN:?run directory}"
cd "${UNIVTAC_L1_SOURCE}"
python_bin=/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python
export PYTHONPATH="${UNIVTAC_L1_SOURCE}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export UNIVTAC_STAGE2_STOP_FILE="${UNIVTAC_L1_RUN}/STOP_REQUEST"
export TMPDIR="/dev/shm/univtac_l1_uvip_${SLURM_JOB_ID}"
mkdir -p "${UNIVTAC_L1_RUN}" "${TMPDIR}"
exec 9>"${UNIVTAC_L1_RUN}/pipeline.lock"
flock -n 9 || exit 0
[[ ! -f "${UNIVTAC_L1_RUN}/train/complete.json" ]] || exit 0
rm -f -- "${UNIVTAC_STAGE2_STOP_FILE}"
trap 'touch "${UNIVTAC_STAGE2_STOP_FILE}"' USR1 TERM
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
"${python_bin}" - <<'PY'
import torch
assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
free,total=torch.cuda.mem_get_info()
assert free > 90*2**30,(free,total)
PY
guard="${UNIVTAC_L1_RUN}/gpu_memory_guard/${SLURM_JOB_ID}"
mkdir -p "${guard}"
ln -sfn "${UNIVTAC_L1_RUN}/train/complete.json" "${guard}/COMPLETE.json"
ln -sfn "${UNIVTAC_STAGE2_STOP_FILE}" "${guard}/SAFE_STOP.json"
rm -f -- "${guard}/gpu_memory_reservation.json"
"${python_bin}" -u "${UNIVTAC_L1_RUN}/reserve_gpu_memory.py" --hold --fraction .16 --output "${guard}" > "${guard}/holder.log" 2>&1 &
holder_pid=$!
monitor_pid=''
cleanup() {
    kill "${holder_pid}" ${monitor_pid:+"${monitor_pid}"} 2>/dev/null || true
    rmdir "${TMPDIR}" 2>/dev/null || true
}
trap cleanup EXIT
ready=0
for attempt in $(seq 1 60); do
    kill -0 "${holder_pid}" 2>/dev/null || exit 1
    if [[ -s "${guard}/gpu_memory_reservation.json" ]]; then ready=1; break; fi
    sleep 1
done
[[ "${ready}" == 1 ]] || exit 1
"${python_bin}" - "${guard}/gpu_memory_reservation.json" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
assert str(r['job_id'])==os.environ['SLURM_JOB_ID']
assert r['reserved_bytes']/r['total_bytes'] > .15
print(json.dumps(r),flush=True)
PY
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
        if ! kill -0 "${holder_pid}" 2>/dev/null; then touch "${UNIVTAC_STAGE2_STOP_FILE}"; exit 1; fi
        sleep 60
    done
) >> "${UNIVTAC_L1_RUN}/gpu_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${python_bin}" -m policy.multitask_dp.l1_uvip --config "${UNIVTAC_L1_RUN}/launch_config.yaml" --output "${UNIVTAC_L1_RUN}" &
trainer_pid=$!
result=0
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    break
done
if [[ -f "${UNIVTAC_L1_RUN}/train/complete.json" ]]; then exit 0
elif [[ -f "${UNIVTAC_STAGE2_STOP_FILE}" ]]; then scontrol requeue "${SLURM_JOB_ID}"
else
    if [[ "${result}" == 0 ]]; then result=1; fi
    exit "${result}"
fi
