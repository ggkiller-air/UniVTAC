#!/usr/bin/env bash
# Two partition requests share one immutable source, output and exclusive training lock.
set -euo pipefail
: "${UNIVTAC_STAGE2_SOURCE:?Set immutable Transformer source}"
: "${UNIVTAC_STAGE2_RUN:?Set persistent training root}"
mkdir -p "${UNIVTAC_STAGE2_RUN}/queue" /data/wangzihao/experiments/univtac/slurm
wave="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
queue_file="${UNIVTAC_STAGE2_RUN}/queue/${wave}.jobs"
job_ids=()
cleanup_failed_submission() {
    for job_id in "${job_ids[@]}"; do scancel "${job_id}" 2>/dev/null || true; done
}
trap cleanup_failed_submission ERR
for partition in gpu gpux; do
    max_time="$(scontrol show partition "${partition}" -o | sed -n 's/.* MaxTime=\([^ ]*\).*/\1/p')"
    [[ -n "${max_time}" && "${max_time}" != UNLIMITED ]]
    options=(--hold --parsable --account=yangyc --partition="${partition}" --time="${max_time}")
    if [[ -n "${UNIVTAC_STAGE2_DEPENDENCY:-}" ]]; then
        options+=(--dependency="${UNIVTAC_STAGE2_DEPENDENCY}")
    fi
    export_spec="ALL,UNIVTAC_STAGE2_SOURCE=${UNIVTAC_STAGE2_SOURCE},UNIVTAC_STAGE2_RUN=${UNIVTAC_STAGE2_RUN},UNIVTAC_STAGE2_QUEUE=${queue_file}"
    job_id="$(sbatch "${options[@]}" --export="${export_spec}" "${UNIVTAC_STAGE2_SOURCE}/scripts/stage2_transformer.sbatch")"
    job_ids+=("${job_id%%;*}")
    echo "${partition} ${max_time} ${job_id}"
done
printf '%s\n' "${job_ids[@]}" > "${queue_file}.partial"
mv "${queue_file}.partial" "${queue_file}"
for job_id in "${job_ids[@]}"; do scontrol release "${job_id}"; done
trap - ERR
