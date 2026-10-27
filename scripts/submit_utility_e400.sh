#!/usr/bin/env bash
# Race two held allocations for one label run, then release only after IDs are persisted.
set -euo pipefail
: "${UNIVTAC_UTILITY_SOURCE:?Set immutable source directory}"
: "${UNIVTAC_UTILITY_RUN:?Set persistent output directory}"
mkdir -p "${UNIVTAC_UTILITY_RUN}/queue" /data/wangzihao/experiments/univtac/slurm
wave="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
queue_file="${UNIVTAC_UTILITY_RUN}/queue/${wave}.jobs"
job_ids=()
cleanup_failed_submission() {
    for job_id in "${job_ids[@]}"; do scancel "${job_id}" 2>/dev/null || true; done
}
trap cleanup_failed_submission ERR
for partition in gpu gpux; do
    max_time="$(scontrol show partition "${partition}" -o | sed -n 's/.* MaxTime=\([^ ]*\).*/\1/p')"
    [[ -n "${max_time}" && "${max_time}" != UNLIMITED ]]
    options=(--hold --parsable --account=yangyc --partition="${partition}" --time="${max_time}")
    if [[ -n "${UNIVTAC_UTILITY_DEPENDENCY:-}" ]]; then
        options+=(--dependency="${UNIVTAC_UTILITY_DEPENDENCY}")
    fi
    export_spec="ALL,UNIVTAC_UTILITY_SOURCE=${UNIVTAC_UTILITY_SOURCE},UNIVTAC_UTILITY_RUN=${UNIVTAC_UTILITY_RUN},UNIVTAC_UTILITY_QUEUE=${queue_file}"
    job_id="$(sbatch "${options[@]}" --export="${export_spec}" "${UNIVTAC_UTILITY_SOURCE}/scripts/utility_e400.sbatch")"
    job_ids+=("${job_id%%;*}")
    echo "${partition} ${max_time} ${job_id}"
done
printf '%s\n' "${job_ids[@]}" > "${queue_file}.partial"
mv "${queue_file}.partial" "${queue_file}"
for job_id in "${job_ids[@]}"; do scontrol release "${job_id}"; done
trap - ERR
