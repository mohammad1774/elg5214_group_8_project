#!/bin/bash
# Student A helper runner for MountainCar dense+sparse on DQN Entropy and DQN RND.
# Use the thin wrappers for CPU/GPU instead of calling this directly.

set -euo pipefail

DEVICE=""
PARALLEL=1

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Usage: $0 --device cpu|gpu [--parallel N]"
    exit 1
fi

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
    # Optional: export ENV_ACTIVATE=/path/to/venv/bin/activate before running.
    # shellcheck disable=SC1090
    source "${ENV_ACTIVATE}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_ENTROPY="configs/dqn_entropy_mountaincar_8000.yaml"
CONFIG_RND="configs/dqn_rnd_mountaincar_8000.yaml"

if [[ "$DEVICE" == "cpu" ]]; then
    export JAX_PLATFORMS=cpu
    unset XLA_PYTHON_CLIENT_PREALLOCATE
    unset XLA_PYTHON_CLIENT_MEM_FRACTION
    unset CUDA_VISIBLE_DEVICES

    TOTAL_CORES="$(nproc)"
    THREADS_PER_JOB=$(( TOTAL_CORES / PARALLEL ))
    if [[ "${THREADS_PER_JOB}" -lt 1 ]]; then
        THREADS_PER_JOB=1
    fi

    export OMP_NUM_THREADS="${THREADS_PER_JOB}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}"
    export MKL_NUM_THREADS="${THREADS_PER_JOB}"
    export NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}"

    echo "Device: CPU"
    echo "Parallel jobs: ${PARALLEL}"
    echo "Total CPU cores: ${TOTAL_CORES}"
    echo "Threads per job: ${THREADS_PER_JOB}"
else
    export JAX_PLATFORMS=gpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    if [[ "${PARALLEL}" -gt 1 ]]; then
        export XLA_PYTHON_CLIENT_MEM_FRACTION="$(python3 -c "print(round(0.9 / ${PARALLEL}, 2))")"
    else
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    fi

    echo "Device: GPU"
    echo "Parallel jobs: ${PARALLEL}"
    echo "GPU mem fraction per job: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
fi

parse_scalar() {
    local config="$1"
    local expr="$2"
    python3 - "$config" "$expr" <<'PY'
import sys
import yaml

config_path, expr = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

value = data
for part in expr.split("."):
    value = value[part]

print(value)
PY
}

parse_list() {
    local config="$1"
    local expr="$2"
    python3 - "$config" "$expr" <<'PY'
import sys
import yaml

config_path, expr = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

value = data
for part in expr.split("."):
    value = value[part]

for item in value:
    print(item)
PY
}

parse_envs() {
    local config="$1"
    python3 - "$config" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

for env_item in data["sweep"]["environments"]:
    print(f"{env_item['name']} {env_item['reward']}")
PY
}

run_command_batch() {
    local -n commands_ref=$1
    local remaining="${#commands_ref[@]}"
    local fails=0

    if [[ "${remaining}" -eq 0 ]]; then
        echo "All runs already completed."
        return 0
    fi

    if [[ "${PARALLEL}" -eq 1 ]]; then
        local idx=0
        for cmd in "${commands_ref[@]}"; do
            idx=$((idx + 1))
            echo "[${idx}/${remaining}] ${cmd}"
            if ! eval "${cmd}"; then
                echo "[${idx}/${remaining}] FAILED"
                fails=$((fails + 1))
            fi
            echo
        done
    else
        local idx=0
        while [[ "${idx}" -lt "${remaining}" ]]; do
            local pids=()
            local labels=()

            for ((j=0; j<PARALLEL && idx<remaining; j++)); do
                local cmd="${commands_ref[$idx]}"
                idx=$((idx + 1))
                echo "[${idx}/${remaining}] START: ${cmd}"
                eval "${cmd}" &
                pids+=($!)
                labels+=("${idx}/${remaining}")
            done

            for k in "${!pids[@]}"; do
                if ! wait "${pids[$k]}"; then
                    echo "[${labels[$k]}] FAILED"
                    fails=$((fails + 1))
                fi
            done

            echo "--- batch done ---"
            echo
        done
    fi

    return "${fails}"
}

run_algo() {
    local algo_label="$1"
    local module_name="$2"
    local config="$3"
    local mechanism_key="$4"
    local mechanism_flag="$5"
    local token="$6"

    local results_dir
    results_dir="$(parse_scalar "${config}" "results_dir")"

    mapfile -t seeds < <(parse_list "${config}" "sweep.seeds")
    mapfile -t lrs < <(parse_list "${config}" "sweep.learning_rates")
    mapfile -t gammas < <(parse_list "${config}" "sweep.gammas")
    mapfile -t mechs < <(parse_list "${config}" "sweep.${mechanism_key}")
    mapfile -t envs < <(parse_envs "${config}")

    local total=0
    local skipped=0
    local commands=()

    echo "============================================"
    echo "  ${algo_label}"
    echo "  Config: ${config}"
    echo "  Results root: ${results_dir}"
    echo "============================================"

    for env_line in "${envs[@]}"; do
        read -r env_name reward_type <<< "${env_line}"
        for gamma in "${gammas[@]}"; do
            for lr in "${lrs[@]}"; do
                for mech in "${mechs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        total=$((total + 1))
                        output_file="${results_dir}/${env_name}_${reward_type}/lr${lr}_g${gamma}_${token}${mech}_seed${seed}.csv"
                        if [[ -f "${output_file}" ]]; then
                            skipped=$((skipped + 1))
                            continue
                        fi

                        commands+=("python3 -m ${module_name} --seed ${seed} --lr ${lr} --gamma ${gamma} ${mechanism_flag} ${mech} --env ${env_name} --reward ${reward_type} --config ${config}")
                    done
                done
            done
        done
    done

    echo "Total: ${total} | Skipped: ${skipped} | Remaining: ${#commands[@]}"
    echo

    if ! run_command_batch commands; then
        echo "${algo_label} finished with failures."
        return 1
    fi

    echo "${algo_label} finished successfully."
    echo
}

run_algo \
    "DQN + Entropy MountainCar 8000" \
    "src.test.test_dqn_entropy_agent" \
    "${CONFIG_ENTROPY}" \
    "alphas" \
    "--alpha" \
    "a"

run_algo \
    "DQN + RND MountainCar 8000" \
    "src.test.test_dqn_rnd_agent" \
    "${CONFIG_RND}" \
    "betas" \
    "--beta" \
    "b"
