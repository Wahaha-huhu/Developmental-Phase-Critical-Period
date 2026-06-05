#!/usr/bin/env bash
# Overnight E1 large-scale runner for an A40 machine.
# Runs one Pythia model size at a time, backs up raw CSVs, analyses results,
# optionally deletes the Hugging Face cache for that model, and continues.
#
# Usage from repo root:
#   bash scripts/run_e1_large_models_overnight_a40.sh
# or if downloaded directly:
#   bash run_e1_large_models_overnight_a40.sh
#
# Optional environment variables:
#   HF_HOME=/workspace/hf_cache
#   DELETE_CACHE=1              # delete model cache after each model, default 1
#   CONTINUE_ON_FAIL=1          # continue to next model if one fails, default 1
#   INCLUDE_12B=0               # include sparse 12B run, default 0
#   MODEL_SEQUENCE="1p4b 2p8b 6p9b"  # override model order
#   ZIP_NAME=e1_large_overnight_a40_for_review.zip

set -u

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

DELETE_CACHE="${DELETE_CACHE:-1}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
INCLUDE_12B="${INCLUDE_12B:-0}"
ZIP_NAME="${ZIP_NAME:-e1_large_overnight_a40_for_review.zip}"

# Default: do not include 12B unless explicitly requested.
MODEL_SEQUENCE="${MODEL_SEQUENCE:-1p4b 2p8b 6p9b}"
if [[ "${INCLUDE_12B}" == "1" ]] && [[ " ${MODEL_SEQUENCE} " != *" 12b "* ]]; then
  MODEL_SEQUENCE="${MODEL_SEQUENCE} 12b"
fi

mkdir -p logs results/e1_large_scale_backups "${HF_HUB_CACHE}"

log_global="logs/e1_large_overnight_a40_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${log_global}") 2>&1

echo "============================================================"
echo "E1 large-model overnight A40 runner"
echo "Started: $(date)"
echo "Repo: $(pwd)"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "MODEL_SEQUENCE=${MODEL_SEQUENCE}"
echo "DELETE_CACHE=${DELETE_CACHE}"
echo "CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL}"
echo "INCLUDE_12B=${INCLUDE_12B}"
echo "============================================================"

echo "GPU status:"
nvidia-smi || true

echo "Disk status:"
df -h || true

echo "Generating large-scale E1 configs..."
python scripts/generate_e1_large_scale_configs.py

# Model-specific settings. Config names match the earlier A40 patch.
model_name() {
  case "$1" in
    1p4b) echo "EleutherAI/pythia-1.4b-deduped" ;;
    2p8b) echo "EleutherAI/pythia-2.8b-deduped" ;;
    6p9b) echo "EleutherAI/pythia-6.9b-deduped" ;;
    12b)  echo "EleutherAI/pythia-12b-deduped" ;;
    *) echo "UNKNOWN" ;;
  esac
}

config_path() {
  case "$1" in
    1p4b) echo "configs/e1_large_1p4b_dense_early_sparse_late.yaml" ;;
    2p8b) echo "configs/e1_large_2p8b_dense_early_sparse_late.yaml" ;;
    6p9b) echo "configs/e1_large_6p9b_dense_early_sparse_late.yaml" ;;
    12b)  echo "configs/e1_large_12b_sparse_key.yaml" ;;
    *) echo "" ;;
  esac
}

result_root() {
  case "$1" in
    1p4b) echo "results/e1_large_1p4b_dense_early_sparse_late" ;;
    2p8b) echo "results/e1_large_2p8b_dense_early_sparse_late" ;;
    6p9b) echo "results/e1_large_6p9b_dense_early_sparse_late" ;;
    12b)  echo "results/e1_large_12b_sparse_key" ;;
    *) echo "" ;;
  esac
}

analysis_root() {
  case "$1" in
    1p4b) echo "results/e1_large_1p4b_boundary_analysis" ;;
    2p8b) echo "results/e1_large_2p8b_boundary_analysis" ;;
    6p9b) echo "results/e1_large_6p9b_boundary_analysis" ;;
    12b)  echo "results/e1_large_12b_boundary_analysis" ;;
    *) echo "" ;;
  esac
}

cache_dir() {
  case "$1" in
    1p4b) echo "${HF_HUB_CACHE}/models--EleutherAI--pythia-1.4b-deduped" ;;
    2p8b) echo "${HF_HUB_CACHE}/models--EleutherAI--pythia-2.8b-deduped" ;;
    6p9b) echo "${HF_HUB_CACHE}/models--EleutherAI--pythia-6.9b-deduped" ;;
    12b)  echo "${HF_HUB_CACHE}/models--EleutherAI--pythia-12b-deduped" ;;
    *) echo "" ;;
  esac
}

backup_name() {
  case "$1" in
    1p4b) echo "results/e1_large_scale_backups/e1_large_1p4b_dense_early_sparse_late.csv" ;;
    2p8b) echo "results/e1_large_scale_backups/e1_large_2p8b_dense_early_sparse_late.csv" ;;
    6p9b) echo "results/e1_large_scale_backups/e1_large_6p9b_dense_early_sparse_late.csv" ;;
    12b)  echo "results/e1_large_scale_backups/e1_large_12b_sparse_key.csv" ;;
    *) echo "" ;;
  esac
}

run_one_model() {
  local key="$1"
  local cfg="$(config_path "${key}")"
  local root="$(result_root "${key}")"
  local analysis="$(analysis_root "${key}")"
  local model="$(model_name "${key}")"
  local cache="$(cache_dir "${key}")"
  local backup="$(backup_name "${key}")"
  local per_log="logs/e1_large_${key}_$(date +%Y%m%d_%H%M%S).log"

  echo ""
  echo "============================================================"
  echo "Starting ${key}: ${model}"
  echo "Config: ${cfg}"
  echo "Result root: ${root}"
  echo "Analysis root: ${analysis}"
  echo "Time: $(date)"
  echo "============================================================"

  if [[ ! -f "${cfg}" ]]; then
    echo "ERROR: Missing config ${cfg}"
    return 10
  fi

  echo "Disk before ${key}:"
  df -h || true
  du -sh "${HF_HOME}" "${HF_HUB_CACHE}" 2>/dev/null || true

  # Run collection. Capture exit rather than exiting whole overnight job.
  set +e
  python scripts/run_e1_collect_spectra.py --config "${cfg}" 2>&1 | tee "${per_log}"
  local run_code=${PIPESTATUS[0]}
  set -e

  if [[ ${run_code} -ne 0 ]]; then
    echo "ERROR: run_e1_collect_spectra failed for ${key} with code ${run_code}"
    echo "Log: ${per_log}"
    if [[ "${DELETE_CACHE}" == "1" ]]; then
      echo "Deleting cache for failed ${key} to protect disk: ${cache}"
      rm -rf "${cache}" || true
      find "${HF_HUB_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null || true
    fi
    return ${run_code}
  fi

  local metrics="${root}/raw/e1_spectral_metrics.csv"
  if [[ ! -f "${metrics}" ]]; then
    echo "ERROR: Expected metrics CSV not found: ${metrics}"
    return 11
  fi

  echo "Backing up metrics to ${backup}"
  mkdir -p "$(dirname "${backup}")"
  cp "${metrics}" "${backup}"

  echo "Analyzing ${key}..."
  set +e
  python scripts/analyze_e1_dense_boundary.py \
    --metrics "${metrics}" \
    --out "${analysis}" \
    --model "${model}"
  local ana_code=$?
  set -e

  if [[ ${ana_code} -ne 0 ]]; then
    echo "WARNING: analysis failed for ${key} with code ${ana_code}. Metrics backup still exists."
  else
    echo "Analysis complete: ${analysis}"
    if [[ -f "${analysis}/reports/e1_dense_boundary_report.md" ]]; then
      echo "--- ${key} report preview ---"
      sed -n '1,120p' "${analysis}/reports/e1_dense_boundary_report.md" || true
      echo "--- end preview ---"
    fi
  fi

  if [[ "${DELETE_CACHE}" == "1" ]]; then
    echo "Deleting model cache for ${key}: ${cache}"
    rm -rf "${cache}" || true
    find "${HF_HUB_CACHE}" -type d -name "*.incomplete" -exec rm -rf {} + 2>/dev/null || true
  else
    echo "DELETE_CACHE=0, keeping cache for ${key}."
  fi

  echo "Disk after ${key}:"
  df -h || true
  du -sh "${HF_HOME}" "${HF_HUB_CACHE}" 2>/dev/null || true

  return 0
}

FAILED_MODELS=()
SUCCEEDED_MODELS=()

# Make bash return non-zero from run_one_model without killing loop.
set +e
for key in ${MODEL_SEQUENCE}; do
  run_one_model "${key}"
  code=$?
  if [[ ${code} -ne 0 ]]; then
    FAILED_MODELS+=("${key}:${code}")
    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      echo "Stopping because CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL}."
      break
    fi
  else
    SUCCEEDED_MODELS+=("${key}")
  fi
 done
set -e

echo ""
echo "============================================================"
echo "Run summary"
echo "Succeeded: ${SUCCEEDED_MODELS[*]:-none}"
echo "Failed: ${FAILED_MODELS[*]:-none}"
echo "Finished: $(date)"
echo "============================================================"

# Create review zip. Include only paths that exist.
ZIP_PATH="${ZIP_NAME}"
echo "Creating review ZIP: ${ZIP_PATH}"
paths_to_zip=(
  "results/e1_large_scale_backups"
  "configs/e1_large_1p4b_dense_early_sparse_late.yaml"
  "configs/e1_large_2p8b_dense_early_sparse_late.yaml"
  "configs/e1_large_6p9b_dense_early_sparse_late.yaml"
  "configs/e1_large_12b_sparse_key.yaml"
  "logs"
)
for key in 1p4b 2p8b 6p9b 12b; do
  a="$(analysis_root "${key}")"
  r="$(result_root "${key}")"
  [[ -d "${a}" ]] && paths_to_zip+=("${a}")
  # Do not include huge raw roots if backup exists, but include reports/manifests if small.
  [[ -d "${r}/reports" ]] && paths_to_zip+=("${r}/reports")
  [[ -d "${r}/tables" ]] && paths_to_zip+=("${r}/tables")
  [[ -d "${r}/processed" ]] && paths_to_zip+=("${r}/processed")
  [[ -d "${r}/manifests" ]] && paths_to_zip+=("${r}/manifests")
 done

# shellcheck disable=SC2048,SC2086
zip -r "${ZIP_PATH}" ${paths_to_zip[*]} 2>/dev/null || true

echo "Review ZIP: ${ZIP_PATH}"
echo "Global log: ${log_global}"
