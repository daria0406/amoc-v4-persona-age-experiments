#!/bin/bash
#SBATCH --job-name=amoc_qwen_512g_baseline
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
INPUT_FILE="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks/baseline_000.csv"
STORY_FILE="${1:-}"

export HF_HOME="/export/projects/nlp/.cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_ID="run_${SLURM_JOB_ID}"
BASE_OUTPUT_DIR="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/baseline_output_qwen"
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_OUTPUT_DIR}"

echo "Running Qwen 512g baseline (no persona)"
echo "Processing file: ${INPUT_FILE}"

if [[ -n "${STORY_FILE}" ]]; then
    echo "Using story file: ${STORY_FILE}"
else
    echo "No story file provided"
fi

STORY_ARG=""
if [[ -n "${STORY_FILE}" ]]; then
    STORY_ARG="--story-text ${STORY_FILE}"
fi

bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
    --models "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8" \
    --tp 4 \
    --max-rows 1 \
    --plot-after-each-sentence \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --file "${INPUT_FILE}" \
    --strict-reactivate-function \
    ${STORY_ARG}
