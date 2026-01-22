#!/bin/bash
#SBATCH --job-name=amoc_keefe_validation
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A.err

set -euo pipefail

# =========================
# Paths
# =========================
PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_DIR="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/keefe_validation"

# =========================
# Experiment parameters
# =========================
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
RUNS_PER_REGIME=5        # increase to 10+ for final paper runs

# =========================
# Logging
# =========================
echo "Running Keefe validation (Section 4.2)"
echo "Model: ${MODEL}"
echo "Runs per regime: ${RUNS_PER_REGIME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Host: $(hostname)"
echo "Start time: $(date)"

# =========================
# Activate environment if needed
# =========================
# source /export/home/acs/stud/a/ana_daria.zahaleanu/venv/bin/activate

# =========================
# Run experiment
# =========================
python "${PROJECT_ROOT}/analysis/keefe_validation.py" \
    --model "${MODEL}" \
    --runs-per-regime "${RUNS_PER_REGIME}" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished Keefe validation"
echo "End time: $(date)"
