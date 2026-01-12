#!/bin/bash
#SBATCH --job-name=amoc_qwen30b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-31%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/chunks"

# Define regimes explicitly
REGIMES=(primary highschool secondary university)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#REGIMES[@]}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

REGIME="${REGIMES[$SLURM_ARRAY_TASK_ID]}"

echo "Running Qwen 30B"
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing educational regime: ${REGIME}"

bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc-run.sh" \
    --models "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --tp 2 \
    --educational-regime "${REGIME}"
