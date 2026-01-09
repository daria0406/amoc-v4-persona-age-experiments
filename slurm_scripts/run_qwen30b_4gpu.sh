#!/bin/bash
#SBATCH --job-name=amoc_qwen30b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-31%1
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

PERSONAS_PER_JOB=50
START_INDEX=$((SLURM_ARRAY_TASK_ID * PERSONAS_PER_JOB))
END_INDEX=$((START_INDEX + PERSONAS_PER_JOB))

echo "Running Qwen 30B..."
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing personas ${START_INDEX} → ${END_INDEX}"

bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc-run.sh" \
    --models "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --tp 2 \
    --start-index ${START_INDEX} \
    --end-index ${END_INDEX}
    # --max-rows 2
    # --replace-pronouns \
