#!/bin/bash
#SBATCH --job-name=amoc_llama70b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4  
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-29%1
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

PERSONAS_PER_JOB=50
START_INDEX=$((SLURM_ARRAY_TASK_ID * PERSONAS_PER_JOB))
END_INDEX=$((START_INDEX + PERSONAS_PER_JOB))
echo "Running Llama 70B..."

bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc-run.sh" \
    --models "meta-llama/Llama-3.3-70B-Instruct" \
    --tp 4 \
    --start-index $START_INDEX \
    --end-index $END_INDEX
