#!/bin/bash
#SBATCH --job-name=amoc_qwen30b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

echo "Running Qwen 30B..."

bash "$HOME/amoc-run.sh" \
    "$HOME/vllm_train_minimal_v4.py" \
    --models "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --tp 2 \
    --replace-pronouns
