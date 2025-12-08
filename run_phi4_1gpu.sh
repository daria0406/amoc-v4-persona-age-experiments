#!/bin/bash
#SBATCH --job-name=amoc_phi4
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

echo "Running Phi-4..."

bash "$HOME/amoc-run.sh" \
    "$HOME/vllm_train_minimal_v4.py" \
    --models "microsoft/phi-4" \
    --tp 1 \
    --replace-pronouns
