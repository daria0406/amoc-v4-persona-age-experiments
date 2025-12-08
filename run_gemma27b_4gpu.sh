#!/bin/bash
#SBATCH --job-name=amoc_gemma3_27b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

export HUGGING_FACE_HUB_TOKEN="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"

echo "Running Gemma 3 27B..."

bash "$HOME/amoc-run.sh" \
    "$HOME/vllm_train_minimal_v4.py" \
    --models "google/gemma-3-27b-it" \
    --tp 4 \
    --replace-pronouns

# Optional: Unset the variable after use for security
unset HUGGING_FACE_HUB_TOKEN