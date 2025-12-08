#!/bin/bash
#SBATCH --job-name=amoc_gptoss120b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4  
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

echo "Running GPT-OSS-120B..."

bash "$HOME/amoc-run.sh" \
    "$HOME/vllm_train_minimal_v4.py" \
    --models "openai/gpt-oss-120b" \
    --tp 4 \
    --replace-pronouns
