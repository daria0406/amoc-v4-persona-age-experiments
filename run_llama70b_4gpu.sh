#!/bin/bash
#SBATCH --job-name=amoc_llama70b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4  
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

echo "Running Llama 70B..."

bash /export/home/acs/stud/a/ana_daria.zahaleanu/amoc-run.sh \
    /export/home/acs/stud/a/ana_daria.zahaleanu/vllm_train_minimal_v4.py \
    --models "meta-llama/Llama-3.3-70B-Instruct" \
    --tp 4 \
    --replace-pronouns
