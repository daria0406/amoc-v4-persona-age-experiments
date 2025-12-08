#!/bin/bash
#SBATCH --job-name=amoc_generate_dfs
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4  
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

OUT_FILE="$1"
echo "Running Llama 70b..."

bash "$HOME/amoc-run.sh" \
    "$HOME/generate_dfs_edu_personas_gpu_2.py" \
    --model "meta-llama/Llama-3.3-70B-Instruct" \
    --file "$OUT_FILE" \
    --tensor_parallel_size 4 \
    --batch_size 32 \
    --min_confidence 80
