#!/bin/bash
#SBATCH --job-name=model_download_gemma
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --mem=128G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

export HF_TOKEN_DOWNLOAD="REMOVEDvBCzdZsVncIIfOerInHfKRtkAFKnXnxtay"

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_ROOT="$HOME/to_transfer/amoc-v4-persona-age-experiments/other_helpers"
MODEL_ID=$1

if [ -z "$MODEL_ID" ]; then
    echo "ERROR: No model ID provided. Usage: sbatch job_script.sh <model/id>"
    exit 1
fi


echo "Downloading $MODEL_ID..."

apptainer exec --nv \
  -B "$PROJECT_ROOT:$PROJECT_ROOT" \
  -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
  "$SIF_IMAGE" \
  python "$PROJECT_ROOT/download_model_not_in_cache.py" \
  --model_name "$MODEL_ID"

unset HF_TOKEN_DOWNLOAD

