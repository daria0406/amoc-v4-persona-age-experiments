#!/bin/bash
#SBATCH --job-name=amoc_repeated_measures
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

# Cross-text repeated-measures analysis (Friedman + Wilcoxon post-hoc).
#
# This is the PAIRED axis: the SAME personas read all texts, so we compare a
# metric ACROSS texts within each education regime. It is the complement to the
# per-text Mann-Whitney + BH that runs inside each normal stats run.

set -euo pipefail

MODEL="meta-llama/Llama-3.3-70B-Instruct"
INPUT_DIRS=("$@")

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: pass one or more input dirs." >&2
    echo "Example: sbatch $0 /path/run1 /path/run2 /path/run3 /path/run4" >&2
    exit 1
fi

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_DIR="$HOME/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/to_transfer/output/amoc_analysis}"
TEXT_DIR="${TEXT_DIR:-$PROJECT_DIR/tusa_text/min_drp_texts}"

echo "Model:       ${MODEL}"
echo "Input dirs:  ${INPUT_DIRS[*]}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Text dir:    ${TEXT_DIR}"

# Run as a module (python -m) from the project root, NOT by file path.
# Running the .py by path would put amoc/analysis/ first on sys.path, where our
# own statistics.py shadows the stdlib 'statistics' module that torch/seaborn
# import -> circular-import crash. -m + --pwd avoids that.
apptainer exec --nv \
    --pwd "$PROJECT_DIR" \
    --env PYTHONPATH="$PROJECT_DIR" \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    python -m amoc.analysis.repeated_measures \
        --input-dir "${INPUT_DIRS[@]}" \
        --model "${MODEL}" \
        --output-dir "${OUTPUT_DIR}" \
        --text-dir "${TEXT_DIR}"
