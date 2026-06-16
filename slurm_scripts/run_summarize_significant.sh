#!/bin/bash
#SBATCH --job-name=amoc_summarize_significant
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

set -euo pipefail

INPUT_DIRS=("$@")

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: pass one or more input dirs (analysis dir + per-text run dirs)." >&2
    echo "Example: sbatch $0 /path/amoc_analysis /path/run1 /path/run2 /path/run3 /path/run4" >&2
    exit 1
fi

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_DIR="$HOME/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/to_transfer/output/amoc_analysis}"
TEXT_DIR="${TEXT_DIR:-$PROJECT_DIR/tusa_text/min_drp_texts}"
OUT_CSV="${OUT_CSV:-$OUTPUT_DIR/combined_significant.csv}"

echo "Input dirs:  ${INPUT_DIRS[*]}"
echo "Text dir:    ${TEXT_DIR}"
echo "Out CSV:     ${OUT_CSV}"

apptainer exec --nv \
    --pwd "$PROJECT_DIR" \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    python "$PROJECT_DIR/helpers_run_analytics/summarize_significant.py" \
        --input-dir "${INPUT_DIRS[@]}" \
        --text-dir "${TEXT_DIR}" \
        --out "${OUT_CSV}"
