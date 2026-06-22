#!/bin/bash
#SBATCH --job-name=amoc_within_text_persona
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

# Within-text persona effect: how large is the reader-age (regime) effect when
# TEXT is held fixed? Per text: Kruskal-Wallis epsilon^2 (+bootstrap CI),
# Jonckheere-Terpstra ordered-trend test, Spearman rho (+CI), and the FULL
# pairwise rank-biserial distribution (BH-corrected within text).

set -euo pipefail

MODEL="meta-llama/Llama-3.3-70B-Instruct"
INPUT_DIRS=("$@")

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: pass one or more input dirs (one per text level)." >&2
    echo "Example: sbatch $0 /path/run_primary /path/run_secondary /path/run_high /path/run_college" >&2
    exit 1
fi

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_DIR="$HOME/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/to_transfer/output/amoc_analysis}"
TEXT_DIR="${TEXT_DIR:-$PROJECT_DIR/tusa_text/min_drp_texts}"
LABELS="${LABELS:-}"
N_BOOT="${N_BOOT:-2000}"

echo "Model:       ${MODEL}"
echo "Input dirs:  ${INPUT_DIRS[*]}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Text dir:    ${TEXT_DIR}"
echo "Labels:      ${LABELS}"
echo "Bootstraps:  ${N_BOOT}"

apptainer exec --nv \
    --pwd "$PROJECT_DIR" \
    --env PYTHONPATH="$PROJECT_DIR" \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    python -m amoc.analysis.within_text_persona \
        --input-dir "${INPUT_DIRS[@]}" \
        --model "${MODEL}" \
        --output-dir "${OUTPUT_DIR}" \
        --text-dir "${TEXT_DIR}" \
        --n-boot "${N_BOOT}" \
        ${LABELS:+--label $LABELS}
