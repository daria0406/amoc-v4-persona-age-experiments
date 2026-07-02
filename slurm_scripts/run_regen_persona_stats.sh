#!/bin/bash
#SBATCH --job-name=amoc_regen_stats
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

# Usage: sbatch run_regen_persona_stats.sh <triplet_dir> <out_csv> [model]
#   triplet_dir : path to the triplets_final_state directory
#   out_csv     : full path for the output persona_stats_raw.csv
#   model       : (optional) defaults to meta-llama/Llama-3.3-70B-Instruct

TRIPLET_DIR="${1:?ERROR: pass triplet_dir as \$1}"
OUT_CSV="${2:?ERROR: pass out_csv path as \$2}"
MODEL="${3:-meta-llama/Llama-3.3-70B-Instruct}"

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
SCRIPT="$HOME/to_transfer/amoc-v4-persona-age-experiments/other_helpers/regen_persona_stats.py"

echo "Triplet dir : $TRIPLET_DIR"
echo "Output CSV  : $OUT_CSV"
echo "Model       : $MODEL"

PROJECT_ROOT="$HOME/to_transfer/amoc-v4-persona-age-experiments"

apptainer exec \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    bash -c '
        export PYTHONPATH="$1:${PYTHONPATH:-}"
        python "$2" --triplet-dir "$3" --model "$4" --out "$5"
    ' bash "$PROJECT_ROOT" "$SCRIPT" "$TRIPLET_DIR" "$MODEL" "$OUT_CSV"
