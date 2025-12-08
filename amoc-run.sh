#!/bin/bash
SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_DIR="/export/projects/nlp"
INPUT_DIR="/export/home/acs/stud/a/ana_daria.zahaleanu/personas_dfs"
OUTPUT_DIR="/export/projects/nlp/daria_amoc_output"

mkdir -p "${OUTPUT_DIR}/extracted_triplets"
mkdir -p "${OUTPUT_DIR}/amoc_graphs"
mkdir -p "${OUTPUT_DIR}/amoc_analysis"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Container image not found at $SIF_IMAGE"
    exit 1
fi

echo "Starting Container..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# HuggingFace cache (inside container too)
export HF_HOME=/export/projects/nlp/.cache

apptainer exec --nv \
    -B "${PROJECT_DIR}:${PROJECT_DIR}" \
    -B "${INPUT_DIR}:${INPUT_DIR}" \
    "$SIF_IMAGE" \
    python3 "$@"
