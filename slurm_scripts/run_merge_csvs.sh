#!/bin/bash
set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"


if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Container image not found at $SIF_IMAGE"
    exit 1
fi

echo "Running triplet file merge script"
echo "Project root : $PROJECT_ROOT"

apptainer exec \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "$SIF_IMAGE" \
    bash -c "
        cd '${PROJECT_ROOT}' || exit 1
        python3 merge_csv_triplet_chunks.py
    "
