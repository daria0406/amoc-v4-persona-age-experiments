#!/bin/bash
# Wrapper to execute a Python script inside the AMoC container
#!/bin/bash
#SBATCH --job-name=amoc_stats_run
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err


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

echo "Running statistical analysis inside container..."
echo "Model: $1"

export HF_HOME=/export/projects/nlp/.cache

apptainer exec --nv \
    -B "${PROJECT_DIR}:${PROJECT_DIR}" \
    -B "${INPUT_DIR}:${INPUT_DIR}" \
    "$SIF_IMAGE" \
    python3 "$@"
