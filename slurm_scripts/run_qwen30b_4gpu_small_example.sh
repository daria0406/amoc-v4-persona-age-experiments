#!/bin/bash
#SBATCH --job-name=amoc_qwen30b_small_example
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-13%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks"

# list of chunk files
CHUNK_FILES=($(ls ${CHUNKS_DIR}/*.csv | sort))
NUM_CHUNKS=${#CHUNK_FILES[@]}

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_CHUNKS}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds number of chunks (${NUM_CHUNKS})"
    exit 1
fi

INPUT_FILE="${CHUNK_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Running Qwen 30B for a small example"
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing chunk file: ${INPUT_FILE}"

bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
    --models "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --tp 2 \
    --max-rows 1 \
    --plot-after-each-sentence \
    --output-dir "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/small_example_output" \
    --file "${INPUT_FILE}" \
    --strict-reactivate-function \
    --strict-attachament-constraint \
    --story-text "Who were the first Americans? Many, many years ago, perhaps 35,000
years ago, life was very different than it is today. At that time, 
the earth was in the grip of the last ice age. There were few people
anywhere in the world, and none lived in the Americas. People did
live in Asia, however. And some of them wandered into North
America. The firstcomers did not know they had found a new
continent. Like all ice age peoples, they were hunters. They had
to move from place to place in search of their food. Sometimes they
killed giant elephants called mammoths. Some of their spearpoints
have been found in several places in North America. Scientists say
these are about 30,000 years old. Besides hunting, men and women of
the ice age fished for their food. They also gathered wild fruits,
roots, and seeds to eat. Farming had not yet been invented. Neither
had writing. The firstcomers, therefore, did not leave any written
records. But they did leave other evidence, which scientists can
date and study. All of this evidence is important. Each item is
like a piece to a giant jigsaw puzzle. We still do not have all of
the pieces. Perhaps we never shall. But we now have enough to tell
a fairly accurate story. It is safe to say that the first
Americans were Asians. But no one is sure just when the first
group arrived. It was probably about 35,000 years ago. Other
groups followed. Even so the peopling of America from Asia was a
slow process."