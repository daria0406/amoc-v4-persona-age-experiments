#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=test_qwen_%j.out
#SBATCH --error=test_qwen_%j.err

set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"
PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
export HF_HOME="/export/projects/nlp/.cache"

# Create a temporary test script (or copy from a permanent location)
cat > /tmp/test_qwen.py << 'EOF'
from amoc.llm.vllm_client import VLLMClient

client = VLLMClient(
    model_name="Qwen/Qwen3.5-122B-A10B-FP8",
    tp_size=4,
    debug=True
)

prompt = """Extract relationships from the sentence: 'The knight rode through the forest.'
Output as a Python list of tuples, e.g., [('knight', 'rode through', 'forest')].
Do not output anything else."""

response = client.generate_raw(prompt, temperature=0.0)
print("=== TEST 1 ===")
print("Raw response:", repr(response))

from amoc.prompts.amoc_prompts import NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT
nodes = "charlemagne, famous, king"
text = "A man very close to Charlemagne wrote most of the things we know about this famous king."
prompt2 = NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT.format(nodes_from_text=nodes, text=text)
response2 = client.call_vllm(prompt2, persona="16-year-old Kiwi high school student")
print("\n=== TEST 2 ===")
print("Response:", response2)
EOF

apptainer exec --nv \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${HF_HOME}:${HF_HOME}" \
    "${SIF_IMAGE}" \
    python3 /tmp/test_qwen.py