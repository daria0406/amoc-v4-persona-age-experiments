import os
import sys
import time
import logging
import argparse
from typing import List

# --- Multiprocessing safety (vLLM + CUDA) ---
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"

# --- AMoC imports ---
from amoc.config import INPUT_DIR, OUTPUT_DIR, BLUE_NODES
from amoc.pipeline.runner import process_persona_csv
from amoc.analysis.statistics import run_statistical_analysis
from amoc.nlp import load_spacy


# ==========================================
# CLI ARGUMENTS
# ==========================================


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run AMoCv4 over persona CSVs using age-aware, persona-aware prompts."
        )
    )

    p.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated list of vLLM model names "
            "(e.g. 'Qwen/Qwen3-30B-A3B-Instruct-2507,openai/gpt-oss-120b')"
        ),
    )

    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on rows per CSV (for testing).",
    )

    p.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index for persona slicing.",
    )

    p.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for persona slicing (exclusive).",
    )

    p.add_argument(
        "--replace-pronouns",
        action="store_true",
        help="Enable pronoun resolution in AMoC.",
    )

    p.add_argument(
        "--tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tp_size",
        help="Tensor parallel size for vLLM.",
    )

    p.add_argument(
        "--resume-only",
        action="store_true",
        help="Only process personas not yet completed (checkpoint-based).",
    )

    return p.parse_args(argv)


# ==========================================
# SHARD COORDINATION
# ==========================================


def write_shard_done_marker():
    shard_id = os.environ.get("SLURM_ARRAY_TASK_ID", "single")
    done_dir = os.path.join(OUTPUT_DIR, "shard_done")
    os.makedirs(done_dir, exist_ok=True)

    marker = os.path.join(done_dir, f"shard_{shard_id}.done")
    with open(marker, "w") as f:
        f.write("done\n")


def all_shards_done(expected_shards: int) -> bool:
    done_dir = os.path.join(OUTPUT_DIR, "shard_done")
    if not os.path.isdir(done_dir):
        return False

    done_files = [
        f
        for f in os.listdir(done_dir)
        if f.startswith("shard_") and f.endswith(".done")
    ]
    return len(done_files) >= expected_shards


def is_leader() -> bool:
    return os.environ.get("SLURM_ARRAY_TASK_ID") in (None, "0")


# ==========================================
# MAIN
# ==========================================


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise SystemExit("--models must contain at least one model")

    # --- Load spaCy ONCE per process ---
    spacy_nlp = load_spacy()
    if spacy_nlp is None:
        raise RuntimeError("spaCy failed to load")

    # --- Discover input files ---
    files_to_process = sorted(
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".csv")
    )

    if not files_to_process:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    print(f"Discovered {len(files_to_process)} persona CSV files")
    print(f"Models: {model_names}")
    print(f"Output directory: {OUTPUT_DIR}")

    total_start = time.time()

    try:
        for filename in files_to_process:
            print(f"\n=== Processing file: {os.path.basename(filename)} ===")
            process_persona_csv(
                filename=filename,
                model_names=model_names,
                spacy_nlp=spacy_nlp,
                output_dir=OUTPUT_DIR,
                max_rows=args.max_rows,
                replace_pronouns=args.replace_pronouns,
                tensor_parallel_size=args.tp_size,
                resume_only=args.resume_only,
                start_index=args.start_index,
                end_index=args.end_index,
                plot_after_each_sentence=False,
                graphs_output_dir=OUTPUT_DIR,
                highlight_nodes=BLUE_NODES,
            )
    finally:
        elapsed = time.time() - total_start
        print(f"\nExtraction phase finished in {elapsed:.2f} seconds")

        # --- Mark shard completion ---
        write_shard_done_marker()

        # --- Statistics (leader-only, after all shards) ---
        EXPECTED_SHARDS = 32  # adjust

        if is_leader():
            print("Leader waiting for all shards to complete...")
            while not all_shards_done(EXPECTED_SHARDS):
                time.sleep(30)

            print("All shards completed. Running statistics.")

            for model in model_names:
                try:
                    run_statistical_analysis(model)
                except Exception as e:
                    logging.error(
                        f"Statistical analysis failed for {model}: {e}",
                        exc_info=True,
                    )
                    print(f"[ERROR] Statistics failed for {model}")


if __name__ == "__main__":
    main(sys.argv[1:])
