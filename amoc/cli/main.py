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
from amoc.config import INPUT_DIR, OUTPUT_DIR, OUTPUT_ANALYSIS_DIR, BLUE_NODES
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

    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for extracted triplets (overrides config).",
    )

    p.add_argument(
        "--plot-after-each-sentence",
        action="store_true",
        help="Plot a graph after each sentence for a specific persona.",
    )

    p.add_argument(
        "--plot-final-graph",
        action="store_true",
        help="Plot a single final graph per persona (disables per-sentence plotting).",
    )

    p.add_argument(
        "--plot-largest-component-only",
        action="store_true",
        help="When plotting final graph, keep only the largest connected component (default: plot all).",
    )

    # p.add_argument(
    #     "--educational-regime",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Educational regime to process (e.g. primary, highschool). "
    #         "If set, only CSVs matching <regime>_*.csv are processed."
    #     ),
    # )

    p.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to a single persona CSV chunk file to process.",
    )

    return p.parse_args(argv)


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
    if not os.path.isfile(args.file):
        raise RuntimeError(f"Input file does not exist: {args.file}")

    files_to_process = [args.file]

    if not files_to_process:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    print(f"Discovered {len(files_to_process)} persona CSV files")
    print(f"Models: {model_names}")
    print(f"Output directory: {OUTPUT_DIR}")

    total_start = time.time()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir
        print(f"Overriding output directory to: {output_dir}")
    else:
        output_dir = OUTPUT_DIR

    try:
        for filename in files_to_process:
            print(f"\n=== Processing file: {os.path.basename(filename)} ===")
            process_persona_csv(
                filename=filename,
                model_names=model_names,
                spacy_nlp=spacy_nlp,
                output_dir=output_dir,
                max_rows=args.max_rows,
                replace_pronouns=args.replace_pronouns,
                tensor_parallel_size=args.tp_size,
                resume_only=args.resume_only,
                plot_after_each_sentence=args.plot_after_each_sentence,
                graphs_output_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "graphs"),
                highlight_nodes=BLUE_NODES,
                plot_final_graph=args.plot_final_graph,
                plot_largest_component_only=args.plot_largest_component_only,
            )
    finally:
        elapsed = time.time() - total_start
        print(f"\nExtraction phase finished in {elapsed:.2f} seconds")

        # Leader-only cumulative statistics:
        if is_leader():
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
