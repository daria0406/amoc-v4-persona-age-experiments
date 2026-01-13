import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd

from amoc.pipeline import AgeAwareAMoCEngine
from amoc.llm.vllm_client import VLLMClient
from amoc.config.constants import DEBUG
from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from amoc.pipeline.io import (
    robust_read_persona_csv,
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    infer_regime_from_filename,
)
from amoc.viz.graph_plots import plot_amoc_triplets

VLLM_CLIENT_CACHE: Dict[str, VLLMClient] = {}

CONTROL_TOKENS = {
    "|eot_id|",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<s>",
    "</s>",
    "assistant",
    "user",
    "system",
}

CSV_HEADERS = [
    "original_index",
    "age_refined",
    "persona_text",
    "model_name",
    "subject",
    "relation",
    "object",
    "regime",
    "active",
]


def format_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def is_bad(x: str) -> bool:
    if not isinstance(x, str):
        return False
    return any(tok in x for tok in CONTROL_TOKENS)


def repair_triplet(e1: str, e2: str, e3: str):
    e1, e2, e3 = str(e1).strip(), str(e2).strip(), str(e3).strip()

    # Fix subject
    if is_bad(e1):
        e1 = e3 if not is_bad(e3) else "UNKNOWN"

    # Fix object
    if is_bad(e3):
        e3 = e1 if not is_bad(e1) else "UNKNOWN"

    # Fix relation
    if is_bad(e2) or e2 == "":
        e2 = "related_to"

    return e1, e2, e3


def process_persona_csv(
    filename: str,
    model_names: List[str],
    spacy_nlp,
    output_dir: str,
    max_rows: Optional[int] = None,
    replace_pronouns: bool = False,
    tensor_parallel_size: int = 1,
    resume_only: bool = False,
    plot_after_each_sentence: bool = False,
    graphs_output_dir: Optional[str] = None,
    highlight_nodes: Optional[List[str]] = None,
    plot_final_graph: bool = False,
    plot_largest_component_only: bool = False,
) -> None:
    short_filename = os.path.basename(filename)
    print(f"\n=== Processing File (chunk): {short_filename} ===")

    # --- Path normalization (CRITICAL FIX) ---
    output_dir = Path(output_dir)

    # 1. Load data (entire chunk)
    df = robust_read_persona_csv(filename)

    # Ensure required columns
    if "persona_text" not in df.columns or "age_refined" not in df.columns:
        print(
            f"   [Skip] File {short_filename} missing "
            f"'persona_text' or 'age_refined' columns."
        )
        return

    regime = infer_regime_from_filename(filename)
    df["age_refined"] = pd.to_numeric(df["age_refined"], errors="coerce")

    # Optional row cap (debug / testing only)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    if df.empty:
        print(f"   [Skip] File {short_filename} has no valid rows.")
        return

    # 2. Initialize engines
    engines: Dict[str, AgeAwareAMoCEngine] = {}

    for model_name in model_names:
        if model_name not in VLLM_CLIENT_CACHE:
            VLLM_CLIENT_CACHE[model_name] = VLLMClient(
                model_name=model_name,
                tp_size=tensor_parallel_size,
                debug=DEBUG,
            )

        engines[model_name] = AgeAwareAMoCEngine(
            vllm_client=VLLM_CLIENT_CACHE[model_name],
            spacy_nlp=spacy_nlp,
        )
    # 3. For each model, collect triplets into a table (incremental write)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Process per model
    for model_name, engine in engines.items():
        safe_model_name = model_name.replace(":", "-").replace("/", "-")
        output_filename = f"model_{safe_model_name}_triplets_{short_filename}"
        output_path = output_dir / output_filename

        ckpt_path = get_checkpoint_path(
            output_dir=str(output_dir),
            short_filename=short_filename,
            model_name=model_name,
        )

        ckpt = load_checkpoint(ckpt_path)
        processed_indices = set(ckpt.get("processed_indices", []))
        failures = ckpt.get("failures", [])
        personas_processed = ckpt.get("personas_processed", 0)

        print(f"[Model] {model_name}")
        print(f"  → Output: {output_path}")
        print(f"  → Checkpoint: {ckpt_path}")

        if not output_path.exists():
            pd.DataFrame([], columns=CSV_HEADERS).to_csv(
                output_path, index=False, encoding="utf-8"
            )

        start_model_time = time.time()
        total_rows = len(df)

        try:
            for idx, (row_idx, row) in enumerate(df.iterrows(), start=1):
                if row_idx in processed_indices:
                    continue

                persona_text = str(row["persona_text"])
                age_refined_int = (
                    int(row["age_refined"]) if pd.notna(row["age_refined"]) else -1
                )

                try:
                    triplets = engine.run(
                        persona_text=persona_text,
                        age_refined=age_refined_int,
                        replace_pronouns=replace_pronouns,
                        plot_after_each_sentence=plot_after_each_sentence,
                        graphs_output_dir=graphs_output_dir,
                        highlight_nodes=highlight_nodes,
                    )

                    records = []
                    for trip in triplets:
                        # Support both legacy 3-tuple and new 4-tuple with active flag
                        if len(trip) == 4:
                            s, r, o, active = trip
                        else:
                            s, r, o = trip
                            active = True
                        s, r, o = repair_triplet(s, r, o)
                        records.append(
                            {
                                "original_index": row_idx,
                                "age_refined": age_refined_int,
                                "persona_text": persona_text,
                                "model_name": model_name,
                                "subject": s,
                                "relation": r,
                                "object": o,
                                "regime": regime,
                                "active": bool(active),
                            }
                        )

                    if records:
                        pd.DataFrame(records).to_csv(
                            output_path,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                    personas_processed += 1
                    processed_indices.add(row_idx)

                    ckpt["personas_processed"] = personas_processed
                    ckpt["processed_indices"] = sorted(processed_indices)
                    save_checkpoint(ckpt_path, ckpt)

                    if plot_final_graph and records:
                        active_triplets = [
                            (rec["subject"], rec["relation"], rec["object"])
                            for rec in records
                            if rec.get("active", True)
                        ]
                        if active_triplets:
                            plot_dir = graphs_output_dir or os.path.join(
                                OUTPUT_ANALYSIS_DIR, "graphs"
                            )
                            plot_amoc_triplets(
                                triplets=active_triplets,
                                persona=persona_text,
                                model_name=model_name,
                                age=age_refined_int,
                                blue_nodes=highlight_nodes,
                                output_dir=plot_dir,
                                step_tag="final",
                                largest_component_only=plot_largest_component_only,
                            )

                except Exception as e:
                    logging.error(
                        f"Failure idx={row_idx}, model={model_name}",
                        exc_info=True,
                    )
                    failures.append(
                        {
                            "row_index": int(row_idx),
                            "persona_snippet": persona_text[:80],
                            "error": str(e),
                            "time": datetime.utcnow().isoformat(),
                        }
                    )
                    ckpt["failures"] = failures
                    save_checkpoint(ckpt_path, ckpt)

        finally:
            ckpt["elapsed_seconds"] = time.time() - start_model_time
            ckpt["failures"] = failures
            save_checkpoint(ckpt_path, ckpt)

            print(
                f"[Model] {model_name}: processed {personas_processed} "
                f"personas from {short_filename}"
            )
