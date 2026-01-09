import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


import pandas as pd

from amoc.pipeline import AgeAwareAMoCEngine
from amoc.llm.vllm_client import VLLMClient
from amoc.config.constants import DEBUG
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
    start_index: int = 0,
    end_index: Optional[int] = None,
    plot_after_each_sentence: bool = False,
    graphs_output_dir: Optional[str] = None,
    highlight_nodes: Optional[List[str]] = None,
) -> None:
    short_filename = os.path.basename(filename)
    print(f"\n=== Processing File: {short_filename} ===")

    # 1. Load Data
    df = robust_read_persona_csv(filename)
    # --- Apply slicing ---
    end_index = end_index or len(df)
    df = df.iloc[start_index:end_index]

    # Ensure required columns
    if "persona_text" not in df.columns or "age_refined" not in df.columns:
        print(
            f"   [Skip] File {short_filename} missing 'persona_text' or 'age_refined' columns."
        )
        return

    # Extract age bin:
    regime = infer_regime_from_filename(filename)

    # Ensure valid age_refined
    df["age_refined"] = pd.to_numeric(df["age_refined"], errors="coerce")

    # Apply max_rows limit if provided
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    if df.empty:
        print(
            f"   [Skip] File {short_filename} has no valid rows after age_refined filtering."
        )
        return

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
    os.makedirs(output_dir, exist_ok=True)

    for model_name, engine in engines.items():
        safe_model_name = model_name.replace(":", "-").replace("/", "-")
        output_filename = f"model_{safe_model_name}_triplets_{short_filename}"
        if not output_filename.lower().endswith(".csv"):
            output_filename += ".csv"
        output_path = os.path.join(output_dir, output_filename)

        # checkpoint path for this model+file
        ckpt_path = get_checkpoint_path(
            output_dir,
            short_filename,
            model_name,
            start_index=start_index,
            end_index=end_index,
        )
        ckpt = load_checkpoint(ckpt_path)
        processed_indices = set(ckpt.get("processed_indices", []))
        failures = ckpt.get("failures", [])
        personas_processed = ckpt.get("personas_processed", 0)

        print(f"   [Model] {model_name}: writing to {output_path}")
        print(
            f"   [Model] {model_name}: checkpoint at {ckpt_path} "
            f"(already processed {personas_processed} personas; "
            f"{len(processed_indices)} indices)"
        )

        # Ensure CSV exists with header (once)
        if not os.path.isfile(output_path):
            pd.DataFrame([], columns=CSV_HEADERS).to_csv(
                output_path, index=False, encoding="utf-8"
            )
            print(f"   [Model] {model_name}: initialized empty CSV at {output_path}")

        # all_extracted_data: List[Dict[str, Any]] = []
        start_model_time = time.time()
        total_personas_in_slice = len(df)
        last_progress_log = 0
        PROGRESS_LOG_EVERY = 10
        MAX_RUNTIME = 60 * 60 * 3.5  # 3.5 hours

        try:
            for idx, (row_idx, row) in enumerate(df.iterrows(), start=1):
                if time.time() - start_model_time > MAX_RUNTIME:
                    print("Approaching walltime, exiting safely.")
                    return
                # Skip if already processed in a previous run
                if resume_only and start_index != 0:
                    raise RuntimeError(
                        "resume_only is not supported with array slicing unless checkpoints are slice-scoped"
                    )
                else:
                    # Normal mode → process everything
                    # But still skip already-processed rows to avoid double-writing
                    if row_idx in processed_indices:
                        continue

                persona_text = str(row["persona_text"])
                age_refined = row["age_refined"]

                try:
                    age_refined_int = int(age_refined)
                except Exception:
                    age_refined_int = (
                        int(float(age_refined)) if pd.notna(age_refined) else -1
                    )

                print(
                    f"      [{idx}/{len(df)}] "
                    f"Age: {age_refined_int} | Persona: {persona_text[:50]}..."
                )

                start_time = time.time()

                try:
                    triplets = engine.run(
                        persona_text=persona_text,
                        age_refined=age_refined_int,
                        replace_pronouns=replace_pronouns,
                        plot_after_each_sentence=plot_after_each_sentence,
                        graphs_output_dir=graphs_output_dir,
                        highlight_nodes=highlight_nodes,
                    )
                    row_records: List[Dict[str, Any]] = []
                    for s, r, o in triplets:
                        s, r, o = repair_triplet(s, r, o)
                        rec = {
                            "original_index": row_idx,
                            "age_refined": age_refined_int,
                            "persona_text": persona_text,
                            "model_name": model_name,
                            "subject": s,
                            "relation": r,
                            "object": o,
                            "regime": regime,
                        }
                        # all_extracted_data.append(rec)
                        row_records.append(rec)

                    # Incremental flush per persona
                    if row_records:
                        df_chunk = pd.DataFrame(row_records)
                        # header=False because we ensured file exists above
                        df_chunk.to_csv(
                            output_path,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                    time_taken = time.time() - start_time
                    print(
                        f"         -> extracted {len(triplets)} triplets in {time_taken:.2f}s",
                        flush=True,
                    )

                    # Update checkpoint
                    personas_processed += 1
                    processed_indices.add(row_idx)
                    # ---- Job-level progress logging ----
                    if personas_processed % PROGRESS_LOG_EVERY == 0:
                        elapsed = time.time() - start_model_time
                        avg_time = elapsed / max(personas_processed, 1)
                        remaining = max(total_personas_in_slice - personas_processed, 0)
                        eta = remaining * avg_time

                        print(
                            f"[JOB PROGRESS] model={model_name} | "
                            f"slice={start_index}-{end_index or 'end'} | "
                            f"processed={personas_processed}/{total_personas_in_slice} "
                            f"({(personas_processed / total_personas_in_slice) * 100:.1f}%) | "
                            f"elapsed={format_seconds(elapsed)} | "
                            f"avg={avg_time:.1f}s/persona | "
                            f"ETA={format_seconds(eta)}",
                            flush=True,
                        )
                    # ---- Job-level progress logging ----
                    ckpt["personas_processed"] = personas_processed
                    ckpt["processed_indices"] = sorted(processed_indices)
                    save_checkpoint(ckpt_path, ckpt)

                except Exception as e:
                    time_taken = time.time() - start_time
                    print(
                        f"{persona_text[:10]:<10} | {model_name:<15} | "
                        f"{time_taken:<10.2f} | !! FAILED !!",
                        flush=True,
                    )
                    err_info = {
                        "row_index": int(row_idx),
                        "age_refined": age_refined_int,
                        "persona_snippet": persona_text[:80],
                        "error": str(e),
                        "time": datetime.utcnow().isoformat(),
                    }
                    failures.append(err_info)
                    ckpt["failures"] = failures
                    save_checkpoint(ckpt_path, ckpt)
                    logging.error(
                        f"Failed run: idx={row_idx}, model={model_name}. Error: {e}",
                        exc_info=True,
                    )
                    # continue to next persona

        finally:
            elapsed_model = time.time() - start_model_time
            ckpt["elapsed_seconds"] = elapsed_model
            ckpt["failures"] = failures
            ckpt["personas_processed"] = personas_processed
            save_checkpoint(ckpt_path, ckpt)

            print(
                f"   [Model] {model_name}: processed {personas_processed} personas "
                f"(skipped {len(processed_indices) - personas_processed} already done) "
                f"in {elapsed_model:.2f}s. "
                f"Checkpoint: {ckpt_path}"
            )

        # # 4. In-memory summary
        # if all_extracted_data:
        #     print(
        #         f"   [Model] {model_name}: Total triplets in memory this run: "
        #         f"{len(all_extracted_data)}. CSV was written incrementally to: {output_path}"
        #     )
        # else:
        #     print(
        #         f"   [Model] {model_name}: No triplets extracted for {short_filename}."
        #     )
