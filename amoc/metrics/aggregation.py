import logging
from typing import Dict, Any

import pandas as pd

from amoc.metrics.lexical import simple_sentiment_score
from amoc.metrics.lexical import compute_lexical_metrics
from amoc.metrics.graph_metrics import compute_graph_metrics


def process_triplets_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame()

    required_cols = {
        "original_index",
        "age_refined",
        "persona_text",
        "subject",
        "object",
        "regime",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logging.warning(f"Missing required columns in {path}: {missing}")
        return pd.DataFrame()

    if "model_name" not in df.columns:
        df["model_name"] = None

    group_cols = [
        "original_index",
        "age_refined",
        "persona_text",
        "model_name",
        "regime",
    ]
    if "education_level" in df.columns:
        group_cols.append("education_level")

    records = []

    for keys, g in df.groupby(group_cols, dropna=False):
        ctx = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

        num_triplets = len(g)
        if num_triplets == 0:
            continue

        subjects = g["subject"].astype(str)
        objects = g["object"].astype(str)
        relations = (
            g["relation"].astype(str)
            if "relation" in g.columns
            else pd.Series(["<NO_RELATION>"] * num_triplets, index=g.index)
        )

        # --- Unique counts ---
        num_unique_subjects = subjects.nunique()
        num_unique_objects = objects.nunique()
        num_unique_relations = relations.nunique()
        num_unique_concepts = len(set(subjects) | set(objects))

        # --- Triplet repetition ---
        triplets = list(zip(subjects, relations, objects))
        num_unique_triplets = len(set(triplets))
        triplet_repetition_ratio = 1.0 - (num_unique_triplets / num_triplets)

        # --- Persona text ---
        persona_text = ctx["persona_text"] or ""
        persona_tokens = persona_text.split()
        persona_num_tokens = len(persona_tokens)
        triplets_per_100_tokens = (
            (num_triplets / persona_num_tokens) * 100 if persona_num_tokens > 0 else 0.0
        )

        # --- Sentiment + lexical ---
        sentiment_score = simple_sentiment_score(persona_text)
        lex = compute_lexical_metrics(persona_text)

        # --- Graph metrics ---
        edges = list(zip(subjects.tolist(), objects.tolist()))
        graph = compute_graph_metrics(edges)

        # --- Age regime ---
        age_refined = ctx["age_refined"]
        try:
            age_refined_int = int(age_refined)
        except Exception:
            age_refined_int = None

        record: Dict[str, Any] = {
            "original_index": ctx["original_index"],
            "regime": ctx["regime"],
            "persona_text": persona_text,
            "subject": "; ".join(subjects),
            "relation": "; ".join(relations),
            "object": "; ".join(objects),
            "num_triplets": num_triplets,
            "num_unique_triplets": num_unique_triplets,
            "num_unique_subjects": num_unique_subjects,
            "num_unique_objects": num_unique_objects,
            "num_unique_concepts": num_unique_concepts,
            "num_unique_relations": num_unique_relations,
            "triplet_repetition_ratio": triplet_repetition_ratio,
            "persona_num_tokens": persona_num_tokens,
            "triplets_per_100_tokens": triplets_per_100_tokens,
            "sentiment_score": sentiment_score,
            "lexical_ttr": lex["lexical_ttr"],
            "lexical_avg_word_len": lex["lexical_avg_word_len"],
            **graph,
        }

        if "education_level" in ctx:
            record["education_level"] = ctx["education_level"]

        records.append(record)

    return pd.DataFrame(records)
