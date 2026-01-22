#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from amoc.config import OUTPUT_ANALYSIS_DIR, STORY_TEXT
from amoc.metrics.graph_metrics import compute_graph_metrics
from amoc.nlp import load_spacy
from amoc.pipeline.engine import AgeAwareAMoCEngine
from amoc.llm.vllm_client import VLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("keefe_validation")

KEEFE_PROMPT = (
    "A knight was traveling through a dark forest.\n"
    "He rode his horse cautiously, aware of unfamiliar sounds around him.\n"
    "Suddenly, he encountered a dragon blocking his path.\n\n"
    "Explain what is happening in this situation and describe the relationships\n"
    "between the characters and elements in the scene."
)


# Documentation-only mapping for paper Table 3 (not used in code paths)
TABLE3_MAPPING = {
    "hierarchical_depth_proxy": "graph_largest_component_ratio",
    "edge_type_entropy": "relation_entropy",
    "cross_sentence_reuse": "cross_sentence_reuse",
    "compression_ratio_proxy": "concepts_per_triplet_ratio",
}

REGIME_ORDER = ["rote", "standard", "inquiry"]
PAIRWISE_COMPARISONS = [
    ("rote", "standard"),
    ("standard", "inquiry"),
]
FIXED_REGIMES = REGIME_ORDER


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(
        -sum((c / total) * math.log(c / total + 1e-12) for c in counter.values())
    )


def z_test(mu1: float, mu2: float, sd1: float, sd2: float, n1: int, n2: int) -> float:
    denom = math.sqrt((sd1**2) / max(n1, 1) + (sd2**2) / max(n2, 1))
    if denom == 0:
        return float("inf") if mu1 != mu2 else 0.0
    return (mu1 - mu2) / denom


def p_from_z(z: float) -> float:
    # Two-tailed p-value from standard normal
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


@dataclass
class RunConfig:
    model_name: str
    regimes: List[str]
    output_dir: Path
    runs_per_regime: int


def compute_metrics_from_triplets(
    final_triplets: List[tuple],
    sentence_triplets: List[tuple],
    story_text: str,
    regime: str,
    model_name: str,
) -> Dict[str, Any]:
    # Use final active edges for structural metrics
    edges = []
    relations = []
    concepts = set()
    for trip in final_triplets:
        if len(trip) >= 3:
            s, r, o = trip[0], trip[1], trip[2]
        else:
            continue
        edges.append((s, o))
        relations.append(r)
        concepts.update([s, o])

    graph = compute_graph_metrics(edges)
    num_triplets = len(edges)
    num_unique_concepts = len(concepts)
    story_tokens = story_text.split()
    triplets_per_100_tokens = (
        (num_triplets / len(story_tokens)) * 100 if story_tokens else 0.0
    )

    # Relation entropy
    relation_entropy = entropy(Counter(relations))

    # Cross-sentence reuse: fraction of unique triplets that appear in >1 sentence
    triplet_to_sentences: Dict[tuple, set] = {}
    for sent_idx, sent_text, subj, rel, obj, active, anchor_kept, introduced_at in (
        sentence_triplets or []
    ):
        key = (subj, rel, obj)
        triplet_to_sentences.setdefault(key, set()).add(sent_idx)
    unique_triplets = len(triplet_to_sentences)
    reuse = sum(1 for sents in triplet_to_sentences.values() if len(sents) > 1)
    cross_sentence_reuse = (reuse / unique_triplets) if unique_triplets else 0.0

    # Compression proxy: concepts per triplet
    concepts_per_triplet = (num_unique_concepts / num_triplets) if num_triplets else 0.0

    return {
        "regime": regime.lower().strip(),
        "model_name": model_name,
        "num_triplets": num_triplets,
        "num_unique_concepts": num_unique_concepts,
        "triplets_per_100_tokens": triplets_per_100_tokens,
        "relation_entropy": relation_entropy,
        "cross_sentence_reuse": cross_sentence_reuse,
        "concepts_per_triplet_ratio": concepts_per_triplet,
        # Proxy for hierarchical depth: largest component ratio
        # Operationalizes structural consolidation (largest component ratio) as a stand-in
        # for hierarchical depth; does not compute literal tree depth.
        "hierarchical_depth_proxy": graph.get("graph_largest_component_ratio", 0.0),
        "graph_avg_degree": graph.get("graph_avg_degree", 0.0),
    }


def aggregate_by_regime(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    grouped = df.groupby("regime")
    agg = grouped[metrics].agg(["mean", "std", "count"])
    # flatten columns
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    return agg.reset_index()


def pairwise_tests(
    df: pd.DataFrame, metrics: List[str], comparisons: List[tuple]
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for r1, r2 in comparisons:
            mu1 = df.loc[df["regime"] == r1, metric].mean()
            mu2 = df.loc[df["regime"] == r2, metric].mean()
            sd1 = df.loc[df["regime"] == r1, metric].std()
            sd2 = df.loc[df["regime"] == r2, metric].std()
            n1 = df.loc[df["regime"] == r1, metric].count()
            n2 = df.loc[df["regime"] == r2, metric].count()
            z = z_test(mu1, mu2, sd1 or 0.0, sd2 or 0.0, n1, n2)
            p = p_from_z(z)
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{r1} vs {r2}",
                    "z_score": z,
                    "p_value": p,
                    "n1": n1,
                    "n2": n2,
                }
            )
    return pd.DataFrame(rows)


def run_experiment(cfg: RunConfig) -> None:
    logger.info("Loading spaCy")
    spacy_nlp = load_spacy()
    if spacy_nlp is None:
        raise RuntimeError("spaCy failed to load")

    VLLM_CACHE: Dict[str, VLLMClient] = {}

    records: List[Dict[str, Any]] = []

    for regime in cfg.regimes:
        for run_idx in range(cfg.runs_per_regime):
            logger.info("Regime=%s run=%d", regime, run_idx + 1)
            if cfg.model_name not in VLLM_CACHE:
                VLLM_CACHE[cfg.model_name] = VLLMClient(
                    model_name=cfg.model_name,
                    tp_size=1,
                    debug=False,
                )
            engine = AgeAwareAMoCEngine(VLLM_CACHE[cfg.model_name], spacy_nlp)
            final_triplets, sentence_triplets, cumulative_triplets = engine.run(
                persona_text=f"Keefe validation persona [{regime}]",
                age_refined=-1,
                replace_pronouns=True,
                plot_after_each_sentence=False,
                graphs_output_dir=None,
                highlight_nodes=None,
                largest_component_only=True,
                strict_reactivate_function=True,
                strict_attachament_constraint=True,
                single_anchor_hub=True,
                edge_forget=None,
                story_text=KEEFE_PROMPT,
            )
            rec = compute_metrics_from_triplets(
                final_triplets=final_triplets,
                sentence_triplets=sentence_triplets,
                story_text=KEEFE_PROMPT,
                regime=regime,
                model_name=cfg.model_name,
            )
            records.append(rec)

    df = pd.DataFrame(records)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "hierarchical_depth_proxy",
        "relation_entropy",
        "cross_sentence_reuse",
        "concepts_per_triplet_ratio",
    ]

    summary = aggregate_by_regime(df, metrics)
    tests = pairwise_tests(df, metrics, PAIRWISE_COMPARISONS)

    summary_path = cfg.output_dir / "keefe_regime_summary.csv"
    tests_path = cfg.output_dir / "keefe_regime_pairwise.csv"
    df_path = cfg.output_dir / "keefe_runs_raw.csv"

    df.to_csv(df_path, index=False)
    # Reporting-friendly metric labels
    REPORTING_NAMES = {
        "hierarchical_depth_proxy": "Hierarchical depth",
        "relation_entropy": "Edge type entropy",
        "cross_sentence_reuse": "Cross-sentence reuse",
        "concepts_per_triplet_ratio": "Compression ratio",
    }

    def rename_for_output(df_in: pd.DataFrame, metric_keys: List[str]) -> pd.DataFrame:
        df_out = df_in.copy()
        for mk, friendly in REPORTING_NAMES.items():
            df_out.columns = [
                col.replace(mk, friendly) if mk in col else col
                for col in df_out.columns
            ]
        return df_out

    summary_out = rename_for_output(summary, metrics)
    tests_out = rename_for_output(tests, metrics)

    summary_out.to_csv(summary_path, index=False)
    tests_out.to_csv(tests_path, index=False)

    logger.info("Saved run-level data to %s", df_path.resolve())
    logger.info("Saved regime summary to %s", summary_path.resolve())
    logger.info("Saved pairwise tests to %s", tests_path.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keefe validation (Section 4.2) runner")
    p.add_argument("--model", required=True, help="LLM model name")
    p.add_argument(
        "--runs-per-regime",
        type=int,
        default=1,
        help="Number of responses per regime",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(OUTPUT_ANALYSIS_DIR, "keefe_validation"),
        help="Output directory for results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        model_name=args.model,
        regimes=FIXED_REGIMES,
        output_dir=Path(args.output_dir),
        runs_per_regime=args.runs_per_regime,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
