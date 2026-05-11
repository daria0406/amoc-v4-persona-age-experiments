import json
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.formula.api import mixedlm
import re
import os
from itertools import combinations

from amoc.pipeline.orchestrator import AMoCv4
from amoc.llm.vllm_client import VLLMClient
from amoc.output.recorder import graph_edges_to_triplets
from amoc.utils.spacy_utils import load_spacy
from amoc.utils.io import robust_read_persona_csv
from amoc.admission.node_admission import NodeAdmission as _NodeAdmission

# allow nodes created with TEXT_FALLBACK provenance -  same as in the Keefe experiment
_orig_admit = _NodeAdmission.admit_node
def _permissive_admit(self, lemma, node_type, provenance="STORY_EXPLICIT", **kw):
    if provenance == "TEXT_FALLBACK":
        return True
    return _orig_admit(self, lemma, node_type, provenance=provenance, **kw)
_NodeAdmission.admit_node = _permissive_admit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prompt for support / contradict 
SUPPORT_CONTRADICT_PROMPT = """You have the following edges from a knowledge graph in the format: node - edge - node.
{edges}

Target sentence: "{target_sentence}"

Using the graph and the story it tells, tell me which edges SUPPORT or CONTRADICT the target sentence.
Return a JSON object with two lists: "support" contains the numbers of edges that support the target sentence, and "contradict" contains the numbers of edges that contradict it.
Example: {{"support": [1, 3], "contradict": [2, 5]}}
Only output the JSON object."""

def get_support_contradict(amoc, target_sentence):
    triplets = graph_edges_to_triplets(amoc.graph, only_active=True)
    if not triplets:
        return {"support": [], "contradict": []}

    edges_str_parts = []
    for i, (s, r, o) in enumerate(triplets, 1):
        edges_str_parts.append(f"{i}. {s} - {r} - {o}")
    edges_str = "\n".join(edges_str_parts)

    prompt = SUPPORT_CONTRADICT_PROMPT.format(
        edges=edges_str,
        target_sentence=target_sentence
    )

    try:
        response = amoc.client.call_vllm(prompt, persona=amoc.persona)
        print(f"[DEBUG] LLM response: '{response}'", flush=True)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        return {"support": [], "contradict": []}

    # Parse JSON response (handle possible markdown fences)
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(clean)
        if not isinstance(result, dict):
            raise ValueError("Not a dict")
        return result
    except Exception as e:
        logger.warning(f"Could not parse LLM response: {response}, error: {e}")
        # Fallback regex extraction
        support_match = re.search(r"\"support\"\s*:\s*\[(.*?)\]", clean, re.DOTALL)
        contradict_match = re.search(r"\"contradict\"\s*:\s*\[(.*?)\]", clean, re.DOTALL)
        if support_match and contradict_match:
            try:
                support_nums = [int(x.strip()) for x in support_match.group(1).split(",") if x.strip().isdigit()]
                contradict_nums = [int(x.strip()) for x in contradict_match.group(1).split(",") if x.strip().isdigit()]
                return {"support": support_nums, "contradict": contradict_nums}
            except:
                pass
        return {"support": [], "contradict": []}


def lme_all_pairs(df, metric_col, condition_col="condition", item_col="item_id"):
    df = df.copy()
    df[condition_col] = pd.Categorical(df[condition_col])
    model = mixedlm(f"{metric_col} ~ C({condition_col}) - 1", df, groups=df[item_col])
    result = model.fit()

    levels = df[condition_col].cat.categories.tolist()
    # Build contrast vectors and apply t_test
    rows = []
    for cond1, cond2 in combinations(levels, 2):
        # Contrast vector: +1 for cond1, -1 for cond2, 0 otherwise
        contrast = np.zeros(len(result.params))
        param_names = result.params.index.tolist()
        idx1 = param_names.index(f"C({condition_col})[{cond1}]")
        idx2 = param_names.index(f"C({condition_col})[{cond2}]")
        contrast[idx1] = 1.0
        contrast[idx2] = -1.0
        t_res = result.t_test(contrast)
        est = t_res.effect[0]  # difference estimate
        se = t_res.sd[0]
        z = t_res.tvalue[0]    
        p = t_res.pvalue[0]
        rows.append({
            "Comparison": f"{cond1} - {cond2}",
            "Estimate": est,
            "z": z,
            "p": p
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="O'Brien experiment with personas (SLURM array compatible)")
    parser.add_argument("--persona-csv", required=True, help="CSV chunk file with personas (must have 'persona_text' and 'age_refined')")
    parser.add_argument("--obrien-json", required=True, help="JSON file with O'Brien items (combined)")
    parser.add_argument("--output-csv", required=True, help="Output CSV file for this chunk (will include all personas)")
    parser.add_argument("--stats-output", default=None, help="Optional file to save LME pairwise tables")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", help="vLLM model name")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of personas (for testing)")
    parser.add_argument("--max-items", type=int, default=None, help="Limit number of O'Brien items (for testing)")
    args = parser.parse_args()

    # Load O'Brien files
    with open(args.obrien_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    logger.info(f"Loaded {len(items)} O'Brien items from {args.obrien_json}")
    if args.max_items:
        items = items[:args.max_items]

    # Load persona chunk
    df = robust_read_persona_csv(args.persona_csv)
    if "persona_text" not in df.columns or "age_refined" not in df.columns:
        raise ValueError(f"CSV must contain 'persona_text' and 'age_refined' columns. Found: {df.columns.tolist()}")
    df["age_refined"] = pd.to_numeric(df["age_refined"], errors="coerce")
    if args.max_rows:
        df = df.head(args.max_rows)
    logger.info(f"Loaded {len(df)} personas from {args.persona_csv}")

    # spaCy
    spacy_nlp = load_spacy()
    if spacy_nlp is None:
        raise RuntimeError("Failed to load spaCy model.")

    client = VLLMClient(model_name=args.model, tp_size=args.tp, debug=False)

    # Collect rows
    all_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing personas"):
        persona_text = str(row["persona_text"])
        age = int(row["age_refined"]) if pd.notna(row["age_refined"]) else -1
        persona_description = f"Age: {age} years old.\n{persona_text}"

        for item in items:
            item_id = item["id"]
            intro = item["introduction"]
            target = item["target_sentence_1"]
            for cond in ["consistent", "inconsistent",
                         "qualified1", "qualified2", "qualified3", "qualified4"]:
                elaboration = item.get(cond, "")
                if not elaboration or not isinstance(elaboration, str) or not elaboration.strip():
                    continue  # skip missing conditions
                story = intro.strip() + " " + elaboration.strip()

                amoc = AMoCv4(
                    persona_description=persona_description,
                    story_text=story,
                    vllm_client=client,
                    max_distance_from_active_nodes=2,
                    max_new_concepts=15,
                    max_new_properties=15,
                    context_length=2,
                    edge_visibility=3,
                    nr_relevant_edges=15,
                    spacy_nlp=spacy_nlp,
                    debug=False,
                    persona_age=age,
                    strict_reactivate_function=True,
                    single_anchor_hub=True,
                    matrix_dir_base=None,
                    checkpoint=False,
                )

                # Suppress unnecessary recordings and outputs 
                amoc.record_activation_matrix_wrapper = lambda *a, **kw: None
                amoc._activation_ops.record_sentence_activation_matrix = lambda *a, **kw: None
                amoc._activation_ops.export_activation_matrix_csv = lambda *a, **kw: None
                amoc._output_ops.finalize_outputs = lambda *a, **kw: (None, None, None)
                amoc._plot_ops.plot_sentence_views = lambda *a, **kw: None
                amoc._plot_ops.plot_graph_snapshot_full = lambda *a, **kw: None

                amoc.analyze(replace_pronouns=False, plot_after_each_sentence=False)

                triplets = graph_edges_to_triplets(amoc.graph, only_active=False)
                active_triplets = graph_edges_to_triplets(amoc.graph, only_active=True)

                print(f"\n{'='*60}")
                print(f"Item {item_id} | Condition: {cond} | Persona idx: {idx}")
                print(f"Story (intro + elaboration): {story[:200]}...")
                print(f"Target sentence: {target}")
                print(f"\nTotal nodes in graph: {len(amoc.graph.nodes)}")
                print(f"Total edges in graph: {len(amoc.graph.edges)}")
                print(f"Active edges: {len(active_triplets)}")
                print(f"All edges (active + inactive): {len(triplets)}")

                print(f"\nActive edges:")
                for i, (s, r, o) in enumerate(active_triplets, 1):
                    print(f"  {i}. {s} --{r}--> {o}")

                print(f"\nAll edges (including inactive):")
                for i, (s, r, o) in enumerate(triplets, 1):
                    print(f"  {i}. {s} --{r}--> {o}")

                print(f"{'='*60}\n")

                res = get_support_contradict(amoc, target)
                support_count = len(res.get("support", []))
                contradict_count = len(res.get("contradict", []))
                diff = support_count - contradict_count

                all_rows.append({
                    "global_persona_index": row.get("original_index", idx),
                    "age": age,
                    "persona_text": persona_text,
                    "item_id": item_id,
                    "condition": cond,
                    "support": support_count,
                    "contradict": contradict_count,
                    "difference": diff
                })

    # compute LME on the entire chunk
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(args.output_csv, index=False)
        print(f"Saved {len(df_out)} observations to {args.output_csv}")

        all_stats = []
        for metric in ["support", "contradict", "difference"]:
            print(f"\n=== LME for {metric} ===")
            conditions_present = df_out[df_out[metric].notna()]['condition'].unique()
            sub_df = df_out[df_out['condition'].isin(conditions_present)]
            
            print(f"  Observations: {len(sub_df)}")
            print(f"  Unique items: {sub_df['item_id'].nunique()}")
            print(f"  Conditions: {sorted(sub_df['condition'].unique())}")
            print(f"  Mean {metric} by condition:")
            print(sub_df.groupby('condition')[metric].mean().to_string())
            
            try:
                stats = lme_all_pairs(sub_df, metric)
                print(stats.to_string(index=False))
                stats["metric"] = metric
                all_stats.append(stats)
            except Exception as e:
                print(f"  LME failed: {e}")
                # Fallback: simple t-tests between conditions
                from scipy.stats import ttest_ind
                print(f"  Falling back to pairwise t-tests...")
                rows = []
                levels = sorted(sub_df['condition'].unique())
                for cond1, cond2 in combinations(levels, 2):
                    vals1 = sub_df[sub_df['condition'] == cond1][metric].dropna().values
                    vals2 = sub_df[sub_df['condition'] == cond2][metric].dropna().values
                    if len(vals1) > 1 and len(vals2) > 1:
                        t_stat, p_val = ttest_ind(vals1, vals2)
                        rows.append({
                            "Comparison": f"{cond1} - {cond2}",
                            "Estimate": np.mean(vals1) - np.mean(vals2),
                            "z": t_stat,
                            "p": p_val
                        })
                if rows:
                    fallback_df = pd.DataFrame(rows)
                    print(fallback_df.to_string(index=False))
                    fallback_df["metric"] = metric
                    all_stats.append(fallback_df)

        if args.stats_output and all_stats:
            final_stats = pd.concat(all_stats, ignore_index=True)
            final_stats.to_csv(args.stats_output, index=False)
            print(f"\nSaved LME pairwise tables to {args.stats_output}")

if __name__ == "__main__":
    main()