import argparse
import glob
import pandas as pd
import os
# scipy used only for legacy spearmanr path; binary_agreement does not need it
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

TOKEN_MAP = {
    # ── rode ──────────────────────────────────────────────────────────────────
    "rode": "rode",
    "ride": "rode",
    "rides": "rode",
    "ride through": "rode",
    "rides through": "rode",
    "travels through": "rode",
    "traveled through": "rode",
    "journeyed through": "rode",
    "explores": "rode",
    "hurried after": "rode",        # preschooler: "knight hurried after dragon"
    "hurries after": "rode",
    "hurry after": "rode",
    "chased": "rode",
    "chases": "rode",
    "galloped": "rode",
    "gallops": "rode",

    # ── appeared ──────────────────────────────────────────────────────────────
    "appeared": "appeared",
    "appears": "appeared",
    "appears in": "appeared",
    "appears at": "appeared",
    "appears near": "appeared",
    "appears on": "appeared",
    "enters": "appeared",
    "entered": "appeared",
    "showed up": "appeared",
    "came out": "appeared",
    "emerged": "appeared",

    # ── kidnapping ────────────────────────────────────────────────────────────
    "kidnapping": "kidnapping",
    "kidnap": "kidnapping",
    "kidnaps": "kidnapping",
    "kidnapped by": "kidnapping",
    "kidnaped": "kidnapping",        # typo variant in AMoC output
    "kidnaped by": "kidnapping",     # typo variant in AMoC output
    "kidnaping": "kidnapping",       # typo variant
    "abducts": "kidnapping",
    "abducted": "kidnapping",
    "captures": "kidnapping",
    "captured": "kidnapping",
    "taken from": "kidnapping",
    "takes": "kidnapping",
    "took": "kidnapping",

    # ── scorched ──────────────────────────────────────────────────────────────
    "scorched": "scorched",
    "scorch": "scorched",
    "scorches": "scorched",
    "scorched by": "scorched",
    "burned by": "scorched",
    "burns": "scorched",

    # ── killed ────────────────────────────────────────────────────────────────
    "killed": "killed",
    "kill": "killed",
    "kills": "killed",
    "killed by": "killed",
    "defeat": "killed",
    "defeats": "killed",
    "defeated by": "killed",
    "dies from": "killed",
    "died at": "killed",
    "dies at": "killed",
    "slays": "killed",
    "slain by": "killed",
    "overcomes": "killed",

    # ── freed ─────────────────────────────────────────────────────────────────
    "freed": "freed",
    "free": "freed",
    "frees": "freed",
    "freed by": "freed",
    "frees from": "freed",
    "to be freed by": "freed",
    "defends against": "freed",
    "rescued by": "freed",
    "rescued at": "freed",
    "rescue from": "freed",
    "rescues from": "freed",
    "rescues": "freed",
    "seeks to rescue": "freed",
    "wants to free": "freed",
    "protects from": "freed",
    "saved by": "freed",
    "saved from": "freed",
    "saves from": "freed",
    "saves": "freed",
    "liberates": "freed",

    # ── married ───────────────────────────────────────────────────────────────
    "married": "married",
    "marries": "married",
    "marry": "married",
    "married to": "married",
    "wants to marry": "married",
    "gets married": "married",
    "gets married in": "married",
    "gets married at": "married",
    "got married in": "married",     # preschooler typo variant
    "got married at": "married",
    "to be married in": "married",
    "to be married to": "married",
    "to be wed at": "married",
    "wed": "married",
    "weds": "married",
    "wants to marry in": "married",  # seen in preschooler output

    # ── fought ────────────────────────────────────────────────────────────────
    "fought": "fought",
    "fight": "fought",
    "fights": "fought",
    "fight for": "fought",
    "fights for": "fought",
    "fights against": "fought",
    "fights at": "fought",
    "fights in": "fought",
    "fights with": "fought",
    "fights from": "fought",
    "fights on": "fought",
    "fights to avoid": "fought",
    "fights to protect": "fought",
    "fought for": "fought",
    "fought against": "fought",
    "fought in": "fought",
    "fought with": "fought",
    "fought by": "fought",
    "fought for by": "fought",
    "battles in": "fought",
    "battled": "fought",
    "battles": "fought",
    "battle": "fought",
    "engages in": "fought",
    "participated in": "fought",
    "participates in": "fought",

    # ── thankful ──────────────────────────────────────────────────────────────
    "thankful": "thankful",
    "grateful": "thankful",
    "grateful to": "thankful",
    "grateful for": "thankful",
    "grateful to knight on": "thankful",
    "receives gratitude from": "thankful",
    "thanks": "thankful",

    # ── unfamiliar ────────────────────────────────────────────────────────────
    "unfamiliar": "unfamiliar",
    "unfamiliar with": "unfamiliar",

    # ── landscape nouns / adjectives (identity) ───────────────────────────────
    "knight": "knight",
    "princess": "princess",
    "dragon": "dragon",
    "forest": "forest",
    "country": "country",
    "beautiful": "beautiful",
    "armor": "armor",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter AMoC matrix to Landscape tokens, then compute per-sentence Spearman correlation."
    )
    parser.add_argument(
        "amoc_csv",
        nargs="?",
        help="Path to a single AMoC activation matrix CSV (wide format). "
        "Omit when using --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Process every matrix in this directory instead of a single file. "
        "Files whose name starts with 'landscape_' are skipped automatically.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern within --input-dir (default: *.csv). "
        "Use e.g. 'amoc_matrix_*.csv' to restrict to canonical matrices.",
    )
    parser.add_argument(
        "--landscape",
        default="./matrix/landscape_paper_no_inference.csv",
        help="Path to Landscape reference matrix (space-separated). Default: ./matrix/landscape_paper.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="./matrix_results/run_233630",
        help="Directory to save output files (default: ./matrix_results).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Salience threshold for binary presence: score > threshold → present (default: 0.0).",
    )
    args = parser.parse_args()
    if bool(args.amoc_csv) == bool(args.input_dir):
        parser.error("provide exactly one of: a single amoc_csv path OR --input-dir")
    return args


def normalise_amoc_token(token: str) -> str:
    token_clean = token.strip()
    return TOKEN_MAP.get(token_clean, token_clean)


def load_amoc_matrix(file_path):
    df = pd.read_csv(file_path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "token"})
    df = df[df["token"] != "story_text"]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["token"] = df["token"].str.strip()
    # Apply normalisation
    df["token"] = df["token"].apply(normalise_amoc_token)
    # Aggregate: for each token, take the maximum score per sentence
    token_col = "token"
    sentence_cols = [c for c in df.columns if c != token_col]
    df = df.groupby(token_col, as_index=False)[sentence_cols].max()
    return df


def load_landscape_matrix(file_path):
    tokens = []
    scores = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("word/sentence"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token = parts[0]
            score_vals = parts[1:14]
            if len(score_vals) != 13:
                print(f"Warning: row for '{token}' has {len(score_vals)} scores, expected 13.")
                score_vals += [""] * (13 - len(score_vals))
            tokens.append(token)
            scores.append(score_vals)
    df = pd.DataFrame(scores, columns=[str(i) for i in range(1, 14)])
    df.insert(0, "token", tokens)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def wide_to_long(df):
    token_col = "token"
    sentence_cols = [c for c in df.columns if c != token_col]
    long_df = pd.melt(
        df,
        id_vars=[token_col],
        value_vars=sentence_cols,
        var_name="sentence",
        value_name="score",
    )
    long_df["sentence"] = long_df["sentence"].astype(int)
    long_df = long_df.dropna(subset=["score"])
    return long_df


def binary_agreement(amoc_vec, land_vec, threshold):
    """
    Per-token binary presence agreement metrics.

    Both vectors have 13 values (one per sentence).
    A sentence is 'present' if score > threshold.

    Returns:
        agreement_rate  – fraction of sentences where both agree (TP+TN / 13)
        jaccard         – |both present| / |either present|  (NaN if union=0)
        phi             – phi coefficient (binary Pearson correlation)
    """
    n = len(amoc_vec)
    a = [1 if v > threshold else 0 for v in amoc_vec]
    b = [1 if v > threshold else 0 for v in land_vec]

    tp = sum(1 for i in range(n) if a[i] == 1 and b[i] == 1)
    tn = sum(1 for i in range(n) if a[i] == 0 and b[i] == 0)
    fp = sum(1 for i in range(n) if a[i] == 1 and b[i] == 0)
    fn = sum(1 for i in range(n) if a[i] == 0 and b[i] == 1)

    agreement_rate = (tp + tn) / n

    union = tp + fp + fn
    jaccard = tp / union if union > 0 else float("nan")

    # phi coefficient
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    phi = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")

    return agreement_rate, jaccard, phi


def process_amoc_file(amoc_csv, landscape_wide, landscape_tokens, output_dir, threshold=0.0):
    input_name = os.path.splitext(os.path.basename(amoc_csv))[0]

    amoc_wide = load_amoc_matrix(amoc_csv)
    # Keep only tokens that appear in Landscape (after normalisation)
    amoc_filtered = amoc_wide[amoc_wide["token"].isin(landscape_tokens)].copy()
    print(f"[{input_name}] AMoC tokens (after aggregation): {len(amoc_wide)}; "
          f"filtered to {len(amoc_filtered)} common tokens.")

    # Report tokens that did not match across the two matrices (both directions)
    amoc_tokens = set(amoc_wide["token"])
    unmatched_amoc = sorted(amoc_tokens - landscape_tokens)
    missing_landscape = sorted(landscape_tokens - amoc_tokens)
    print(f"  AMoC tokens not in Landscape ({len(unmatched_amoc)}): {unmatched_amoc}")
    print(f"  Landscape tokens missing from AMoC ({len(missing_landscape)}): {missing_landscape}")

    if amoc_filtered.empty:
        print("  No overlapping tokens found; skipping.")
        return None

    print(f"  Common tokens: {sorted(amoc_filtered['token'].tolist())}")

    # Save filtered matrix (optional, for inspection)
    formatted_path = os.path.join(output_dir, f"formatted_{input_name}.csv")
    amoc_filtered.to_csv(formatted_path, index=False)
    print(f"  Saved filtered AMoC matrix to {formatted_path}")

    # Convert to long format
    amoc_long = wide_to_long(amoc_filtered)
    landscape_long = wide_to_long(landscape_wide)

    # Left join on landscape: all 17 landscape (sentence, token) pairs are kept.
    # AMoC score is 0 for tokens the model didn't generate (a real miss, not a skip).
    merged = landscape_long.merge(
        amoc_long,
        on=["sentence", "token"],
        how="left",
        suffixes=("_land", "_amoc"),
    )
    merged["score_amoc"] = merged["score_amoc"].fillna(0.0)

    if merged.empty:
        print("  No overlapping (sentence, token) pairs found; skipping.")
        return None

    # ── Binary presence agreement per token ───────────────────────────────────
    # For each landscape token, compare its 13-sentence presence profile in
    # AMoC vs landscape using a score > threshold threshold.
    # Metrics: agreement_rate (TP+TN/13), Jaccard (TP/TP+FP+FN), phi coefficient.
    # This avoids the rank-ordering issue that collapses knight r to ~0.
    results = []
    for token in sorted(merged["token"].unique()):
        sub = merged[merged["token"] == token].sort_values("sentence")
        amoc_vec = sub["score_amoc"].tolist()
        land_vec = sub["score_land"].tolist()
        agr, jac, phi = binary_agreement(amoc_vec, land_vec, threshold)
        results.append({
            "token":          token,
            "agreement_rate": round(agr, 4),
            "jaccard":        round(jac, 4) if jac == jac else float("nan"),
            "phi":            round(phi, 4) if phi == phi else float("nan"),
        })

    df_results = pd.DataFrame(results)

    output_path = os.path.join(output_dir, f"presence_{input_name}.csv")
    df_results.to_csv(output_path, index=False)
    print(f"  Saved per-token presence metrics to {output_path}")

    avg_agr = df_results["agreement_rate"].mean(skipna=True)
    avg_jac = df_results["jaccard"].mean(skipna=True)
    print(f"  Mean agreement={avg_agr:.3f}  mean Jaccard={avg_jac:.3f}  (threshold={threshold})")

    return {
        "file":             input_name,
        "avg_agreement":    round(avg_agr, 4) if pd.notna(avg_agr) else float("nan"),
        "avg_jaccard":      round(avg_jac, 4) if pd.notna(avg_jac) else float("nan"),
        "n_tokens":         len(df_results),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the Landscape reference once and reuse across all files
    landscape_wide = load_landscape_matrix(args.landscape)
    landscape_tokens = set(landscape_wide["token"])

    # --- Single-file mode -----------------------------------------------------
    if args.amoc_csv:
        summary = process_amoc_file(
            args.amoc_csv, landscape_wide, landscape_tokens, args.output_dir,
            threshold=args.threshold,
        )
        if summary is not None:
            df = pd.read_csv(
                os.path.join(args.output_dir, f"presence_{summary['file']}.csv")
            )
            print("\nBinary presence metrics by token:")
            print(df.to_string(index=False))
        return

    # --- Batch / directory mode ----------------------------------------------
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    files = [f for f in files if not os.path.basename(f).startswith("landscape_")]
    if not files:
        print(f"No files matching '{args.pattern}' in {args.input_dir} "
              f"(after skipping landscape_*).")
        return

    print(f"Processing {len(files)} matrices from {args.input_dir} "
          f"(pattern='{args.pattern}')\n")

    summaries = []
    for f in files:
        print(f"=== {f} ===")
        res = process_amoc_file(f, landscape_wide, landscape_tokens, args.output_dir, threshold=args.threshold)
        if res is not None:
            summaries.append(res)
        print()

    if not summaries:
        print("No files produced correlations (no overlapping tokens anywhere).")
        return

    summary_df = pd.DataFrame(summaries).sort_values(
        "avg_agreement", ascending=False, na_position="last"
    )
    summary_path = os.path.join(args.output_dir, "presence_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=" * 70)
    print(f"SUMMARY: {len(summaries)}/{len(files)} files produced metrics")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nMean avg_agreement across files: "
          f"{summary_df['avg_agreement'].mean(skipna=True):.3f}")
    print(f"Mean avg_jaccard  across files: "
          f"{summary_df['avg_jaccard'].mean(skipna=True):.3f}")
    print(f"Saved combined summary to {summary_path}")


if __name__ == "__main__":
    main()