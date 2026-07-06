import argparse
import glob
import pandas as pd
import os
from scipy.stats import spearmanr

TOKEN_MAP = {
    "rode": "rode",
    "ride": "rode",
    "rides": "rode",
    "ride through": "rode",
    "rides through": "rode",
    "travels through": "rode",
    "traveled through": "rode",
    "journeyed through": "rode",
    "galloped": "rode",
    "gallops": "rode",

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

    "kidnapping": "kidnapping",
    "kidnap": "kidnapping",
    "kidnaps": "kidnapping",
    "kidnapped by": "kidnapping",
    "kidnaped": "kidnapping",
    "kidnaped by": "kidnapping",
    "kidnaping": "kidnapping",
    "abducts": "kidnapping",
    "abducted": "kidnapping",
    "captures": "kidnapping",
    "captured": "kidnapping",
    "taken from": "kidnapping",
    "kidnapped": "kidnapping",

    "scorched": "scorched",
    "scorch": "scorched",
    "scorches": "scorched",
    "scorched by": "scorched",
    "burned by": "scorched",
    "burned": "scorched",
    "burns": "scorched",

    "killed": "killed",
    "kill": "killed",
    "kills": "killed",
    "killed by": "killed",
    "defeat": "killed",
    "defeats": "killed",
    "defeated": "killed",
    "defeated by": "killed",
    "dies from": "killed",
    "died at": "killed",
    "dies at": "killed",
    "slays": "killed",
    "slain by": "killed",
    "slain": "killed",
    "slew": "killed",

    "freed": "freed",
    "free": "freed",
    "frees": "freed",
    "freed by": "freed",
    "freed from": "freed",
    "frees from": "freed",
    "to be freed by": "freed",
    "rescued by": "freed",
    "rescued at": "freed",
    "rescued": "freed",
    "rescue from": "freed",
    "rescues from": "freed",
    "rescues": "freed",
    "saved by": "freed",
    "saved from": "freed",
    "saved": "freed",
    "saves from": "freed",
    "saves": "freed",
    "liberates": "freed",
    "married": "married",
    "marries": "married",
    "marry": "married",
    "married to": "married",
    "gets married": "married",
    "gets married in": "married",
    "gets married at": "married",
    "got married in": "married",
    "got married at": "married",
    "to be married in": "married",
    "to be married to": "married",
    "to be wed at": "married",
    "wed": "married",
    "weds": "married",

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

    "thankful": "thankful",
    "grateful": "thankful",
    "grateful to": "thankful",
    "grateful for": "thankful",
    "grateful to knight on": "thankful",
    "receives gratitude from": "thankful",
    "thanks": "thankful",
    "thanked": "thankful",

    "unfamiliar": "unfamiliar",
    "unfamiliar with": "unfamiliar",

    "knight": "knight",
    "princess": "princess",
    "dragon": "dragon",
    "forest": "forest",
    "country": "country",
    "beautiful": "beautiful",
    "armor": "armor",
}

MAX_SCORE = 5.0
N_SENTENCES = 13
SENTENCE_COLS = [str(i) for i in range(1, N_SENTENCES + 1)]

_HERE = os.path.dirname(os.path.abspath(__file__))


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
        default="amoc_matrix_*.csv",
        help="Glob pattern within --input-dir (default: amoc_matrix_*.csv, i.e. "
        "canonical raw matrices only — avoids re-processing aligned_/presence_/"
        "spearman_ outputs living in the same tree).",
    )
    parser.add_argument(
        "--landscape",
        default=os.path.join(_HERE, "matrix/landscape_paper_no_inference.csv"),
        help="Path to Landscape reference matrix (space-separated). "
        "Default: <script dir>/matrix/landscape_paper_no_inference.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_HERE, "matrix_results/run_233965_fixed"),
        help="Directory to save output files. Regime subfolders of --input-dir "
        "(primary/secondary/highschool/college) are mirrored here.",
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


_TRAILING_PREPS = {
    "by", "at", "in", "for", "with", "from", "to", "on", "against", "into", "of",
}


def normalise_amoc_token(token: str) -> str:
    token_clean = token.strip()
    key = token_clean.lower()
    if key in TOKEN_MAP:
        return TOKEN_MAP[key]
    parts = key.split()
    while len(parts) > 1 and parts[-1] in _TRAILING_PREPS:
        parts = parts[:-1]
        candidate = " ".join(parts)
        if candidate in TOKEN_MAP:
            return TOKEN_MAP[candidate]
    return token_clean


def load_amoc_matrix(file_path):
    df = pd.read_csv(file_path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "token"})
    df = df[df["token"] != "story_text"]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["token"] = df["token"].str.strip()
    df["token"] = df["token"].apply(normalise_amoc_token)
    token_col = "token"
    sentence_cols = [c for c in df.columns if c != token_col]
    df = df.groupby(token_col, as_index=False)[sentence_cols].max()
    df[sentence_cols] = df[sentence_cols].clip(upper=MAX_SCORE)
    return df


def build_aligned_matrix(amoc_wide, landscape_tokens):
    df = amoc_wide.copy()
    df["token"] = df["token"].apply(normalise_amoc_token)
    num_cols = [c for c in df.columns if c != "token"]
    df = df.groupby("token", as_index=False)[num_cols].max()
    df = df.set_index("token")

    df = df.rename(columns={c: str(c).strip() for c in df.columns if str(c).strip().isdigit()})
    for c in SENTENCE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df[SENTENCE_COLS]

    df = df.reindex(list(landscape_tokens)).fillna(0.0).clip(upper=MAX_SCORE)
    df.index.name = "token"
    return df.reset_index()


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


PAPER_R = {
    1: 0.77, 2: 0.98, 3: 0.66, 4: 0.91, 5: 0.69, 6: 0.74,
    7: 0.68, 8: 0.69, 9: 0.93, 10: 0.76, 11: 0.68, 12: 0.79, 13: 0.79,
}


def process_amoc_file(amoc_csv, landscape_wide, landscape_tokens, output_dir, threshold=0.0):
    input_name = os.path.splitext(os.path.basename(amoc_csv))[0]

    try:
        amoc_wide = load_amoc_matrix(amoc_csv)
    except (pd.errors.EmptyDataError, ValueError) as e:
        print(f"  SKIP [{input_name}]: unreadable/empty matrix ({e})")
        return None
    amoc_filtered = amoc_wide[amoc_wide["token"].isin(landscape_tokens)].copy()
    print(f"[{input_name}] AMoC tokens (after aggregation): {len(amoc_wide)}; "
          f"filtered to {len(amoc_filtered)} common tokens.")

    amoc_tokens = set(amoc_wide["token"])
    missing_landscape = sorted(landscape_tokens - amoc_tokens)
    sent_cols = [c for c in amoc_wide.columns if c != "token"]
    unmatched_mass = (
        amoc_wide[~amoc_wide["token"].isin(landscape_tokens)]
        .set_index("token")[sent_cols]
        .sum(axis=1)
        .sort_values(ascending=False)
    )
    unmatched_mass = unmatched_mass[unmatched_mass > 0]
    top_unmatched = ", ".join(
        f"{t} ({m:.1f})" for t, m in unmatched_mass.head(15).items()
    )
    print(f"  Unmatched AMoC tokens with mass>0 ({len(unmatched_mass)}), "
          f"top by total activation: {top_unmatched}")
    print(f"  Landscape tokens missing from AMoC ({len(missing_landscape)}): {missing_landscape}")

    if amoc_filtered.empty:
        print("  No overlapping tokens found; skipping.")
        return None

    print(f"  Common tokens: {sorted(amoc_filtered['token'].tolist())}")

    formatted_path = os.path.join(output_dir, f"formatted_{input_name}.csv")
    amoc_filtered.to_csv(formatted_path, index=False)
    print(f"  Saved filtered AMoC matrix to {formatted_path}")

    aligned = build_aligned_matrix(amoc_wide, landscape_wide["token"].tolist())
    aligned_path = os.path.join(output_dir, f"aligned_{input_name}.csv")
    aligned.to_csv(aligned_path, index=False)
    print(f"  Saved aligned appendix matrix to {aligned_path}")

    amoc_long = wide_to_long(amoc_filtered)
    landscape_long = wide_to_long(landscape_wide)

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

    results = []
    for sentence in sorted(merged["sentence"].unique()):
        sub = merged[merged["sentence"] == sentence]
        amoc_vec = sub["score_amoc"].to_numpy()
        land_vec = sub["score_land"].to_numpy()
        if len(sub) < 2 or amoc_vec.std() == 0 or land_vec.std() == 0:
            r, p = float("nan"), float("nan")
        else:
            r, p = spearmanr(amoc_vec, land_vec)
        results.append({
            "sentence":     int(sentence),
            "spearman_r":   round(r, 2) if pd.notna(r) else float("nan"),
            "paper_r":      PAPER_R.get(int(sentence)),
            "p_value":      p,
            "significance": (
                "not significant" if pd.isna(p)
                else ("significant" if p < 0.05 else "not significant")
            ),
        })

    df_results = pd.DataFrame(
        results, columns=["sentence", "spearman_r", "paper_r", "p_value", "significance"]
    )

    output_path = os.path.join(output_dir, f"spearman_{input_name}.csv")
    df_results.to_csv(output_path, index=False)
    print(f"  Saved per-sentence Spearman correlations to {output_path}")

    avg_r = df_results["spearman_r"].mean(skipna=True)
    n_valid = int(df_results["spearman_r"].notna().sum())
    print(f"  Average Spearman rho = {avg_r:.4f}  (over {n_valid} valid sentences)")

    return {
        "file":          input_name,
        "avg_spearman":  round(avg_r, 4) if pd.notna(avg_r) else float("nan"),
        "n_sentences":   n_valid,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    landscape_wide = load_landscape_matrix(args.landscape)
    landscape_tokens = set(landscape_wide["token"])

    if args.amoc_csv:
        summary = process_amoc_file(
            args.amoc_csv, landscape_wide, landscape_tokens, args.output_dir,
            threshold=args.threshold,
        )
        if summary is not None:
            df = pd.read_csv(
                os.path.join(args.output_dir, f"spearman_{summary['file']}.csv")
            )
            print("\nPer-sentence Spearman correlations:")
            print(df.to_string(index=False))
            print(f"\nAverage Spearman rho: {summary['avg_spearman']}")
        return

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True)
        )
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
        rel = os.path.relpath(os.path.dirname(f), args.input_dir)
        file_out_dir = (
            args.output_dir if rel == "." else os.path.join(args.output_dir, rel)
        )
        os.makedirs(file_out_dir, exist_ok=True)
        res = process_amoc_file(
            f, landscape_wide, landscape_tokens, file_out_dir, threshold=args.threshold
        )
        if res is not None:
            res["regime_dir"] = "" if rel == "." else rel
            summaries.append(res)
        print()

    if not summaries:
        print("No files produced correlations (no overlapping tokens anywhere).")
        return

    summary_df = pd.DataFrame(summaries).sort_values(
        "avg_spearman", ascending=False, na_position="last"
    )
    summary_path = os.path.join(args.output_dir, "spearman_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=" * 70)
    print(f"SUMMARY: {len(summaries)}/{len(files)} files produced metrics")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nMean avg_spearman across files: "
          f"{summary_df['avg_spearman'].mean(skipna=True):.3f}")
    print(f"Saved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
