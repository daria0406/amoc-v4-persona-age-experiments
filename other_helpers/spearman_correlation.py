import argparse
import pandas as pd
import os
from scipy.stats import spearmanr

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter AMoC matrix to Landscape tokens, then compute per-sentence Spearman correlation."
    )
    parser.add_argument("amoc_csv", help="Path to AMoC activation matrix CSV (wide format).")
    parser.add_argument("--landscape", default="./matrix/landscape_paper.csv", 
                        help="Path to Landscape reference matrix (space-separated). Default: ./matrix/landscape_paper.csv")
    parser.add_argument("--output-dir", default="./matrix_results", 
                        help="Directory to save output files (default: ./matrix_results).")
    return parser.parse_args()

def load_amoc_matrix(file_path):
    df = pd.read_csv(file_path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: 'token'})
    df = df[df['token'] != 'story_text']
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['token'] = df['token'].str.strip()
    return df

def load_landscape_matrix(file_path):
    tokens = []
    scores = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('word/sentence'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token = parts[0]
            score_vals = parts[1:14]
            if len(score_vals) != 13:
                print(f"Warning: row for '{token}' has {len(score_vals)} scores, expected 13.")
                score_vals += [''] * (13 - len(score_vals))
            tokens.append(token)
            scores.append(score_vals)
    df = pd.DataFrame(scores, columns=[str(i) for i in range(1, 14)])
    df.insert(0, 'token', tokens)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def wide_to_long(df):
    token_col = 'token'
    sentence_cols = [c for c in df.columns if c != token_col]
    long_df = pd.melt(df, id_vars=[token_col], value_vars=sentence_cols,
                      var_name="sentence", value_name="score")
    long_df["sentence"] = long_df["sentence"].astype(int)
    long_df = long_df.dropna(subset=["score"])
    return long_df

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    landscape_wide = load_landscape_matrix(args.landscape)
    landscape_tokens = set(landscape_wide['token'])

    amoc_wide = load_amoc_matrix(args.amoc_csv)
    amoc_filtered = amoc_wide[amoc_wide['token'].isin(landscape_tokens)].copy()
    print(f"Original AMoC tokens: {len(amoc_wide)}; Filtered to {len(amoc_filtered)} common tokens.")

    if amoc_filtered.empty:
        print("No overlapping tokens found.")
        return

    input_basename = os.path.basename(args.amoc_csv)
    input_name = os.path.splitext(input_basename)[0]
    formatted_path = os.path.join(args.output_dir, f"formatted_{input_name}.csv")
    amoc_filtered.to_csv(formatted_path, index=False)
    print(f"Saved filtered AMoC matrix to {formatted_path}")

    amoc_long = wide_to_long(amoc_filtered)
    landscape_long = wide_to_long(landscape_wide)

    merged = amoc_long.merge(landscape_long, on=["sentence", "token"],
                             suffixes=("_amoc", "_land"))

    if merged.empty:
        print("No overlapping (sentence, token) pairs found.")
        return

    # Print detailed token scores for each sentence
    print("\nDetailed token scores (AMoC vs Landscape) by sentence:")
    for sent in sorted(merged["sentence"].unique()):
        sub = merged[merged["sentence"] == sent][['token', 'score_amoc', 'score_land']]
        print(f"\nSentence {sent}:")
        # Print as a table with a simple format
        print(sub.to_string(index=False))
        
    results = []
    for sentence in sorted(merged["sentence"].unique()):
        sub = merged[merged["sentence"] == sentence]
        if len(sub) < 2:
            r, p = float('nan'), float('nan')
        else:
            r, p = spearmanr(sub["score_amoc"], sub["score_land"])
        results.append({"sentence": sentence, "spearman_r": r, "p_value": p})

    # Add paper correlations
    paper_corrs = {
        1: 0.77, 2: 0.98, 3: 0.66, 4: 0.91, 5: 0.69, 6: 0.74,
        7: 0.68, 8: 0.69, 9: 0.93, 10: 0.76, 11: 0.68, 12: 0.79, 13: 0.79
    }
    for res in results:
        res['paper_r'] = paper_corrs.get(res['sentence'], None)

    # Add significance column
    for res in results:
        if pd.isna(res['p_value']):
            res['significance'] = 'not significant'
        else:
            res['significance'] = 'significant' if res['p_value'] < 0.05 else 'not significant'

    df_results = pd.DataFrame(results)

    # Round spearman_r and paper_r to 2 decimal places
    df_results['spearman_r'] = df_results['spearman_r'].round(2)
    df_results['paper_r'] = df_results['paper_r'].round(2)

    # Reorder columns: sentence, spearman_r, paper_r, p_value, significance
    df_results = df_results[['sentence', 'spearman_r', 'paper_r', 'p_value', 'significance']]

    output_filename = f"spearman_{input_name}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved per-sentence correlations to {output_path}")

    print("\nSpearman correlations by sentence:")
    # Use float_format to ensure 2 decimals for spearman_r and paper_r when printing
    # But since we already rounded, simple to_string is fine.
    print(df_results.to_string(index=False))

    avg_r = df_results["spearman_r"].mean(skipna=True)
    print(f"\nAverage correlation (valid sentences): {avg_r:.2f}")

if __name__ == "__main__":
    main()