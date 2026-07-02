import argparse
import os
import sys

from amoc.analysis.statistics import canonicalize_model_name
from amoc.outliers.io import find_triplet_files
from amoc.outliers.stats import build_persona_stats


def main():
    parser = argparse.ArgumentParser(description="Regenerate persona_stats_raw.csv from triplet CSVs.")
    parser.add_argument("--triplet-dir", required=True,
                        help="Directory containing model_*_final_triplets_*.csv files")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--out", default=None,
                        help="Output path for persona_stats_raw.csv. "
                             "Defaults to <triplet-dir>/../persona_stats_raw.csv")
    args = parser.parse_args()

    triplet_dir = os.path.abspath(args.triplet_dir)
    if not os.path.isdir(triplet_dir):
        sys.exit(f"ERROR: --triplet-dir not found: {triplet_dir}")

    safe_tag = canonicalize_model_name(args.model).replace("/", "-")
    triplet_files = find_triplet_files(triplet_dir, safe_tag)

    if not triplet_files:
        sys.exit(f"ERROR: no files matching model_{safe_tag}*_final_triplets_*.csv in {triplet_dir}")

    print(f"Found {len(triplet_files)} triplet file(s):")
    for f in sorted(triplet_files):
        print(f"  {os.path.basename(f)}")

    df = build_persona_stats(triplet_files)
    if "original_index" in df.columns:
        df = df.rename(columns={"original_index": "idx"})

    print("\nPersonas per regime:")
    print(df.groupby("regime").size().to_string())
    print(f"Total: {len(df)}")

    out_path = args.out or os.path.join(triplet_dir, "..", "persona_stats_raw.csv")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
