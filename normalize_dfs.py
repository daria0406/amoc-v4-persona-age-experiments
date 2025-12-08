import os
import re
import pandas as pd

# ------------------------------------------------------------
# Regex patterns to detect educational groups from filenames
# ------------------------------------------------------------
PATTERNS = {
    "primary": re.compile(r"(primary|young|elementary|preschool|kindergarten)", re.I),
    "secondary": re.compile(r"(secondary|middle[_-]?school|junior[_-]?high)", re.I),
    "highschool": re.compile(r"(high[_-]?school|highschool)", re.I),
    "university": re.compile(r"(university|college|uni\b|freshman)", re.I),
}


def detect_group(filename: str):
    """Return the educational group based on regex patterns."""
    lower = filename.lower()
    for group, pattern in PATTERNS.items():
        if pattern.search(lower):
            return group
    return None


def balance_persona_csvs_separately(
    folder: str,
    output_folder: str = "balanced_output",
    text_col: str = "persona_text",
):
    """
    Automatically:
      - Finds CSV files in a folder
      - Detects category (primary/secondary/highschool/university)
      - Downsamples each category to the smallest size
      - Saves each balanced category to its own CSV
      - Also returns a single concatenated balanced dataframe

    No ArgParser needed.
    """

    # Storage for loaded dataframes
    group_dfs = {
        "primary": [],
        "secondary": [],
        "highschool": [],
        "university": [],
    }

    # ------------------------------------------------------------
    # Step 1 — Scan folder for files and load them
    # ------------------------------------------------------------
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue

        group = detect_group(fname)
        if group is None:
            print(f"Ignoring file (no educational group match): {fname}")
            continue

        path = os.path.join(folder, fname)
        print(f"Loading: {fname} → detected group = {group}")

        df = pd.read_csv(path)

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' missing in {fname}")

        df["edu_group"] = group
        group_dfs[group].append(df)

    # ------------------------------------------------------------
    # Step 2 — Merge files per group
    # ------------------------------------------------------------
    final_groups = {}
    for group, dfs in group_dfs.items():
        if len(dfs) == 0:
            print(f"⚠️ WARNING: No files found for group '{group}'")
            continue
        final_groups[group] = pd.concat(dfs, ignore_index=True)
        print(f"{group}: {len(final_groups[group])} samples")

    # Ensure all groups exist
    if any(len(final_groups[g]) == 0 for g in final_groups):
        raise ValueError("At least one educational group is missing.")

    # ------------------------------------------------------------
    # Step 3 — Determine minimum group size (balanced target)
    # ------------------------------------------------------------
    sizes = {g: len(df) for g, df in final_groups.items()}
    print("\nOriginal Sizes:", sizes)

    target_n = min(sizes.values())
    print(f"Balancing to sample size: {target_n} per group\n")

    # ------------------------------------------------------------
    # Step 4 — Downsample each group & save individually
    # ------------------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)

    balanced_groups = {}

    for group, df in final_groups.items():
        print(f"Balancing {group}...")
        df_bal = df.sample(n=target_n, random_state=42)
        balanced_groups[group] = df_bal

        out_path = os.path.join(output_folder, f"{group}_balanced.csv")
        df_bal.to_csv(out_path, index=False)
        print(f"  → saved to {out_path} ({len(df_bal)} rows)")

    # ------------------------------------------------------------
    # Step 5 — Return a combined dataset if needed
    # ------------------------------------------------------------
    df_combined = pd.concat(list(balanced_groups.values()), ignore_index=True)
    combined_path = os.path.join(output_folder, "all_balanced_combined.csv")
    df_combined.to_csv(combined_path, index=False)

    print(f"\nCombined balanced dataset saved to: {combined_path}")
    print(f"Total rows = {len(df_combined)}")

    return balanced_groups, df_combined


# ------------------------------------------------------------
# Run automatically
# ------------------------------------------------------------
if __name__ == "__main__":
    INPUT_FOLDER = "./amoc_output/personas_dfs"            # put your CSVs in this directory
    OUTPUT_FOLDER = "./amoc_output/balanced_dfs"   # where to save balanced files

    balance_persona_csvs_separately(INPUT_FOLDER, OUTPUT_FOLDER)
