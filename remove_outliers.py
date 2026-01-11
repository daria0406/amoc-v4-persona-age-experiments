import argparse
import os
from amoc.analysis.statistics import canonicalize_model_name
from amoc.outliers.io import find_triplet_files, save_persona_outputs
from amoc.outliers.stats import build_persona_stats
from amoc.outliers.trimming import quantile_trim, iqr_cap
from amoc.outliers.triplets import filter_triplets_by_persona
from amoc.outliers.cleaned_regime_analysis import run_cleaned_regime_analysis


LOWER_Q = 0.05
UPPER_Q = 0.95

TRIM_METRICS = [
    "num_triplets",
    "num_unique_concepts",
    "triplets_per_100_tokens",
    "graph_num_nodes",
    "graph_num_edges",
]

CAP_METRICS = [
    "graph_density",
    "graph_avg_degree",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate statistical (violin, etc.) plots per regime after outlier removal",
    )
    parser.add_argument(
        "--plots-age",
        action="store_true",
        help="Generate plots per regime after outlier removal for age",
    )
    parser.add_argument(
        "--refined-age-dir",
        default=None,
        help="Directory containing *_age_refined.csv files ",
    )
    args = parser.parse_args()

    model_name = canonicalize_model_name(args.model)
    safe_tag = model_name.replace("/", "-")

    triplet_files = find_triplet_files(args.input_dir, safe_tag)
    if not triplet_files:
        raise RuntimeError("No triplet CSVs found.")

    df_stats_all = build_persona_stats(triplet_files)

    if "original_index" in df_stats_all.columns:
        df_stats_all = df_stats_all.rename(columns={"original_index": "idx"})

    # Save raw stats as a CSV (not a directory) to avoid clashing with the output dir
    df_stats_all.to_csv(
        os.path.join(args.input_dir, "persona_stats_raw.csv"), index=False
    )
    # trimming by quantiles
    df_trimmed = quantile_trim(
        df_stats_all,
        TRIM_METRICS,
        LOWER_Q,
        UPPER_Q,
    )

    # windsorization
    df_clean = iqr_cap(df_trimmed, CAP_METRICS)

    all_personas = set(df_stats_all["idx"])
    kept_personas = set(df_trimmed["idx"])
    removed_personas = all_personas - kept_personas

    model_tag = canonicalize_model_name(args.model).replace("/", "-")
    save_persona_outputs(
        df_trimmed,
        df_clean,
        removed_personas,
        model_tag,
        out_dir=os.path.join(args.input_dir, "persona_stats"),
    )

    filter_triplets_by_persona(
        triplet_files,
        kept_personas,
        model_tag,
        out_dir=os.path.join(args.input_dir, "triplets_processed"),
    )

    print(
        "Trimmed personas:",
        len(df_trimmed),
        "of",
        len(df_stats_all),
    )

    run_cleaned_regime_analysis(
        input_dir=args.input_dir,
        model=model_tag,
        output_dir=os.path.join(
            args.input_dir,
            "amoc_analysis",
            "cleaned_triplet_plots",
        ),
        plots=args.plots,
        plots_age=args.plots_age,
        refined_age_dir=args.refined_age_dir,
    )


if __name__ == "__main__":
    main()
