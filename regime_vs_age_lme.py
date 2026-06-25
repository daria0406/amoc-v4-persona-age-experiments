"""
regime_vs_age_lme.py
────────────────────
LME model: graph_metric ~ C(regime) + age_refined + (1|persona_id)

For each of the 8 AMoC graph metrics, fits two models (REML=False):
  - Full:    metric ~ C(regime) + age_refined + (1|persona_id)
  - Age-only: metric ~ age_refined            + (1|persona_id)

Regime significance: Likelihood Ratio Test (LRT), chi2(df=3).
Age significance:    t-test coefficient from the full model.

Output
------
  {output_dir}/{tag}_lme_regime_vs_age.csv   — per-metric results table
  stdout                                      — summary

Usage
-----
  python regime_vs_age_lme.py \
      --input-dir path/to/run_dirs ... \
      --model meta-llama/llama-3.3-70b-instruct \
      --output-dir results/ \
      [--text-dir path/to/texts] \
      [--zone on]    # optional: restrict to a single zone (below/on/above/all)
"""

import os
import glob
import hashlib
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from amoc.analysis.repeated_measures import build_long, METRICS, _ordered_texts

REGIME_ORDER = ["primary", "secondary", "high_school", "university"]

LEVELS = {
    "primary": 0, "secondary": 1,
    "high_school": 2, "highschool": 2,
    "college": 3, "university": 3,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_age_map(input_dirs):
    """
    Re-reads raw triplet CSVs to recover age_refined per persona_id.
    Returns dict: persona_id (sha1 hex) -> mean age_refined.
    """
    records = []
    for d in input_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
            try:
                df = pd.read_csv(path, engine="python", on_bad_lines="warn",
                                 usecols=lambda c: c in
                                 {"persona_text", "age_refined"})
                if "persona_text" not in df.columns or \
                   "age_refined" not in df.columns:
                    continue
                df = df[["persona_text", "age_refined"]].dropna()
                df["persona_id"] = df["persona_text"].astype(str).map(
                    lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
                )
                records.append(df[["persona_id", "age_refined"]])
            except Exception:
                continue

    if not records:
        raise RuntimeError(
            "No age_refined column found in any CSV under --input-dir. "
            "Check that the raw triplet files contain 'age_refined'."
        )

    combined = pd.concat(records, ignore_index=True)
    age_map = combined.groupby("persona_id")["age_refined"].mean()
    print(f"[lme] age loaded for {len(age_map)} distinct personas "
          f"(range {age_map.min():.1f}–{age_map.max():.1f})")
    return age_map


def _annotate_zone(long):
    df = long.copy()
    df["regime_level_num"] = df["regime"].map(LEVELS)
    df["text_level_num"]   = df["text"].map(LEVELS)
    df = df.dropna(subset=["regime_level_num", "text_level_num"]).copy()
    df["regime_level_num"] = df["regime_level_num"].astype(int)
    df["text_level_num"]   = df["text_level_num"].astype(int)
    df["gap"]  = df["text_level_num"] - df["regime_level_num"]
    df["zone"] = np.select(
        [df["gap"] < 0, df["gap"] == 0, df["gap"] > 0],
        ["below", "on", "above"], default="on",
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Core LME fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_lme(df, metric):
    """
    Fits full and reduced models for one metric.
    Returns a dict of results or None if fitting fails.
    """
    sub = df[["persona_id", "regime", "age_refined", metric]].dropna()
    if sub["regime"].nunique() < 2 or len(sub) < 10:
        return None

    # Regime reference level = primary (lowest)
    sub = sub.copy()
    sub["regime"] = pd.Categorical(
        sub["regime"], categories=REGIME_ORDER, ordered=False
    )

    # Centre age to improve convergence
    sub["age_c"] = sub["age_refined"] - sub["age_refined"].mean()

    formula_full    = f"{metric} ~ C(regime, Treatment('primary')) + age_c"
    formula_age     = f"{metric} ~ age_c"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fit_full = smf.mixedlm(
                formula_full, data=sub, groups=sub["persona_id"]
            ).fit(reml=False, method="lbfgs")

            fit_age = smf.mixedlm(
                formula_age, data=sub, groups=sub["persona_id"]
            ).fit(reml=False, method="lbfgs")

    except Exception as e:
        print(f"  [lme] WARNING: {metric} failed to converge — {e}")
        return None

    # ── LRT for regime (comparing full vs age-only) ──────────────────────────
    lrt_stat = 2.0 * (fit_full.llf - fit_age.llf)
    lrt_stat = max(lrt_stat, 0.0)   # guard against tiny numerical negatives
    df_diff  = len(REGIME_ORDER) - 1  # 3 parameters (4 levels - reference)
    p_regime = float(stats.chi2.sf(lrt_stat, df=df_diff))

    # ── Age coefficient from full model ──────────────────────────────────────
    age_key  = "age_c"
    age_coef = float(fit_full.params.get(age_key, np.nan))
    age_se   = float(fit_full.bse.get(age_key, np.nan))
    age_p    = float(fit_full.pvalues.get(age_key, np.nan))
    age_z    = float(fit_full.tvalues.get(age_key, np.nan))

    # ── Individual regime coefficients (vs primary) ───────────────────────────
    regime_coefs = {}
    for r in REGIME_ORDER[1:]:
        key = f"C(regime, Treatment('primary'))[T.{r}]"
        regime_coefs[r] = {
            "coef": float(fit_full.params.get(key, np.nan)),
            "p":    float(fit_full.pvalues.get(key, np.nan)),
        }

    return {
        "metric":       metric,
        "n":            len(sub),
        "n_personas":   sub["persona_id"].nunique(),
        # Regime LRT
        "lrt_chi2":     round(lrt_stat, 4),
        "lrt_df":       df_diff,
        "p_regime":     round(p_regime, 6),
        "regime_sig":   p_regime < 0.05,
        # Age t-test
        "age_coef":     round(age_coef, 6),
        "age_se":       round(age_se, 6),
        "age_z":        round(age_z, 4),
        "p_age":        round(age_p, 6),
        "age_sig":      age_p < 0.05,
        # Pairwise regime coefficients vs primary
        **{f"b_{r}":   round(regime_coefs[r]["coef"], 4)
           for r in REGIME_ORDER[1:]},
        **{f"p_{r}":   round(regime_coefs[r]["p"],    6)
           for r in REGIME_ORDER[1:]},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze(input_dirs, model_name, output_dir, text_dir,
            label_overrides, zone_filter):

    os.makedirs(output_dir, exist_ok=True)

    # 1. Build long-format data
    long = build_long(input_dirs, model_name, text_dir,
                      label_overrides=label_overrides)
    if long.empty:
        print("[lme] no data from build_long; aborting.")
        return

    long = long[long["regime"].isin(REGIME_ORDER)].copy()

    # 2. Attach age
    age_map  = _load_age_map(input_dirs)
    long["age_refined"] = long["persona_id"].map(age_map)
    n_missing = long["age_refined"].isna().sum()
    if n_missing > 0:
        print(f"[lme] WARNING: {n_missing} rows missing age_refined — dropped.")
    long = long.dropna(subset=["age_refined"])

    # 3. Attach zone and optionally filter
    long = _annotate_zone(long)
    zone_label = "all zones"
    if zone_filter and zone_filter != "all":
        long = long[long["zone"] == zone_filter].copy()
        zone_label = f"{zone_filter} zone"

    print(f"\n[lme] {len(long)} rows | zone filter: {zone_label}")
    print(f"[lme] regime distribution:\n"
          f"{long.groupby('regime')['persona_id'].nunique().to_string()}")
    print(f"[lme] age_refined: mean={long['age_refined'].mean():.1f}, "
          f"std={long['age_refined'].std():.1f}, "
          f"range=[{long['age_refined'].min():.0f}, "
          f"{long['age_refined'].max():.0f}]")

    # 4. Correlation check: age vs regime level (multicollinearity warning)
    long["regime_num"] = long["regime"].map(
        {r: i for i, r in enumerate(REGIME_ORDER)}
    )
    rho, rho_p = stats.spearmanr(long["regime_num"], long["age_refined"])
    print(f"\n[lme] age–regime Spearman ρ = {rho:.3f} (p={rho_p:.4f})")
    if abs(rho) > 0.5:
        print("  NOTE: high age–regime correlation — interpret age coefficient "
              "with caution (partial effect after regime is partialled out).")

    # 5. Fit LME for each metric
    metrics  = [m for m in METRICS if m in long.columns]
    results  = []
    for metric in metrics:
        print(f"  fitting {metric}...", end=" ", flush=True)
        res = fit_lme(long, metric)
        if res:
            results.append(res)
            sig_r = "***" if res["p_regime"] < 0.001 else \
                    "**"  if res["p_regime"] < 0.01  else \
                    "*"   if res["p_regime"] < 0.05  else "ns"
            sig_a = "***" if res["p_age"]    < 0.001 else \
                    "**"  if res["p_age"]    < 0.01  else \
                    "*"   if res["p_age"]    < 0.05  else "ns"
            print(f"regime LRT χ²={res['lrt_chi2']:.2f} p={res['p_regime']:.4f}{sig_r}  "
                  f"| age β={res['age_coef']:.4f} p={res['p_age']:.4f}{sig_a}")
        else:
            print("SKIPPED")

    if not results:
        print("[lme] no results produced.")
        return

    out_df = pd.DataFrame(results)

    # 6. Summary
    n_regime_sig = out_df["regime_sig"].sum()
    n_age_sig    = out_df["age_sig"].sum()
    print(f"\n{'─'*60}")
    print(f"REGIME significant (LRT p<.05): {n_regime_sig}/{len(out_df)} metrics")
    print(f"AGE    significant (t   p<.05): {n_age_sig}/{len(out_df)} metrics")
    print(f"{'─'*60}")

    # 7. Save
    tag = model_name.replace("/", "_").replace("-", "_")
    if zone_filter and zone_filter != "all":
        tag += f"_{zone_filter}zone"
    out_path = os.path.join(output_dir, f"{tag}_lme_regime_vs_age.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[lme] saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="LME: graph metric ~ regime + age per persona"
    )
    ap.add_argument("--input-dir", nargs="+", required=True,
                    help="One or more directories containing raw triplet CSVs")
    ap.add_argument("--model", required=True,
                    help="Model name string (used for output file naming)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-dir", default=None)
    ap.add_argument("--label", nargs="*", default=None,
                    help="run_dir=text_label overrides, e.g. run1=primary")
    ap.add_argument("--zone", default="all",
                    choices=["all", "below", "on", "above"],
                    help="Restrict analysis to a competence zone (default: all)")
    args = ap.parse_args()

    label_overrides = None
    if args.label:
        label_overrides = {}
        for item in args.label:
            key, val = item.split("=", 1)
            label_overrides[os.path.basename(os.path.normpath(key.strip()))] = \
                val.strip()

    analyze(args.input_dir, args.model, args.output_dir,
            args.text_dir, label_overrides, args.zone)


if __name__ == "__main__":
    main()
