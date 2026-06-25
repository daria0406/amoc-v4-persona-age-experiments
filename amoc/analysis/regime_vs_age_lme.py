"""
regime_vs_age_lme.py
--------------------
LME model: graph_metric ~ C(regime) + age_refined + (1|persona_id)

For each AMoC graph metric, fits two models (REML=False):
  - Full:     metric ~ C(regime) + age_c + (1|persona_id)
  - Age-only: metric ~ age_c             + (1|persona_id)

Regime significance: Likelihood Ratio Test (LRT), chi2(df = n_regimes - 1).
Age significance:     t-test on the age coefficient from the full model.

Disentangles whether a metric tracks the persona's *education regime* or simply
their *age*, which are correlated in the persona set. Reuses the competence-gap
level mapping so regimes/texts are ordered consistently and label aliases
(highschool/high_school, college/university) are all accepted.

Output
------
  {output_dir}/{tag}_lme_regime_vs_age[_<zone>zone].csv  -- per-metric table
  stdout                                                  -- summary

Usage
-----
  python -m amoc.analysis.regime_vs_age_lme \
      --input-dir path/to/run_dirs ... \
      --model meta-llama/Llama-3.3-70B-Instruct \
      --output-dir results/ \
      [--text-dir path/to/texts] \
      [--zone on]    # restrict to a single competence zone (below/on/above/all)
"""

import os
import glob
import hashlib
import argparse
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from amoc.analysis.repeated_measures import (
    build_long, METRICS, _safe_tag, parse_label_overrides)
from amoc.analysis.competence_gap import _annotate_levels, _level


def _regimes_by_level(regimes) -> List[str]:
    """Unique regime labels ordered by education level (primary..university).

    Accepts every alias in the competence-gap LEVELS map; labels with no mapped
    level are dropped (they'd break the ordered reference coding anyway).
    """
    mapped = [(r, _level(r)) for r in dict.fromkeys(regimes)]
    return [r for r, lvl in sorted(
        (m for m in mapped if m[1] is not None), key=lambda m: m[1])]


def _load_age_map(input_dirs: List[str]) -> pd.Series:
    """Re-read raw triplet CSVs to recover mean age_refined per persona_id."""
    records = []
    for d in input_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
            try:
                df = pd.read_csv(
                    path, engine="python", on_bad_lines="warn",
                    usecols=lambda c: c in {"persona_text", "age_refined"})
                if "persona_text" not in df.columns or \
                   "age_refined" not in df.columns:
                    continue
                df = df[["persona_text", "age_refined"]].dropna()
                df["persona_id"] = df["persona_text"].astype(str).map(
                    lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
                records.append(df[["persona_id", "age_refined"]])
            except Exception:
                continue

    if not records:
        raise RuntimeError(
            "No age_refined column found in any CSV under --input-dir. "
            "Check that the raw triplet files contain 'age_refined'.")

    combined = pd.concat(records, ignore_index=True)
    age_map = combined.groupby("persona_id")["age_refined"].mean()
    print(f"[lme] age loaded for {len(age_map)} distinct personas "
          f"(range {age_map.min():.1f}-{age_map.max():.1f})")
    return age_map


def fit_lme(df: pd.DataFrame, metric: str, regime_order: List[str]) -> Optional[Dict]:
    """Fit full and age-only models for one metric. Returns a result dict or None.

    The regime categorical is reference-coded against the lowest-level regime
    actually present, and the LRT df equals (n_present_regimes - 1).
    """
    sub = df[["persona_id", "regime", "age_refined", metric]].dropna().copy()
    present = [r for r in regime_order if r in set(sub["regime"].unique())]
    if len(present) < 2 or len(sub) < 10:
        return None

    ref = present[0]
    sub["regime"] = pd.Categorical(sub["regime"], categories=present, ordered=False)
    # Centre age to improve convergence
    sub["age_c"] = sub["age_refined"] - sub["age_refined"].mean()

    formula_full = f"{metric} ~ C(regime, Treatment('{ref}')) + age_c"
    formula_age = f"{metric} ~ age_c"

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
        print(f"  [lme] WARNING: {metric} failed to converge -- {e}")
        return None

    # LRT for regime (full vs age-only)
    lrt_stat = max(2.0 * (fit_full.llf - fit_age.llf), 0.0)
    df_diff = len(present) - 1
    p_regime = float(stats.chi2.sf(lrt_stat, df=df_diff))

    # Age coefficient from full model
    age_coef = float(fit_full.params.get("age_c", np.nan))
    age_se = float(fit_full.bse.get("age_c", np.nan))
    age_p = float(fit_full.pvalues.get("age_c", np.nan))
    age_z = float(fit_full.tvalues.get("age_c", np.nan))

    # Individual regime coefficients (vs reference)
    regime_coefs = {}
    for r in present[1:]:
        key = f"C(regime, Treatment('{ref}'))[T.{r}]"
        regime_coefs[r] = {
            "coef": float(fit_full.params.get(key, np.nan)),
            "p": float(fit_full.pvalues.get(key, np.nan)),
        }

    return {
        "metric": metric,
        "n": len(sub),
        "n_personas": sub["persona_id"].nunique(),
        "regime_ref": ref,
        "n_regimes": len(present),
        # Regime LRT
        "lrt_chi2": round(lrt_stat, 4),
        "lrt_df": df_diff,
        "p_regime": round(p_regime, 6),
        "regime_sig": p_regime < 0.05,
        # Age t-test
        "age_coef": round(age_coef, 6),
        "age_se": round(age_se, 6),
        "age_z": round(age_z, 4),
        "p_age": round(age_p, 6),
        "age_sig": age_p < 0.05,
        # Pairwise regime coefficients vs reference
        **{f"b_{r}": round(regime_coefs[r]["coef"], 4) for r in present[1:]},
        **{f"p_{r}": round(regime_coefs[r]["p"], 6) for r in present[1:]},
    }


def analyze(input_dir: Union[str, List[str]], model_name: str, output_dir: str,
            text_dir: Optional[str] = "tusa_text/min_drp_texts",
            zone_filter: str = "all", model_tag: Optional[str] = None,
            label_overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    input_dirs = [input_dir] if isinstance(input_dir, str) else list(input_dir)

    # 1. Build long-format data
    long = build_long(input_dirs, model_name, text_dir,
                      require_min_texts=2, label_overrides=label_overrides)
    if long.empty:
        print("[lme] no data from build_long; aborting.")
        return pd.DataFrame()

    # 2. Annotate competence zone (also drops unmapped regimes/texts)
    long, bad_reg, bad_txt = _annotate_levels(long)
    if bad_reg or bad_txt:
        print(f"[lme] WARNING unmapped levels dropped -- regimes={bad_reg} "
              f"texts={bad_txt}. Use --label to fix.")
    if long.empty:
        print("[lme] no rows left after level mapping; aborting.")
        return pd.DataFrame()

    # 3. Attach age
    age_map = _load_age_map(input_dirs)
    long["age_refined"] = long["persona_id"].map(age_map)
    n_missing = long["age_refined"].isna().sum()
    if n_missing > 0:
        print(f"[lme] WARNING: {n_missing} rows missing age_refined -- dropped.")
    long = long.dropna(subset=["age_refined"])

    # 4. Optional zone filter
    zone_label = "all zones"
    if zone_filter and zone_filter != "all":
        long = long[long["zone"] == zone_filter].copy()
        zone_label = f"{zone_filter} zone"

    if long.empty:
        print(f"[lme] no rows left after zone filter ({zone_label}); aborting.")
        return pd.DataFrame()

    regime_order = _regimes_by_level(long["regime"].unique())
    print(f"\n[lme] {len(long)} rows | zone filter: {zone_label}")
    print(f"[lme] regime order (by level): {regime_order}")
    print(f"[lme] regime distribution:\n"
          f"{long.groupby('regime')['persona_id'].nunique().to_string()}")
    print(f"[lme] age_refined: mean={long['age_refined'].mean():.1f}, "
          f"std={long['age_refined'].std():.1f}, "
          f"range=[{long['age_refined'].min():.0f}, "
          f"{long['age_refined'].max():.0f}]")

    # 5. Multicollinearity check: age vs regime level
    long = long.copy()
    long["regime_num"] = long["regime"].map({r: i for i, r in enumerate(regime_order)})
    rho, rho_p = stats.spearmanr(long["regime_num"], long["age_refined"])
    print(f"\n[lme] age-regime Spearman rho = {rho:.3f} (p={rho_p:.4f})")
    if abs(rho) > 0.5:
        print("  NOTE: high age-regime correlation -- interpret the age "
              "coefficient with caution (partial effect after regime).")

    # 6. Fit LME per metric
    results = []
    for metric in [m for m in METRICS if m in long.columns]:
        print(f"  fitting {metric}...", end=" ", flush=True)
        res = fit_lme(long, metric, regime_order)
        if res:
            results.append(res)
            sig_r = _stars(res["p_regime"])
            sig_a = _stars(res["p_age"])
            print(f"regime LRT chi2={res['lrt_chi2']:.2f} "
                  f"p={res['p_regime']:.4f}{sig_r}  | "
                  f"age b={res['age_coef']:.4f} p={res['p_age']:.4f}{sig_a}")
        else:
            print("SKIPPED")

    if not results:
        print("[lme] no results produced.")
        return pd.DataFrame()

    out_df = pd.DataFrame(results)

    # 7. Summary
    n_regime_sig = int(out_df["regime_sig"].sum())
    n_age_sig = int(out_df["age_sig"].sum())
    print(f"\n{'-' * 60}")
    print(f"REGIME significant (LRT p<.05): {n_regime_sig}/{len(out_df)} metrics")
    print(f"AGE    significant (t   p<.05): {n_age_sig}/{len(out_df)} metrics")
    print(f"{'-' * 60}")

    # 8. Save
    tag = model_tag or _safe_tag(model_name)
    if zone_filter and zone_filter != "all":
        tag += f"_{zone_filter}zone"
    out_path = os.path.join(output_dir, f"{tag}_lme_regime_vs_age.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[lme] saved -> {out_path}")
    return out_df


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    ap = argparse.ArgumentParser(
        description="LME: graph metric ~ regime + age per persona "
                    "(LRT for regime, t-test for age).")
    ap.add_argument("--input-dir", nargs="+", required=True,
                    help="One or more directories containing raw triplet CSVs.")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-dir", default="tusa_text/min_drp_texts")
    ap.add_argument("--zone", default="all",
                    choices=["all", "below", "on", "above"],
                    help="Restrict to a competence zone (default: all).")
    ap.add_argument("--label", nargs="*", default=None,
                    help="Explicit run_dir=label overrides, "
                         "e.g. --label run_228474=primary")
    args = ap.parse_args()

    analyze(args.input_dir, args.model, args.output_dir,
            text_dir=args.text_dir, zone_filter=args.zone,
            label_overrides=parse_label_overrides(args.label))


if __name__ == "__main__":
    main()
