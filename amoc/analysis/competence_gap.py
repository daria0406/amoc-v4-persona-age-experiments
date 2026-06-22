import os
import argparse
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata
import statsmodels.formula.api as smf

from amoc.analysis.repeated_measures import (
    build_long, METRICS, _safe_tag, parse_label_overrides)

LEVELS = {
    "primary": 0, "secondary": 1,
    "high_school": 2, "highschool": 2,
    "college": 3, "university": 3,
}


def _level(name) -> Optional[int]:
    return LEVELS.get(str(name).strip().lower())


def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def _annotate_levels(long: pd.DataFrame):
    df = long.copy()
    df["regime_level"] = df["regime"].map(_level)
    df["text_level"] = df["text"].map(_level)
    bad_reg = sorted(df.loc[df["regime_level"].isna(), "regime"].unique())
    bad_txt = sorted(df.loc[df["text_level"].isna(), "text"].unique())
    df = df.dropna(subset=["regime_level", "text_level"]).copy()
    df["regime_level"] = df["regime_level"].astype(int)
    df["text_level"] = df["text_level"].astype(int)
    df["gap"] = df["text_level"] - df["regime_level"]
    df["above"] = df["gap"].clip(lower=0)
    df["zone"] = np.select(
        [df["gap"] < 0, df["gap"] == 0, df["gap"] > 0],
        ["below", "on", "above"], default="on")
    return df, bad_reg, bad_txt


def competence_gap(long: pd.DataFrame, alpha: float = 0.05):
    df0, bad_reg, bad_txt = _annotate_levels(long)
    if bad_reg or bad_txt:
        print(f"[competence-gap] WARNING unmapped levels dropped -- "
              f"regimes={bad_reg} texts={bad_txt}. Use --label to fix "
              f"(e.g. --label run_228474=primary).")

    model_rows: List[Dict] = []
    zone_rows: List[Dict] = []

    for metric in METRICS:
        if metric not in df0.columns:
            continue
        d = df0[["persona_id", "regime_level", "text_level", "gap", "above",
                 "zone", metric]].dropna().copy()
        if d[metric].nunique() < 3 or d["above"].nunique() < 2:
            continue

        d["zt"] = d.groupby("text_level")[metric].transform(_zscore)
        means = d.groupby("zone")["zt"].mean()
        groups = [d.loc[d["zone"] == z, "zt"].values
                  for z in ("below", "on", "above") if (d["zone"] == z).any()]
        if len(groups) >= 2:
            h, p = kruskal(*groups)
        else:
            h, p = np.nan, np.nan
        zone_rows.append({
            "metric": metric,
            "mean_below": round(float(means.get("below", np.nan)), 4),
            "mean_on": round(float(means.get("on", np.nan)), 4),
            "mean_above": round(float(means.get("above", np.nan)), 4),
            "kruskal_h": round(float(h), 4) if np.isfinite(h) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
            "significant": bool(np.isfinite(p) and p < alpha),
        })

        d["y"] = rankdata(d[metric].values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = smf.mixedlm(
                    "y ~ regime_level + text_level + above + above:regime_level",
                    data=d, groups=d["persona_id"]).fit(reml=False)
            except Exception as exc:
                model_rows.append({"metric": metric, "term": "(all)",
                                   "coef": np.nan, "std_err": np.nan,
                                   "z": np.nan, "p_value": np.nan,
                                   "significant": False, "note": f"fit failed: {exc}"})
                continue

        for term in ("regime_level", "above", "above:regime_level"):
            if term not in model.params.index:
                continue
            coef = float(model.params[term])
            se = float(model.bse[term])
            z = float(model.tvalues[term])
            pv = float(model.pvalues[term])
            model_rows.append({"metric": metric, "term": term,
                               "coef": round(coef, 4), "std_err": round(se, 4),
                               "z": round(z, 4), "p_value": pv,
                               "significant": bool(pv < alpha), "note": ""})

    model_df = pd.DataFrame(model_rows)
    zone_df = pd.DataFrame(zone_rows)
    return model_df, zone_df


def analyze(input_dir: Union[str, List[str]], model_name: str, output_dir: str,
            text_dir: Optional[str] = "tusa_text/min_drp_texts",
            alpha: float = 0.05, model_tag: Optional[str] = None,
            label_overrides: Optional[Dict[str, str]] = None):
    long = build_long(input_dir, model_name, text_dir, require_min_texts=2,
                      label_overrides=label_overrides)
    if long.empty:
        print("[competence-gap] no data (need >=2 texts and matching personas).")
        return pd.DataFrame(), pd.DataFrame()

    print(f"[competence-gap] {len(long)} persona x text rows | "
          f"regimes={sorted(long['regime'].unique())} | "
          f"texts={sorted(long['text'].unique())}")

    model_df, zone_df = competence_gap(long, alpha=alpha)

    os.makedirs(output_dir, exist_ok=True)
    tag = model_tag or _safe_tag(model_name)
    model_path = os.path.join(output_dir, f"{tag}_competence_gap_model.csv")
    zone_path = os.path.join(output_dir, f"{tag}_competence_gap_zones.csv")
    model_df.to_csv(model_path, index=False)
    zone_df.to_csv(zone_path, index=False)
    print(f"[competence-gap] saved model -> {model_path}")
    print(f"[competence-gap] saved zones -> {zone_path}")

    if not model_df.empty:
        above = model_df[(model_df["term"] == "above") & model_df["significant"]]
        print(f"\n[competence-gap] developmental ceiling (above-level penalty): "
              f"{len(above)} metric(s) significant")
        if not above.empty:
            print(above[["metric", "coef", "z", "p_value"]].to_string(index=False))
        resist = model_df[(model_df["term"] == "above:regime_level")
                          & model_df["significant"]]
        print(f"[competence-gap] regime resistance (above x regime_level): "
              f"{len(resist)} metric(s) significant")
        if not resist.empty:
            print(resist[["metric", "coef", "z", "p_value"]].to_string(index=False))

    if not zone_df.empty:
        print("\n[competence-gap] within-text z by zone (below / on / above):")
        print(zone_df[["metric", "mean_below", "mean_on", "mean_above",
                       "p_value", "significant"]].to_string(index=False))

    return model_df, zone_df


def main():
    ap = argparse.ArgumentParser(
        description="Competence-gap analysis: does extraction degrade when the "
                    "text level exceeds the persona's education level (above = "
                    "max(text_level - regime_level, 0)), and do higher regimes "
                    "resist that penalty (above x regime_level)?")
    ap.add_argument("--input-dir", required=True, nargs="+")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-dir", default="tusa_text/min_drp_texts")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--label", nargs="*", default=None,
                    help="Explicit run_dir=label overrides, "
                         "e.g. --label run_228474=primary")
    args = ap.parse_args()
    analyze(args.input_dir, args.model, args.output_dir,
            text_dir=args.text_dir, alpha=args.alpha,
            label_overrides=parse_label_overrides(args.label))


if __name__ == "__main__":
    main()
