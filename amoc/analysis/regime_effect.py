import os
import argparse
import itertools
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from amoc.analysis.repeated_measures import build_long, METRICS, _safe_tag

ABSTRACTION_METRICS = ["abstract_concept_ratio", "abstract_relation_ratio"]


def _zscore_within_text(df: pd.DataFrame, metric: str) -> pd.Series:
    def z(x):
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd > 0 else x * 0.0
    return df.groupby("text")[metric].transform(z)


def _epsilon_squared(h: float, n: int, k: int) -> float:
    if n - k <= 0:
        return float("nan")
    return float((h - k + 1) / (n - k))


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    u, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - (2.0 * u) / (nx * ny))


def regime_main_effect(long: pd.DataFrame, alpha: float = 0.05):
    omnibus: List[Dict] = []
    posthoc: List[Dict] = []

    for metric in METRICS:
        if metric not in long.columns:
            continue
        df = long[["persona_id", "regime", "text", metric]].dropna()
        if df.empty:
            continue
        df = df.assign(z=_zscore_within_text(df, metric))
        agg = (df.groupby(["persona_id", "regime"], as_index=False)["z"].mean())

        regimes = sorted(agg["regime"].dropna().unique())
        groups = [agg.loc[agg["regime"] == r, "z"].values for r in regimes]
        groups = [g for g in groups if len(g) >= 2]
        n = sum(len(g) for g in groups)
        k = len(groups)
        if k < 2 or n - k <= 0:
            omnibus.append({"metric": metric, "n_personas": int(n), "n_regimes": k,
                            "kruskal_h": np.nan, "df": max(k - 1, 0),
                            "p_value": np.nan, "epsilon_sq": np.nan,
                            "significant": False,
                            "note": "skipped: need >=2 regimes with >=2 personas"})
            continue

        h, p = kruskal(*groups)
        omnibus.append({"metric": metric, "n_personas": int(n), "n_regimes": k,
                        "kruskal_h": round(float(h), 4), "df": k - 1,
                        "p_value": float(p),
                        "epsilon_sq": round(_epsilon_squared(h, n, k), 4),
                        "significant": bool(p < alpha), "note": ""})

        pair_rows: List[Dict] = []
        for r1, r2 in itertools.combinations(regimes, 2):
            a = agg.loc[agg["regime"] == r1, "z"].values
            b = agg.loc[agg["regime"] == r2, "z"].values
            if len(a) < 2 or len(b) < 2:
                continue
            _, p_raw = mannwhitneyu(a, b, alternative="two-sided")
            pair_rows.append({"metric": metric, "regime_1": r1, "regime_2": r2,
                              "n1": len(a), "n2": len(b), "p_raw": float(p_raw),
                              "effect_size_r": round(_rank_biserial(a, b), 4)})
        if pair_rows:
            _, p_corr, _, _ = multipletests([r["p_raw"] for r in pair_rows],
                                            method="fdr_bh")
            for r, pc in zip(pair_rows, p_corr):
                r["p_corrected"] = round(float(pc), 6)
                r["significant"] = bool(pc < alpha)
            posthoc.extend(pair_rows)

    omnibus_df = pd.DataFrame(omnibus)
    posthoc_df = pd.DataFrame(posthoc) if posthoc else pd.DataFrame(
        columns=["metric", "regime_1", "regime_2", "n1", "n2",
                 "p_raw", "p_corrected", "effect_size_r", "significant"])
    return omnibus_df, posthoc_df


def _aligned(values: pd.Series, regime: pd.Series, text: pd.Series, effect: str):
    gm = values.mean()
    a_i = values.groupby(regime).transform("mean")
    b_j = values.groupby(text).transform("mean")
    cell = values.groupby([regime, text]).transform("mean")
    resid = values - cell
    if effect == "regime":
        return resid + (a_i - gm)
    if effect == "text":
        return resid + (b_j - gm)
    return resid + (cell - a_i - b_j + gm)


def _term_indices(names: List[str], effect: str) -> List[int]:
    idx = []
    for i, nm in enumerate(names):
        if nm == "Intercept":
            continue
        is_inter = ":" in nm
        has_regime = "C(regime)" in nm
        has_text = "C(text)" in nm
        if effect == "regime" and has_regime and not is_inter:
            idx.append(i)
        elif effect == "text" and has_text and not is_inter:
            idx.append(i)
        elif effect == "regime:text" and is_inter:
            idx.append(i)
    return idx


def art_factorial(long: pd.DataFrame, alpha: float = 0.05):
    rows: List[Dict] = []
    for metric in METRICS:
        if metric not in long.columns:
            continue
        df = long[["persona_id", "regime", "text", metric]].dropna().copy()
        if df["regime"].nunique() < 2 or df["text"].nunique() < 2:
            continue

        for effect in ("regime", "text", "regime:text"):
            df["_aligned"] = _aligned(df[metric], df["regime"], df["text"], effect)
            df["_rank"] = rankdata(df["_aligned"].values)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = smf.mixedlm("_rank ~ C(regime) * C(text)", data=df,
                                        groups=df["persona_id"]).fit(reml=False)
                except Exception as exc:
                    rows.append({"metric": metric, "effect": effect,
                                 "wald_chi2": np.nan, "df": np.nan,
                                 "p_value": np.nan, "eta_sq": np.nan,
                                 "significant": False, "note": f"fit failed: {exc}"})
                    continue

            names = list(model.model.exog_names)
            idx = _term_indices(names, effect)
            if not idx:
                continue
            # MixedLM params = fixed effects (first, aligned with exog_names) then
            # variance components; the constraint matrix must span all of them.
            n_params = len(model.params)
            R = np.zeros((len(idx), n_params))
            for r_i, c_i in enumerate(idx):
                R[r_i, c_i] = 1.0
            wald = model.wald_test(R, scalar=False)
            chi2 = float(np.ravel(wald.statistic)[0])
            pval = float(np.ravel(wald.pvalue)[0])

            ss_effect = _rank_ss(df, effect)
            eta2 = ss_effect / df["_rank"].var(ddof=0) / len(df) if len(df) else np.nan

            rows.append({"metric": metric, "effect": effect,
                         "wald_chi2": round(chi2, 4), "df": len(idx),
                         "p_value": pval,
                         "eta_sq": round(float(eta2), 4) if np.isfinite(eta2) else np.nan,
                         "significant": bool(pval < alpha), "note": ""})
    return pd.DataFrame(rows)


def _rank_ss(df: pd.DataFrame, effect: str) -> float:
    gm = df["_rank"].mean()
    if effect == "regime":
        means = df.groupby("regime")["_rank"].transform("mean")
        return float(((means - gm) ** 2).sum())
    if effect == "text":
        means = df.groupby("text")["_rank"].transform("mean")
        return float(((means - gm) ** 2).sum())
    cell = df.groupby(["regime", "text"])["_rank"].transform("mean")
    a_i = df.groupby("regime")["_rank"].transform("mean")
    b_j = df.groupby("text")["_rank"].transform("mean")
    return float(((cell - a_i - b_j + gm) ** 2).sum())


def analyze(input_dir: Union[str, List[str]], model_name: str, output_dir: str,
            text_dir: Optional[str] = "tusa_text/min_drp_texts",
            alpha: float = 0.05, model_tag: Optional[str] = None):
    long = build_long(input_dir, model_name, text_dir, require_min_texts=2)
    if long.empty:
        print("[regime-effect] no data (need >=2 texts and matching personas).")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"[regime-effect] {len(long)} persona x text rows | "
          f"regimes={sorted(long['regime'].unique())} | "
          f"texts={sorted(long['text'].unique())}")

    main_df, posthoc_df = regime_main_effect(long, alpha=alpha)
    art_df = art_factorial(long, alpha=alpha)

    os.makedirs(output_dir, exist_ok=True)
    tag = model_tag or _safe_tag(model_name)
    main_path = os.path.join(output_dir, f"{tag}_regime_main_effect.csv")
    posthoc_path = os.path.join(output_dir, f"{tag}_regime_posthoc.csv")
    art_path = os.path.join(output_dir, f"{tag}_regime_art_factorial.csv")
    main_df.to_csv(main_path, index=False)
    posthoc_df.to_csv(posthoc_path, index=False)
    art_df.to_csv(art_path, index=False)
    print(f"[regime-effect] saved main effect   -> {main_path}")
    print(f"[regime-effect] saved post-hoc      -> {posthoc_path}")
    print(f"[regime-effect] saved ART factorial -> {art_path}")

    if not main_df.empty:
        sig = main_df[main_df["significant"]]
        print(f"\n[regime-effect] Kruskal-Wallis regime main effect: "
              f"{len(sig)}/{len(main_df)} metrics significant (alpha={alpha})")
        cols = ["metric", "kruskal_h", "p_value", "epsilon_sq"]
        print((sig if not sig.empty else main_df)[cols].to_string(index=False))
        ab = main_df[main_df["metric"].isin(ABSTRACTION_METRICS)]
        if not ab.empty:
            print("\n[regime-effect] abstraction metrics (theory-relevant):")
            print(ab[cols + ["significant"]].to_string(index=False))

    if not art_df.empty:
        sig_int = art_df[(art_df["effect"] == "regime:text") & art_df["significant"]]
        print(f"\n[regime-effect] ART regime x text interaction: "
              f"{len(sig_int)} metric(s) significant")
        if not sig_int.empty:
            print(sig_int[["metric", "wald_chi2", "p_value",
                           "eta_sq"]].to_string(index=False))

    return main_df, posthoc_df, art_df


def main():
    ap = argparse.ArgumentParser(
        description="Test the regime main effect (Kruskal-Wallis on per-persona, "
                    "within-text z-scored aggregates) and the regime x text "
                    "interaction (aligned rank transform + mixed model).")
    ap.add_argument("--input-dir", required=True, nargs="+",
                    help="Run dirs (and/or a shared parent) with "
                         "triplets/triplets_final_state/*.csv per text.")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-dir", default="tusa_text/min_drp_texts")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    analyze(args.input_dir, args.model, args.output_dir,
            text_dir=args.text_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
