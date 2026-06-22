import os
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, norm
from statsmodels.stats.multitest import multipletests

from amoc.analysis.repeated_measures import build_long, METRICS, _ordered_texts

REGIME_ORDER = ["primary", "secondary", "high_school", "university"]
REGIME_LEVEL = {r: i for i, r in enumerate(REGIME_ORDER)}

LEVELS = {
    "primary": 0, "secondary": 1,
    "high_school": 2, "highschool": 2,
    "college": 3, "university": 3,
}

EPS_BANDS = [(0.01, "small"), (0.06, "medium"), (0.14, "large")]
RHO_BANDS = [(0.10, "small"), (0.30, "medium"), (0.50, "large")]


def _band(value, bands):
    v = abs(value)
    if np.isnan(v):
        return "na"
    out = "negligible"
    for thr, name in bands:
        if v >= thr:
            out = name
    return out


def _epsilon_squared(h, n, k):
    return float("nan") if n - k <= 0 else float((h - k + 1) / (n - k))


def _count_concordant(a, b):
    a = np.sort(np.asarray(a, float))
    left = np.searchsorted(a, b, side="left")
    right = np.searchsorted(a, b, side="right")
    less = left
    equal = right - left
    return float(less.sum() + 0.5 * equal.sum())


def jonckheere_terpstra(groups_ordered):
    sizes = [len(g) for g in groups_ordered]
    N = sum(sizes)
    if len([s for s in sizes if s > 0]) < 2 or N < 3:
        return np.nan, np.nan, np.nan
    JT = 0.0
    for i in range(len(groups_ordered)):
        for j in range(i + 1, len(groups_ordered)):
            JT += _count_concordant(groups_ordered[i], groups_ordered[j])
    sum_n2 = sum(s * s for s in sizes)
    mean = (N * N - sum_n2) / 4.0
    var = (N * N * (2 * N + 3) - sum(s * s * (2 * s + 3) for s in sizes)) / 72.0
    if var <= 0:
        return JT, np.nan, np.nan
    z = (JT - mean) / np.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return JT, float(z), float(p)


def _kruskal_eps(groups):
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return np.nan, np.nan, np.nan
    try:
        h, p = kruskal(*groups)
    except ValueError:
        return np.nan, np.nan, np.nan
    return float(h), float(p), _epsilon_squared(h, n, k)


def _signed_rank_biserial(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, np.nan
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    rb = 2.0 * u / (nx * ny) - 1.0
    return float(rb), float(p)


def bootstrap_ci(value_fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = value_fn(rng)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------
# Zone annotation: labels each observation as below / on / above
# based on whether text_level exceeds regime_level.
# Used for RQ4 zone-stratified analysis.
# ------------------------------------------------------------------
def _annotate_gap(long: pd.DataFrame) -> pd.DataFrame:
    df = long.copy()
    df["regime_level_num"] = df["regime"].map(LEVELS)
    df["text_level_num"] = df["text"].map(LEVELS)
    df = df.dropna(subset=["regime_level_num", "text_level_num"]).copy()
    df["regime_level_num"] = df["regime_level_num"].astype(int)
    df["text_level_num"] = df["text_level_num"].astype(int)
    df["gap"] = df["text_level_num"] - df["regime_level_num"]
    df["above"] = df["gap"].clip(lower=0)
    df["zone"] = np.select(
        [df["gap"] < 0, df["gap"] == 0, df["gap"] > 0],
        ["below", "on", "above"],
        default="on",
    )
    return df


# ------------------------------------------------------------------
# Zone-stratified Kruskal-Wallis + pairwise Mann-Whitney (RQ4).
# Filters to above-level observations (text > persona level) and
# tests whether the 4 regime groups produce different graph metrics
# under the same difficulty condition.
# ------------------------------------------------------------------
def zone_regime_analysis(long: pd.DataFrame, alpha: float = 0.05):
    df = _annotate_gap(long)
    metrics = [m for m in METRICS if m in df.columns]

    kw_rows = []
    mw_rows = []

    for zone in ["below", "on", "above"]:
        sub_z = df[df["zone"] == zone]
        if sub_z.empty:
            continue

        for metric in metrics:
            sub = sub_z[["regime", metric]].dropna()
            if sub.empty:
                continue

            present = [r for r in REGIME_ORDER if r in set(sub["regime"])]
            groups = [sub.loc[sub["regime"] == r, metric].to_numpy()
                      for r in present]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue

            n = int(sum(len(g) for g in groups))
            k = len(groups)
            h, p_kw, eps = _kruskal_eps(groups)

            kw_rows.append({
                "zone": zone,
                "metric": metric,
                "n": n,
                "k": k,
                "kruskal_h": round(float(h), 4) if not np.isnan(h) else np.nan,
                "p_value": float(p_kw) if not np.isnan(p_kw) else np.nan,
                "epsilon_sq": round(float(eps), 4) if not np.isnan(eps) else np.nan,
                "significant": bool(p_kw < alpha) if not np.isnan(p_kw) else False,
            })

            # pairwise Mann-Whitney between all regime pairs
            raw = []
            arrs = {r: sub.loc[sub["regime"] == r, metric].to_numpy()
                    for r in present}
            for ra, rb in itertools.combinations(present, 2):
                x, y = arrs[ra], arrs[rb]
                rbis, pv = _signed_rank_biserial(x, y)
                direction = (f"{ra} > {rb}" if np.median(x) > np.median(y)
                             else f"{rb} > {ra}")
                raw.append((ra, rb, len(x), len(y), rbis, pv, direction))

            if raw:
                ps = [r[5] for r in raw]
                p_bh = multipletests(ps, method="fdr_bh")[1]
                for (ra, rb, na, nb, rbis, pv, direction), pb in zip(raw, p_bh):
                    mw_rows.append({
                        "zone": zone,
                        "metric": metric,
                        "regime_a": ra,
                        "regime_b": rb,
                        "n_a": na,
                        "n_b": nb,
                        "rank_biserial": round(rbis, 4),
                        "effect_band": _band(rbis, RHO_BANDS),
                        "direction": direction,
                        "p_raw": float(pv),
                        "p_bh": float(pb),
                        "significant": bool(pb < alpha),
                    })

    kw_df = pd.DataFrame(kw_rows)
    mw_df = pd.DataFrame(mw_rows)
    return kw_df, mw_df


def analyze(input_dir, model_name, output_dir, text_dir, label_overrides,
            n_boot, seed):
    long = build_long(input_dir, model_name, text_dir,
                      label_overrides=label_overrides)
    if long.empty:
        print("[within-text] no data after build_long; aborting.")
        return
    long = long[long["regime"].isin(REGIME_ORDER)].copy()
    long["regime_level"] = long["regime"].map(REGIME_LEVEL)
    texts = _ordered_texts(sorted(long["text"].unique()))
    metrics = [m for m in METRICS if m in long.columns]

    n_personas = long["persona_id"].nunique()
    print(f"[within-text] {len(long)} rows | {n_personas} distinct personas | "
          f"texts={texts} | metrics={len(metrics)}")
    print("[within-text] personas per regime:")
    print(long.groupby("regime")["persona_id"].nunique().to_string())

    eff_rows, pair_rows = [], []
    for text in texts:
        sub_t = long[long["text"] == text]
        for metric in metrics:
            sub = sub_t[["regime", "regime_level", metric]].dropna()
            if sub.empty:
                continue
            present = [r for r in REGIME_ORDER if r in set(sub["regime"])]
            groups = [sub.loc[sub["regime"] == r, metric].to_numpy()
                      for r in present]
            if len([g for g in groups if len(g) > 0]) < 2:
                continue
            n = int(sum(len(g) for g in groups)); k = len(groups)

            h, p_kw, eps = _kruskal_eps(groups)
            jt, jt_z, jt_p = jonckheere_terpstra(groups)
            rho, rho_p = spearmanr(sub["regime_level"], sub[metric])

            arrs = {r: sub.loc[sub["regime"] == r, metric].to_numpy()
                    for r in present}
            lvl = {r: REGIME_LEVEL[r] for r in present}

            def boot_eps(rng):
                gs = [rng.choice(arrs[r], size=len(arrs[r]), replace=True)
                      for r in present]
                hh, _, ee = _kruskal_eps(gs)
                return ee

            def boot_rho(rng):
                xs, ys = [], []
                for r in present:
                    bs = rng.choice(arrs[r], size=len(arrs[r]), replace=True)
                    xs.append(np.full(len(bs), lvl[r])); ys.append(bs)
                xs = np.concatenate(xs); ys = np.concatenate(ys)
                if np.ptp(ys) == 0:
                    return np.nan
                return spearmanr(xs, ys).correlation

            eps_lo, eps_hi = bootstrap_ci(boot_eps, n_boot, seed)
            rho_lo, rho_hi = bootstrap_ci(boot_rho, n_boot, seed + 1)

            eff_rows.append({
                "text": text, "metric": metric, "n": n, "k": k,
                "kruskal_h": round(h, 4), "kruskal_p": p_kw,
                "epsilon_sq": round(eps, 4),
                "eps_ci_lo": round(eps_lo, 4), "eps_ci_hi": round(eps_hi, 4),
                "eps_band": _band(eps, EPS_BANDS),
                "jt_z": None if np.isnan(jt_z) else round(jt_z, 4), "jt_p": jt_p,
                "spearman_rho": round(rho, 4), "rho_p": rho_p,
                "rho_ci_lo": round(rho_lo, 4), "rho_ci_hi": round(rho_hi, 4),
                "rho_band": _band(rho, RHO_BANDS),
            })

            raw = []
            for ra, rb in itertools.combinations(present, 2):
                x = sub.loc[sub["regime"] == ra, metric].to_numpy()
                y = sub.loc[sub["regime"] == rb, metric].to_numpy()
                rbis, pv = _signed_rank_biserial(x, y)
                direction = (f"{ra} > {rb}" if np.median(x) > np.median(y)
                             else f"{rb} > {ra}")
                raw.append((ra, rb, len(x), len(y), rbis, pv, direction))
            if raw:
                ps = [r[5] for r in raw]
                p_bh = multipletests(ps, method="fdr_bh")[1]
                for (ra, rb, na, nb, rbis, pv, direction), pb in zip(raw, p_bh):
                    pair_rows.append({
                        "text": text, "metric": metric,
                        "regime_a": ra, "regime_b": rb, "n_a": na, "n_b": nb,
                        "rank_biserial": round(rbis, 4),
                        "effect_band": _band(rbis, RHO_BANDS),
                        "direction": direction,
                        "p_raw": pv, "p_bh": pb,
                        "significant": bool(pb < 0.05),
                    })

    os.makedirs(output_dir, exist_ok=True)
    tag = model_name.replace("/", "-").replace(":", "-").replace(" ", "_").lower()
    eff = pd.DataFrame(eff_rows)
    pair = pd.DataFrame(pair_rows)
    eff_path = os.path.join(output_dir, f"{tag}_within_text_effects.csv")
    pair_path = os.path.join(output_dir, f"{tag}_within_text_pairwise.csv")
    eff.to_csv(eff_path, index=False)
    pair.to_csv(pair_path, index=False)
    print(f"[within-text] saved effects  -> {eff_path}")
    print(f"[within-text] saved pairwise -> {pair_path}")

    print("\n[within-text] per-text persona effect size (epsilon^2 [95% CI], "
          "Spearman rho [95% CI], JT trend p):")
    with pd.option_context("display.width", 160, "display.max_rows", None):
        show = eff[["text", "metric", "n", "epsilon_sq", "eps_ci_lo", "eps_ci_hi",
                    "eps_band", "spearman_rho", "rho_ci_lo", "rho_ci_hi",
                    "rho_band", "jt_p"]]
        print(show.to_string(index=False))

    # ------------------------------------------------------------------
    # RQ4: Zone-stratified Kruskal-Wallis + pairwise Mann-Whitney
    # Tests whether the 4 regime groups differ on graph metrics when
    # text difficulty exceeds persona level (above zone).
    # ------------------------------------------------------------------
    print("\n[within-text] running zone-stratified regime analysis (RQ4)...")
    kw_df, mw_df = zone_regime_analysis(long, alpha=0.05)

    kw_path = os.path.join(output_dir, f"{tag}_zone_kruskal.csv")
    mw_path = os.path.join(output_dir, f"{tag}_zone_mannwhitney.csv")
    kw_df.to_csv(kw_path, index=False)
    mw_df.to_csv(mw_path, index=False)
    print(f"[within-text] saved zone KW      -> {kw_path}")
    print(f"[within-text] saved zone MW pairs -> {mw_path}")

    # print summary focused on the above zone
    above_kw = kw_df[kw_df["zone"] == "above"].copy()
    n_sig = above_kw["significant"].sum()
    print(f"\n[within-text] ABOVE zone: {n_sig}/{len(above_kw)} metrics significant "
          f"(Kruskal-Wallis, alpha=0.05)")
    print(above_kw[["metric", "n", "kruskal_h", "p_value",
                     "epsilon_sq", "significant"]].to_string(index=False))

    above_mw = mw_df[(mw_df["zone"] == "above") & mw_df["significant"]].copy()
    print(f"\n[within-text] ABOVE zone: {len(above_mw)} significant pairwise "
          f"Mann-Whitney pairs (BH-corrected):")
    if not above_mw.empty:
        print(above_mw[["metric", "regime_a", "regime_b",
                         "rank_biserial", "effect_band",
                         "direction", "p_bh"]].to_string(index=False))
    else:
        print("  none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-dir", default=None)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    label_overrides = None
    if args.label:
        label_overrides = {}
        for item in args.label:
            key, val = item.split("=", 1)
            label_overrides[os.path.basename(os.path.normpath(key.strip()))] = val.strip()

    analyze(args.input_dir, args.model, args.output_dir, args.text_dir,
            label_overrides, args.n_boot, args.seed)


if __name__ == "__main__":
    main()