import os
import itertools
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


class PairwiseAnalysis:
    def __init__(self, df: pd.DataFrame, alpha: float = 0.05):
        self.df = df
        self.alpha = alpha
        self._results: List[Dict] = []

    def run(self, metrics: List[str]) -> "PairwiseAnalysis":
        self._results = []
        regimes = sorted(self.df["regime"].dropna().unique())
        pairs = list(itertools.combinations(regimes, 2))

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            raw: List[Dict] = []
            for g1, g2 in pairs:
                x = self.df.loc[self.df["regime"] == g1, metric].dropna().values
                y = self.df.loc[self.df["regime"] == g2, metric].dropna().values
                if len(x) < 2 or len(y) < 2:
                    continue
                u, p = mannwhitneyu(x, y, alternative="two-sided")
                r = 1.0 - (2.0 * u) / (len(x) * len(y))
                raw.append({
                    "metric": metric,
                    "group_1": g1,
                    "group_2": g2,
                    "U": u,
                    "n1": len(x),
                    "n2": len(y),
                    "p_raw": p,
                    "effect_size_r": round(r, 4),
                })

            if not raw:
                continue

            _, p_corrected, _, _ = multipletests(
                [r["p_raw"] for r in raw], method="fdr_bh"
            )
            for rec, p_corr in zip(raw, p_corrected):
                rec["p_corrected"] = round(float(p_corr), 6)
                rec["significant"] = p_corr < self.alpha
                self._results.append(rec)

        return self

    def to_dataframe(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        cols = [
            "metric", "group_1", "group_2",
            "n1", "n2", "U",
            "p_raw", "p_corrected", "effect_size_r", "significant",
        ]
        return pd.DataFrame(self._results)[cols]

    def save(self, output_dir: str, model_tag: str) -> Optional[str]:
        df = self.to_dataframe()
        if df.empty:
            print("[pairwise] No results to save.")
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{model_tag}_mannwhitney_pairwise.csv")
        df.to_csv(path, index=False)
        print(f"[pairwise] Saved {len(df)} comparisons to {path}")
        return path

    def print_summary(self) -> None:
        df = self.to_dataframe()
        if df.empty:
            print("[pairwise] No results.")
            return
        sig = df[df["significant"]]
        print(
            f"\n[pairwise] {len(sig)}/{len(df)} pairs significant "
            f"after BH correction (α={self.alpha})"
        )
        if not sig.empty:
            print(
                sig[["metric", "group_1", "group_2", "p_corrected", "effect_size_r"]]
                .to_string(index=False)
            )
