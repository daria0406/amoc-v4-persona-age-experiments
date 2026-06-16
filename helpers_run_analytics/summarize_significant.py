import os
import re
import glob
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _load_text_labels(text_dir: Optional[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not text_dir or not os.path.isdir(text_dir):
        return labels
    for path in glob.glob(os.path.join(text_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8") as fh:
                labels[_norm(fh.read())] = stem
        except OSError:
            continue
    return labels


def _story_label(story_text: str, label_map: Dict[str, str]) -> Optional[str]:
    norm = _norm(story_text)
    if norm in label_map:
        return label_map[norm]
    for known_norm, stem in label_map.items():
        if norm.startswith(known_norm[:80]) or known_norm.startswith(norm[:80]):
            return stem
    return None


def _recover_story(run_dir: str) -> str:
    for mp in sorted(glob.glob(os.path.join(run_dir, "matrix", "amoc_matrix_*.csv"))):
        try:
            mdf = pd.read_csv(mp, index_col=0, nrows=1)
        except Exception:
            continue
        if len(mdf.index) and str(mdf.index[0]) == "story_text" and mdf.shape[1] > 0:
            val = str(mdf.iloc[0, 0])
            if val.strip():
                return val
    return ""


def _mw_text_label(csv_path: str, label_map: Dict[str, str],
                   uniquifier: str) -> str:
    d = os.path.dirname(os.path.abspath(csv_path))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "matrix")) or \
           os.path.isdir(os.path.join(d, "triplets")):
            story = _recover_story(d)
            if story:
                label = _story_label(story, label_map)
                if label:
                    return label
            return os.path.basename(os.path.normpath(d))
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return uniquifier


def _unique_suffixes(paths: List[str]) -> Dict[str, str]:
    dirs = [os.path.dirname(os.path.abspath(p)) for p in paths]
    common = os.path.commonpath(dirs) if len(dirs) > 1 else os.path.dirname(dirs[0])
    out = {}
    for p, d in zip(paths, dirs):
        rel = os.path.relpath(d, common)
        out[p] = rel if rel and rel != "." else os.path.basename(d)
    return out


def _band(abs_r: float) -> str:
    if abs_r >= 0.5:
        return "large"
    if abs_r >= 0.3:
        return "medium"
    if abs_r >= 0.1:
        return "small"
    return "negligible"


UNIFIED_COLS = ["axis", "test", "regime", "text", "metric", "comparison",
                "effect", "abs_effect", "effect_band", "direction", "p"]


def _from_friedman(path: str) -> List[Dict]:
    df = pd.read_csv(path)
    if "significant" not in df.columns:
        return []
    sig = df[df["significant"] == True]
    rows = []
    for _, r in sig.iterrows():
        w = float(r.get("kendall_w", np.nan))
        rows.append({
            "axis": "text (omnibus)", "test": "Friedman",
            "regime": r["regime"], "text": "(all texts)",
            "metric": r["metric"], "comparison": "across texts",
            "effect": w, "abs_effect": abs(w), "effect_band": _band(abs(w)),
            "direction": "", "p": float(r.get("p_value", np.nan)),
        })
    return rows


def _from_wilcoxon(path: str) -> List[Dict]:
    df = pd.read_csv(path)
    if "significant" not in df.columns:
        return []
    sig = df[df["significant"] == True]
    rows = []
    for _, r in sig.iterrows():
        rb = float(r.get("effect_size_rb", np.nan))
        hi, lo = (r["text_1"], r["text_2"]) if rb >= 0 else (r["text_2"], r["text_1"])
        rows.append({
            "axis": "text (pairwise)", "test": "Wilcoxon",
            "regime": r["regime"], "text": f"{r['text_1']} vs {r['text_2']}",
            "metric": r["metric"], "comparison": f"{r['text_1']} vs {r['text_2']}",
            "effect": rb, "abs_effect": abs(rb), "effect_band": _band(abs(rb)),
            "direction": f"{hi} > {lo}", "p": float(r.get("p_corrected", np.nan)),
        })
    return rows


def _from_mannwhitney(path: str, text_label: str) -> List[Dict]:
    df = pd.read_csv(path)
    if "significant" not in df.columns:
        return []
    sig = df[df["significant"] == True]
    rows = []
    for _, r in sig.iterrows():
        rr = float(r.get("effect_size_r", np.nan))
        hi, lo = (r["group_2"], r["group_1"]) if rr >= 0 else (r["group_1"], r["group_2"])
        rows.append({
            "axis": "regime (pairwise)", "test": "Mann-Whitney",
            "regime": f"{r['group_1']} vs {r['group_2']}", "text": text_label,
            "metric": r["metric"], "comparison": f"{r['group_1']} vs {r['group_2']}",
            "effect": rr, "abs_effect": abs(rr), "effect_band": _band(abs(rr)),
            "direction": f"{hi} > {lo}", "p": float(r.get("p_corrected", np.nan)),
        })
    return rows


def _find(input_dirs: List[str], suffix: str) -> List[str]:
    out: List[str] = []
    for d in input_dirs:
        out.extend(glob.glob(os.path.join(d, "**", f"*{suffix}"), recursive=True))
    seen, uniq = set(), []
    for p in sorted(out):
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def summarize(input_dirs: List[str], text_dir: Optional[str]) -> pd.DataFrame:
    label_map = _load_text_labels(text_dir)
    rows: List[Dict] = []

    for p in _find(input_dirs, "_friedman_omnibus.csv"):
        rows.extend(_from_friedman(p))
    for p in _find(input_dirs, "_wilcoxon_posthoc.csv"):
        rows.extend(_from_wilcoxon(p))

    mw_paths = _find(input_dirs, "_mannwhitney_pairwise.csv")
    uniq = _unique_suffixes(mw_paths) if mw_paths else {}
    for p in mw_paths:
        rows.extend(_from_mannwhitney(p, _mw_text_label(p, label_map, uniq[p])))

    if not rows:
        return pd.DataFrame(columns=UNIFIED_COLS)
    df = pd.DataFrame(rows, columns=UNIFIED_COLS)
    df = df.sort_values(["axis", "abs_effect"], ascending=[True, False])
    df["effect"] = df["effect"].round(4)
    df["abs_effect"] = df["abs_effect"].round(4)
    df["p"] = df["p"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description="Combine significant rows from the Friedman, Wilcoxon and "
                    "Mann-Whitney outputs into one ranked table.")
    ap.add_argument("--input-dir", required=True, nargs="+",
                    help="One or more dirs to scan recursively for "
                         "*_friedman_omnibus.csv, *_wilcoxon_posthoc.csv and "
                         "*_mannwhitney_pairwise.csv. Pass the analysis dir and "
                         "all per-text run dirs.")
    ap.add_argument("--text-dir", default="tusa_text/min_drp_texts",
                    help="Dir of *.txt passages, used to label Mann-Whitney texts.")
    ap.add_argument("--out", default=None, help="Optional path to write the CSV.")
    ap.add_argument("--top", type=int, default=0,
                    help="Only print the top-N rows per axis (0 = all).")
    args = ap.parse_args()

    df = summarize(args.input_dir, args.text_dir)
    if df.empty:
        print("[summary] No significant rows found.")
        return

    shown = df
    if args.top > 0:
        shown = df.groupby("axis", group_keys=False).head(args.top)

    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 40):
        for axis in ["text (omnibus)", "text (pairwise)", "regime (pairwise)"]:
            block = shown[shown["axis"] == axis]
            if block.empty:
                continue
            print(f"\n=== {axis} === ({len(df[df['axis'] == axis])} significant)")
            print(block[["test", "regime", "text", "metric", "effect",
                         "effect_band", "direction", "p"]].to_string(index=False))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n[summary] wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
