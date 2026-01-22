from __future__ import annotations

import os
from typing import List

import pandas as pd


def lifecycle_snapshot(df: pd.DataFrame, sentence_idx: int) -> pd.DataFrame:
    if "sentence_index" not in df.columns:
        raise ValueError("Input DataFrame must contain 'sentence_index'.")
    if not {"subject", "relation", "object", "active"}.issubset(df.columns):
        raise ValueError("Input DataFrame missing required triplet columns.")

    df = df.copy()
    df["active_bool"] = df["active"].astype(bool)
    df_filt = df[df["sentence_index"] <= sentence_idx]
    rows: List[dict] = []
    if df_filt.empty:
        return pd.DataFrame(
            columns=[
                "subject",
                "relation",
                "object",
                "introduced_at",
                "last_active",
                "currently_active",
            ]
        )

    grouped = df_filt.groupby(["subject", "relation", "object"], sort=False)
    for (subj, rel, obj), g in grouped:
        introduced_at = int(g["sentence_index"].min())
        active_rows = g[g["active_bool"]]
        last_active = (
            int(active_rows["sentence_index"].max()) if not active_rows.empty else None
        )
        current_rows = g[g["sentence_index"] == sentence_idx]
        currently_active = (
            bool(current_rows["active_bool"].any()) if not current_rows.empty else False
        )
        rows.append(
            {
                "subject": subj,
                "relation": rel,
                "object": obj,
                "introduced_at": introduced_at,
                "last_active": last_active,
                "currently_active": currently_active,
            }
        )

    return pd.DataFrame(rows)


def validate_snapshot(
    df_events: pd.DataFrame, df_snapshot: pd.DataFrame, sentence_idx: int
) -> None:
    events_at_s = df_events[df_events["sentence_index"] == sentence_idx]
    actives_at_s = {
        (row["subject"], row["relation"], row["object"])
        for _, row in events_at_s.iterrows()
        if bool(row["active"])
    }

    for _, row in df_snapshot.iterrows():
        subj, rel, obj = row["subject"], row["relation"], row["object"]
        introduced_at = int(row["introduced_at"])
        last_active = row["last_active"]
        currently_active = bool(row["currently_active"])

        if last_active is not None:
            if not (introduced_at <= last_active <= sentence_idx):
                raise ValueError(
                    f"Lifecycle invariant violated for ({subj}, {rel}, {obj}): "
                    f"introduced_at={introduced_at}, last_active={last_active}, "
                    f"S={sentence_idx}"
                )
        if currently_active and (subj, rel, obj) not in actives_at_s:
            raise ValueError(
                f"Currently active in snapshot but not active in events at S={sentence_idx}: "
                f"({subj}, {rel}, {obj})"
            )


def export_lifecycle_snapshots(
    csv_path: str, out_dir: str, sentence_indices: List[int]
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df_events = pd.read_csv(csv_path)
    if "sentence_index" not in df_events.columns:
        raise ValueError("Input CSV must contain 'sentence_index'.")

    for s in sentence_indices:
        snap = lifecycle_snapshot(df_events, s)
        validate_snapshot(df_events, snap, s)
        out_path = os.path.join(out_dir, f"lifecycle_triplets_sentence_{s}.csv")
        snap.to_csv(out_path, index=False)


__all__ = ["lifecycle_snapshot", "validate_snapshot", "export_lifecycle_snapshots"]
