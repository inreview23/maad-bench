#!/usr/bin/env python3
"""f2_tempsr_investigation.py

Investigates TempSR (PyOD SpectralResidual on 1D frame-to-frame spectral flux):
1. Performance of TempSR across all 7 datasets and both settings.
2. Comparison against the entire 25 clip-level detector roster.
3. Rationale for why no valid clip-level analogue exists in the 25-detector roster.
"""

import os
import numpy as np
import pandas as pd

DATA_ROOT = "results"
DATASETS = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
SETTINGS = ["primary", "secondary"]


def investigate_tempsr():
    rows = []
    for setting in SETTINGS:
        for d in DATASETS:
            t = pd.read_parquet(os.path.join(DATA_ROOT, d, "rot_temporal_runs.parquet"))
            c = pd.read_parquet(os.path.join(DATA_ROOT, d, "rot_runs.parquet"))

            t = t[(t.setting == setting) & (t.detector == "TempSR")]
            c = c[c.setting == setting]

            t_fold = t.groupby("group").auc_roc.mean()
            m_tsr = float(t_fold.mean())
            s_tsr = float(t_fold.std(ddof=1))
            k = len(t_fold)

            # Compare against 25 clip detectors
            c_fold = c.groupby(["detector", "group"]).auc_roc.mean().unstack(level=0)
            c_means = c_fold.mean(axis=0)

            # How many clip detectors does TempSR beat?
            clip_beat = int((m_tsr > c_means).sum())
            clip_total = len(c_means)
            rank_in_25 = clip_total - clip_beat + 1

            rows.append({
                "setting": setting,
                "dataset": d,
                "k_folds": k,
                "tempsr_mean": m_tsr,
                "tempsr_sd": s_tsr,
                "clip_detectors_beaten": f"{clip_beat}/{clip_total}",
                "effective_rank": f"{rank_in_25} of {clip_total + 1}",
                "below_chance": m_tsr < 0.5,
                "min_clip_det": f"{c_means.min():.3f} ({c_means.idxmin()})",
                "med_clip_det": f"{c_means.median():.3f}",
                "max_clip_det": f"{c_means.max():.3f} ({c_means.idxmax()})",
            })

    res = pd.DataFrame(rows)
    return res


if __name__ == "__main__":
    df = investigate_tempsr()
    df.to_csv("f2_tempsr_analysis.csv", index=False)
    for setting in SETTINGS:
        print(f"\n=== TempSR vs Clip Roster: {setting.upper()} ===")
        sub = df[df.setting == setting]
        cols = ["dataset", "k_folds", "tempsr_mean", "tempsr_sd", "below_chance",
                "clip_detectors_beaten", "effective_rank", "med_clip_det", "max_clip_det"]
        print(sub[cols].to_string(index=False))
