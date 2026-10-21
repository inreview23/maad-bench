#!/usr/bin/env python3
"""f2_protocol_fair_comparison.py

Evaluates the comparison protocols between temporal and clip-level models:
1. Paper's Table 10 protocol: Best of 25 clip-level detectors vs Best of 2 temporal detectors.
2. Fair alternative A: Best of 2 temporal detectors vs Median clip-level detector.
3. Fair alternative B: Best of 2 temporal detectors vs Clip-level Isolation Forest.
4. Fair alternative C: Strict matched algorithm: Frame-level IForest (TempIF) vs Clip-level IForest.

Evaluated on all seven sub-datasets for both primary and secondary settings.
All comparisons verified on unrounded floating-point values.
"""

import os
import numpy as np
import pandas as pd

DATA_ROOT = "results"
DATASETS = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
SETTINGS = ["primary", "secondary"]


def compute_protocol_comparisons():
    all_results = []

    for setting in SETTINGS:
        for d in DATASETS:
            c = pd.read_parquet(os.path.join(DATA_ROOT, d, "rot_runs.parquet"))
            t = pd.read_parquet(os.path.join(DATA_ROOT, d, "rot_temporal_runs.parquet"))

            c = c[c.setting == setting]
            t = t[t.setting == setting]

            # Fold-level seed-mean AUC per detector
            c_f = c.groupby(["detector", "group"]).auc_roc.mean().unstack(level=0)
            t_f = t.groupby(["detector", "group"]).auc_roc.mean().unstack(level=0)

            # Overall unweighted mean across folds
            c_means = c_f.mean(axis=0)
            t_means = t_f.mean(axis=0)

            # Clip-level roster statistics (25 detectors)
            best_clip_det = c_means.idxmax()
            best_clip_val = float(c_means.max())
            median_clip_val = float(c_means.median())
            clip_if_val = float(c_means["IForest"])

            # Temporal roster statistics (2 detectors: TempSR, TempIF)
            best_temp_det = t_means.idxmax()
            best_temp_val = float(t_means.max())
            temp_if_val = float(t_means["TempIF"])
            temp_sr_val = float(t_means["TempSR"])

            # Differences and winner checks on UNROUNDED values
            # 1. Best-of-25 clip vs Best-of-2 temporal (Paper Table 10)
            diff_best25 = best_temp_val - best_clip_val
            winner_best25 = "Temporal" if best_temp_val > best_clip_val else "Clip"

            # 2. Best-of-2 temporal vs Median clip
            diff_med25 = best_temp_val - median_clip_val
            winner_med25 = "Temporal" if best_temp_val > median_clip_val else "Clip"

            # 3. Best-of-2 temporal vs Clip IForest
            diff_best2_cif = best_temp_val - clip_if_val
            winner_best2_cif = "Temporal" if best_temp_val > clip_if_val else "Clip"

            # 4. Matched TempIF vs Clip IForest
            diff_matched_if = temp_if_val - clip_if_val
            winner_matched_if = "TempIF" if temp_if_val > clip_if_val else "ClipIF"

            all_results.append({
                "setting": setting,
                "dataset": d,
                # Values
                "best_clip_det": best_clip_det,
                "best_clip_val": best_clip_val,
                "median_clip_val": median_clip_val,
                "clip_if_val": clip_if_val,
                "best_temp_det": best_temp_det,
                "best_temp_val": best_temp_val,
                "temp_if_val": temp_if_val,
                "temp_sr_val": temp_sr_val,
                # Comparisons
                "diff_best25": diff_best25,
                "winner_best25": winner_best25,
                "diff_med25": diff_med25,
                "winner_med25": winner_med25,
                "diff_best2_cif": diff_best2_cif,
                "winner_best2_cif": winner_best2_cif,
                "diff_matched_if": diff_matched_if,
                "winner_matched_if": winner_matched_if,
            })

    df = pd.DataFrame(all_results)
    return df


if __name__ == "__main__":
    df = compute_protocol_comparisons()
    df.to_csv("f2_protocol_comparison.csv", index=False)

    for setting in SETTINGS:
        sub = df[df.setting == setting]
        print(f"\n==================== PROTOCOL COMPARISON: {setting.upper()} ====================")
        print(f"{'Dataset':<14} | {'Paper Table 10 (Best-2 vs Best-25)':<36} | {'Best-2 vs Med-25':<22} | {'Matched TempIF vs ClipIF':<26}")
        print("-" * 105)
        for _, r in sub.iterrows():
            p_str = f"{r['best_temp_val']:.3f} ({r['best_temp_det']}) vs {r['best_clip_val']:.3f} ({r['best_clip_det']}) -> {r['winner_best25']}"
            m_str = f"{r['best_temp_val']:.3f} vs {r['median_clip_val']:.3f} -> {r['winner_med25']}"
            mat_str = f"{r['temp_if_val']:.3f} vs {r['clip_if_val']:.3f} -> {r['winner_matched_if']}"
            print(f"{r['dataset']:<14} | {p_str:<36} | {m_str:<22} | {mat_str:<26}")
