#!/usr/bin/env python3
"""Task 2: ICBHI Inversion Mechanism Analysis.
Examines:
1. Mean detector score by device and label (Normal vs Anomaly) across all 6,898 clips
   for illustrative detectors: LOF (best pooled), Sampling (median), KDE, COPOD.
2. Within-device score difference: Delta = Score(Anom) - Score(Norm).
3. Between-device score offsets: Device overall mean vs device anomaly prevalence.
4. Explains whether pooled inversion is driven by within-device inversion (AKGC417L)
   or between-device score offsets interacting with prevalence.
"""

import os
import numpy as np
import pandas as pd

SNAP_DIR = "./snap"
OUT_DIR = "results/analysis_r3/F5"

def parse_device_from_path(p):
    fname = os.path.basename(p)
    return os.path.splitext(fname)[0].split("_")[4]

def main():
    rcs_path = os.path.join(SNAP_DIR, "release/results/icbhi/rot_clip_scores.parquet")
    rcs = pd.read_parquet(rcs_path)
    rcs = rcs[rcs["setting"] == "primary"].copy()
    rcs["device"] = rcs["path"].apply(parse_device_from_path)
    
    target_dets = ["LOF", "Sampling", "KDE", "COPOD"]
    devices = ["AKGC417L", "Litt3200", "LittC2SE", "Meditron"]
    
    rows = []
    for det in target_dets:
        sub = rcs[rcs["detector"] == det]
        # Average score across seeds for each unique clip (row_index)
        clip_scores = sub.groupby(["row_index", "device", "anomaly_label"])["score"].mean().reset_index()
        
        # Pooled stats
        pooled_norm = clip_scores[clip_scores["anomaly_label"] == 0]["score"]
        pooled_anom = clip_scores[clip_scores["anomaly_label"] == 1]["score"]
        
        rows.append({
            "detector": det,
            "device": "POOLED (All)",
            "n_total": len(clip_scores),
            "n_norm": len(pooled_norm),
            "n_anom": len(pooled_anom),
            "prevalence": len(pooled_anom) / len(clip_scores),
            "mean_score_norm": pooled_norm.mean(),
            "mean_score_anom": pooled_anom.mean(),
            "delta_anom_minus_norm": pooled_anom.mean() - pooled_norm.mean(),
            "mean_score_overall": clip_scores["score"].mean()
        })
        
        for d in devices:
            d_clips = clip_scores[clip_scores["device"] == d]
            norm_s = d_clips[d_clips["anomaly_label"] == 0]["score"]
            anom_s = d_clips[d_clips["anomaly_label"] == 1]["score"]
            
            rows.append({
                "detector": det,
                "device": d,
                "n_total": len(d_clips),
                "n_norm": len(norm_s),
                "n_anom": len(anom_s),
                "prevalence": len(anom_s) / len(d_clips),
                "mean_score_norm": norm_s.mean(),
                "mean_score_anom": anom_s.mean(),
                "delta_anom_minus_norm": anom_s.mean() - norm_s.mean(),
                "mean_score_overall": d_clips["score"].mean()
            })
            
    df_mech = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "icbhi_mechanism_score_offsets.csv")
    df_mech.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    
    print("\n=== ICBHI SCORE BREAKDOWN BY DEVICE AND ANOMALY LABEL ===")
    print(df_mech.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
