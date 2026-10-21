#!/usr/bin/env python3
"""Task 1: ICBHI Full Roster (25 PyOD detectors + rms0 baseline).
Computes:
- Pooled fold-mean AUC
- Within-device fold-mean AUC for each device (AKGC417L, Litt3200, LittC2SE, Meditron)
- Device-balanced AUC:
  1) Per-fold pair-weighted fold mean (dev_bal_pair_weighted)
  2) Overall pair-weighted across all (fold, device) cells
  3) Equal-weighted mean across the 4 devices (dev_bal_equal_weighted)
- Counts of detectors below 0.5 under each setting
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SNAP_DIR = "./snap"
OUT_DIR = "results/analysis_r3/F5"

def parse_device_from_path(p):
    fname = os.path.basename(p)
    parts = os.path.splitext(fname)[0].split("_")
    return parts[4]

def main():
    meta_path = os.path.join(SNAP_DIR, "release/results/icbhi/metadata.parquet")
    rcs_path = os.path.join(SNAP_DIR, "release/results/icbhi/rot_clip_scores.parquet")
    feat_path = os.path.join(SNAP_DIR, "release/results/icbhi/features.parquet")
    
    meta = pd.read_parquet(meta_path)
    feat = pd.read_parquet(feat_path)
    rcs = pd.read_parquet(rcs_path)
    
    # Filter to primary setting
    rcs = rcs[rcs["setting"] == "primary"].copy()
    rcs["device"] = rcs["path"].apply(parse_device_from_path)
    meta["device"] = meta["path"].apply(parse_device_from_path)
    
    # Add rms0 absolute-z baseline as a pseudo-detector
    # Compute z-scores fold by fold on features
    df_feat_meta = pd.concat([meta, feat[["rms0_mean"]]], axis=1)
    folds = sorted(meta["group"].unique())
    seeds = sorted(rcs["seed"].unique())
    devices = ["AKGC417L", "Litt3200", "LittC2SE", "Meditron"]
    
    rms0_rows = []
    for f in folds:
        train_mask = (df_feat_meta["group"] != f) & (df_feat_meta["anomaly_label"] == 0)
        mu = df_feat_meta.loc[train_mask, "rms0_mean"].mean()
        sigma = df_feat_meta.loc[train_mask, "rms0_mean"].std()
        
        test_rows = df_feat_meta[df_feat_meta["group"] == f].copy()
        test_rows["score"] = (test_rows["rms0_mean"] - mu).abs() / sigma
        test_rows["detector"] = "rms0_mean_abs_z"
        test_rows["setting"] = "primary"
        test_rows["dataset"] = "icbhi"
        
        for s in seeds:
            t_s = test_rows.copy()
            t_s["seed"] = s
            rms0_rows.append(t_s[["dataset", "detector", "seed", "anomaly_label", "score", "setting", "group", "path", "device"]])
            
    df_rms0 = pd.concat(rms0_rows, ignore_index=True)
    
    # Combine rcs with rms0
    all_scores = pd.concat([rcs, df_rms0], ignore_index=True)
    all_detectors = sorted(all_scores["detector"].unique())
    
    # Store results per detector
    table_rows = []
    
    for det in all_detectors:
        sub_det = all_scores[all_scores["detector"] == det]
        
        fold_pooled_aucs = []
        fold_dev_aucs = {d: [] for d in devices}
        fold_dev_pairs = {d: [] for d in devices}
        fold_pair_weighted_aucs = []
        
        for f in folds:
            # Pooled AUC across seeds for fold f
            s_pooled = []
            for s in seeds:
                df_fs = sub_det[(sub_det["group"] == f) & (sub_det["seed"] == s)]
                s_pooled.append(roc_auc_score(df_fs["anomaly_label"], df_fs["score"]))
            auc_pooled_f = float(np.mean(s_pooled))
            fold_pooled_aucs.append(auc_pooled_f)
            
            # Within-device AUC across seeds for fold f
            dev_aucs_f = {}
            dev_pairs_f = {}
            for d in devices:
                s_dev = []
                for s in seeds:
                    df_fsd = sub_det[(sub_det["group"] == f) & (sub_det["seed"] == s) & (sub_det["device"] == d)]
                    if len(df_fsd["anomaly_label"].unique()) == 2:
                        s_dev.append(roc_auc_score(df_fsd["anomaly_label"], df_fsd["score"]))
                
                # Check pairs
                meta_fd = meta[(meta["group"] == f) & (meta["device"] == d)]
                n_norm = np.sum(meta_fd["anomaly_label"] == 0)
                n_anom = np.sum(meta_fd["anomaly_label"] == 1)
                pairs = n_norm * n_anom
                
                if s_dev:
                    mean_dev_auc = float(np.mean(s_dev))
                    fold_dev_aucs[d].append(mean_dev_auc)
                    fold_dev_pairs[d].append(pairs)
                    dev_aucs_f[d] = mean_dev_auc
                    dev_pairs_f[d] = pairs
                else:
                    fold_dev_aucs[d].append(np.nan)
                    fold_dev_pairs[d].append(0)
                    
            # Per-fold pair-weighted device-balanced AUC
            w_sum = sum(dev_pairs_f.values())
            if w_sum > 0:
                weighted_auc = sum(dev_aucs_f[d] * dev_pairs_f[d] for d in dev_aucs_f) / w_sum
                fold_pair_weighted_aucs.append(weighted_auc)
            else:
                fold_pair_weighted_aucs.append(np.nan)
                
        # Aggregate across folds
        mean_pooled = float(np.mean(fold_pooled_aucs))
        mean_dev = {d: float(np.nanmean(fold_dev_aucs[d])) for d in devices}
        mean_pair_weighted = float(np.mean(fold_pair_weighted_aucs))
        
        # Overall pair-weighted
        total_w = sum(sum(fold_dev_pairs[d]) for d in devices)
        total_w_auc = sum(sum(np.nan_to_num(fold_dev_aucs[d]) * np.array(fold_dev_pairs[d])) for d in devices)
        overall_pair_weighted = float(total_w_auc / total_w)
        
        # Equal-weighted across 4 devices
        mean_equal_weighted = float(np.mean([mean_dev[d] for d in devices]))
        
        table_rows.append({
            "detector": det,
            "pooled": mean_pooled,
            "AKGC417L": mean_dev["AKGC417L"],
            "Litt3200": mean_dev["Litt3200"],
            "LittC2SE": mean_dev["LittC2SE"],
            "Meditron": mean_dev["Meditron"],
            "dev_bal_pair_weighted": mean_pair_weighted,
            "dev_bal_equal_weighted": mean_equal_weighted,
            "dev_bal_overall_pair_weighted": overall_pair_weighted
        })
        
    df_res = pd.DataFrame(table_rows)
    # Sort with PyOD detectors first (descending pooled AUC), then rms0 baseline
    pyod_df = df_res[df_res["detector"] != "rms0_mean_abs_z"].sort_values("pooled", ascending=False).reset_index(drop=True)
    baseline_df = df_res[df_res["detector"] == "rms0_mean_abs_z"].reset_index(drop=True)
    df_sorted = pd.concat([pyod_df, baseline_df], ignore_index=True)
    
    # Save CSV
    out_csv = os.path.join(OUT_DIR, "icbhi_full_roster_device_stratified.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    
    print("\n=== ICBHI FULL ROSTER (25 PyOD DETECTORS + RMS0 BASELINE) ===")
    print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    print("\n=== COUNTS BELOW 0.5 (25 PyOD DETECTORS) ===")
    print(f"Pooled < 0.5: {sum(pyod_df['pooled'] < 0.5)} / 25")
    print(f"AKGC417L < 0.5: {sum(pyod_df['AKGC417L'] < 0.5)} / 25")
    print(f"Litt3200 < 0.5: {sum(pyod_df['Litt3200'] < 0.5)} / 25")
    print(f"LittC2SE < 0.5: {sum(pyod_df['LittC2SE'] < 0.5)} / 25")
    print(f"Meditron < 0.5: {sum(pyod_df['Meditron'] < 0.5)} / 25")
    print(f"Device-Balanced (Pair-Weighted Fold Mean) < 0.5: {sum(pyod_df['dev_bal_pair_weighted'] < 0.5)} / 25")
    print(f"Device-Balanced (Overall Pair-Weighted) < 0.5: {sum(pyod_df['dev_bal_overall_pair_weighted'] < 0.5)} / 25")
    print(f"Device-Balanced (Equal-Weighted Mean) < 0.5: {sum(pyod_df['dev_bal_equal_weighted'] < 0.5)} / 25")

    print("\n=== COUNTS BELOW 0.5 (ALL 26 DETECTORS INCL RMS0) ===")
    print(f"Pooled < 0.5: {sum(df_sorted['pooled'] < 0.5)} / 26")
    print(f"AKGC417L < 0.5: {sum(df_sorted['AKGC417L'] < 0.5)} / 26")
    print(f"Device-Balanced (Pair-Weighted Fold Mean) < 0.5: {sum(df_sorted['dev_bal_pair_weighted'] < 0.5)} / 26")
    print(f"Device-Balanced (Equal-Weighted Mean) < 0.5: {sum(df_sorted['dev_bal_equal_weighted'] < 0.5)} / 26")

if __name__ == "__main__":
    main()
