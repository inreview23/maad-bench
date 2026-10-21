#!/usr/bin/env python3
"""Task 3: PhysioNet Source Confound Analysis.
Computes:
1. Per-source (per-fold) N, anomaly count, and prevalence.
2. Per-fold AUC for top detectors (SOD, ECOD, GMM, COPOD, QMCD).
3. Correlation (Pearson r, Spearman rho, and p-values) across the six folds (n=6)
   between fold AUC and:
   - Source prevalence
   - Source size (N)
   - Source log-size (log N)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SNAP_DIR = "./snap"
OUT_DIR = "results/analysis_r3/F5"

def main():
    meta_path = os.path.join(SNAP_DIR, "release/results/physionet/metadata.parquet")
    runs_path = os.path.join(SNAP_DIR, "release/results/physionet/rot_runs.parquet")
    
    meta = pd.read_parquet(meta_path)
    runs = pd.read_parquet(runs_path)
    runs = runs[runs["setting"] == "primary"].copy()
    
    group_to_source = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"}
    
    # 1. Source stats
    source_stats = []
    for g in sorted(meta["group"].unique()):
        sub_m = meta[meta["group"] == g]
        n_tot = len(sub_m)
        n_anom = (sub_m["anomaly_label"] == 1).sum()
        prev = n_anom / n_tot
        source_stats.append({
            "group": g,
            "source": group_to_source[g],
            "n": n_tot,
            "n_anom": n_anom,
            "prevalence": prev,
            "log_n": np.log(n_tot)
        })
    df_sources = pd.DataFrame(source_stats)
    
    # Top 5 detectors
    top_dets = ["SOD", "ECOD", "GMM", "COPOD", "QMCD"]
    
    # Extract fold AUC per detector (averaged over seeds)
    det_fold_aucs = {}
    for det in top_dets:
        sub_r = runs[runs["detector"] == det]
        mean_by_group = sub_r.groupby("group")["auc_roc"].mean().to_dict()
        det_fold_aucs[det] = mean_by_group
        df_sources[f"auc_{det}"] = df_sources["group"].map(mean_by_group)
        
    out_sources_csv = os.path.join(OUT_DIR, "physionet_source_breakdown.csv")
    df_sources.to_csv(out_sources_csv, index=False)
    print(f"Saved: {out_sources_csv}")
    
    print("\n=== PHYSIONET SOURCE-LEVEL METRICS (6 FOLDS) ===")
    print(df_sources[["source", "group", "n", "n_anom", "prevalence"] + [f"auc_{d}" for d in top_dets]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # 2. Compute correlations across the 6 folds
    corr_rows = []
    for det in top_dets:
        y = df_sources[f"auc_{det}"].values
        prev = df_sources["prevalence"].values
        n = df_sources["n"].values
        log_n = df_sources["log_n"].values
        
        # Pearson
        r_prev, p_prev = pearsonr(prev, y)
        r_n, p_n = pearsonr(n, y)
        r_logn, p_logn = pearsonr(log_n, y)
        
        # Spearman
        rho_prev, sp_prev = spearmanr(prev, y)
        rho_n, sp_n = spearmanr(n, y)
        rho_logn, sp_logn = spearmanr(log_n, y)
        
        corr_rows.append({
            "detector": det,
            "fold_mean_auc": y.mean(),
            "pearson_r_prev": r_prev,
            "p_val_prev": p_prev,
            "spearman_rho_prev": rho_prev,
            "spearman_p_prev": sp_prev,
            "pearson_r_n": r_n,
            "p_val_n": p_n,
            "spearman_rho_n": rho_n,
            "spearman_p_n": sp_n,
            "pearson_r_logn": r_logn,
            "p_val_logn": p_logn
        })
        
    df_corr = pd.DataFrame(corr_rows)
    out_corr_csv = os.path.join(OUT_DIR, "physionet_source_correlations.csv")
    df_corr.to_csv(out_corr_csv, index=False)
    print(f"Saved: {out_corr_csv}")
    
    print("\n=== CORRELATIONS WITH SOURCE PREVALENCE AND SIZE (N=6 FOLDS) ===")
    print(df_corr[["detector", "fold_mean_auc", "pearson_r_prev", "p_val_prev", "pearson_r_n", "p_val_n", "pearson_r_logn", "p_val_logn"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
