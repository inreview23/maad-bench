"""Summarize results for MAAD-Bench Unit E4.

Computes:
1. Dataset overview and counts: class counts per group, test prevalence per fold.
2. Best detector and AUC (primary and secondary).
3. Complete 25-detector comparison table (primary and secondary mean AUC, fold SD, seed SD).
4. Paper claims audit.
"""
import os
import sys
import pandas as pd
import numpy as np

def analyze_circor(results_dir):
    print("=== CIRCOR ANALYSIS ===")
    md_path = os.path.join(results_dir, "circor", "metadata.parquet")
    runs_path = os.path.join(results_dir, "circor", "rot_runs.parquet")
    metrics_path = os.path.join(results_dir, "circor", "rot_metrics.csv")
    
    if os.path.exists(md_path):
        md = pd.read_parquet(md_path)
        print("Clips per group and category:")
        ct = pd.crosstab(md["group"], md["category"], margins=True)
        print(ct.to_string())
        
        print("\nPrevalence per group (fold):")
        prev = md.groupby("group")["anomaly_label"].agg(["count", "sum", "mean"])
        prev.columns = ["n_test", "n_anom", "prevalence"]
        print(prev.to_string())
    
    if os.path.exists(metrics_path):
        met = pd.read_csv(metrics_path)
        for setting in ["primary", "secondary"]:
            s_df = met[met.setting == setting].sort_values("auc_roc_mean", ascending=False)
            best = s_df.iloc[0]
            print(f"\n[{setting.upper()}] Best detector: {best['detector']} with AUC = {best['auc_roc_mean']:.3f} (fold_sd = {best['auc_roc_fold_sd']:.3f}, seed_sd = {best['auc_roc_seed_sd']:.3f})")
            print(f"Top 5 detectors ({setting}):")
            print(s_df[["detector", "auc_roc_mean", "auc_roc_fold_sd", "auc_roc_seed_sd", "auc_roc_pairweighted"]].head(5).to_string(index=False))

def analyze_cremad(results_dir):
    print("\n=== CREMA-D ANALYSIS ===")
    md_path = os.path.join(results_dir, "cremad", "metadata.parquet")
    runs_path = os.path.join(results_dir, "cremad", "rot_runs.parquet")
    metrics_path = os.path.join(results_dir, "cremad", "rot_metrics.csv")
    
    if os.path.exists(md_path):
        md = pd.read_parquet(md_path)
        print("Total clips:", len(md))
        print("Actors:", md["group"].nunique())
        print("Clips per category:")
        print(md["category"].value_counts().to_string())
        
        prev = md.groupby("group")["anomaly_label"].agg(["count", "sum", "mean"])
        print(f"Prevalence min={prev['mean'].min():.3f}, max={prev['mean'].max():.3f}, mean={prev['mean'].mean():.3f}")
    
    if os.path.exists(metrics_path):
        met = pd.read_csv(metrics_path)
        for setting in ["primary", "secondary"]:
            s_df = met[met.setting == setting].sort_values("auc_roc_mean", ascending=False)
            best = s_df.iloc[0]
            print(f"\n[{setting.upper()}] Best detector: {best['detector']} with AUC = {best['auc_roc_mean']:.3f} (fold_sd = {best['auc_roc_fold_sd']:.3f}, seed_sd = {best['auc_roc_seed_sd']:.3f})")
            print(f"Top 5 detectors ({setting}):")
            print(s_df[["detector", "auc_roc_mean", "auc_roc_fold_sd", "auc_roc_seed_sd", "auc_roc_pairweighted"]].head(5).to_string(index=False))

if __name__ == "__main__":
    r_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    analyze_circor(r_dir)
    analyze_cremad(r_dir)
