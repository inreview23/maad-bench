"""Analysis of Hyperparameter Sensitivity Results (UNIT E3).

Computes:
(a) Each tuned detector's fold-mean AUC under each config per sub-dataset.
(b) The 25-detector ordering under defaults vs under each detector's best config (oracle, optimistic)
    vs under reasonable config fixed in advance; Spearman correlation with default ordering; winner changes.
(c) Spearman-Brown inter-fold reliability (exactly as analysis.py computes it) of default vs reasonable config.
(d) Largest single-detector gain from tuning and where it occurs.
"""
import itertools
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATASETS = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
RESULTS_DIR = os.path.join(HERE, "results")
HPSENS_DIR = os.path.join(RESULTS_DIR, "hpsens")
HPSENS_FILE = os.path.join(HPSENS_DIR, "hpsens_runs.parquet")

REASONABLE_CONFIG = {
    "AutoEncoder": "epochs=100",
    "VAE": "epochs=100",
    "GMM": "nc=4_cov=diag",
    "KNN": "k=10",
    "LOF": "k=10",
    "COF": "k=10",
    "IForest": "n_est=500",
    "OCSVM": "gamma=scale_nu=0.1",
    "DIF": "n_est=12",
    "LUNAR": "epochs=400",
}

TUNED_DETECTORS = list(REASONABLE_CONFIG.keys())


def load_runs():
    if not os.path.exists(HPSENS_FILE):
        raise FileNotFoundError(f"Missing {HPSENS_FILE}. Run e3_hpsens.py first.")
    df = pd.read_parquet(HPSENS_FILE)
    return df


def compute_fold_mean_auc(df):
    """Aggregate seeds within (dataset, detector, config, fold) first, then compute fold-level mean and std."""
    per_fold = (
        df.groupby(["dataset", "detector", "config", "is_default", "fold"])["auc_roc"]
        .mean()
        .reset_index()
    )
    summ = (
        per_fold.groupby(["dataset", "detector", "config", "is_default"])["auc_roc"]
        .agg(auc_mean="mean", auc_sd="std", n_folds="count")
        .reset_index()
    )
    return per_fold, summ


def compute_spearman_brown_reliability(per_fold_subset):
    """Compute Spearman-Brown inter-fold reliability exactly as analysis.py computes it."""
    groups = sorted(per_fold_subset["fold"].unique())
    k = len(groups)
    if k < 2:
        return np.nan, np.nan

    per_fold_map = {g: per_fold_subset[per_fold_subset.fold == g].set_index("detector")["auc_roc"] for g in groups}

    pair_rhos = []
    for a, b in itertools.combinations(groups, 2):
        x, y = per_fold_map[a], per_fold_map[b]
        common = x.index.intersection(y.index)
        if len(common) < 3:
            continue
        r = spearmanr(x[common], y[common]).statistic
        if np.isfinite(r):
            pair_rhos.append(r)

    mean_pair = float(np.mean(pair_rhos)) if pair_rhos else np.nan
    denom = 1.0 + (k - 1) * mean_pair
    rel = (k * mean_pair / denom) if np.isfinite(mean_pair) and denom > 0 else np.nan
    return mean_pair, rel


def run_full_analysis():
    df = load_runs()
    per_fold, summ = compute_fold_mean_auc(df)

    # -------------------------------------------------------------
    # (a) Tuned detector fold-mean AUC under each config
    # -------------------------------------------------------------
    tuned_summ = summ[summ["detector"].isin(TUNED_DETECTORS)].copy()
    tuned_summ = tuned_summ.sort_values(["dataset", "detector", "config"])

    # -------------------------------------------------------------
    # (b) 25-detector orderings: Defaults vs Oracle Best vs Reasonable
    # -------------------------------------------------------------
    orderings_record = []
    reliability_record = []
    gains_record = []

    for ds in DATASETS:
        ds_pf = per_fold[per_fold["dataset"] == ds]
        ds_summ = summ[summ["dataset"] == ds]

        # 1. Defaults
        def_summ = ds_summ[ds_summ["is_default"] == True].copy()
        def_map = dict(zip(def_summ["detector"], def_summ["auc_mean"]))
        def_pf = ds_pf[ds_pf["is_default"] == True].copy()

        # 2. Reasonable Config
        # for tuned: REASONABLE_CONFIG[det], for others: is_default==True
        reason_rows = []
        for det in ds_summ["detector"].unique():
            if det in REASONABLE_CONFIG:
                target_cfg = REASONABLE_CONFIG[det]
                match = ds_summ[(ds_summ["detector"] == det) & (ds_summ["config"] == target_cfg)]
                if not match.empty:
                    reason_rows.append(match.iloc[0])
                else:
                    # fallback to default if missing
                    match_def = ds_summ[(ds_summ["detector"] == det) & (ds_summ["is_default"] == True)]
                    reason_rows.append(match_def.iloc[0])
            else:
                match_def = ds_summ[(ds_summ["detector"] == det) & (ds_summ["is_default"] == True)]
                reason_rows.append(match_def.iloc[0])
        reason_summ = pd.DataFrame(reason_rows)
        reason_map = dict(zip(reason_summ["detector"], reason_summ["auc_mean"]))

        reason_pf_rows = []
        for det in ds_summ["detector"].unique():
            if det in REASONABLE_CONFIG:
                target_cfg = REASONABLE_CONFIG[det]
                match = ds_pf[(ds_pf["detector"] == det) & (ds_pf["config"] == target_cfg)]
                if not match.empty:
                    reason_pf_rows.append(match)
                else:
                    reason_pf_rows.append(ds_pf[(ds_pf["detector"] == det) & (ds_pf["is_default"] == True)])
            else:
                reason_pf_rows.append(ds_pf[(ds_pf["detector"] == det) & (ds_pf["is_default"] == True)])
        reason_pf = pd.concat(reason_pf_rows, ignore_index=True)

        # 3. Oracle Best (optimistic)
        oracle_rows = []
        for det in ds_summ["detector"].unique():
            det_options = ds_summ[ds_summ["detector"] == det]
            best_idx = det_options["auc_mean"].idxmax()
            best_row = det_options.loc[best_idx]
            oracle_rows.append(best_row)
        oracle_summ = pd.DataFrame(oracle_rows)
        oracle_map = dict(zip(oracle_summ["detector"], oracle_summ["auc_mean"]))

        # Common detectors across all three
        all_dets = sorted(list(set(def_map.keys()) & set(reason_map.keys()) & set(oracle_map.keys())))

        s_def = pd.Series({d: def_map[d] for d in all_dets})
        s_reason = pd.Series({d: reason_map[d] for d in all_dets})
        s_oracle = pd.Series({d: oracle_map[d] for d in all_dets})

        # Winners
        win_def = s_def.idxmax()
        win_def_val = s_def.max()
        win_reason = s_reason.idxmax()
        win_reason_val = s_reason.max()
        win_oracle = s_oracle.idxmax()
        win_oracle_val = s_oracle.max()

        rho_reason = spearmanr(s_def, s_reason).statistic
        rho_oracle = spearmanr(s_def, s_oracle).statistic

        orderings_record.append({
            "dataset": ds,
            "n_detectors": len(all_dets),
            "default_winner": win_def,
            "default_best_auc": win_def_val,
            "reasonable_winner": win_reason,
            "reasonable_best_auc": win_reason_val,
            "reasonable_winner_changed": win_def != win_reason,
            "spearman_rho_reasonable": rho_reason,
            "oracle_winner": win_oracle,
            "oracle_best_auc": win_oracle_val,
            "oracle_winner_changed": win_def != win_oracle,
            "spearman_rho_oracle": rho_oracle,
        })

        # -------------------------------------------------------------
        # (c) Inter-fold reliability
        # -------------------------------------------------------------
        k_folds = ds_pf["fold"].nunique()
        def_pair_rho, def_rel = compute_spearman_brown_reliability(def_pf)
        reason_pair_rho, reason_rel = compute_spearman_brown_reliability(reason_pf)

        reliability_record.append({
            "dataset": ds,
            "n_folds": k_folds,
            "default_interfold_rho": def_pair_rho,
            "default_reliability": def_rel,
            "reasonable_interfold_rho": reason_pair_rho,
            "reasonable_reliability": reason_rel,
            "delta_reliability": reason_rel - def_rel if (np.isfinite(reason_rel) and np.isfinite(def_rel)) else np.nan,
        })

        # -------------------------------------------------------------
        # (d) Gains from tuning per detector
        # -------------------------------------------------------------
        for det in TUNED_DETECTORS:
            if det in def_map and det in oracle_map:
                def_auc = def_map[det]
                best_auc = oracle_map[det]
                best_cfg = oracle_summ[oracle_summ["detector"] == det]["config"].values[0]
                reason_auc = reason_map[det]
                gains_record.append({
                    "dataset": ds,
                    "detector": det,
                    "default_config": def_summ[def_summ["detector"] == det]["config"].values[0],
                    "default_auc": def_auc,
                    "best_config": best_cfg,
                    "best_auc": best_auc,
                    "gain_oracle": best_auc - def_auc,
                    "reasonable_config": REASONABLE_CONFIG[det],
                    "reasonable_auc": reason_auc,
                    "gain_reasonable": reason_auc - def_auc,
                })

    orderings_df = pd.DataFrame(orderings_record)
    reliability_df = pd.DataFrame(reliability_record)
    gains_df = pd.DataFrame(gains_record)

    # Save summary tables to CSV under results/hpsens/
    tuned_summ.to_csv(os.path.join(HPSENS_DIR, "tuned_configs_auc.csv"), index=False)
    orderings_df.to_csv(os.path.join(HPSENS_DIR, "orderings_comparison.csv"), index=False)
    reliability_df.to_csv(os.path.join(HPSENS_DIR, "reliability_comparison.csv"), index=False)
    gains_df.to_csv(os.path.join(HPSENS_DIR, "tuning_gains.csv"), index=False)

    print("\n" + "=" * 80)
    print("TABLE 1: 25-DETECTOR ORDERING & WINNER COMPARISON")
    print("=" * 80)
    cols1 = ["dataset", "default_winner", "default_best_auc", "reasonable_winner", "reasonable_best_auc", "spearman_rho_reasonable", "oracle_winner", "oracle_best_auc", "spearman_rho_oracle"]
    print(orderings_df[cols1].to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 2: SPEARMAN-BROWN INTER-FOLD RELIABILITY (DEFAULT vs REASONABLE)")
    print("=" * 80)
    print(reliability_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 3: TOP 10 SINGLE-DETECTOR GAINS FROM TUNING (ORACLE)")
    print("=" * 80)
    top_gains = gains_df.sort_values("gain_oracle", ascending=False).head(10)
    cols3 = ["dataset", "detector", "default_auc", "best_config", "best_auc", "gain_oracle"]
    print(top_gains[cols3].to_string(index=False))

    max_gain_row = gains_df.loc[gains_df["gain_oracle"].idxmax()]
    print(f"\nLargest single-detector tuning gain: {max_gain_row['detector']} on {max_gain_row['dataset']}: "
          f"+{max_gain_row['gain_oracle']:.4f} ({max_gain_row['default_auc']:.4f} -> {max_gain_row['best_auc']:.4f}) "
          f"under config '{max_gain_row['best_config']}'")

    return tuned_summ, orderings_df, reliability_df, gains_df


if __name__ == "__main__":
    run_full_analysis()
