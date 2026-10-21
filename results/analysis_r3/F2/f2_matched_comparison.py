#!/usr/bin/env python3
"""f2_matched_comparison.py

Matched-algorithm comparison: clip-level Isolation Forest (IForest) vs
frame-level sliding-window Isolation Forest (TempIF) across all seven
sub-datasets on both primary and secondary settings.

Computes:
- Seed-mean AUC per fold for clip IForest and TempIF
- Per-fold differences (TempIF - clipIF)
- Fold wins (TempIF > clipIF), losses, and ties
- Mean difference and between-fold standard deviation
- Naive paired Student's t-test (df = K - 1)
- Nadeau-Bengio (2003) corrected paired t-test:
    var_corrected = (1/K + n_test/n_train) * s^2, df = K - 1
- Exact two-sided sign-flip permutation test:
    enumerating all 2^K sign flips (using meet-in-the-middle for K=24)
"""

import itertools
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_ROOT = "results"
DATASETS = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
SETTINGS = ["primary", "secondary"]


def exact_sign_flip_p(diff: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation test p-value."""
    k = len(diff)
    obs = float(abs(diff.sum()))
    if k <= 12:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=k)))
        perms = np.abs(np.dot(signs, diff))
        return float((perms >= obs - 1e-12).mean())
    elif k <= 24:
        # Meet-in-the-middle split to avoid 2^24 full matrix allocation
        k1 = k // 2
        k2 = k - k1
        s1 = np.array(list(itertools.product([-1.0, 1.0], repeat=k1)))
        s2 = np.array(list(itertools.product([-1.0, 1.0], repeat=k2)))
        sums1 = np.dot(s1, diff[:k1])
        sums2 = np.dot(s2, diff[k1:])
        sums2_sorted = np.sort(sums2)
        # Count |sums1 + sums2| >= obs
        # sums1 + sums2 >= obs  <=>  sums2 >= obs - sums1
        # sums1 + sums2 <= -obs <=>  sums2 <= -obs - sums1
        c1 = len(sums2_sorted) - np.searchsorted(sums2_sorted, obs - sums1 - 1e-12)
        c2 = np.searchsorted(sums2_sorted, -obs - sums1 + 1e-12, side="right")
        return float(np.sum(c1 + c2) / (2**k))
    else:
        rng = np.random.default_rng(42)
        signs = rng.choice([-1.0, 1.0], size=(500_000, k))
        perms = np.abs(np.dot(signs, diff))
        return float((perms >= obs - 1e-12).mean())


def run_matched_comparison():
    summary_rows = []
    per_fold_dict = {}

    for setting in SETTINGS:
        per_fold_dict[setting] = {}
        for d in DATASETS:
            c = pd.read_parquet(os.path.join(DATA_ROOT, d, "rot_runs.parquet"))
            t = pd.read_parquet(os.path.join(DATA_ROOT, d, "rot_temporal_runs.parquet"))

            c = c[(c.setting == setting) & (c.detector == "IForest")]
            t = t[(t.setting == setting) & (t.detector == "TempIF")]

            # Normalize group IDs
            c = c.assign(g=c.group.astype(str).str.extract(r"(\d+)$")[0].astype(int))
            t = t.assign(g=t.group.astype(str).str.extract(r"(\d+)$")[0].astype(int))

            # Group fold sizes
            g_stats = c.groupby("g")[["n_train", "n_test"]].first()
            k = len(g_stats)
            m_train = float(g_stats.n_train.mean())
            m_test = float(g_stats.n_test.mean())
            var_factor = (1.0 / k) + (m_test / m_train)

            # Fold-level seed-mean AUC
            cf = c.groupby("g").auc_roc.mean()
            tf = t.groupby("g").auc_roc.mean()
            common_groups = sorted(list(set(cf.index) & set(tf.index)))
            assert len(common_groups) == k, f"Group mismatch on {d}: {len(common_groups)} vs {k}"

            cf_vals = cf.loc[common_groups].values
            tf_vals = tf.loc[common_groups].values
            diff = tf_vals - cf_vals

            # Save per-fold table
            per_fold_df = pd.DataFrame({
                "group": common_groups,
                "clip_iforest": cf_vals,
                "temp_iforest": tf_vals,
                "diff": diff
            })
            per_fold_dict[setting][d] = per_fold_df

            # Metrics
            m_clip = float(np.mean(cf_vals))
            m_temp = float(np.mean(tf_vals))
            m_diff = float(np.mean(diff))
            s_diff = float(np.std(diff, ddof=1)) if k > 1 else 0.0

            se_naive = s_diff / np.sqrt(k) if k > 1 else 0.0
            t_naive = m_diff / se_naive if se_naive > 0 else 0.0
            p_naive = float(2 * (1 - stats.t.cdf(abs(t_naive), df=k-1))) if k > 1 else np.nan

            se_nb = s_diff * np.sqrt(var_factor) if k > 1 else 0.0
            t_nb = m_diff / se_nb if se_nb > 0 else 0.0
            p_nb = float(2 * (1 - stats.t.cdf(abs(t_nb), df=k-1))) if k > 1 else np.nan

            p_perm = exact_sign_flip_p(diff)

            wins = int((diff > 0).sum())
            losses = int((diff < 0).sum())
            ties = int((diff == 0).sum())

            summary_rows.append({
                "setting": setting,
                "dataset": d,
                "k_folds": k,
                "n_train_mean": m_train,
                "n_test_mean": m_test,
                "var_factor": var_factor,
                "clip_if_mean": m_clip,
                "temp_if_mean": m_temp,
                "diff_mean": m_diff,
                "diff_sd": s_diff,
                "temp_wins": wins,
                "temp_losses": losses,
                "temp_ties": ties,
                "win_ratio": f"{wins}/{k}",
                "t_naive": t_naive,
                "p_naive": p_naive,
                "t_nb": t_nb,
                "p_nb": p_nb,
                "p_perm": p_perm,
            })

    res_df = pd.DataFrame(summary_rows)
    return res_df, per_fold_dict


if __name__ == "__main__":
    summary_df, per_fold = run_matched_comparison()
    summary_df.to_csv("f2_matched_summary.csv", index=False)
    print("=== SUMMARY: PRIMARY SETTING ===")
    p_sub = summary_df[summary_df.setting == "primary"]
    cols_show = ["dataset", "k_folds", "clip_if_mean", "temp_if_mean", "diff_mean",
                 "diff_sd", "win_ratio", "t_naive", "p_naive", "t_nb", "p_nb", "p_perm"]
    print(p_sub[cols_show].to_string(index=False))

    print("\n=== SUMMARY: SECONDARY SETTING ===")
    s_sub = summary_df[summary_df.setting == "secondary"]
    print(s_sub[cols_show].to_string(index=False))
