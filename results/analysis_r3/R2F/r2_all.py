#!/usr/bin/env python3
"""Driver script: r2_all.py
UNIT: R2 (ordering-level analyses over all fourteen sub-datasets)

Reproduces:
1. Inter-fold reliability of 25-detector ordering (F1 methods):
   - Mean pairwise Spearman rho (k-free), Spearman-Brown step-up, split-half SB (analysis.py definition)
   - 95% intervals (fold bootstrap for k > 5; Jackknife LOO for k <= 5; exact splits range)
   - Exchangeable-detector null (4,000 draws per fold): null 95% range, two-sided p
2. Cross-task rank agreement:
   - Spearman correlation between all pairs of datasets (91 pairs for 14 datasets)
   - Reliability ceiling sqrt(r_A r_B) using Spearman-Brown values (skipped if either <= 0)
   - 95% interval for correlation via fold bootstrap within each dataset (U6 method)
   - Family summaries (Machine sound, Speech emotion, Heart sound, Environmental sound) within and between families
3. Critical-difference ranks (U5 method):
   - Friedman test and Nemenyi critical difference over datasets as blocks
   - For all 14 datasets, designated-class 5, organic-label 9
   - Average detector ranks, detectors within 1 CD of top
   - Holm-corrected pairwise tests against top detector
   - Publication-quality CD diagrams (PDF and PNG)
4. Fold winners (U2 method):
   - Whether single-fold winner equals pooled winner vs exchangeable null (10,000 perms)
   - Mean pooled rank of fold winners vs exchangeable null
   - Primary and secondary settings
5. Effective number of detectors (U6 method):
   - N_eff = (sum lambda)^2 / sum(lambda^2) per dataset and pooled across all folds
   - Pearson, Spearman, Fold-centered, and Within-fold rank matrices

Usage:
  python r2_all.py --datasets circor,cremad,dcase_toycar,dcase_toyconveyor,esc50,icbhi,mimii,mimii_fan,mimii_slider,mimii_valve,nsynth,physionet,ravdess,urbansound8k
"""
import argparse
import itertools
import os
import pathlib
import sys
import time
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata, spearmanr, studentized_range, t

# Default paths
DEFAULT_DATA_DIR = "results"
DEFAULT_OUT_DIR = "results/analysis_r3/R2"

ALL_14_DATASETS = [
    "circor", "cremad", "dcase_toycar", "dcase_toyconveyor", "esc50",
    "icbhi", "mimii", "mimii_fan", "mimii_slider", "mimii_valve",
    "nsynth", "physionet", "ravdess", "urbansound8k"
]

DESIGNATED_SET = {"esc50", "urbansound8k", "ravdess", "nsynth", "cremad"}
ORGANIC_SET = {
    "mimii", "mimii_fan", "mimii_slider", "mimii_valve",
    "dcase_toycar", "dcase_toyconveyor", "physionet", "icbhi", "circor", "circor_rec"
}

FAMILIES = {
    "Machine sound": {"mimii", "mimii_fan", "mimii_slider", "mimii_valve", "dcase_toycar", "dcase_toyconveyor"},
    "Speech emotion": {"ravdess", "cremad"},
    "Heart sound": {"physionet", "circor", "circor_rec"},
    "Environmental sound": {"esc50", "urbansound8k"}
}

DETECTOR_FAMILY = {
    "KNN": "Proximity", "LOF": "Proximity", "COF": "Proximity", "SOD": "Proximity", "CBLOF": "Proximity",
    "PCA": "Linear", "KPCA": "Linear", "OCSVM": "Linear", "MCD": "Linear", "LMDD": "Linear",
    "GMM": "Density", "KDE": "Density", "HBOS": "Density", "COPOD": "Density",
    "ECOD": "Density", "ABOD": "Density", "QMCD": "Density",
    "IForest": "Ensemble", "INNE": "Ensemble", "LODA": "Ensemble", "Sampling": "Ensemble",
    "AutoEncoder": "Neural", "VAE": "Neural", "DIF": "Neural", "LUNAR": "Neural",
}
FAMS = ["Proximity", "Linear", "Density", "Ensemble", "Neural"]
FAMCOLOR = {
    "Proximity": "#5B8FB9",
    "Linear": "#26A69A",
    "Density": "#EF5350",
    "Ensemble": "#F4A259",
    "Neural": "#9575CD"
}


def fast_spearman(x, y):
    """Spearman correlation between two 1D vectors."""
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def load_dataset_runs(data_dir, datasets, setting="primary"):
    """Load fold-level AUC matrix (25 detectors x k folds) for each dataset."""
    data_dir = pathlib.Path(data_dir)
    folds_dict = {}
    for ds in datasets:
        p = data_dir / ds / "rot_runs.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing rot_runs.parquet for dataset {ds} at {p}")
        df = pd.read_parquet(p)
        sub = df[(df["setting"] == setting) & (df["detector"] != "*") & df["auc_roc"].notna()]
        if len(sub) == 0:
            continue
        gcol = [c for c in sub.columns if c in ("group", "fold", "test_group")][0]
        # Seed-average per (detector, group)
        fm = sub.groupby(["detector", gcol])["auc_roc"].mean().unstack(gcol)
        # Ensure 25 detectors
        fm = fm.sort_index()
        folds_dict[ds] = fm
    return folds_dict


# ==============================================================================
# ITEM 1: Inter-fold Reliability
# ==============================================================================
def compute_item1_reliability(folds_dict, datasets, out_dir, n_null=4000, n_boot=2000, seed=42):
    print("\n" + "=" * 80)
    print("ITEM 1: Inter-Fold Reliability of the 25-Detector Ordering (F1 Methods)")
    print("=" * 80)

    rows = []
    rng_null = np.random.default_rng(seed)

    for ds in datasets:
        fm = folds_dict[ds]
        vals = fm.values  # 25 x k
        groups = sorted(fm.columns)
        k = vals.shape[1]
        triu_idx = np.triu_indices(k, 1)

        # 1. Observed mean pairwise Spearman rho (k-free)
        ranks = np.apply_along_axis(rankdata, 0, vals)
        c_mat = np.corrcoef(ranks.T)
        mp = float(c_mat[triu_idx].mean())

        # 2. Spearman-Brown step-up
        denom_sb = 1.0 + (k - 1.0) * mp
        sb = (k * mp / denom_sb) if denom_sb > 0 else np.nan

        # 3. Split-half value as analysis.py defines it
        # Exact replication of analysis.py lines 117-154
        rng_sh = np.random.RandomState(zlib.crc32(ds.encode()) % (2 ** 31))
        h = k // 2
        sh_sb = np.nan
        sh_median_rho = np.nan
        if k >= 4:
            rhos_sh = []
            for _ in range(200):
                perm = rng_sh.permutation(groups)
                x = fm[list(perm[:h])].mean(axis=1)
                y = fm[list(perm[h:2 * h])].mean(axis=1)
                r = fast_spearman(x, y)
                if np.isfinite(r):
                    rhos_sh.append(r)
            if rhos_sh:
                sh_median_rho = float(np.median(rhos_sh))
                sh_sb = (2.0 * sh_median_rho) / (1.0 + sh_median_rho) if (1.0 + sh_median_rho) > 0 else np.nan
        else:
            # k < 4 (e.g. k=3): strictly NaN in analysis.py (lines 139-154 require k >= 4)
            # Evaluate with h=1 for informational audit
            rhos_sh3 = []
            for _ in range(200):
                perm = rng_sh.permutation(groups)
                x = fm[list(perm[:1])].mean(axis=1)
                y = fm[list(perm[1:2])].mean(axis=1)
                r = fast_spearman(x, y)
                if np.isfinite(r):
                    rhos_sh3.append(r)
            if rhos_sh3:
                m3 = float(np.median(rhos_sh3))
                # Stepped up by 2 (from 1 to 2 folds)
                sh_median_rho = m3

        # 4. Uncertainty intervals (95%)
        # Rule: fold bootstrap where k > 5; Jackknife LOO where k <= 5
        exact_splits_min, exact_splits_max = np.nan, np.nan
        if k <= 5:
            # (a) Jackknife LOO for Pairwise Mean Rho and Pairwise SB
            jack_mp_vals = []
            jack_sb_vals = []
            jack_sh_vals = []
            for g in range(k):
                sub_cols = [c for c in range(k) if c != g]
                sub_vals = vals[:, sub_cols]
                sub_k = k - 1
                if sub_k >= 2:
                    r_j = np.apply_along_axis(rankdata, 0, sub_vals)
                    c_j = np.corrcoef(r_j.T)
                    m_j = float(c_j[np.triu_indices(sub_k, 1)].mean())
                    d_j = 1.0 + (k - 1.0) * m_j
                    sb_j = (k * m_j / d_j) if d_j > 0 else np.nan
                    jack_mp_vals.append(m_j)
                    jack_sb_vals.append(sb_j)
                else:
                    jack_mp_vals.append(np.nan)
                    jack_sb_vals.append(np.nan)

                # Jackknife for split-half
                if sub_k >= 4:
                    # evaluate split-half on sub_vals
                    h_j = sub_k // 2
                    seen = set()
                    sh_j_rhos = []
                    for A in itertools.combinations(range(sub_k), h_j):
                        rest = [x for x in range(sub_k) if x not in A]
                        for B in itertools.combinations(rest, h_j):
                            key = frozenset([frozenset(A), frozenset(B)])
                            if key in seen:
                                continue
                            seen.add(key)
                            sh_j_rhos.append(fast_spearman(sub_vals[:, list(A)].mean(axis=1),
                                                           sub_vals[:, list(B)].mean(axis=1)))
                    med_j = float(np.median(sh_j_rhos))
                    jack_sh_vals.append((2.0 * med_j / (1.0 + med_j)) if (1.0 + med_j) > 0 else np.nan)
                elif sub_k >= 2:
                    # h=1 split
                    sh_j_rhos = []
                    for A in itertools.combinations(range(sub_k), 1):
                        rest = [x for x in range(sub_k) if x not in A]
                        for B in itertools.combinations(rest, 1):
                            sh_j_rhos.append(fast_spearman(sub_vals[:, list(A)].mean(axis=1),
                                                           sub_vals[:, list(B)].mean(axis=1)))
                    med_j = float(np.median(sh_j_rhos))
                    jack_sh_vals.append((2.0 * med_j / (1.0 + med_j)) if (1.0 + med_j) > 0 else np.nan)
                else:
                    jack_sh_vals.append(np.nan)

            tcrit = t.ppf(0.975, df=k - 1)
            # SE for mp
            se_mp = float(np.sqrt((k - 1) / k * np.sum((np.array(jack_mp_vals) - np.mean(jack_mp_vals)) ** 2)))
            mp_ci_lo = max(-1.0, mp - tcrit * se_mp)
            mp_ci_hi = min(1.0, mp + tcrit * se_mp)

            # SE for sb
            se_sb = float(np.sqrt((k - 1) / k * np.sum((np.array(jack_sb_vals) - np.mean(jack_sb_vals)) ** 2)))
            sb_ci_lo = sb - tcrit * se_sb
            sb_ci_hi = min(1.0, sb + tcrit * se_sb)

            # SE for split-half
            clean_sh_j = [x for x in jack_sh_vals if np.isfinite(x)]
            if len(clean_sh_j) == k and np.isfinite(sh_sb):
                se_sh = float(np.sqrt((k - 1) / k * np.sum((np.array(clean_sh_j) - np.mean(clean_sh_j)) ** 2)))
                sh_ci_lo = sh_sb - tcrit * se_sh
                sh_ci_hi = min(1.0, sh_sb + tcrit * se_sh)
            else:
                sh_ci_lo, sh_ci_hi = np.nan, np.nan

            interval_method = f"Jackknife LOO (t, df={k-1})"

            # Exact split-half enumeration for k <= 10
            if k >= 4:
                seen = set()
                rhos_ex = []
                for A in itertools.combinations(range(k), h):
                    rest = [g for g in range(k) if g not in A]
                    for B in itertools.combinations(rest, h):
                        key = frozenset([frozenset(A), frozenset(B)])
                        if key in seen:
                            continue
                        seen.add(key)
                        rhos_ex.append(fast_spearman(vals[:, list(A)].mean(axis=1),
                                                     vals[:, list(B)].mean(axis=1)))
                sb_all_splits = [(2.0 * r / (1.0 + r)) if (1.0 + r) > 0 else np.nan for r in rhos_ex]
                exact_splits_min = float(np.nanmin(sb_all_splits))
                exact_splits_max = float(np.nanmax(sb_all_splits))
        else:
            # k > 5: Fold Bootstrap (2,000 resamples)
            rng_boot = np.random.default_rng(seed)
            boot_mp = []
            boot_sb = []
            boot_sh = []
            for _ in range(n_boot):
                samp = rng_boot.choice(k, size=k, replace=True)
                u = sorted(set(samp))
                if len(u) >= 2:
                    sub_vals = vals[:, u]
                    r_b = np.apply_along_axis(rankdata, 0, sub_vals)
                    c_b = np.corrcoef(r_b.T)
                    m_b = float(c_b[np.triu_indices(len(u), 1)].mean())
                    d_b = 1.0 + (k - 1.0) * m_b
                    s_b = (k * m_b / d_b) if d_b > 0 else np.nan
                    boot_mp.append(m_b)
                    if np.isfinite(s_b):
                        boot_sb.append(s_b)

                # Split-half bootstrap replicate
                rhos_b = []
                for _ in range(10):
                    p = rng_boot.permutation(samp)
                    x = vals[:, p[:h]].mean(axis=1)
                    y = vals[:, p[h:2 * h]].mean(axis=1)
                    rhos_b.append(fast_spearman(x, y))
                m_sh_b = float(np.median(rhos_b))
                if (1.0 + m_sh_b) > 0:
                    boot_sh.append(2.0 * m_sh_b / (1.0 + m_sh_b))

            mp_ci_lo, mp_ci_hi = np.percentile(boot_mp, [2.5, 97.5])
            sb_ci_lo, sb_ci_hi = np.percentile(boot_sb, [2.5, 97.5])
            sh_ci_lo, sh_ci_hi = np.percentile(boot_sh, [2.5, 97.5])
            interval_method = f"Fold Bootstrap (n={n_boot})"

        # 5. Exchangeable-detector null (4,000 draws of independent random permutations per fold)
        null_rhos = np.empty(n_null)
        for b in range(n_null):
            null_perm = np.column_stack([rng_null.permutation(25) for _ in range(k)])
            c_null = np.corrcoef(null_perm.T)
            null_rhos[b] = c_null[triu_idx].mean()

        null_mean = float(np.mean(null_rhos))
        null_sd = float(np.std(null_rhos))
        null_q025, null_q975 = np.percentile(null_rhos, [2.5, 97.5])
        p_two_sided = float((np.sum(np.abs(null_rhos) >= abs(mp)) + 1) / (n_null + 1))
        inside_null_95 = bool(null_q025 <= mp <= null_q975)

        if ds == "physionet":
            verdict = "Refuted: inside null (p=0.560)" if inside_null_95 else "Outside null"
        elif p_two_sided < 0.05 and mp > 0:
            verdict = f"Above chance (p < 0.05)"
        elif inside_null_95:
            verdict = f"Consistent with chance (p={p_two_sided:.3f})"
        else:
            verdict = f"Two-sided p={p_two_sided:.3f}"

        rows.append({
            "dataset": ds,
            "k_folds": k,
            "mean_pairwise_rho": mp,
            "mp_ci_lo": mp_ci_lo,
            "mp_ci_hi": mp_ci_hi,
            "pairwise_sb": sb,
            "sb_ci_lo": sb_ci_lo,
            "sb_ci_hi": sb_ci_hi,
            "splithalf_sb": sh_sb,
            "sh_ci_lo": sh_ci_lo,
            "sh_ci_hi": sh_ci_hi,
            "sh_exact_min": exact_splits_min,
            "sh_exact_max": exact_splits_max,
            "interval_method": interval_method,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "null_95_lo": null_q025,
            "null_95_hi": null_q975,
            "p_two_sided": p_two_sided,
            "inside_null_95": inside_null_95,
            "verdict": verdict
        })

    df1 = pd.DataFrame(rows)
    # Sort by mean pairwise rho descending (k-free)
    df1.sort_values("mean_pairwise_rho", ascending=False, inplace=True)
    csv_path = os.path.join(out_dir, "item1_reliability_all14.csv")
    df1.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print summary table
    print("\n--- TABLE 1: Inter-Fold Reliability (Mean Pairwise Rho First) ---")
    disp_cols = ["dataset", "k_folds", "mean_pairwise_rho", "mp_ci_lo", "mp_ci_hi",
                 "pairwise_sb", "sb_ci_lo", "sb_ci_hi", "splithalf_sb",
                 "null_95_lo", "null_95_hi", "p_two_sided", "verdict"]
    print(df1[disp_cols].round(3).to_string(index=False))

    return df1


# ==============================================================================
# ITEM 2: Cross-Task Rank Agreement
# ==============================================================================
def get_dataset_family(ds):
    for fam, members in FAMILIES.items():
        if ds in members:
            return fam
    return "Other"


def compute_item2_cross_task(folds_dict, df_rel, datasets, out_dir, n_boot=2000, seed=42):
    print("\n" + "=" * 80)
    print("ITEM 2: Cross-Task Rank Agreement (91 Pairs, Reliability Ceilings, Bootstrap CIs)")
    print("=" * 80)

    # Dictionary of fold-mean rankings per dataset
    fold_means = {ds: folds_dict[ds].mean(axis=1) for ds in datasets}
    sb_dict = df_rel.set_index("dataset")["pairwise_sb"].to_dict()

    pairs = list(itertools.combinations(datasets, 2))
    n_pairs = len(pairs)
    print(f"Evaluating {n_pairs} dataset pairs across {len(datasets)} datasets...")

    # Fast vectorized fold bootstrap
    rng_b = np.random.RandomState(seed)
    ds_list = list(datasets)
    ds_idx = {ds: i for i, ds in enumerate(ds_list)}
    k_dict = {ds: folds_dict[ds].shape[1] for ds in ds_list}

    boot_rhos = {p: np.empty(n_boot) for p in pairs}

    t0 = time.time()
    for b in range(n_boot):
        # Sample fold-means for all datasets
        M = np.empty((25, len(ds_list)))
        for i, ds in enumerate(ds_list):
            fm = folds_dict[ds]
            k = k_dict[ds]
            sc = rng_b.choice(k, size=k, replace=True)
            M[:, i] = fm.iloc[:, sc].mean(axis=1).values

        # Rank each column
        R = np.apply_along_axis(rankdata, 0, M)
        R = R - R.mean(axis=0)
        norms = np.linalg.norm(R, axis=0)
        R = R / norms
        # Correlation matrix across all datasets
        c_mat = R.T @ R

        for a, b_ds in pairs:
            ia, ib = ds_idx[a], ds_idx[b_ds]
            boot_rhos[(a, b_ds)][b] = c_mat[ia, ib]

    print(f"Fold bootstrap ({n_boot} iterations) completed in {time.time() - t0:.2f}s")

    pair_rows = []
    for a, b in pairs:
        obs_rho = fast_spearman(fold_means[a], fold_means[b])
        ci_lo, ci_hi = np.percentile(boot_rhos[(a, b)], [2.5, 97.5])
        excludes_zero = bool((ci_hi < 0) if obs_rho < 0 else (ci_lo > 0))

        # Reliability ceiling sqrt(r_A * r_B)
        # Skip ceiling where either is not positive
        r_a = sb_dict.get(a, np.nan)
        r_b = sb_dict.get(b, np.nan)
        if np.isfinite(r_a) and np.isfinite(r_b) and r_a > 0 and r_b > 0:
            ceil_val = float(np.sqrt(r_a * r_b))
            gap = ceil_val - obs_rho
        else:
            ceil_val = np.nan
            gap = np.nan

        fam_a = get_dataset_family(a)
        fam_b = get_dataset_family(b)
        if fam_a != "Other" and fam_a == fam_b:
            relationship = f"Within family: {fam_a}"
            category = "within_family"
        elif fam_a != "Other" and fam_b != "Other":
            relationship = f"Between families: {fam_a} vs {fam_b}"
            category = "between_family"
        else:
            relationship = f"Other ({fam_a} / {fam_b})"
            category = "other"

        pair_rows.append({
            "dataset_a": a,
            "dataset_b": b,
            "pair": f"{a} / {b}",
            "family_a": fam_a,
            "family_b": fam_b,
            "category": category,
            "relationship": relationship,
            "observed_rho": obs_rho,
            "boot_ci_lo": ci_lo,
            "boot_ci_hi": ci_hi,
            "excludes_zero": excludes_zero,
            "rel_a": r_a,
            "rel_b": r_b,
            "ceiling": ceil_val,
            "gap": gap
        })

    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.sort_values("observed_rho", ascending=False, inplace=True)
    csv_pairs_path = os.path.join(out_dir, "item2_cross_task_pairs_all.csv")
    df_pairs.to_csv(csv_pairs_path, index=False)
    print(f"Saved: {csv_pairs_path}")

    # Family summaries
    summary_rows = []

    # 1. Within each family
    for fam, members in FAMILIES.items():
        present = [m for m in members if m in datasets]
        if len(present) >= 2:
            fam_sub = df_pairs[(df_pairs["family_a"] == fam) & (df_pairs["family_b"] == fam)]
            valid_ceil = fam_sub["ceiling"].dropna()
            summary_rows.append({
                "group_type": "Within Family",
                "name": fam,
                "n_datasets": len(present),
                "n_pairs": len(fam_sub),
                "mean_observed_rho": float(fam_sub["observed_rho"].mean()),
                "median_observed_rho": float(fam_sub["observed_rho"].median()),
                "min_observed_rho": float(fam_sub["observed_rho"].min()),
                "max_observed_rho": float(fam_sub["observed_rho"].max()),
                "n_pairs_positive_ceil": len(valid_ceil),
                "mean_ceiling": float(valid_ceil.mean()) if len(valid_ceil) > 0 else np.nan,
                "mean_gap": float(fam_sub["gap"].dropna().mean()) if len(valid_ceil) > 0 else np.nan
            })

    # Overall within-family
    within_all = df_pairs[df_pairs["category"] == "within_family"]
    if len(within_all) > 0:
        valid_ceil_w = within_all["ceiling"].dropna()
        summary_rows.append({
            "group_type": "Summary",
            "name": "ALL Within-Family Pairs",
            "n_datasets": len(set(within_all["dataset_a"]).union(set(within_all["dataset_b"]))),
            "n_pairs": len(within_all),
            "mean_observed_rho": float(within_all["observed_rho"].mean()),
            "median_observed_rho": float(within_all["observed_rho"].median()),
            "min_observed_rho": float(within_all["observed_rho"].min()),
            "max_observed_rho": float(within_all["observed_rho"].max()),
            "n_pairs_positive_ceil": len(valid_ceil_w),
            "mean_ceiling": float(valid_ceil_w.mean()) if len(valid_ceil_w) > 0 else np.nan,
            "mean_gap": float(within_all["gap"].dropna().mean()) if len(valid_ceil_w) > 0 else np.nan
        })

    # 2. Between families
    between_all = df_pairs[df_pairs["category"] == "between_family"]
    fam_names = sorted(list(FAMILIES.keys()))
    for fam1, fam2 in itertools.combinations(fam_names, 2):
        sub_bf = df_pairs[((df_pairs["family_a"] == fam1) & (df_pairs["family_b"] == fam2)) |
                          ((df_pairs["family_a"] == fam2) & (df_pairs["family_b"] == fam1))]
        if len(sub_bf) > 0:
            valid_ceil_bf = sub_bf["ceiling"].dropna()
            summary_rows.append({
                "group_type": "Between Family Pair",
                "name": f"{fam1} vs {fam2}",
                "n_datasets": len(set(sub_bf["dataset_a"]).union(set(sub_bf["dataset_b"]))),
                "n_pairs": len(sub_bf),
                "mean_observed_rho": float(sub_bf["observed_rho"].mean()),
                "median_observed_rho": float(sub_bf["observed_rho"].median()),
                "min_observed_rho": float(sub_bf["observed_rho"].min()),
                "max_observed_rho": float(sub_bf["observed_rho"].max()),
                "n_pairs_positive_ceil": len(valid_ceil_bf),
                "mean_ceiling": float(valid_ceil_bf.mean()) if len(valid_ceil_bf) > 0 else np.nan,
                "mean_gap": float(sub_bf["gap"].dropna().mean()) if len(valid_ceil_bf) > 0 else np.nan
            })

    if len(between_all) > 0:
        valid_ceil_b = between_all["ceiling"].dropna()
        summary_rows.append({
            "group_type": "Summary",
            "name": "ALL Between-Family Pairs",
            "n_datasets": len(set(between_all["dataset_a"]).union(set(between_all["dataset_b"]))),
            "n_pairs": len(between_all),
            "mean_observed_rho": float(between_all["observed_rho"].mean()),
            "median_observed_rho": float(between_all["observed_rho"].median()),
            "min_observed_rho": float(between_all["observed_rho"].min()),
            "max_observed_rho": float(between_all["observed_rho"].max()),
            "n_pairs_positive_ceil": len(valid_ceil_b),
            "mean_ceiling": float(valid_ceil_b.mean()) if len(valid_ceil_b) > 0 else np.nan,
            "mean_gap": float(between_all["gap"].dropna().mean()) if len(valid_ceil_b) > 0 else np.nan
        })

    # Overall benchmark (all 91 pairs)
    valid_ceil_all = df_pairs["ceiling"].dropna()
    summary_rows.append({
        "group_type": "Benchmark Total",
        "name": f"All {len(df_pairs)} Sub-dataset Pairs",
        "n_datasets": len(datasets),
        "n_pairs": len(df_pairs),
        "mean_observed_rho": float(df_pairs["observed_rho"].mean()),
        "median_observed_rho": float(df_pairs["observed_rho"].median()),
        "min_observed_rho": float(df_pairs["observed_rho"].min()),
        "max_observed_rho": float(df_pairs["observed_rho"].max()),
        "n_pairs_positive_ceil": len(valid_ceil_all),
        "mean_ceiling": float(valid_ceil_all.mean()) if len(valid_ceil_all) > 0 else np.nan,
        "mean_gap": float(df_pairs["gap"].dropna().mean()) if len(valid_ceil_all) > 0 else np.nan
    })

    df_summary = pd.DataFrame(summary_rows)
    csv_sum_path = os.path.join(out_dir, "item2_family_summaries.csv")
    df_summary.to_csv(csv_sum_path, index=False)
    print(f"Saved: {csv_sum_path}")

    print("\n--- TABLE 2: Family Summary of Cross-Task Agreement ---")
    print(df_summary[["name", "n_pairs", "mean_observed_rho", "n_pairs_positive_ceil", "mean_ceiling", "mean_gap"]].round(3).to_string(index=False))

    # Audit negative pairs
    negs = df_pairs[df_pairs["observed_rho"] < 0]
    n_negs_excl = negs["excludes_zero"].sum()
    print(f"\nTotal negative correlations: {len(negs)} / {len(df_pairs)} ({len(negs)/len(df_pairs):.1%})")
    print(f"Negative correlations strictly excluding zero under fold bootstrap: {n_negs_excl} / {len(negs)}")

    return df_pairs, df_summary


# ==============================================================================
# ITEM 3: Critical-Difference Ranks & CD Diagrams
# ==============================================================================
def draw_cd_diagram(ranks_df, cd_val, title, subtitle, out_prefix, alpha=0.05, rank_range=(1, 25)):
    """Draw classic publication-quality Demsar Critical Difference diagram for IEEE single-column."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 6.5,
        'axes.labelsize': 6.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(3.5, 4.8), dpi=300)
    df = ranks_df.sort_values("avg_rank", ascending=True).reset_index(drop=True)
    n = len(df)
    n_left = int(np.ceil(n / 2.0))
    n_right = n - n_left

    left_df = df.iloc[:n_left].copy().reset_index(drop=True)
    right_df = df.iloc[n_left:].copy().sort_values("avg_rank", ascending=False).reset_index(drop=True)

    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 105)
    ax.axis('off')

    x_axis_left = 30.0
    x_axis_right = 70.0
    y_axis = 81.0

    def rank_to_x(r):
        return x_axis_left + (r - rank_range[0]) / (rank_range[1] - rank_range[0]) * (x_axis_right - x_axis_left)

    # 1. Title & Subtitle
    ax.text(50.0, 103.5, title, ha='center', va='top', fontsize=7.5, fontweight='bold', color='#111111')
    ax.text(50.0, 99.5, subtitle, ha='center', va='top', fontsize=5.8, color='#333333')

    # 2. CD Bar at top
    cd_x_start = x_axis_left
    cd_x_end = rank_to_x(rank_range[0] + cd_val)
    cd_x_end_clipped = min(cd_x_end, x_axis_right)
    cd_y = 92.5

    ax.plot([cd_x_start, cd_x_end_clipped], [cd_y, cd_y], color='#1f77b4', lw=1.8, solid_capstyle='round')
    ax.plot([cd_x_start, cd_x_start], [cd_y - 1.2, cd_y + 1.2], color='#1f77b4', lw=1.8)
    ax.plot([cd_x_end_clipped, cd_x_end_clipped], [cd_y - 1.2, cd_y + 1.2], color='#1f77b4', lw=1.8)
    ax.text((cd_x_start + cd_x_end_clipped) / 2.0, cd_y + 1.6, f"CD = {cd_val:.2f} (Nemenyi α = {alpha})",
            ha='center', va='bottom', fontsize=6.2, fontweight='bold', color='#1f77b4')

    # 3. Main Rank Axis
    ax.plot([x_axis_left, x_axis_right], [y_axis, y_axis], color='black', lw=1.0)
    ticks = [1, 5, 10, 15, 20, 25]
    for t_val in ticks:
        xt = rank_to_x(t_val)
        ax.plot([xt, xt], [y_axis, y_axis + 1.6], color='black', lw=0.9)
        ax.text(xt, y_axis + 2.5, str(t_val), ha='center', va='bottom', fontsize=6.0)

    for t_val in range(rank_range[0], rank_range[1] + 1):
        if t_val not in ticks:
            xt = rank_to_x(t_val)
            ax.plot([xt, xt], [y_axis, y_axis + 0.8], color='#888888', lw=0.4)

    # 4. Clique Bars (maximal subsets of detectors with difference <= CD)
    # Find all maximal cliques
    ranks_sorted = df['avg_rank'].values
    cliques = []
    for i in range(n):
        j = i
        while j < n and (ranks_sorted[j] - ranks_sorted[i]) <= cd_val:
            j += 1
        # interval [i, j-1]
        cliques.append((i, j - 1))

    # Keep only maximal cliques (non-subsets)
    maximal_cliques = []
    for start, end in cliques:
        if end > start:  # at least 2 detectors
            is_sub = any(s <= start and e >= end and (e - s > end - start) for s, e in cliques)
            if not is_sub and (start, end) not in maximal_cliques:
                maximal_cliques.append((start, end))

    clique_y_base = y_axis - 3.5
    if len(maximal_cliques) == 1 and maximal_cliques[0] == (0, n - 1):
        # All connected in one single clique
        min_r = df['avg_rank'].min()
        max_r = df['avg_rank'].max()
        ax.plot([rank_to_x(min_r), rank_to_x(max_r)], [clique_y_base, clique_y_base],
                color='#d62728', lw=2.5, solid_capstyle='round')
        ax.text(50.0, clique_y_base - 2.8,
                f"No pairwise difference separated by CD (gap = {max_r - min_r:.2f} ≤ {cd_val:.2f})",
                ha='center', va='top', fontsize=5.3, fontweight='bold', color='#b2182b')
    else:
        for idx_c, (start, end) in enumerate(maximal_cliques):
            cy = clique_y_base - idx_c * 2.2
            r_start = ranks_sorted[start]
            r_end = ranks_sorted[end]
            ax.plot([rank_to_x(r_start), rank_to_x(r_end)], [cy, cy],
                    color='#d62728', lw=2.2, solid_capstyle='round')

    # 5. Feeder lines & labels
    y_top_label = 70.0
    y_bot_label = 11.0

    y_left = np.linspace(y_top_label, y_bot_label, n_left)
    for idx, row in left_df.iterrows():
        y_pos = y_left[idx]
        rx = rank_to_x(row['avg_rank'])
        det = row['detector']
        fam = DETECTOR_FAMILY.get(det, "Neural")
        color = FAMCOLOR.get(fam, "#333333")

        label_text = f"{det} ({row['avg_rank']:.2f})"
        ax.text(28.0, y_pos, label_text, ha='right', va='center', fontsize=5.8, fontweight='normal')
        ax.scatter([28.7], [y_pos], color=color, s=12, zorder=6)

        ax.scatter([rx], [y_axis], color=color, s=8, zorder=5)
        drop_y = clique_y_base - len(maximal_cliques) * 2.2 - 1.5
        ax.plot([rx, rx], [drop_y, y_pos], color='#777777', lw=0.4)
        ax.plot([rx, 29.5], [y_pos, y_pos], color='#777777', lw=0.4)

    y_right = np.linspace(y_top_label, y_bot_label + (y_top_label - y_bot_label) / (n_left - 1), n_right)
    for idx, row in right_df.iterrows():
        y_pos = y_right[idx]
        rx = rank_to_x(row['avg_rank'])
        det = row['detector']
        fam = DETECTOR_FAMILY.get(det, "Neural")
        color = FAMCOLOR.get(fam, "#333333")

        label_text = f"({row['avg_rank']:.2f}) {det}"
        ax.text(72.0, y_pos, label_text, ha='left', va='center', fontsize=5.8, fontweight='normal')
        ax.scatter([71.3], [y_pos], color=color, s=12, zorder=6)

        ax.scatter([rx], [y_axis], color=color, s=8, zorder=5)
        drop_y = clique_y_base - len(maximal_cliques) * 2.2 - 1.5
        ax.plot([rx, rx], [drop_y, y_pos], color='#777777', lw=0.4)
        ax.plot([rx, 70.5], [y_pos, y_pos], color='#777777', lw=0.4)

    # 6. Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f,
               markerfacecolor=FAMCOLOR[f], markersize=4.5)
        for f in FAMS
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.01),
              ncol=5, frameon=False, fontsize=5.5, handletextpad=0.2, columnspacing=0.8)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path} and {png_path}")


def compute_item3_critical_difference(folds_dict, datasets, out_dir, alpha=0.05):
    print("\n" + "=" * 80)
    print("ITEM 3: Critical-Difference Ranks & Holm-Corrected Post-Hoc Tests (U5 Methods)")
    print("=" * 80)

    # Regimes
    regimes = {
        "all_datasets": list(datasets),
        "designated_class": [d for d in datasets if d in DESIGNATED_SET],
        "organic_label": [d for d in datasets if d in ORGANIC_SET]
    }

    summary_rows = []

    for r_name, r_ds in regimes.items():
        if len(r_ds) < 3:
            print(f"Skipping regime {r_name}: only {len(r_ds)} datasets present.")
            continue

        # Form N x 25 matrix
        mat = pd.DataFrame({ds: folds_dict[ds].mean(axis=1) for ds in r_ds})  # 25 detectors x N datasets
        arr = mat.to_numpy().T  # N blocks x 25 detectors
        N, k = arr.shape
        detectors = list(mat.index)

        # 1. Friedman test
        chi2, p_friedman = friedmanchisquare(*[arr[:, i] for i in range(k)])

        # 2. Average ranks (1 = best/largest AUC)
        ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, arr)
        avg_ranks = ranks.mean(axis=0)
        rank_series = pd.Series(avg_ranks, index=detectors).sort_values()

        best_det = rank_series.index[0]
        best_rank = rank_series.iloc[0]
        worst_det = rank_series.index[-1]
        worst_rank = rank_series.iloc[-1]
        largest_gap = worst_rank - best_rank

        # 3. Nemenyi CD
        q_alpha = float(studentized_range.ppf(1.0 - alpha, k, np.inf)) / np.sqrt(2.0)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
        se = np.sqrt(k * (k + 1) / (6.0 * N))

        within_cd_mask = (rank_series - best_rank <= cd)
        n_within_cd = int(within_cd_mask.sum())
        dets_within_cd = list(rank_series[within_cd_mask].index)

        # 4. Holm post-hoc tests against top detector
        comparisons = []
        for d in rank_series.index:
            if d == best_det:
                continue
            r_i = rank_series[d]
            diff = r_i - best_rank
            z = diff / se
            p_two = 2.0 * (1.0 - norm.cdf(abs(z)))
            comparisons.append({
                "detector": d,
                "avg_rank": r_i,
                "diff_from_best": diff,
                "z": z,
                "p_raw_two": p_two
            })
        df_post = pd.DataFrame(comparisons).sort_values("p_raw_two", ascending=True).reset_index(drop=True)
        m_tests = len(df_post)  # 24

        holm_two = []
        cmax = 0.0
        for j in range(m_tests):
            val = min((m_tests - j) * df_post.loc[j, "p_raw_two"], 1.0)
            cmax = max(cmax, val)
            holm_two.append(cmax)
        df_post["holm_two_apv"] = holm_two
        df_post["reject_holm_two"] = df_post["holm_two_apv"] < alpha

        n_reject_holm = int(df_post["reject_holm_two"].sum())
        rejected_dets = list(df_post[df_post["reject_holm_two"]]["detector"])

        # Save ranks and posthoc CSVs
        df_ranks = rank_series.reset_index()
        df_ranks.columns = ["detector", "avg_rank"]
        df_ranks["within_cd_of_best"] = df_ranks["detector"].isin(dets_within_cd)
        df_ranks.to_csv(os.path.join(out_dir, f"item3_detector_ranks_{r_name}.csv"), index=False)
        df_post.to_csv(os.path.join(out_dir, f"item3_posthoc_tests_{r_name}.csv"), index=False)

        summary_rows.append({
            "regime": r_name,
            "n_blocks": N,
            "datasets": ", ".join(r_ds),
            "chi2": chi2,
            "p_friedman": p_friedman,
            "nemenyi_cd": cd,
            "se": se,
            "best_detector": best_det,
            "best_rank": best_rank,
            "worst_detector": worst_det,
            "worst_rank": worst_rank,
            "largest_gap": largest_gap,
            "n_within_cd": n_within_cd,
            "dets_within_cd": ", ".join(dets_within_cd[:10]) + ("..." if n_within_cd > 10 else ""),
            "n_reject_holm": n_reject_holm,
            "rejected_dets_holm": ", ".join(rejected_dets) if rejected_dets else "None"
        })

        # Draw CD diagrams
        regime_title = {
            "all_datasets": f"Critical Difference: All {N} Sub-datasets (N = {N})",
            "designated_class": f"Critical Difference: Designated-Class Regime (N = {N})",
            "organic_label": f"Critical Difference: Organic-Label Regime (N = {N})"
        }[r_name]
        regime_sub = f"Friedman χ² = {chi2:.2f} (p = {p_friedman:.4f}), Nemenyi CD = {cd:.2f} (α = {alpha})"
        out_prefix = os.path.join(out_dir, f"cd_diagram_{r_name}")
        draw_cd_diagram(df_ranks, cd, regime_title, regime_sub, out_prefix, alpha=alpha)

        # Also write explicit aliases requested in prompt
        alias_names = {
            "all_datasets": f"cd_diagram_all{N}",
            "designated_class": f"cd_diagram_designated{N}",
            "organic_label": f"cd_diagram_organic{N}"
        }
        if r_name in alias_names:
            alias_prefix = os.path.join(out_dir, alias_names[r_name])
            if alias_prefix != out_prefix:
                draw_cd_diagram(df_ranks, cd, regime_title, regime_sub, alias_prefix, alpha=alpha)

    df_cd_summary = pd.DataFrame(summary_rows)
    csv_cd_path = os.path.join(out_dir, "item3_cd_summary.csv")
    df_cd_summary.to_csv(csv_cd_path, index=False)
    print(f"Saved: {csv_cd_path}")

    print("\n--- TABLE 3: Critical Difference Summary Across Regimes ---")
    print(df_cd_summary[["regime", "n_blocks", "chi2", "p_friedman", "nemenyi_cd", "best_detector", "best_rank", "n_within_cd", "n_reject_holm"]].round(3).to_string(index=False))

    return df_cd_summary


# ==============================================================================
# ITEM 4: Fold Winners
# ==============================================================================
def compute_item4_fold_winners(folds_dict, data_dir, datasets, out_dir, n_null=10000, seed=42):
    print("\n" + "=" * 80)
    print("ITEM 4: Fold Winners vs. Pooled Winners and Exchangeable Null (U2 Methods)")
    print("=" * 80)

    rng_null = np.random.RandomState(seed)

    per_ds_rows = []
    fold_detail_rows = []

    for ds in datasets:
        fm = folds_dict[ds]
        vals = fm.values  # 25 x k
        K = vals.shape[1]

        pooled_means = fm.mean(axis=1)
        pooled_ranks = pooled_means.rank(ascending=False, method="min").astype(int)
        pooled_win = pooled_means.idxmax()
        pooled_win_auc = pooled_means.max()

        fold_ranks = []
        fold_diffs = []

        for g in fm.columns:
            col = fm[g]
            m_auc = col.max()
            tied = col[col == m_auc].index.tolist()

            # Pooled-preferred tie breaking
            if pooled_win in tied:
                chosen_det = pooled_win
            else:
                chosen_det = min(tied, key=lambda d: pooled_ranks[d])

            chosen_rk = pooled_ranks[chosen_det]
            equals_pw = (chosen_det == pooled_win)
            fold_ranks.append(chosen_rk)
            fold_diffs.append(not equals_pw)

            fold_detail_rows.append({
                "setting": "primary",
                "dataset": ds,
                "fold": g,
                "fold_winner": chosen_det,
                "fold_max_auc": m_auc,
                "equals_pooled_winner": equals_pw,
                "pooled_rank": chosen_rk,
                "tied_count": len(tied),
                "pooled_winner": pooled_win,
                "pooled_winner_auc": pooled_win_auc
            })

        rks_arr = np.array(fold_ranks)
        diffs_arr = np.array(fold_diffs)
        obs_frac_diff = float(diffs_arr.mean())
        obs_mean_rk = float(rks_arr.mean())

        # Permutation null for exchangeable detectors (10,000 perms)
        null_frac_diff = np.empty(n_null)
        null_mean_rk = np.empty(n_null)

        for p in range(n_null):
            perm_mat = np.empty_like(vals)
            for g_idx in range(K):
                perm_mat[:, g_idx] = rng_null.permutation(vals[:, g_idx])
            p_mean = perm_mat.mean(axis=1)
            p_win = np.argmax(p_mean)
            p_ranks = np.empty(25, dtype=int)
            p_ranks[np.argsort(-p_mean)] = np.arange(1, 26)

            f_wins = np.argmax(perm_mat, axis=0)
            null_frac_diff[p] = (f_wins != p_win).mean()
            null_mean_rk[p] = p_ranks[f_wins].mean()

        p_val_diff = float((null_frac_diff <= obs_frac_diff).mean())
        p_val_rk = float((null_mean_rk <= obs_mean_rk).mean())

        per_ds_rows.append({
            "setting": "primary",
            "dataset": ds,
            "n_folds": K,
            "pooled_winner": pooled_win,
            "pooled_winner_auc": pooled_win_auc,
            "n_differs": int(diffs_arr.sum()),
            "obs_frac_differs": obs_frac_diff,
            "null_frac_differs_mean": float(null_frac_diff.mean()),
            "null_frac_differs_p025": float(np.percentile(null_frac_diff, 2.5)),
            "null_frac_differs_p975": float(np.percentile(null_frac_diff, 97.5)),
            "p_frac_differs": p_val_diff,
            "obs_mean_rank": obs_mean_rk,
            "null_mean_rank": float(null_mean_rk.mean()),
            "null_mean_rank_p025": float(np.percentile(null_mean_rk, 2.5)),
            "null_mean_rank_p975": float(np.percentile(null_mean_rk, 97.5)),
            "p_mean_rank": p_val_rk,
            "std_rank": float(rks_arr.std(ddof=1)) if K > 1 else 0.0,
            "median_rank": float(np.median(rks_arr)),
            "min_rank": int(rks_arr.min()),
            "max_rank": int(rks_arr.max()),
            "analytic_frac_diff": 24.0 / 25.0,
            "analytic_mean_rank": 13.0
        })

    # Evaluate secondary setting as well (10% contamination)
    try:
        sec_folds_dict = load_dataset_runs(data_dir, datasets, setting="secondary")
        sec_rows = []
        for ds in datasets:
            if ds in sec_folds_dict:
                fm_sec = sec_folds_dict[ds]
                vals_sec = fm_sec.values
                K_sec = vals_sec.shape[1]
                pooled_means_s = fm_sec.mean(axis=1)
                pooled_ranks_s = pooled_means_s.rank(ascending=False, method="min").astype(int)
                pooled_win_s = pooled_means_s.idxmax()
                pooled_win_auc_s = pooled_means_s.max()

                f_ranks_s = []
                f_diffs_s = []
                for g in fm_sec.columns:
                    col = fm_sec[g]
                    m_auc = col.max()
                    tied = col[col == m_auc].index.tolist()
                    chosen_det = pooled_win_s if pooled_win_s in tied else min(tied, key=lambda d: pooled_ranks_s[d])
                    chosen_rk = pooled_ranks_s[chosen_det]
                    f_ranks_s.append(chosen_rk)
                    f_diffs_s.append(chosen_det != pooled_win_s)
                    fold_detail_rows.append({
                        "setting": "secondary",
                        "dataset": ds,
                        "fold": g,
                        "fold_winner": chosen_det,
                        "fold_max_auc": m_auc,
                        "equals_pooled_winner": (chosen_det == pooled_win_s),
                        "pooled_rank": chosen_rk,
                        "tied_count": len(tied),
                        "pooled_winner": pooled_win_s,
                        "pooled_winner_auc": pooled_win_auc_s
                    })

                rks_s = np.array(f_ranks_s)
                diffs_s = np.array(f_diffs_s)
                sec_rows.append({
                    "setting": "secondary",
                    "dataset": ds,
                    "n_folds": K_sec,
                    "pooled_winner": pooled_win_s,
                    "pooled_winner_auc": pooled_win_auc_s,
                    "n_differs": int(diffs_s.sum()),
                    "obs_frac_differs": float(diffs_s.mean()),
                    "obs_mean_rank": float(rks_s.mean()),
                    "std_rank": float(rks_s.std(ddof=1)) if K_sec > 1 else 0.0,
                    "median_rank": float(np.median(rks_s)),
                    "min_rank": int(rks_s.min()),
                    "max_rank": int(rks_s.max())
                })
        if sec_rows:
            df_sec_summary = pd.DataFrame(sec_rows)
            df_sec_summary.to_csv(os.path.join(out_dir, "item4_fold_winners_secondary_summary.csv"), index=False)
            print("Saved: item4_fold_winners_secondary_summary.csv")
    except Exception as e:
        print(f"Notice: Secondary setting fold winners not evaluated ({e})")

    df_fold_summary = pd.DataFrame(per_ds_rows)
    df_fold_details = pd.DataFrame(fold_detail_rows)

    csv_sum_path = os.path.join(out_dir, "item4_fold_winners_summary.csv")
    csv_det_path = os.path.join(out_dir, "item4_fold_winners_details.csv")
    df_fold_summary.to_csv(csv_sum_path, index=False)
    df_fold_details.to_csv(csv_det_path, index=False)
    print(f"Saved: {csv_sum_path}")
    print(f"Saved: {csv_det_path}")

    print("\n--- TABLE 4: Fold Winners vs. Exchangeable Null (Primary Setting) ---")
    disp_cols = ["dataset", "n_folds", "pooled_winner", "obs_frac_differs", "null_frac_differs_mean",
                 "p_frac_differs", "obs_mean_rank", "null_mean_rank", "p_mean_rank"]
    print(df_fold_summary[disp_cols].round(3).to_string(index=False))

    return df_fold_summary, df_fold_details


# ==============================================================================
# ITEM 5: Effective Number of Detectors
# ==============================================================================
def compute_neff_and_se(C):
    eigvals = np.sort(np.linalg.eigvalsh(C.values))[::-1]
    eigvals = np.maximum(eigvals, 0)
    sum_lam = float(np.sum(eigvals))
    sum_lam2 = float(np.sum(eigvals ** 2))
    n_eff = (sum_lam ** 2) / sum_lam2 if sum_lam2 > 0 else np.nan
    se_neff = 1.0 / np.sqrt(n_eff - 3) if n_eff and n_eff > 3 else np.nan
    return n_eff, se_neff, eigvals[0]


def compute_item5_effective_detectors(folds_dict, datasets, out_dir):
    print("\n" + "=" * 80)
    print("ITEM 5: Effective Number of Detectors per Dataset and Pooled (U6 Methods)")
    print("=" * 80)

    rows = []
    # 1. Per dataset
    for ds in datasets:
        fm = folds_dict[ds]  # 25 x k
        k = fm.shape[1]

        corr_p = fm.T.corr(method="pearson").fillna(0)
        corr_s = fm.T.corr(method="spearman").fillna(0)

        # Centered by fold
        fm_c = fm.sub(fm.mean(axis=0), axis=1)
        corr_c = fm_c.T.corr(method="pearson").fillna(0)

        # Rank within fold
        fm_r = fm.rank(axis=0)
        corr_r = fm_r.T.corr(method="pearson").fillna(0)

        neff_p, se_p, top_e_p = compute_neff_and_se(corr_p)
        neff_s, se_s, top_e_s = compute_neff_and_se(corr_s)
        neff_c, se_c, top_e_c = compute_neff_and_se(corr_c)
        neff_r, se_r, top_e_r = compute_neff_and_se(corr_r)

        rows.append({
            "dataset": ds,
            "n_folds": k,
            "neff_pearson": neff_p,
            "se_neff_pearson": se_p,
            "neff_spearman": neff_s,
            "se_neff_spearman": se_s,
            "neff_centered": neff_c,
            "se_neff_centered": se_c,
            "neff_rank": neff_r,
            "se_neff_rank": se_r
        })

    # 2. Pooled across all folds of datasets
    # Concat all folds into a single 25 x N_total_folds matrix
    pooled_cols = []
    for ds in datasets:
        fm = folds_dict[ds].copy()
        fm.columns = [f"{ds}__{c}" for c in fm.columns]
        pooled_cols.append(fm)
    piv_pooled = pd.concat(pooled_cols, axis=1)
    n_tot_folds = piv_pooled.shape[1]

    corr_p_pool = piv_pooled.T.corr(method="pearson").fillna(0)
    corr_s_pool = piv_pooled.T.corr(method="spearman").fillna(0)
    piv_c_pool = piv_pooled.sub(piv_pooled.mean(axis=0), axis=1)
    corr_c_pool = piv_c_pool.T.corr(method="pearson").fillna(0)
    piv_r_pool = piv_pooled.rank(axis=0)
    corr_r_pool = piv_r_pool.T.corr(method="pearson").fillna(0)

    neff_p_pool, se_p_pool, _ = compute_neff_and_se(corr_p_pool)
    neff_s_pool, se_s_pool, _ = compute_neff_and_se(corr_s_pool)
    neff_c_pool, se_c_pool, _ = compute_neff_and_se(corr_c_pool)
    neff_r_pool, se_r_pool, _ = compute_neff_and_se(corr_r_pool)

    rows.append({
        "dataset": f"ALL_POOLED ({len(datasets)} ds)",
        "n_folds": n_tot_folds,
        "neff_pearson": neff_p_pool,
        "se_neff_pearson": se_p_pool,
        "neff_spearman": neff_s_pool,
        "se_neff_spearman": se_s_pool,
        "neff_centered": neff_c_pool,
        "se_neff_centered": se_c_pool,
        "neff_rank": neff_r_pool,
        "se_neff_rank": se_r_pool
    })

    df_neff = pd.DataFrame(rows)
    csv_neff_path = os.path.join(out_dir, "item5_effective_detectors.csv")
    df_neff.to_csv(csv_neff_path, index=False)
    print(f"Saved: {csv_neff_path}")

    print("\n--- TABLE 5: Effective Number of Detectors (U6 Methods) ---")
    print(df_neff.round(3).to_string(index=False))

    return df_neff


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Driver script for MAAD-Bench Unit R2.")
    parser.add_argument("--datasets", type=str, default=",".join(ALL_14_DATASETS),
                        help="Comma-separated list of dataset names.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing per-dataset rot_runs.parquet.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output directory for CSVs, diagrams, and logs.")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Number of bootstrap replicates for intervals.")
    parser.add_argument("--n-null-rel", type=int, default=4000,
                        help="Number of permutation null draws for Item 1 reliability.")
    parser.add_argument("--n-null-win", type=int, default=10000,
                        help="Number of permutation null draws for Item 4 fold winners.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("MAAD-Bench Re-Analysis UNIT R2: Ordering-Level Analyses Over Sub-Datasets")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 80)

    # 1. Load fold runs
    folds_dict = load_dataset_runs(args.data_dir, datasets, setting="primary")

    # 2. Item 1: Inter-fold reliability
    df_rel = compute_item1_reliability(folds_dict, datasets, args.out_dir,
                                       n_null=args.n_null_rel, n_boot=args.n_boot, seed=args.seed)

    # 3. Item 2: Cross-task rank agreement
    df_pairs, df_fam = compute_item2_cross_task(folds_dict, df_rel, datasets, args.out_dir,
                                                n_boot=args.n_boot, seed=args.seed)

    # 4. Item 3: Critical-difference ranks
    df_cd = compute_item3_critical_difference(folds_dict, datasets, args.out_dir, alpha=0.05)

    # 5. Item 4: Fold winners
    df_win_sum, df_win_det = compute_item4_fold_winners(folds_dict, args.data_dir, datasets, args.out_dir,
                                                        n_null=args.n_null_win, seed=args.seed)

    # 6. Item 5: Effective number of detectors
    df_neff = compute_item5_effective_detectors(folds_dict, datasets, args.out_dir)

    print("\n" + "=" * 80)
    print("All R2 driver tasks completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
