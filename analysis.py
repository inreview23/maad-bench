"""Post-rotation analysis for MAAD-Bench.

Every number the revised paper's claims rest on is computed here, from
results/<ds>/rot_runs.parquet. The v1 analyses this replaces, and why:

  R2-5  stats_cd.py pivoted on ["dataset", "seed"], giving 35 blocks from 7
        sub-datasets. Seeds within a sub-dataset are repeated measurements, not
        independent blocks. Here the block is the sub-dataset, after averaging
        folds and seeds.
  R2-4  The reviewer argued that correlations against noisy near-chance ranks
        trend to zero regardless. That is testable: under one shared detector
        ordering, the observed correlation between two sub-datasets cannot
        exceed sqrt(rel_A * rel_B). Reliability is now estimated across folds,
        which is the axis the reviewer asked about, rather than across seeds,
        which shared a training set.
  R2-3  Per-fold AUC distributions and held-out-group atypicality, so
        "inversion" is only claimed where it survives group rotation.
  ---   The winner-count argument is computed against its null so it can be
        retired rather than promoted: six distinct winners across seven
        sub-datasets is what interchangeable detectors produce.

Usage:
  python analysis.py                 # full report to stdout + CSVs under results/
  python analysis.py --metric auc_roc --repr handcrafted
"""
import argparse
import itertools
import json
import os
import zlib

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr

import maad_bench as mb

DESIGNATED = ["esc50", "urbansound8k", "ravdess", "nsynth"]
ORGANIC = ["mimii", "physionet", "icbhi"]
ALL = DESIGNATED + ORGANIC


# ------------------------------------------------------------------ loading
def guard_mixed_protocols():
    """Refuse to emit v1 tables while rotated results sit beside them.

    results/ carries both protocols' outputs during the transition. A script
    that reads metrics.csv while rot_metrics.csv exists prints a LaTeX body with
    no error at all, and a mixed-protocol table in the manuscript is the kind of
    defect that survives a submission cycle. Call this at the top of anything
    that still reads the v1 files.
    """
    import glob as _glob
    rotated = _glob.glob(os.path.join(mb.RESULTS_DIR, "*", "rot_metrics.csv"))
    if rotated:
        raise SystemExit(
            "results/ holds rotated outputs for %d sub-datasets alongside the v1 "
            "files this script reads. Use make_table.py / analysis.py, which read "
            "the rotated results, or move the rot_* files aside deliberately."
            % len(rotated))


def load_runs(repr_name="handcrafted", metric="auc_roc"):
    """One row per (dataset, group, setting, seed, detector)."""
    suffix = "rot_runs.parquet" if repr_name == "handcrafted" \
        else "rot_%s_runs.parquet" % repr_name
    frames = []
    for ds in ALL:
        p = os.path.join(mb.RESULTS_DIR, ds, suffix)
        if not os.path.exists(p):
            print("  (missing %s)" % p)
            continue
        d = pd.read_parquet(p)
        if "dataset" not in d.columns:
            d["dataset"] = ds
        frames.append(d)
    if not frames:
        raise SystemExit("no rotation results found; run rotate.py first")
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["detector"] != "*") & d[metric].notna()]
    return d


def fold_table(runs, metric="auc_roc", setting="primary"):
    """Seed-averaged AUC per (dataset, detector, fold)."""
    d = runs[runs["setting"] == setting]
    return (d.groupby(["dataset", "detector", "group"])[metric]
            .mean().reset_index().rename(columns={metric: "auc"}))


def dataset_matrix(folds):
    """detector x dataset matrix of fold-averaged AUC."""
    return folds.groupby(["dataset", "detector"]).auc.mean().unstack(0)


# -------------------------------------------------------- rank reliability
def fold_reliability(folds, n_splits=200):
    """Rank reliability of each sub-dataset's detector ordering, across FOLDS.

    The v1 answer to R2-4 split across seeds, but the primary training set and
    the held-out group were identical across seeds, so that measured almost
    nothing. Folds are the axis that varies.

    The headline estimator is the mean pairwise rank correlation between single
    folds, stepped up to the full fold count by Spearman-Brown. With four or five
    folds a single random split-half is itself noisy, and averaging every pair
    uses all of them. The split-half value is kept as a cross-check.

    Returns a frame with one row per sub-dataset so the fold count travels with
    the estimate: reliability from four folds should not be read like
    reliability from twenty-four.
    """
    rows = []
    for ds, t in folds.groupby("dataset"):
        groups = sorted(t["group"].unique())
        k = len(groups)
        # A fresh generator per sub-dataset, rather than the module-level one,
        # so the split-half cross-check does not depend on how many sub-datasets
        # were processed before this one. crc32 rather than hash(): Python
        # randomizes string hashing per process, which would make the seed itself
        # vary between runs.
        rng = np.random.RandomState(zlib.crc32(ds.encode()) % (2 ** 31))
        per_fold = {g: t[t.group == g].set_index("detector").auc for g in groups}

        pair = []
        for a, b in itertools.combinations(groups, 2):
            x, y = per_fold[a], per_fold[b]
            common = x.index.intersection(y.index)
            if len(common) < 3:
                continue
            r = spearmanr(x[common], y[common]).statistic
            if np.isfinite(r):
                pair.append(r)
        mean_pair = float(np.mean(pair)) if pair else np.nan
        rel = (k * mean_pair / (1 + (k - 1) * mean_pair)
               if np.isfinite(mean_pair) and (1 + (k - 1) * mean_pair) > 0 else np.nan)

        half = np.nan
        if k >= 4:
            rhos = []
            for _ in range(n_splits):
                perm = rng.permutation(groups)
                h = k // 2
                x = t[t.group.isin(perm[:h])].groupby("detector").auc.mean()
                y = t[t.group.isin(perm[h:2 * h])].groupby("detector").auc.mean()
                common = x.index.intersection(y.index)
                if len(common) >= 3:
                    r = spearmanr(x[common], y[common]).statistic
                    if np.isfinite(r):
                        rhos.append(r)
            if rhos:
                m = float(np.median(rhos))
                half = (2 * m) / (1 + m) if m > -1 else np.nan

        rows.append({"dataset": ds, "n_folds": k, "mean_pairwise_rho": mean_pair,
                     "reliability": rel, "splithalf_reliability": half})
    return pd.DataFrame(rows).set_index("dataset")


def ceiling_table(mat, rel):
    """Observed pairwise rank correlation against its reliability ceiling.

    Under one shared detector ordering the observed correlation between two
    sub-datasets cannot exceed sqrt(rel_a * rel_b). A shortfall is therefore
    reordering that measurement error cannot explain, which is the direct test of
    the mechanism R2-4 proposes. Note the direction: attenuation pulls a
    correlation toward zero, so it cannot manufacture a negative one.

    `rel` is a Series of per-sub-dataset reliabilities.
    """
    rows = []
    for a, b in itertools.combinations(mat.columns, 2):
        obs = spearmanr(mat[a], mat[b]).statistic
        ra, rb = rel.get(a, np.nan), rel.get(b, np.nan)
        ceil = float(np.sqrt(max(ra, 0) * max(rb, 0))) if np.isfinite(ra) and np.isfinite(rb) \
            else float("nan")
        rows.append({"a": a, "b": b, "observed": obs, "ceiling": ceil,
                     "gap": ceil - obs, "reaches_ceiling": bool(obs >= ceil)})
    cols = ["a", "b", "observed", "ceiling", "gap", "reaches_ceiling"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


# -------------------------------------------------------------- transfer
def fisher_ci(rho, n, alpha=0.05):
    if not np.isfinite(rho) or n < 4 or abs(rho) >= 1:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = z - norm.ppf(1 - alpha / 2) * se, z + norm.ppf(1 - alpha / 2) * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def transfer(mat, designated=None, organic=None):
    designated = [c for c in (designated or DESIGNATED) if c in mat.columns]
    organic = [c for c in (organic or ORGANIC) if c in mat.columns]
    if not designated or not organic:
        return None
    des, org = mat[designated].mean(axis=1), mat[organic].mean(axis=1)
    r = spearmanr(des, org)
    lo, hi = fisher_ci(r.statistic, len(des))
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n_detectors": int(len(des)),
            "ci_lo": lo, "ci_hi": hi,
            "designated": designated, "organic": organic}


# ------------------------------------------------------ omnibus + post-hoc
def friedman_block(mat):
    """Friedman over sub-dataset blocks, detectors as treatments."""
    m = mat.dropna(axis=0, how="any")
    arr = m.to_numpy().T                      # blocks x detectors
    N, k = arr.shape
    if N < 3 or k < 3:
        return None
    chi, p = friedmanchisquare(*[arr[:, i] for i in range(k)])
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, arr)
    avg = ranks.mean(0)
    try:
        from scipy.stats import studentized_range
        q = float(studentized_range.ppf(0.95, k, np.inf)) / np.sqrt(2.0)
    except Exception:
        q = 3.658                              # k=25, alpha=0.05 fallback
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    order = np.argsort(avg)
    best = avg[order[0]]
    return {"n_blocks": int(N), "n_detectors": int(k),
            "chi2": float(chi), "p": float(p),
            "critical_difference": float(cd),
            "best_detector": str(m.index[order[0]]), "best_avg_rank": float(best),
            "worst_detector": str(m.index[order[-1]]), "worst_avg_rank": float(avg[order[-1]]),
            "largest_gap": float(avg[order[-1]] - best),
            "n_within_cd_of_best": int((avg - best <= cd).sum()),
            "n_separated_pairs": int(sum(1 for i in range(k) for j in range(i + 1, k)
                                         if abs(avg[i] - avg[j]) > cd)),
            "avg_ranks": {str(m.index[i]): float(avg[i]) for i in order}}


def permutation_block(mat, n_perm=20000):
    """Cluster-aware omnibus: permute detector labels within each sub-dataset.

    The sub-dataset is the resampling unit, so this respects the dependence the
    Friedman blocks assume away and does not need the chi-square approximation
    with 7 blocks and 25 treatments.
    """
    m = mat.dropna(axis=0, how="any")
    arr = m.to_numpy().T
    N, k = arr.shape
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, arr)
    obs = float(np.var(ranks.mean(0)))
    rng = np.random.RandomState(0)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = np.array([rng.permutation(row) for row in ranks])
        null[i] = np.var(p.mean(0))
    return {"statistic": obs, "p": float((1 + (null >= obs).sum()) / (n_perm + 1)),
            "n_blocks": int(N), "n_detectors": int(k), "n_perm": int(n_perm)}


def winner_null(mat):
    """Distinct-winner count against the exchangeable-detector null."""
    m = mat.dropna(axis=0, how="any")
    winners = m.idxmax(axis=0)
    obs, n, k = int(winners.nunique()), int(m.shape[1]), int(m.shape[0])
    dp = np.zeros(k + 1)
    dp[0] = 1.0
    for _ in range(n):
        nd = np.zeros(k + 1)
        for j in range(k + 1):
            if dp[j]:
                nd[j] += dp[j] * (j / k)
                if j + 1 <= k:
                    nd[j + 1] += dp[j] * ((k - j) / k)
        dp = nd
    return {"observed_distinct_winners": obs, "n_blocks": n, "n_detectors": k,
            "expected_under_null": float(k * (1 - ((k - 1) / k) ** n)),
            "p_at_least_observed": float(dp[obs:].sum()),
            "winners": {str(d): str(w) for d, w in winners.items()}}


# ----------------------------------------------------------- group shift
def group_shift(datasets=ORGANIC):
    """How atypical is each held-out group's normal population?

    Answers R2-3(b) across every fold rather than for the one group v1 fixed.
    Ratio > 1 means the held-out normals sit further from the training normals
    than the anomalies do, which inverts any distance-like detector.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    rows = []
    for ds in datasets:
        d = os.path.join(mb.RESULTS_DIR, ds)
        if not os.path.exists(os.path.join(d, "features.parquet")):
            continue
        X = pd.read_parquet(os.path.join(d, "features.parquet")).to_numpy(float)
        md = pd.read_parquet(os.path.join(d, "metadata.parquet"))
        y, g = md.anomaly_label.to_numpy(), md.group.to_numpy()
        for gi in sorted(set(g)):
            te, tr = (g == gi), (~(g == gi)) & (y == 0)
            if tr.sum() < 20 or (te & (y == 1)).sum() < 3 or (te & (y == 0)).sum() < 3:
                continue
            sc = StandardScaler().fit(X[tr])
            c = sc.transform(X[tr]).mean(0)
            Zn, Za = sc.transform(X[te & (y == 0)]), sc.transform(X[te & (y == 1)])
            dn, da = np.linalg.norm(Zn - c, axis=1), np.linalg.norm(Za - c, axis=1)
            rows.append({"dataset": ds, "group": gi,
                         "n_test_normal": int(len(Zn)), "n_test_anom": int(len(Za)),
                         "normal_centroid_dist": float(np.linalg.norm(Zn.mean(0) - c)),
                         "anom_centroid_dist": float(np.linalg.norm(Za.mean(0) - c)),
                         "radial_auc": float(roc_auc_score(
                             np.r_[np.zeros(len(dn)), np.ones(len(da))],
                             np.r_[dn, da]))})
    t = pd.DataFrame(rows)
    if len(t):
        t["normal_over_anom_ratio"] = t.normal_centroid_dist / t.anom_centroid_dist
    return t


# ---------------------------------------------------------------- report
def chance_test(folds, metric="auc_roc", alpha=0.05):
    """Which detectors sit off chance, per sub-dataset, and in which direction.

    Each detector's fold mean is tested against 0.5 using the standard error of
    that mean across folds. That is deliberately conservative: on every
    sub-dataset here the between-fold spread exceeds the within-fold sampling
    error, and the fold is the unit the protocol resamples. A detector counted
    "below" has its whole interval under 0.5, which is what separates a systematic
    inversion from a point estimate that happens to land under the line.
    """
    from scipy import stats
    z = stats.norm.ppf(1 - alpha / 2.0)
    # fold_table renames the metric column to "auc", so the caller's metric name
    # is not the column name here. Reading it as metric raised KeyError and took
    # the whole run down before it wrote the ceiling table and the report, which
    # is why those files sat two days stale while the printed summary was current.
    col = metric if metric in folds.columns else "auc"
    g = folds.groupby(["dataset", "detector"])[col].agg(["mean", "std", "count"])
    g = g[g["count"] > 1]
    se = g["std"] / np.sqrt(g["count"])
    g["below"] = g["mean"] + z * se < 0.5
    g["above"] = g["mean"] - z * se > 0.5
    out = g.groupby("dataset").agg(
        lo=("mean", "min"), hi=("mean", "max"),
        n_below=("below", "sum"), n_above=("above", "sum"),
        n=("mean", "size"))
    out["verdict"] = [
        "inverted" if r.n_below > r.n / 2 else
        "solved" if r.hi >= 0.65 else "at chance"
        for r in out.itertuples()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="auc_roc")
    ap.add_argument("--repr", default="handcrafted")
    ap.add_argument("--setting", default="primary")
    args = ap.parse_args()

    runs = load_runs(args.repr, args.metric)
    folds = fold_table(runs, args.metric, args.setting)
    mat = dataset_matrix(folds)
    report = {"representation": args.repr, "metric": args.metric,
              "setting": args.setting}

    print("=" * 72)
    print("MAAD-Bench post-rotation analysis  [%s / %s / %s]"
          % (args.repr, args.metric, args.setting))
    print("=" * 72)

    print("\n--- 1. coverage ---")
    cov = folds.groupby("dataset").agg(folds_=("group", "nunique"),
                                       detectors=("detector", "nunique"))
    print(cov.to_string())
    report["coverage"] = cov.to_dict("index")

    print("\n--- 2. per-fold spread (best detector per dataset) ---")
    spread = (folds.groupby(["dataset", "detector"])
              .auc.agg(["mean", "std", "min", "max", "count"]).reset_index())
    best = spread.sort_values("mean", ascending=False).groupby("dataset").head(1)
    print(best.round(3).to_string(index=False))
    report["best_per_dataset"] = best.to_dict("records")

    print("\n--- 3. rank reliability across folds (answers R2-4) ---")
    relf = fold_reliability(folds)
    print(relf.round(3).to_string())
    print("  reliability = Spearman-Brown of the mean pairwise inter-fold rho, "
          "stepped up to n_folds")
    report["fold_reliability"] = relf.round(4).to_dict("index")
    rel = relf["reliability"]

    print("\n--- 4. observed pairwise correlation vs its reliability ceiling ---")
    ct = ceiling_table(mat, rel)
    if len(ct):
        print(ct.round(3).to_string(index=False))
        print("  mean ceiling=%.3f  mean observed=%.3f  pairs reaching ceiling: %d/%d"
              % (ct.ceiling.mean(), ct.observed.mean(),
                 int(ct.reaches_ceiling.sum()), len(ct)))
        print("  most negative observed: %.3f (attenuation shrinks toward zero, so "
              "it cannot manufacture a negative)" % ct.observed.min())
        report["ceiling"] = {"mean_ceiling": float(ct.ceiling.mean()),
                             "mean_observed": float(ct.observed.mean()),
                             "n_reaching": int(ct.reaches_ceiling.sum()),
                             "n_pairs": int(len(ct)),
                             "min_observed": float(ct.observed.min())}
    else:
        print("  (needs at least two sub-datasets)")

    print("\n--- 5. designated-to-organic transfer ---")
    full = transfer(mat)
    loo, per_ds = {}, {}
    if full is None:
        print("  (needs at least one sub-dataset on each side of the partition)")
        report["transfer"] = None
    else:
        print("  all organic:      rho=%.4f p=%.4f  95%% CI [%.3f, %.3f]  n=%d"
              % (full["rho"], full["p"], full["ci_lo"], full["ci_hi"],
                 full["n_detectors"]))
    for drop in ORGANIC:
        keep = [o for o in ORGANIC if o != drop and o in mat.columns]
        t = transfer(mat, organic=keep)
        if t:
            loo[drop] = t
            print("  drop %-11s rho=%.4f p=%.4f  95%% CI [%.3f, %.3f]"
                  % (drop, t["rho"], t["p"], t["ci_lo"], t["ci_hi"]))
    per_ds = {}
    des = mat[[c for c in DESIGNATED if c in mat.columns]].mean(axis=1)
    for o in ORGANIC:
        if o in mat.columns:
            r = spearmanr(des, mat[o])
            per_ds[o] = {"rho": float(r.statistic), "p": float(r.pvalue)}
            print("  designated vs %-11s rho=%+.3f p=%.4f" % (o, r.statistic, r.pvalue))
    report["transfer"] = {"full": full, "leave_one_organic_out": loo,
                          "per_organic_dataset": per_ds}

    print("\n--- 6. omnibus (answers R2-5) ---")
    fr = friedman_block(mat)
    if fr:
        print("  Friedman over %d sub-dataset blocks, %d detectors: chi2=%.2f p=%.4f"
              % (fr["n_blocks"], fr["n_detectors"], fr["chi2"], fr["p"]))
        print("  Nemenyi CD=%.2f; best %s (rank %.2f), worst %s (rank %.2f), "
              "largest gap %.2f" % (fr["critical_difference"], fr["best_detector"],
                                    fr["best_avg_rank"], fr["worst_detector"],
                                    fr["worst_avg_rank"], fr["largest_gap"]))
        print("  detectors within CD of the best: %d/%d; separated pairs: %d"
              % (fr["n_within_cd_of_best"], fr["n_detectors"], fr["n_separated_pairs"]))
        report["friedman"] = fr
        pm = permutation_block(mat)
        print("  cluster-aware permutation (sub-dataset as unit): p=%.4f (%d perms)"
              % (pm["p"], pm["n_perm"]))
        report["permutation"] = pm

    print("\n--- 7. winner counts against their null (why this argument is retired) ---")
    wn = winner_null(mat)
    print("  observed distinct winners: %d over %d sub-datasets (%d detectors)"
          % (wn["observed_distinct_winners"], wn["n_blocks"], wn["n_detectors"]))
    print("  exchangeable-detector null: expected %.3f, P(>= observed) = %.3f"
          % (wn["expected_under_null"], wn["p_at_least_observed"]))
    print("  winners: %s" % wn["winners"])
    report["winner_null"] = wn

    print("\n--- 8. held-out-group atypicality (answers R2-3b) ---")
    gs = group_shift()
    if len(gs):
        print(gs.round(3).to_string(index=False))
        report["group_shift"] = gs.to_dict("records")

    out = os.path.join(mb.RESULTS_DIR, "analysis_%s_%s.json" % (args.repr, args.setting))
    print("\n--- 9. detectors off chance, by sub-dataset ---")
    ch = chance_test(folds, args.metric)
    print(ch.to_string(float_format=lambda v: "%.3f" % v))
    print("  below/above count a detector only when its whole 95% interval clears 0.5,")
    print("  using the standard error of its fold mean across folds")
    report["chance_test"] = ch.to_dict("index")

    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    # The setting belongs in the name. Without it the secondary run overwrites
    # the primary table under a filename that says nothing about which one it
    # holds, and the file that ships is whichever setting happened to run last.
    ct.to_csv(os.path.join(mb.RESULTS_DIR, "analysis_ceiling_%s_%s.csv"
                           % (args.repr, args.setting)), index=False)
    if len(gs):
        gs.to_csv(os.path.join(mb.RESULTS_DIR, "analysis_group_shift_%s.csv"
                               % args.setting), index=False)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
