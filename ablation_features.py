"""MAAD-Bench feature-family ablation, group-rotated and over the full roster.

Incremental schema: MFCC only -> + chroma -> + spectral (full).

Two v1 problems this fixes. The released ablation_features.csv covers 11
detectors (CBLOF, COPOD, ECOD, GMM, HBOS, IForest, KNN, LODA, LOF, OCSVM, PCA)
over 3 sub-datasets, while the main table ranks 25 over 7. That is why the two
tables disagreed on ESC-50's full-schema best: the ablation's 0.674 is CBLOF,
which is exactly CBLOF's main-table entry, while the main table's 0.716 is SOD,
a detector the ablation never ran. Both reviewers flagged the gap. Re-running
over the current roster removes the discrepancy at its source rather than
disclosing it. The second fix is the protocol: rotation, and no test-side
anomaly subsample, matching rotate.py.

Reads results/<ds>/features.parquet, so no audio and no feature extraction.

Usage:
  python ablation_features.py --dataset esc50,urbansound8k,ravdess --jobs 8
"""
import argparse
import os

import pandas as pd

import maad_bench as mb
from rotate import ALL, load_tables

# Column-name prefixes the extractor emits (mfcc / chroma / cent / bw / roll /
# zcr / rms), grouped into the three incremental schemas.
FEATURE_SETS = {
    "mfcc": ("mfcc",),
    "mfcc+chroma": ("mfcc", "chroma"),
    "full": ("mfcc", "chroma", "cent", "bw", "roll", "zcr", "rms"),
}


def cols_for(feat, prefixes):
    return feat[[c for c in feat.columns if any(c.startswith(p) for p in prefixes)]]


def ablate_one(name, settings, jobs):
    df, feat = load_tables(name)
    groups = sorted(df["group"].unique().tolist())
    parts = []
    for setname, prefixes in FEATURE_SETS.items():
        sub = cols_for(feat, prefixes)
        if jobs > 1 and len(groups) > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=min(jobs, len(groups)))(
                delayed(mb.run_rotation)(df, sub, groups=[g], settings=settings,
                                         subsample_test=False, collect_scores_for=())
                for g in groups)
            runs = pd.concat([r[0] for r in res], ignore_index=True)
        else:
            runs, _ = mb.run_rotation(df, sub, groups=groups, settings=settings,
                                      subsample_test=False, collect_scores_for=())
        runs.insert(0, "dataset", name)
        s = mb.summarize_rotation(runs, name)
        s["feature_set"] = setname
        s["n_features"] = sub.shape[1]
        parts.append(s)
        print("[%s | %s] %d features, %d detectors done"
              % (name, setname, sub.shape[1],
                 int(s.detector.nunique())), flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esc50,urbansound8k,ravdess",
                    help="comma list, or 'all'")
    ap.add_argument("--settings", default="primary")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    names = ALL if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]
    settings = tuple(s for s in args.settings.split(",") if s)

    rows = []
    for n in names:
        try:
            rows.append(ablate_one(n, settings, args.jobs))
        except Exception as e:
            print("[%s] FAILED: %s: %s" % (n, type(e).__name__, e), flush=True)
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    p = os.path.join(mb.RESULTS_DIR, "rot_ablation_features.csv")
    out.to_csv(p, index=False)

    prim = out[out["setting"] == "primary"]
    print("\n=== feature ablation, group-rotated: primary AUC-ROC, "
          "mean over detectors and folds ===")
    print(prim.pivot_table(index="dataset", columns="feature_set",
                           values="auc_roc_mean")[list(FEATURE_SETS)]
          .round(3).to_string())

    best = prim.loc[prim.groupby(["dataset", "feature_set"])["auc_roc_mean"].idxmax()]
    print("\n=== best detector per feature set (primary), full roster ===")
    print(best[["dataset", "feature_set", "detector", "auc_roc_mean",
                "auc_roc_fold_sd", "n_features"]]
          .sort_values(["dataset", "feature_set"]).to_string(index=False))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
