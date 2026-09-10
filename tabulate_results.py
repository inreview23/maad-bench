"""Consolidate MAAD-Bench results (classical sweep + AE baseline) into pivot
tables for the paper. Handles both schemas: metrics_combined.csv is already
aggregated (<metric>_mean / <metric>_std columns); ae_scores.parquet is
per-seed. Prints mean (and std for AUC) per detector per dataset, split by
setting. Run from the repository root.
"""
import glob
import os

import pandas as pd

import maad_bench as mb
from analysis import guard_mixed_protocols  # noqa: E402
guard_mixed_protocols()


METRICS = ["auc_roc", "avg_precision", "pauc_0.1", "f1_top_q"]
KEYS = ["dataset", "setting", "detector"]


def normalize(df):
    if "auc_roc_mean" in df.columns:
        cols = KEYS + [m + s for m in METRICS for s in ("_mean", "_std")
                       if (m + s) in df.columns]
        return df[cols].copy()
    g = df.groupby(KEYS)
    out = g[METRICS].agg(["mean", "std"])
    out.columns = ["%s_%s" % (m, s) for m, s in out.columns]
    return out.reset_index()


frames = []
# Per-dataset metrics.csv files persist across single-dataset runs (metrics_combined.csv
# is overwritten per invocation), so read those and accumulate all datasets present.
for mc in sorted(glob.glob(os.path.join(mb.RESULTS_DIR, "*", "metrics.csv"))):
    c = pd.read_csv(mc)
    if "name" in c.columns and "dataset" not in c.columns:
        c = c.rename(columns={"name": "dataset"})
    frames.append(normalize(c))

for ae in sorted(glob.glob(os.path.join(mb.RESULTS_DIR, "*", "ae_scores.parquet"))):
    name = os.path.basename(os.path.dirname(ae))
    a = pd.read_parquet(ae)
    a["dataset"] = name
    frames.append(normalize(a))

df = pd.concat(frames, ignore_index=True)
print("detectors:", sorted(df["detector"].unique()))
print("datasets:", sorted(df["dataset"].unique()))

for metric in METRICS:
    mcol, scol = metric + "_mean", metric + "_std"
    if mcol not in df.columns:
        continue
    for setting in ("primary", "secondary"):
        sub = df[df["setting"] == setting]
        if sub.empty:
            continue
        piv_m = sub.pivot_table(index="detector", columns="dataset", values=mcol)
        print("\n=== %s_mean | %s ===" % (metric, setting))
        print(piv_m.round(3).to_string())
        if metric == "auc_roc" and scol in sub.columns:
            piv_s = sub.pivot_table(index="detector", columns="dataset", values=scol)
            print("--- std ---")
            print(piv_s.round(3).to_string())
