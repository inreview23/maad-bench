"""Emit LaTeX table bodies for the MAAD-Bench paper from the rotated results.

Primary and secondary AUC-ROC, formatted as "mean (rank)" per cell (rank within
each sub-dataset column, 1 = best), best per column in bold, an "Avg. rank"
column, and detectors sorted by average rank. This matches the value-(rank)
convention used by ADBench and MetaOD.

Reads rot_metrics.csv, whose auc_roc_mean averages folds after averaging seeds
within each fold. Pass --source metrics.csv to reproduce the v1 tables from the
single fixed held-out group.

The cell uncertainty printed by --with-sd is the BETWEEN-FOLD standard
deviation, not the within-fold seed spread the v1 tables reported. Under the v1
protocol the primary training set and the held-out group were identical across
seeds, so the reported spread measured only the anomaly subsample and detector
randomness; with the subsample dropped it is exactly zero for every
deterministic detector. Between-fold spread is the quantity a reader should see.

Usage:
  python make_table.py                       # rotated, both settings
  python make_table.py --with-sd
  python make_table.py --source metrics.csv  # v1 comparison
"""
import argparse
import glob
import os

import pandas as pd

import maad_bench as mb

DESIGNATED = ["esc50", "urbansound8k", "ravdess", "nsynth"]
ORGANIC = ["mimii", "physionet", "icbhi"]
ORDER = DESIGNATED + ORGANIC
LABEL = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "ravdess": "RAVDESS",
         "nsynth": "NSynth", "mimii": "MIMII", "physionet": "PhysioNet",
         "icbhi": "ICBHI"}


def load(source):
    hits = sorted(glob.glob(os.path.join(mb.RESULTS_DIR, "*", source)))
    if not hits:
        raise SystemExit("no %s under %s; run rotate.py first"
                         % (source, mb.RESULTS_DIR))
    return pd.concat([pd.read_csv(h) for h in hits], ignore_index=True)


def emit(df, setting, with_sd):
    sub = df[df["setting"] == setting]
    if sub.empty:
        return
    m = sub.pivot_table(index="detector", columns="dataset", values="auc_roc_mean")
    present = [d for d in ORDER if d in m.columns]
    m = m[present]
    sd = None
    if with_sd and "auc_roc_fold_sd" in sub.columns:
        sd = sub.pivot_table(index="detector", columns="dataset",
                             values="auc_roc_fold_sd")[present]
    ranks = m.rank(ascending=False, method="min")
    avg_rank = ranks.mean(axis=1)

    n_folds = ""
    if "n_folds" in sub.columns:
        per = sub.groupby("dataset").n_folds.max()
        n_folds = "; folds per column: " + ", ".join(
            "%s=%d" % (LABEL[d], int(per.get(d, 0))) for d in present)
    print("\n%% ===== %s AUC-ROC, mean (rank); cols: %s & Avg. rank%s ====="
          % (setting, " & ".join(LABEL[d] for d in present), n_folds))

    for det in avg_rank.sort_values().index:
        cells = []
        for d in present:
            mv, rk = m.loc[det, d], ranks.loc[det, d]
            if pd.isna(mv):
                cells.append("--")
                continue
            body = "%.3f" % mv
            if sd is not None and pd.notna(sd.loc[det, d]):
                body += "$\\pm$%.3f" % sd.loc[det, d]
            if rk == 1:
                cells.append("\\textbf{%s (1)}" % body)
            elif rk == 2:
                cells.append("\\underline{%s (2)}" % body)
            else:
                cells.append("%s (%d)" % (body, int(rk)))
        ar = avg_rank.loc[det]
        cells.append("\\textbf{%.1f}" % ar if ar == avg_rank.min() else "%.1f" % ar)
        print("%-12s & %s \\\\" % (det, " & ".join(cells)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rot_metrics.csv",
                    help="rot_metrics.csv (rotated) or metrics.csv (v1)")
    ap.add_argument("--with-sd", action="store_true",
                    help="append the between-fold SD to each cell")
    args = ap.parse_args()
    df = load(args.source)
    for s in ("primary", "secondary"):
        emit(df, s, args.with_sd)


if __name__ == "__main__":
    main()
