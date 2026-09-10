"""Training-side contamination robustness for MAAD-Bench.

v1 swept mb.CONTAMINATION to move the TEST-side positive rate, because
make_split subsampled the held-out anomalies down to that rate. Under the
rotated protocol the test side keeps every held-out anomaly, so that knob no
longer touches the test split and the old question ("is AUC-ROC invariant to
test prevalence?") is answered by construction rather than by experiment: AUC-ROC
is prevalence-invariant, which is why the subsample was dropped.

The question worth asking instead is the one the secondary setting already asks
at a single level: how much does the ranking move when the TRAINING data are
contaminated at a rate other than the assumed 10%? This sweeps that at 5%, 10%
and 20% under the rotated protocol.

The level moves the training-side injection, the contamination hyperparameter
passed to each detector, and the top-q flag budget together, so it reads as "the
true training rate and the practitioner's assumption are both X" rather than as
a mis-specification study. Report it that way.

Reuses results/<name>/features.parquet, so no audio is re-read.

Usage:
  python contam_sweep.py --dataset all --jobs 8
"""
import argparse
import os

import pandas as pd

import maad_bench as mb
from rotate import ALL, load_tables

LEVELS = [0.05, 0.10, 0.20]
# 2026-08-28 09:00 local: when the level began travelling as an argument instead
# of as module state. Parts written before it all ran at 0.10 whatever their name.
RESUME_AFTER = 1787932800.0


def sweep_one(name, jobs, levels=LEVELS):
    df, feat = load_tables(name)
    groups = sorted(df["group"].unique().tolist())
    out_dir = os.path.join(mb.RESULTS_DIR, name)
    parts = []
    for c in levels:
        part = os.path.join(out_dir, "rot_sweep_c%02d.csv" % int(round(c * 100)))
        # Resume: a part written after the contamination fix is valid and costs
        # nothing to keep. Anything older ran every level at 0.10 and is refused.
        if os.path.exists(part) and os.path.getmtime(part) >= RESUME_AFTER:
            d = pd.read_csv(part)
            if "dataset" not in d.columns:
                d.insert(0, "dataset", name)
            if "train_contamination" not in d.columns:
                d["train_contamination"] = c
            parts.append(d)
            print("[%s @ train contamination %.2f] reused" % (name, c), flush=True)
            continue
        if jobs > 1 and len(groups) > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=min(jobs, len(groups)))(
                delayed(mb.run_rotation)(df, feat, groups=[g],
                                         settings=("secondary",),
                                         subsample_test=False,
                                         collect_scores_for=(),
                                         contamination=c)
                for g in groups)
            runs = pd.concat([r[0] for r in res], ignore_index=True)
        else:
            runs, _ = mb.run_rotation(df, feat, groups=groups,
                                      settings=("secondary",),
                                      subsample_test=False,
                                      collect_scores_for=(),
                                      contamination=c)
        runs.insert(0, "dataset", name)
        summ = mb.summarize_rotation(runs, name)
        summ["train_contamination"] = c
        summ.to_csv(os.path.join(out_dir, "rot_sweep_c%02d.csv"
                                 % int(round(c * 100))), index=False)
        parts.append(summ)
        print("[%s @ train contamination %.2f] done" % (name, c), flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=ALL + ["all"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--levels", default=",".join("%g" % x for x in LEVELS),
                    help="comma-separated training contamination rates")
    args = ap.parse_args()
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    names = ALL if args.dataset == "all" else [args.dataset]
    rows = []
    for n in names:
        try:
            rows.append(sweep_one(n, args.jobs, levels))
        except Exception as e:
            print("[%s] FAILED: %s: %s" % (n, type(e).__name__, e), flush=True)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        p = os.path.join(mb.RESULTS_DIR, "rot_sweep_combined.csv")
        out.to_csv(p, index=False)
        print("wrote %s (%d rows)" % (p, len(out)), flush=True)


if __name__ == "__main__":
    main()
