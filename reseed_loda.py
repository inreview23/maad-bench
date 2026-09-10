"""Re-run LODA everywhere under the seeding fix, and re-summarise.

Why only LODA. The factory passes `random_state` by signature introspection, and
of the 25 detectors exactly one takes no such argument and draws from the global
NumPy generator instead: PyOD's LODA builds its random cuts with `np.random`
inside `fit`. Measured across six sub-datasets, LODA is the only detector whose
value moves when the driver's call order changes, by up to 0.018; the other 24
agree to within 3e-6. Seeding the global generator before each fit therefore
changes LODA's numbers and nothing else, which is why re-running LODA alone gives
exactly what a full re-run would give. This script checks that claim on one fold
before it rewrites anything.

What it touches: the per-sub-dataset runs files for the core rotation, the
representation arms, and the contamination sweep, replacing LODA's rows and
re-deriving the summary CSVs from them.

Usage:
  python reseed_loda.py --check          # verify LODA-only == full, do not write
  python reseed_loda.py --arm core       # rewrite one arm
  python reseed_loda.py --arm all
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import maad_bench as mb
from rotate import ALL, load_tables

R = mb.RESULTS_DIR
DET = "LODA"


def loda_rows(df, feat, groups, settings, pre=None, contamination=None):
    """Fit LODA alone on every (group, setting, seed), through the same
    _sweep_split the full run uses, so the seeding is identical."""
    X = feat.to_numpy(dtype=float)
    y = df["anomaly_label"].to_numpy()
    facs = {DET: mb.detector_factories()[DET]}
    c = mb.CONTAMINATION if contamination is None else float(contamination)
    rec = []
    for g in groups:
        for setting in settings:
            for seed in mb.SEEDS:
                tr, te = mb.make_split(df, seed, setting, [g],
                                       subsample_test=False, contamination=c)
                if not mb.is_scorable(y[te]):
                    continue
                rows, _ = mb._sweep_split(X, y, tr, te, facs, seed, c,
                                          ctx={"setting": setting, "group": g},
                                          collect=False, pre=pre)
                rec.extend(rows)
    return pd.DataFrame(rec)


def check_one(name="esc50"):
    """LODA alone against LODA inside the full roster, same fold, same seeds."""
    df, feat = load_tables(name)
    g = sorted(df["group"].unique().tolist())[0]
    alone = loda_rows(df, feat, [g], ("primary", "secondary"))
    full, _ = mb.run_rotation(df, feat, groups=[g],
                              settings=("primary", "secondary"),
                              subsample_test=False, collect_scores_for=())
    full = full[full.detector == DET]
    a = alone.set_index(["setting", "seed"]).auc_roc
    b = full.set_index(["setting", "seed"]).auc_roc
    common = a.index.intersection(b.index)
    d = (a[common] - b[common]).abs()
    print("cells compared: %d" % len(common))
    print("max abs delta:  %.12f" % d.max())
    ok = d.max() < 1e-12
    print("VERDICT: %s" % ("LODA alone reproduces LODA in the full roster"
                           if ok else "DIFFERS, do not splice"))
    return ok


def splice(runs_path, new_rows, summary_path, dataset):
    old = pd.read_parquet(runs_path)
    keep = old[old.detector != DET]
    new_rows = new_rows.copy()
    new_rows["dataset"] = dataset
    for col in old.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan
    merged = pd.concat([keep, new_rows[old.columns]], ignore_index=True)
    merged.to_parquet(runs_path, index=False)
    summ = mb.summarize_rotation(merged, dataset)
    if "dataset" not in summ.columns:
        summ.insert(0, "dataset", dataset)
    summ.to_csv(summary_path, index=False)
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arm", default="core", choices=["core", "all"])
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check_one() else 1)

    total = 0
    for name in ALL:
        df, feat = load_tables(name)
        groups = sorted(df["group"].unique().tolist())
        rp = os.path.join(R, name, "rot_runs.parquet")
        sp = os.path.join(R, name, "rot_metrics.csv")
        if os.path.exists(rp):
            rows = loda_rows(df, feat, groups, ("primary", "secondary"))
            n = splice(rp, rows, sp, name)
            total += n
            print("[%s] core: %d LODA rows" % (name, n), flush=True)
    print("rewrote %d LODA rows" % total)


if __name__ == "__main__":
    main()
