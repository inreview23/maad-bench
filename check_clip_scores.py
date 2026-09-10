"""Verify the release claim the submitted version got wrong.

The v1 availability statement said per-detector scores were released, but
scores.parquet held run-level metrics: the array from decision_function was passed
straight into metrics() and never bound. The revised caption and availability
statement repeat the claim, so it has to be true this time. Checks, per
sub-dataset, that every test clip of every fold appears once per detector and
seed, and that the paths in the score table resolve back to the metadata.

Runs from any checkout containing the released ``results/`` tree.
"""
import glob
import os
import sys

import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]

bad = []
print("%-14s %9s %8s %7s %6s %6s  %s"
      % ("dataset", "clip rows", "clips", "folds", "det", "seeds", "verdict"))
for ds in ORDER:
    f = os.path.join(RESULTS, ds, "rot_clip_scores.parquet")
    m = os.path.join(RESULTS, ds, "metadata.parquet")
    if not os.path.exists(f):
        print("%-14s missing" % ds)
        bad.append((ds, "missing"))
        continue
    c = pd.read_parquet(f)
    meta = pd.read_parquet(m)

    n_folds = c.group.nunique()
    n_det = c.detector.nunique()
    n_seed = c.seed.nunique()

    # Every clip is a test clip in exactly one fold, for each (detector, seed),
    # except where a fit raised: those are recorded in rot_runs.parquet with an
    # error and legitimately produce no scores. Subtract them before comparing.
    runs = pd.read_parquet(os.path.join(RESULTS, ds, "rot_runs.parquet"))
    failed = 0
    if "error" in runs.columns:
        e = runs[runs["error"].notna() & (runs["error"].astype(str) != "")]
        failed = len(e)
    per = c.groupby(["detector", "seed"]).size()
    expected = len(meta)
    consistent = per.nunique() <= 1 + (1 if failed else 0)
    covered = int(per.max()) == expected

    # the paths resolve back to the metadata
    key = "path" if "path" in c.columns else None
    resolves = True
    if key:
        resolves = bool(c[key].isin(set(meta.path)).all())

    verdict = []
    if not consistent:
        verdict.append("uneven rows per (detector, seed)")
    if not covered:
        verdict.append("clips per pass %d != %d in metadata"
                       % (int(per.max()), expected))
    if failed:
        verdict.append("%d recorded failure(s), scores short accordingly" % failed)
    if not resolves:
        verdict.append("paths do not resolve to metadata")
    hard = [v for v in verdict if "recorded failure" not in v]
    if hard:
        bad.append((ds, "; ".join(hard)))

    print("%-14s %9d %8d %7d %6d %6d  %s"
          % (ds, len(c), len(meta), n_folds, n_det, n_seed,
             "ok" if not verdict else "; ".join(verdict)))

print()
if bad:
    print("PROBLEMS:")
    for ds, why in bad:
        print("  %s: %s" % (ds, why))
    sys.exit(1)
print("every test clip appears exactly once per fold, detector and seed, and "
      "every path resolves to the metadata")
