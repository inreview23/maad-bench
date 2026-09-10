"""Assemble rot_sweep_combined.csv from the per-sub-dataset sweep files.

contam_sweep.py writes results/<ds>/rot_sweep_c{05,10,20}.csv as it goes and the
combined file only at the end, so a run split across invocations never produces
the combined file.

The freshness guard is the point of this script rather than an extra. The first
sweep ran every level at 0.10 because the level did not cross the joblib process
boundary, and it left 21 wrong parts on disk under the right names. Assembling by
name alone silently mixes those with the corrected ones. So a part older than
--since is refused, and the default is the moment the fix landed.
"""
import argparse
import datetime as dt
import os
import sys

import pandas as pd

ORDER = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
LEVELS = [5, 10, 20]
HERE = os.path.dirname(os.path.abspath(__file__))
# the contamination fix landed at 09:05 on 2026-08-28; anything older ran at 0.10
DEFAULT_SINCE = "2026-08-28 09:00"

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(HERE, "results"))
ap.add_argument("--since", default=DEFAULT_SINCE,
                help="refuse parts older than this local timestamp")
args = ap.parse_args()

cutoff = dt.datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp()

parts, missing, stale = [], [], []
for ds in ORDER:
    for c in LEVELS:
        p = os.path.join(args.results, ds, "rot_sweep_c%02d.csv" % c)
        tag = "%s@%d%%" % (ds, c)
        if not os.path.exists(p):
            missing.append(tag)
            continue
        if os.path.getmtime(p) < cutoff:
            stale.append("%s (%s)" % (tag, dt.datetime.fromtimestamp(
                os.path.getmtime(p)).strftime("%m-%d %H:%M")))
            continue
        d = pd.read_csv(p)
        if "dataset" not in d.columns:
            d.insert(0, "dataset", ds)
        if "train_contamination" not in d.columns:
            d["train_contamination"] = c / 100.0
        parts.append(d)

n_all = len(ORDER) * len(LEVELS)
print("fresh %d of %d parts (cutoff %s)" % (len(parts), n_all, args.since))
if stale:
    print("STALE, refused (%d): %s" % (len(stale), ", ".join(stale)))
if missing:
    print("MISSING (%d): %s" % (len(missing), ", ".join(missing)))
if not parts:
    sys.exit(1)

out = pd.concat(parts, ignore_index=True)
print(out.groupby(["dataset", "train_contamination"]).size()
      .unstack(fill_value=0).to_string())

if stale or missing:
    print("\nnot written: %d parts still to come" % (len(stale) + len(missing)))
    sys.exit(1)

p = os.path.join(args.results, "rot_sweep_combined.csv")
out.to_csv(p, index=False)
print("\nwrote %s (%d rows)" % (p, len(out)))
