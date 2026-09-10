"""Run one fold twice, in two separate processes, and print every detector's AUC.

The temporal arm turned out not to reproduce because its isolation forest drew
from the global NumPy RNG. Two detectors in the main roster are in the same
position: the factory passes random_state by signature introspection, and LODA
and LUNAR expose no such attribute while both show a nonzero between-seed
standard deviation, so both are drawing from somewhere the factory does not
reach. This prints what a second invocation gets, so the two runs can be diffed.

Usage:
  python repro_probe.py esc50 > /tmp/r1.txt
  python repro_probe.py esc50 > /tmp/r2.txt
  diff /tmp/r1.txt /tmp/r2.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import maad_bench as mb
from rotate import load_tables

name = sys.argv[1] if len(sys.argv) > 1 else "esc50"
df, feat = load_tables(name)
group = sorted(df["group"].unique().tolist())[0]

runs, _ = mb.run_rotation(df, feat, groups=[group], settings=("primary",),
                          subsample_test=False, collect_scores_for=())
r = runs[runs.detector != "*"]
for (det, seed), g in r.groupby(["detector", "seed"]):
    print("%-12s seed=%d  auc=%.12f" % (det, seed, float(g.auc_roc.iloc[0])))
