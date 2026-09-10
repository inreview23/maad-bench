"""Per-detector runtime for MAAD-Bench.

Times fit + decision_function for each detector on each sub-dataset (primary
split, seed 0) reusing the CACHED features, so no audio is re-read. Writes
results/runtime.csv with [dataset, detector, seconds, n_train, n_features]. This
gives the classical-vs-neural cost tradeoff for the paper. Neural detectors use
the GPU, so run this from the repository root when the GPU is free.
"""
import os
import sys
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

NAMES = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]

rows = []
facs = mb.detector_factories()
for name in NAMES:
    d = os.path.join(mb.RESULTS_DIR, name)
    feat = pd.read_parquet(os.path.join(d, "features.parquet"))
    df = pd.read_parquet(os.path.join(d, "metadata.parquet"))
    X = feat.to_numpy(dtype=float)
    tr, te = mb.make_split(df, 0, "primary", mb.DATASETS[name]["test_groups"])
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    for dname, make in facs.items():
        try:
            det = make(mb.CONTAMINATION, 0)
            t0 = time.perf_counter()
            det.fit(Xtr)
            det.decision_function(Xte)
            sec = time.perf_counter() - t0
        except Exception as exc:
            print("[%s] %s FAILED %s" % (name, dname, type(exc).__name__), flush=True)
            sec = float("nan")
        rows.append({"dataset": name, "detector": dname, "seconds": sec,
                     "n_train": len(tr), "n_features": X.shape[1]})
        print("[%s] %-12s %.3fs" % (name, dname, sec), flush=True)

pd.DataFrame(rows).to_csv(os.path.join(mb.RESULTS_DIR, "runtime.csv"), index=False)
print("wrote runtime.csv", flush=True)
