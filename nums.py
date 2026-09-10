"""Print the best primary AUC and its detector per sub-dataset for each arm.

Fills the representation and temporal table cells. Reads the group-rotated
outputs; set MAAD_NUMS_SOURCE=v1 to read the fixed-group ones instead.

Two v1 problems fixed here. RES was the relative string "results", so the script
silently read nothing unless run from the repository root. And the judge
breakdown is gone with Section VII, whose 10/10 and 19/19 came from a prompt that
named the normal classes and then asked the model to classify those same names.
"""
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DS = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
_V1 = os.environ.get("MAAD_NUMS_SOURCE", "rotated").lower() == "v1"
_P = "" if _V1 else "rot_"
ARMS = {"handcrafted": "%smetrics.csv" % _P,
        "panns": "%spanns_metrics.csv" % _P,
        "ast": "%sast_metrics.csv" % _P,
        "clap": "%sclap_metrics.csv" % _P,
        "twfr": "%stwfr_metrics.csv" % _P,
        "temporal": "%stemporal_metrics.csv" % _P}


def best(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d["setting"] == "primary"]
    if d.empty:
        return None
    r = d.loc[d["auc_roc_mean"].idxmax()]
    sd = r["auc_roc_fold_sd"] if "auc_roc_fold_sd" in d.columns else float("nan")
    folds = int(r["n_folds"]) if "n_folds" in d.columns else 1
    return str(r["detector"]), float(r["auc_roc_mean"]), float(sd), folds


print("source: %s results" % ("v1 fixed-group" if _V1 else "group-rotated"))
print("%-14s %-12s %-22s %-22s %-22s %-22s %-22s"
      % ("dataset", "", *ARMS.keys()))
for ds in DS:
    cells = []
    for arm, fn in ARMS.items():
        b = best(os.path.join(RES, ds, fn))
        cells.append("--" if b is None
                     else "%s %.3f%s" % (b[0], b[1],
                                         "" if b[3] == 1 else " (+-%.3f, %df)" % (b[2], b[3])))
    print("%-14s %-12s %s" % (ds, "", "  ".join("%-22s" % c for c in cells)))
