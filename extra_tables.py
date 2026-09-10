"""Robustness (contamination), runtime, and representation (PANNs) table bodies
for the MAAD-Bench paper. Reads the cached result CSVs under results/. Uses only
pandas + scipy. Paths are __file__-relative via maad_bench, so run from anywhere:
  python extra_tables.py
"""
import glob
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

R = mb.RESULTS_DIR
DESIGNATED = ["esc50", "urbansound8k", "ravdess", "nsynth"]
ORGANIC = ["mimii", "physionet", "icbhi"]
ORDER = DESIGNATED + ORGANIC
LABEL = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "ravdess": "RAVDESS",
         "nsynth": "NSynth", "mimii": "MIMII", "physionet": "PhysioNet", "icbhi": "ICBHI"}


def _guard_mixed_protocols():
    """Refuse to emit a table that blends the two protocols.

    These files sit side by side under results/: metrics.csv and
    sweep_combined.csv come from the fixed-group protocol, rot_metrics.csv and
    rot_sweep_combined.csv from the rotation. Reading the v1 pair while the
    rotated pair exists would print a LaTeX body with no error at all, which is
    worse than a crash, so it is an error here.
    """
    rotated = glob.glob(os.path.join(R, "*", "rot_metrics.csv"))
    if rotated:
        raise SystemExit(
            "results/ holds rotated outputs (%d sub-datasets) as well as the v1 "
            "files this script reads. Emitting v1 tables next to rotated ones "
            "risks mixing protocols in the manuscript. Use make_table.py and "
            "analysis.py, which read the rotated results, or move the rot_* "
            "files aside deliberately." % len(rotated))


def load_metrics():
    rows = [pd.read_csv(p) for p in glob.glob(os.path.join(R, "*", "metrics.csv"))]
    return pd.concat(rows, ignore_index=True)


_guard_mixed_protocols()

# ---------- robustness: AUC invariant, AP scales with contamination ----------
main = load_metrics()
main_sec = main[main["setting"] == "secondary"].copy()
main_sec["contamination"] = 0.10
sweep = pd.read_csv(os.path.join(R, "sweep_combined.csv"))
sweep_sec = sweep[sweep["setting"] == "secondary"].copy()
allc = pd.concat([main_sec, sweep_sec], ignore_index=True)
print("%% ===== robustness: secondary, mean over 25 detectors x 7 datasets =====")
print("%% level & AUC-ROC & AP & F1@top-q")
for c in [0.05, 0.10, 0.20]:
    s = allc[(allc["contamination"] - c).abs() < 1e-6]
    print("%d\\%% & %.3f & %.3f & %.3f \\\\" % (
        int(round(c * 100)), s["auc_roc_mean"].mean(),
        s["avg_precision_mean"].mean(), s["f1_top_q_mean"].mean()))

# ---------- runtime: per-detector fit+score seconds ----------
rt = pd.read_csv(os.path.join(R, "runtime.csv"))
med = rt.groupby("detector")["seconds"].agg(["median", "min", "max"]).sort_values("median")
print("\n%% ===== runtime: fit+score seconds, median/min/max across 7 datasets =====")
print("%% detector & median & min & max")
for det, row in med.iterrows():
    print("%-12s & %.3f & %.3f & %.3f \\\\" % (det, row["median"], row["min"], row["max"]))

# ---------- representation: handcrafted-74 vs PANNs-64 ----------
hand = main[main["setting"] == "primary"]
prows = []
for p in glob.glob(os.path.join(R, "*", "panns_metrics.csv")):
    ds = os.path.basename(os.path.dirname(p))
    t = pd.read_csv(p)
    t["dataset"] = ds
    prows.append(t)
pann = pd.concat(prows, ignore_index=True)
pann = pann[pann["setting"] == "primary"]
print("\n%% ===== representation: best primary AUC, handcrafted-74 vs PANNs-64 =====")
print("%% dataset & hand AUC & hand det & panns AUC & panns det")
for ds in ORDER:
    h = hand[hand["dataset"] == ds]
    q = pann[pann["dataset"] == ds]
    if len(h) == 0 or len(q) == 0:
        print("%% %s missing (hand=%d panns=%d)" % (ds, len(h), len(q)))
        continue
    hb = h.loc[h["auc_roc_mean"].idxmax()]
    qb = q.loc[q["auc_roc_mean"].idxmax()]
    print("%-13s & %.3f & %s & %.3f & %s \\\\" % (
        LABEL[ds], hb["auc_roc_mean"], hb["detector"], qb["auc_roc_mean"], qb["detector"]))


def pmean(names, frame):
    return frame[frame["dataset"].isin(names)].groupby("detector")["auc_roc_mean"].mean()


pdes = pmean(DESIGNATED, pann)
porg = pmean(ORGANIC, pann)
common = sorted(set(pdes.index) & set(porg.index))
if len(common) >= 3:
    rho, pv = spearmanr(pdes[common].values, porg[common].values)
    print("%% PANNs origin-transfer Spearman rho=%.3f p=%.3f (n=%d)" % (rho, pv, len(common)))
hand_mean_org = hand[hand["dataset"].isin(ORGANIC)].groupby("dataset")["auc_roc_mean"].max()
pann_mean_org = pann[pann["dataset"].isin(ORGANIC)].groupby("dataset")["auc_roc_mean"].max()
print("%% organic best-AUC hand vs panns:")
for ds in ORGANIC:
    if ds in hand_mean_org.index and ds in pann_mean_org.index:
        print("%%   %s hand=%.3f panns=%.3f" % (ds, hand_mean_org[ds], pann_mean_org[ds]))
