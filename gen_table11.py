"""Rebuild Table 11 from the corrected training-contamination sweep.

The submitted table swept a knob that moved the test-side positive rate, so its
story was that AUC-ROC is stable while AP and F1 scale with prevalence. Under the
rotated protocol the test split keeps every anomaly in the held-out group, so the
level no longer touches the test side at all. What it does move is the training
injection, the contamination hyperparameter each detector is given, and the
top-q flag budget. AP should therefore be roughly flat rather than rising, and
whatever AUC-ROC does is a statement about robustness to a mis-set training
assumption rather than about prevalence invariance.

Run this on the DGX once contam_sweep.py finishes, then paste the printed rows.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
P = os.path.join(D, "rot_sweep_combined.csv")
# The first sweep ran every level at 0.10 and left files under the right names,
# so age is the only thing separating a valid combined file from a wrong one.
CUTOFF = 1787932800.0  # 2026-08-28 09:00 local, when the contamination fix landed
if not os.path.exists(P):
    print("no rot_sweep_combined.csv yet"); sys.exit(1)
if os.path.getmtime(P) < CUTOFF:
    print("REFUSED: rot_sweep_combined.csv predates the contamination fix"); sys.exit(1)
S = pd.read_csv(P)
ORDER = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]

missing = sorted(set(ORDER) - set(S.dataset.unique()))
if missing:
    print("INCOMPLETE: no rows for %s" % ", ".join(missing))
    sys.exit(1)

METS = ["auc_roc_mean", "avg_precision_mean", "pauc_0.1_mean", "f1_top_q_mean"]
LEVELS = sorted(S.train_contamination.unique())

# mean over detectors within a sub-dataset first, then over sub-datasets, so a
# sub-dataset with more folds does not weigh more than one with fewer
per_ds = S.groupby(["train_contamination", "dataset"])[METS].mean()
grand = per_ds.groupby("train_contamination").mean()

print("=== Table 11: mean over 25 detectors and 7 sub-datasets ===")
print(grand.round(3).to_string())
print()
print("=== LaTeX rows ===")
for c in LEVELS:
    r = grand.loc[c]
    print("%d\\%%  & %.3f & %.3f & %.3f \\\\"
          % (round(c * 100), r["auc_roc_mean"], r["avg_precision_mean"],
             r["f1_top_q_mean"]))

print()
print("=== spread across the range, per metric ===")
for m in METS:
    v = grand[m]
    print("  %-20s %.3f to %.3f   span %.3f"
          % (m, v.min(), v.max(), v.max() - v.min()))

print()
print("=== per sub-dataset AUC-ROC, so the prose can name the worst case ===")
a = per_ds["auc_roc_mean"].unstack(0)[LEVELS].loc[ORDER]
a["span"] = a[LEVELS[-1]] - a[LEVELS[0]]
print(a.round(3).to_string())

print()
print("=== best detector per sub-dataset at each level, and whether it changes ===")
for ds in ORDER:
    picks = []
    for c in LEVELS:
        d = S[(S.dataset == ds) & (S.train_contamination == c)]
        picks.append(d.set_index("detector").auc_roc_mean.idxmax())
    print("  %-13s %s%s" % (ds, " -> ".join(picks),
                            "" if len(set(picks)) == 1 else "   CHANGES"))

print()
print("=== rank stability of the roster across levels (Spearman) ===")
from scipy import stats
for ds in ORDER:
    cols = {}
    for c in LEVELS:
        d = S[(S.dataset == ds) & (S.train_contamination == c)]
        cols[c] = d.set_index("detector").auc_roc_mean
    f = pd.DataFrame(cols)
    rho = stats.spearmanr(f[LEVELS[0]], f[LEVELS[-1]]).statistic
    print("  %-13s rho(%.2f, %.2f) = %.3f" % (ds, LEVELS[0], LEVELS[-1], rho))
