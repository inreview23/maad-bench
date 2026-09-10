"""The DCASE autoencoder anchor, rotated.

Its seven values were reported only inside the submitted Table 4 caption.
Rewriting that caption removed the paper's one report of a baseline the abstract,
a contribution bullet and the conclusion all still claim, so it has to go back.

This prints the numbers and the comparison against the published fixed-fold ones,
so the sentence that goes into the manuscript is written from data rather than
from the earlier draft.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
ORDER = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
NAME = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "ravdess": "RAVDESS",
        "nsynth": "NSynth", "mimii": "MIMII", "physionet": "PhysioNet",
        "icbhi": "ICBHI"}
# the submitted caption, fixed fold, five seeds
PUB = {"esc50": 0.589, "urbansound8k": 0.768, "ravdess": 0.902, "nsynth": 0.497,
       "mimii": 0.535, "physionet": 0.298, "icbhi": 0.487}

rows, missing = [], []
for ds in ORDER:
    p = os.path.join(D, ds, "ae_rot_metrics.csv")
    if not os.path.exists(p):
        missing.append(ds)
        continue
    d = pd.read_csv(p)
    d = d[d.setting == "primary"] if "setting" in d.columns else d
    r = d.iloc[0]
    rows.append((ds, float(r.auc_roc_mean), float(r.auc_roc_fold_sd), int(r.n_folds)))

if missing:
    print("INCOMPLETE: %s" % ", ".join(missing))

print("%-13s %-22s %-10s %s" % ("sub-dataset", "rotated", "published", "delta"))
for ds, mu, sd, n in rows:
    print("%-13s %.3f +/- %.3f (n=%2d)  %.3f      %+.3f"
          % (NAME[ds], mu, sd, n, PUB[ds], mu - PUB[ds]))

if len(rows) < len(ORDER):
    sys.exit(1)

print()
print("=" * 70)
print("the sentence, with the numbers in sub-dataset order")
print("=" * 70)
vals = ", ".join("%.3f" % mu for _, mu, _, _ in rows)
print("reaches primary AUC-ROC %s on ESC-50, UrbanSound8K, RAVDESS," % vals)
print("NSynth, MIMII, PhysioNet, and ICBHI")

print()
best = max(rows, key=lambda r: r[1])
worst = min(rows, key=lambda r: r[1])
print("strongest: %s at %.3f" % (NAME[best[0]], best[1]))
print("weakest:   %s at %.3f" % (NAME[worst[0]], worst[1]))
solvable = {"esc50", "urbansound8k", "ravdess", "mimii"}
n_above = sum(1 for ds, mu, _, _ in rows if mu >= 0.60)
print("clears 0.60 on %d of 7" % n_above)
print("beats the best clip-level detector on: %s"
      % ([NAME[ds] for ds, mu, _, _ in rows] and "see below"))
