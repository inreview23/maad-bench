"""Audio-detector comparison tables for the MAAD-Bench paper. Emits LaTeX row
bodies and key statistics from the per-dataset metrics CSVs:
  1. representation: best primary AUC per sub-dataset under each backbone
     (Handcrafted / PANNs / AST / CLAP / TWFR), best per row in bold + detector.
  2. per-backbone origin-transfer Spearman rho (designated vs organic), to show
     whether a stronger representation closes the transfer gap.
  3. temporal vs tabular: best tabular AUC vs the temporal detectors per dataset.
Run from anywhere with the project python.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb
from analysis import guard_mixed_protocols  # noqa: E402
guard_mixed_protocols()


R = mb.RESULTS_DIR
DES = ["esc50", "urbansound8k", "ravdess", "nsynth"]
ORG = ["mimii", "physionet", "icbhi"]
ORDER = DES + ORG
LABEL = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "ravdess": "RAVDESS",
         "nsynth": "NSynth", "mimii": "MIMII", "physionet": "PhysioNet", "icbhi": "ICBHI"}
BACK = [("Handcrafted", "metrics.csv"), ("PANNs", "panns_metrics.csv"),
        ("AST", "ast_metrics.csv"), ("CLAP", "clap_metrics.csv"), ("TWFR", "twfr_metrics.csv")]


def load(fn):
    out = {}
    for ds in ORDER:
        p = os.path.join(R, ds, fn)
        if os.path.exists(p):
            t = pd.read_csv(p)
            t = t[t["setting"] == "primary"]
            if len(t):
                out[ds] = t
    return out


data = {lab: load(fn) for lab, fn in BACK}
labs = [lab for lab, _ in BACK if data[lab]]

print("%% ===== representation: best primary AUC per backbone; cols: %s =====" % " & ".join(labs))
for ds in ORDER:
    vals = {}
    for lab in labs:
        t = data[lab].get(ds)
        if t is not None:
            b = t.loc[t["auc_roc_mean"].idxmax()]
            vals[lab] = (float(b["auc_roc_mean"]), str(b["detector"]))
    best = max((v for v, _ in vals.values()), default=None)
    cells = []
    for lab in labs:
        if lab not in vals:
            cells.append("--")
            continue
        v, det = vals[lab]
        s = "\\textbf{%.3f}" % v if best is not None and abs(v - best) < 1e-9 else "%.3f" % v
        cells.append("%s {\\scriptsize %s}" % (s, det))
    print("%-13s & %s \\\\" % (LABEL[ds], " & ".join(cells)))

print("\n%% ===== per-backbone origin-transfer Spearman rho (designated vs organic, primary) =====")
for lab, fn in BACK:
    d = load(fn)
    if not d:
        continue
    rows = []
    for ds, t in d.items():
        for _, r in t.iterrows():
            rows.append((r["detector"], ds, r["auc_roc_mean"]))
    tab = pd.DataFrame(rows, columns=["detector", "dataset", "auc"])
    desv = tab[tab.dataset.isin(DES)].groupby("detector")["auc"].mean()
    orgv = tab[tab.dataset.isin(ORG)].groupby("detector")["auc"].mean()
    common = sorted(set(desv.index) & set(orgv.index))
    nd = sum(1 for x in DES if x in d)
    no = sum(1 for x in ORG if x in d)
    if len(common) >= 3 and nd >= 2 and no >= 2:
        rho, p = spearmanr(desv[common].values, orgv[common].values)
        print("%%   %-12s rho=%+.3f p=%.3f  (det=%d, des=%d/4, org=%d/3)" %
              (lab, rho, p, len(common), nd, no))
    else:
        print("%%   %-12s incomplete (des=%d org=%d)" % (lab, nd, no))

print("\n%% ===== temporal vs tabular: best primary AUC per sub-dataset =====")
hand = load("metrics.csv")
temp = load("temporal_metrics.csv")
print("%% Sub-dataset & Best tabular (det) & TempSR & TempIF")
for ds in ORDER:
    ht, tt = hand.get(ds), temp.get(ds)
    if ht is not None:
        hb = ht.loc[ht["auc_roc_mean"].idxmax()]
        bt, btd = float(hb["auc_roc_mean"]), str(hb["detector"])
    else:
        bt, btd = float("nan"), "--"

    def pick(name):
        if tt is not None and (tt["detector"] == name).any():
            return float(tt[tt["detector"] == name]["auc_roc_mean"].iloc[0])
        return float("nan")

    print("%-13s & %.3f {\\scriptsize %s} & %.3f & %.3f \\\\" %
          (LABEL[ds], bt, btd, pick("TempSR"), pick("TempIF")))
