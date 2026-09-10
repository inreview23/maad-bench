"""Print per-detector mean primary AUC over designated-class vs organic-label
sub-datasets, for embedding as native pgfplots scatter coordinates."""
import os
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DES = ["esc50", "urbansound8k", "ravdess", "nsynth"]
ORG = ["mimii", "physionet", "icbhi"]

# Group-rotated results by default; MAAD_SCATTER_SOURCE=v1 for the
# fixed-group ones. Mixing the two in one figure would be silent.
SRC = ("metrics.csv" if os.environ.get("MAAD_SCATTER_SOURCE", "").lower() == "v1"
       else "rot_metrics.csv")

frames = []
for ds in DES + ORG:
    d = pd.read_csv(os.path.join(RES, ds, SRC))
    d = d[d["setting"] == "primary"][["detector", "auc_roc_mean"]].copy()
    d["dataset"] = ds
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
des = df[df.dataset.isin(DES)].groupby("detector")["auc_roc_mean"].mean()
org = df[df.dataset.isin(ORG)].groupby("detector")["auc_roc_mean"].mean()
common = sorted(set(des.index) & set(org.index))
for d in common:
    print("%s %.3f %.3f" % (d, des[d], org[d]))
