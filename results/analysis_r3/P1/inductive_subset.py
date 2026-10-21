"""Panel follow-up: headline summaries over the 20 detectors that score each clip against the fitted training
model alone, excluding COF, SOD and LMDD (scored from the test batch only in PyOD 3.5.1) and ECOD and COPOD
(empirical distributions computed on training and test clips together)."""
import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1])
BATCH = ["COF", "SOD", "LMDD", "ECOD", "COPOD"]
m = pd.read_csv(S / "out/R1F/appendix_auc_matrix_primary_numeric.csv", index_col=0)
m = m[[c for c in m.columns if c.endswith("_mean")]]
m.columns = [c[:-5] for c in m.columns]
tb = pd.read_csv(S / "out/R1F/trivial_baselines.csv")
loud = tb[(tb.setting == "primary") & (tb.baseline == "rms0_abs_z")].set_index("dataset").auc_mean
sig = pd.read_csv(S / "out/R1F/sig_tests_chance.csv")
sig = sig[sig.setting == "primary"]
null = pd.read_csv(S / "out/R1F/null_max25.csv")
print("null columns:", null.columns.tolist())
rows = []
for ds in m.columns:
    full, sub = m[ds], m[ds].drop(BATCH)
    s_full = sig[sig.dataset == ds]
    s_sub = s_full[~s_full.detector.isin(BATCH)]
    rows.append(dict(dataset=ds, best25=full.max(), det25=full.idxmax(), best20=sub.max(), det20=sub.idxmax(),
                     med25=full.median(), med20=sub.median(), loud=loud[ds],
                     loud_gt_med25=loud[ds] > full.median(), loud_gt_med20=loud[ds] > sub.median(),
                     loud_beats_all20=bool((loud[ds] > sub).all()),
                     nb_above25=int(s_full.nb_sig_above.sum()), nb_above20=int(s_sub.nb_sig_above.sum())))
r = pd.DataFrame(rows)
r.to_csv(S / "out/P1/inductive_subset.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string())
print("loudness > median: 25 ->", int(r.loud_gt_med25.sum()), "| 20 ->", int(r.loud_gt_med20.sum()))
print("loudness beats all 20 on", list(r.loc[r.loud_beats_all20, "dataset"]))
print("tasks with an NB-Holm detector above chance: 25 ->", int((r.nb_above25 > 0).sum()), "| 20 ->", int((r.nb_above20 > 0).sum()))
print("best changes on", list(r.loc[r.det25 != r.det20, "dataset"]), "max drop", (r.best25 - r.best20).max().round(3))
