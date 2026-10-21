"""Aggregates behind Section 4.6, read from tables/repr_numbers.csv and the loudness baseline of out/R1F.

Usage: python repr_summary.py <maad-r3 root>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(sys.argv[1])
rn = pd.read_csv(S / "paper-sensors" / "tables" / "repr_numbers.csv")
tb = pd.read_csv(S / "out" / "R1F" / "trivial_baselines.csv")
print("trivial_baselines columns:", tb.columns.tolist())
loud = tb[(tb.setting == "primary") & (tb.baseline == "rms0_abs_z")].set_index("dataset").auc_mean \
    if "baseline" in tb.columns else None

rn["emb_best"] = rn[["ast_best", "panns_best"]].max(axis=1)
rn["emb_median"] = rn[["ast_median", "panns_median"]].max(axis=1)
rn["lift_best"] = rn.emb_best - rn.hand_best
rn["lift_median"] = rn.emb_median - rn.hand_median
if loud is not None:
    rn["loud"] = rn.dataset.map(loud)
cols = ["dataset", "hand_best", "ast_best", "panns_best", "lift_best", "hand_median", "ast_median",
        "panns_median", "lift_median", "ae"] + (["loud"] if loud is not None else [])
print(rn[cols].round(3).to_string())
print()
print("embedding best > hand best on", int((rn.lift_best > 0).sum()), "of", len(rn),
      "| lift > 0.01 on", int((rn.lift_best > 0.01).sum()))
print("hand best > both embeddings on", list(rn.loc[rn.lift_best < 0, "dataset"]))
print("AST best > PANNs best on", int((rn.ast_best > rn.panns_best).sum()),
      "| PANNs > AST on", list(rn.loc[rn.panns_best > rn.ast_best, "dataset"]))
print("embedding median > hand median on", int((rn.lift_median > 0).sum()),
      "| lower on", list(rn.loc[rn.lift_median < 0, "dataset"]))
if loud is not None:
    for rep in ("hand", "ast", "panns"):
        print(f"loudness > {rep} median on", int((rn.loud > rn[f"{rep}_median"]).sum()),
              list(rn.loc[rn.loud > rn[f"{rep}_median"], "dataset"]))
    print("loudness > best of all three representations on",
          list(rn.loc[rn.loud > rn[["hand_best", "ast_best", "panns_best"]].max(axis=1), "dataset"]))
ae = rn.dropna(subset=["ae"])
print("AE anchor: n =", len(ae), "range", ae.ae.min().round(3), ae.ae.max().round(3),
      "| below 0.5 on", list(ae.loc[ae.ae < 0.5, "dataset"]))
print("AE > hand median on", int((ae.ae > ae.hand_median).sum()), "| AE > hand best on",
      list(ae.loc[ae.ae > ae.hand_best, "dataset"]),
      "| AE > all three bests on", list(ae.loc[ae.ae > ae[["hand_best", "ast_best", "panns_best"]].max(axis=1), "dataset"]))
