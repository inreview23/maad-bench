"""Print per-dataset clip and anomaly counts plus group structure from the
metadata.parquet files, to fill the MAAD-Bench sub-dataset description table.
"""
import glob
import os

import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

for p in sorted(glob.glob(os.path.join(RESULTS, "*", "metadata.parquet"))):
    ds = os.path.basename(os.path.dirname(p))
    d = pd.read_parquet(p)
    cols = list(d.columns)
    label_col = next((x for x in ("anomaly_label", "label", "y") if x in cols), None)
    n = len(d)
    na = int(d[label_col].sum()) if label_col else -1
    gv = sorted(d["group"].unique().tolist()) if "group" in cols else []
    anom_per_group = d.groupby("group")[label_col].agg(["size", "sum"]).to_dict("index") if label_col else {}
    print("%-13s total=%4d anomaly=%4d normal=%4d" % (ds, n, na, n - na))
    print("    groups=%s" % gv)
    print("    per-group (size, anomalies)=%s" % {g: (v["size"], int(v["sum"])) for g, v in anom_per_group.items()})
