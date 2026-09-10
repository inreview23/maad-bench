"""Extend the LODA re-run to the representation arms, the sweep and the ablation.

The core arm is done. These three reach LODA through the same `_sweep_split`, so
they carry the same order-dependence and the same fix applies. Two of them keep
no per-run file, only a summary, but `summarize_rotation` groups by
(setting, detector, group), so one detector's summary row depends on that
detector alone and replacing it in place is exact rather than approximate.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import maad_bench as mb
from rotate import ALL, load_tables
from reseed_loda import DET, loda_rows, splice

R = mb.RESULTS_DIR


def do_repr(backbones=("panns", "ast", "twfr", "clap"), pca_dim=64):
    from rotate_repr import load_embedding
    total = 0
    for backbone in backbones:
        for name in ALL:
            rp = os.path.join(R, name, "rot_%s_runs.parquet" % backbone)
            sp = os.path.join(R, name, "rot_%s_metrics.csv" % backbone)
            if not os.path.exists(rp):
                continue
            df, emb, _ = load_embedding(name, backbone)
            groups = sorted(df["group"].unique().tolist())
            pre = mb.make_pca_pre(pca_dim) if pca_dim > 0 else None
            rows = loda_rows(df, emb, groups, ("primary", "secondary"), pre=pre)
            old = pd.read_parquet(rp)
            keep = old[old.detector != DET]
            rows = rows.copy()
            rows["dataset"] = name
            rows["representation"] = backbone
            for col in old.columns:
                if col not in rows.columns:
                    rows[col] = np.nan
            merged = pd.concat([keep, rows[old.columns]], ignore_index=True)
            merged.to_parquet(rp, index=False)
            summ = mb.summarize_rotation(merged, name)
            if "representation" not in summ.columns:
                summ.insert(1, "representation", backbone)
            summ.to_csv(sp, index=False)
            total += len(rows)
            print("[%s/%s] %d LODA rows" % (name, backbone, len(rows)), flush=True)
    print("representation arms: %d rows" % total)


def patch_summary_row(path, new_row, keyfields):
    """Replace one detector's row in a summary CSV that has no run file behind it."""
    d = pd.read_csv(path)
    mask = d.detector == DET
    for k, v in keyfields.items():
        if k in d.columns:
            mask &= (d[k] == v)
    assert mask.sum() == 1, "expected exactly one %s row in %s, found %d" % (
        DET, os.path.basename(path), mask.sum())
    for col, val in new_row.items():
        if col in d.columns:
            d.loc[mask, col] = val
    d.to_csv(path, index=False)


def do_sweep(levels=(0.05, 0.10, 0.20)):
    total = 0
    for name in ALL:
        df, feat = load_tables(name)
        groups = sorted(df["group"].unique().tolist())
        for c in levels:
            p = os.path.join(R, name, "rot_sweep_c%02d.csv" % int(round(c * 100)))
            if not os.path.exists(p):
                continue
            rows = loda_rows(df, feat, groups, ("secondary",), contamination=c)
            summ = mb.summarize_rotation(rows, name)
            row = summ[summ.detector == DET].iloc[0].to_dict()
            patch_summary_row(p, row, {"setting": "secondary"})
            total += len(rows)
            print("[%s @ %.2f] %d LODA rows" % (name, c, len(rows)), flush=True)
    print("sweep: %d rows" % total)


def do_ablation():
    from ablation_features import FEATURE_SETS, cols_for
    total = 0
    p = os.path.join(R, "rot_ablation_features.csv")
    if not os.path.exists(p):
        print("no ablation file")
        return
    d = pd.read_csv(p)
    for name in sorted(d.dataset.unique()):
        df, feat = load_tables(name)
        groups = sorted(df["group"].unique().tolist())
        for setname in FEATURE_SETS:
            sub = cols_for(feat, FEATURE_SETS[setname])
            rows = loda_rows(df, sub, groups, ("primary",))
            summ = mb.summarize_rotation(rows, name)
            row = summ[summ.detector == DET].iloc[0].to_dict()
            mask = (d.dataset == name) & (d.feature_set == setname) & (d.detector == DET)
            assert mask.sum() == 1, "ablation: %d rows for %s/%s" % (
                mask.sum(), name, setname)
            for col, val in row.items():
                if col in d.columns and col not in ("dataset", "feature_set", "detector"):
                    d.loc[mask, col] = val
            total += len(rows)
            print("[%s | %s] %d LODA rows" % (name, setname, len(rows)), flush=True)
    d.to_csv(p, index=False)
    print("ablation: %d rows" % total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["repr", "sweep", "ablation"])
    a = ap.parse_args()
    {"repr": do_repr, "sweep": do_sweep, "ablation": do_ablation}[a.arm]()
