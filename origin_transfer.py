"""MAAD-Bench origin-transfer analysis, group-rotated.

Do detector rankings on DESIGNATED-CLASS tasks transfer to ORGANIC-LABEL tasks?

Three things changed since v1, and the first two are why the v1 answer was not
safe to lead with.

  1. Leave-one-organic-domain sensitivity. The locked design called for it and it
     was never run. It matters: the pooled correlation is a cancellation of
     domain-specific associations of opposite sign, so dropping one organic
     sub-dataset can move the conclusion across the significance line. Reported
     here by default.
  2. A confidence interval. A non-significant rho is a failure to reject zero,
     not evidence that the association is negligible, and the interval on 25
     detectors is wide enough to admit a moderate association in either
     direction.
  3. The protocol. Rankings now come from every native group in rotation rather
     than one fixed held-out group per sub-dataset, so a partition ranking is no
     longer hostage to which group was pre-registered.

Frame the result as transfer of detector SELECTION across benchmark partitions,
not as causal proof: the organic side is confounded with domain.

This is a thin front end over analysis.py, which holds the implementations.

Usage:
  python origin_transfer.py
  python origin_transfer.py --repr panns
"""
import argparse

import pandas as pd
from scipy.stats import spearmanr

from analysis import (DESIGNATED, ORGANIC, dataset_matrix, fold_table,
                      load_runs, transfer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="auc_roc")
    ap.add_argument("--repr", default="handcrafted")
    ap.add_argument("--setting", default="primary")
    args = ap.parse_args()

    runs = load_runs(args.repr, args.metric)
    mat = dataset_matrix(fold_table(runs, args.metric, args.setting))
    des_sets = [d for d in DESIGNATED if d in mat.columns]
    org_sets = [d for d in ORGANIC if d in mat.columns]
    print("designated:", des_sets)
    print("organic:   ", org_sets)
    if not org_sets:
        raise SystemExit("no organic sub-datasets present")

    full = transfer(mat)
    print("\n=== pooled transfer ===")
    print("Spearman rho = %.4f (p = %.4f), n = %d detectors, 95%% CI [%.3f, %.3f]"
          % (full["rho"], full["p"], full["n_detectors"], full["ci_lo"], full["ci_hi"]))
    if full["p"] > 0.05:
        print("  A non-significant rho is a failure to reject zero. The interval "
              "above is what the data actually exclude.")

    print("\n=== leave-one-organic-domain sensitivity ===")
    for drop in org_sets:
        keep = [o for o in org_sets if o != drop]
        if not keep:
            continue
        t = transfer(mat, organic=keep)
        flag = "  <-- crosses 0.05" if (t["p"] < 0.05) != (full["p"] < 0.05) else ""
        print("  drop %-12s rho = %+.4f (p = %.4f), CI [%.3f, %.3f]%s"
              % (drop, t["rho"], t["p"], t["ci_lo"], t["ci_hi"], flag))

    print("\n=== per organic sub-dataset against the designated ranking ===")
    des = mat[des_sets].mean(axis=1)
    for o in org_sets:
        r = spearmanr(des, mat[o])
        print("  %-12s rho = %+.4f (p = %.4f)" % (o, r.statistic, r.pvalue))
    print("  If these carry opposite signs, the pooled value is a cancellation "
          "rather than an absence of association; say so rather than reporting "
          "the pooled number alone.")

    org = mat[org_sets].mean(axis=1)
    tab = pd.DataFrame({"designated": des, "organic": org})
    tab["des_rank"] = tab["designated"].rank(ascending=False)
    tab["org_rank"] = tab["organic"].rank(ascending=False)
    tab["rank_shift"] = (tab["des_rank"] - tab["org_rank"]).abs()
    print("\n=== detector mean AUC: designated vs organic (sorted by designated) ===")
    print(tab.sort_values("designated", ascending=False).round(3).to_string())
    print("\n=== biggest rank shifts ===")
    print(tab.sort_values("rank_shift", ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
