"""Rebuild Table 8 from the seeded temporal run, and test the three claims the
prose makes about it.

Those claims are: the clip-level summary wins on every task either approach
solves, respiratory sound is the single row where the temporal arm leads, and the
closest gap is PhysioNet at 0.025. The isolation forest was unseeded before, so
all three were read off a draw that does not reproduce, and ICBHI in particular
sat at 0.505 against a clip-level 0.486 with nothing holding it there.
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
SEEDED_AFTER = 1787938800.0  # 2026-08-28 10:40 local, when the seed fix landed
SOLVABLE = {"esc50", "urbansound8k", "ravdess", "mimii"}
# what the manuscript currently prints
OLD = {"esc50": (0.652, "TempSR"), "urbansound8k": (0.619, "TempIF"),
       "ravdess": (0.909, "TempIF"), "nsynth": (0.549, "TempIF"),
       "mimii": (0.456, "TempSR"), "physionet": (0.554, "TempIF"),
       "icbhi": (0.505, "TempIF")}

rows = []
for ds in ORDER:
    tp = os.path.join(D, ds, "rot_temporal_metrics.csv")
    cp = os.path.join(D, ds, "rot_metrics.csv")
    if not os.path.exists(tp):
        print("INCOMPLETE: no temporal result for %s" % ds)
        sys.exit(1)
    # Refuse anything from before the isolation forest was seeded: those files
    # sit under the right names and differ by up to 0.009 on TempIF.
    if os.path.getmtime(tp) < SEEDED_AFTER:
        import datetime as _dt
        print("STALE: %s temporal result is from %s, before the seed fix"
              % (ds, _dt.datetime.fromtimestamp(os.path.getmtime(tp))
                 .strftime("%m-%d %H:%M")))
        sys.exit(1)
    t = pd.read_csv(tp)
    t = t[t.setting == "primary"] if "setting" in t.columns else t
    ti = t.auc_roc_mean.idxmax()
    c = pd.read_csv(cp)
    c = c[c.setting == "primary"]
    ci = c.auc_roc_mean.idxmax()
    rows.append({
        "ds": ds,
        "t_auc": float(t.loc[ti, "auc_roc_mean"]),
        "t_sd": float(t.loc[ti, "auc_roc_fold_sd"]),
        "t_det": t.loc[ti, "detector"],
        "c_auc": float(c.loc[ci, "auc_roc_mean"]),
        "c_det": c.loc[ci, "detector"],
    })

print("%-13s %-22s %-24s %s" % ("sub-dataset", "temporal (seeded)", "clip-level", "was"))
for r in rows:
    print("%-13s %.3f +/- %.3f (%-7s) %.3f (%-11s) %s %.3f (%s)"
          % (NAME[r["ds"]], r["t_auc"], r["t_sd"], r["t_det"],
             r["c_auc"], r["c_det"],
             " " * 2, OLD[r["ds"]][0], OLD[r["ds"]][1]))

print()
print("=" * 70)
print("LaTeX rows for tab:temporal")
print("=" * 70)
for r in rows:
    tcell = "%.3f (%s)" % (r["t_auc"], r["t_det"])
    ccell = "%.3f (%s)" % (r["c_auc"], r["c_det"])
    if r["t_auc"] > r["c_auc"]:
        tcell = "\\textbf{%s}" % tcell
    else:
        ccell = "\\textbf{%s}" % ccell
    print("%-13s& %s & %s \\\\" % (NAME[r["ds"]], tcell, ccell))

print()
print("=" * 70)
print("the three claims")
print("=" * 70)
leads = [r["ds"] for r in rows if r["t_auc"] > r["c_auc"]]
print("1. temporal leads on: %s" % (", ".join(NAME[d] for d in leads) or "nothing"))
solved_lead = [d for d in leads if d in SOLVABLE]
print("   clip-level wins on every solvable task: %s"
      % ("YES" if not solved_lead else "NO, temporal leads " + str(solved_lead)))
print("2. the single leading row is respiratory sound: %s"
      % ("YES" if leads == ["icbhi"] else "NO, leads = %s" % leads))
gaps = {r["ds"]: r["c_auc"] - r["t_auc"] for r in rows if r["ds"] not in leads}
if gaps:
    tightest = min(gaps, key=gaps.get)
    widest = max(gaps, key=gaps.get)
    print("3. closest clip-level lead: %s at %.3f  (widest: %s at %.3f)"
          % (NAME[tightest], gaps[tightest], NAME[widest], gaps[widest]))
icbhi = [r for r in rows if r["ds"] == "icbhi"][0]
print()
print("ICBHI, the row at risk: temporal %.3f +/- %.3f against clip-level %.3f"
      % (icbhi["t_auc"], icbhi["t_sd"], icbhi["c_auc"]))
lo = icbhi["t_auc"] - 1.96 * icbhi["t_sd"] / (5 ** 0.5)
hi = icbhi["t_auc"] + 1.96 * icbhi["t_sd"] / (5 ** 0.5)
print("  its 95%% interval is [%.3f, %.3f], so it is %s chance"
      % (lo, hi, "above" if lo > 0.5 else "below" if hi < 0.5 else "at"))
