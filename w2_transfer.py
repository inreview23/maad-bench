"""Acoustic distance between sub-datasets against observed rank transfer.

The manuscript claims that acoustic similarity does not predict which task pairs
transfer, and quotes a Spearman correlation between the two. No script in the
repository produced that number, so it could not be re-derived after the LODA
reseed moved every observed transfer value. This file fixes that: it states the
distance it uses, computes it from the released feature matrices, and ships with
the release so the claim is checkable.

Distance is the 2-Wasserstein between the two sub-datasets' normal-clip
populations under a Gaussian approximation, the Frechet distance

    W2^2 = ||m1 - m2||^2 + tr(S1 + S2 - 2 (S1^1/2 S2 S1^1/2)^1/2)

computed on the shared 74-feature schema after standardizing each feature on the
pooled normals of the pair, so that no feature's unit dominates the trace. The
Gaussian form is an approximation; it is used because it is closed-form,
deterministic, and does not depend on a sample-size-sensitive coupling estimate.

Run with --old to correlate against the submitted fixed-split ordering instead of
the rotated one, which is how the number in the submitted version is recovered.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import linalg, stats

# Both roots default to the released result tree and remain overridable for
# comparisons against a separate working analysis directory.
HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.environ.get("MAAD_RELEASE", os.path.join(HERE, "results"))
ROT = os.environ.get("MAAD_ROT", os.path.join(HERE, "results"))
ORDER = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]


def normals(ds):
    """The normal clips' feature matrix for one sub-dataset."""
    f = pd.read_parquet(os.path.join(REL, ds, "features.parquet"))
    m = pd.read_parquet(os.path.join(REL, ds, "metadata.parquet"))
    keep = (m["anomaly_label"] == 0).to_numpy()
    cols = [c for c in f.columns if f[c].dtype.kind in "fi"]
    return f.loc[keep, cols].to_numpy(dtype=float)


def w2(a, b):
    """Gaussian 2-Wasserstein distance, on the pair's pooled standardization."""
    mu = np.concatenate([a, b]).mean(axis=0)
    sd = np.concatenate([a, b]).std(axis=0)
    sd[sd == 0] = 1.0
    a, b = (a - mu) / sd, (b - mu) / sd
    m1, m2 = a.mean(axis=0), b.mean(axis=0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = linalg.sqrtm(s1)
    inner = linalg.sqrtm(root @ s2 @ root)
    if np.iscomplexobj(inner):
        inner = inner.real
    d2 = float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * inner))
    return float(np.sqrt(max(d2, 0.0)))


def w2_marginal(a, b):
    """Mean 1-D Wasserstein distance over features, on the same standardization.

    A second definition, reported alongside the Gaussian one so that the claim
    does not rest on a single modelling choice. It ignores the covariance and is
    therefore the more conservative of the two about calling two populations
    close.
    """
    mu = np.concatenate([a, b]).mean(axis=0)
    sd = np.concatenate([a, b]).std(axis=0)
    sd[sd == 0] = 1.0
    a, b = (a - mu) / sd, (b - mu) / sd
    return float(np.mean([stats.wasserstein_distance(a[:, j], b[:, j])
                          for j in range(a.shape[1])]))


def orderings(old):
    out = {}
    for ds in ORDER:
        if old:
            d = pd.read_csv(os.path.join(REL, ds, "metrics.csv"))
        else:
            # the run tree keeps these per sub-dataset, the analysis copy flat
            flat = os.path.join(ROT, "rot_metrics_%s.csv" % ds)
            nested = os.path.join(ROT, ds, "rot_metrics.csv")
            d = pd.read_csv(flat if os.path.exists(flat) else nested)
        out[ds] = d[d.setting == "primary"].set_index("detector").auc_roc_mean
    return out


def main():
    old = "--old" in sys.argv
    X = {ds: normals(ds) for ds in ORDER}
    M = orderings(old)
    rows = []
    for a, b in itertools.combinations(ORDER, 2):
        common = M[a].index.intersection(M[b].index)
        rows.append({
            "a": a, "b": b,
            "w2": w2(X[a], X[b]),
            "w2_marginal": w2_marginal(X[a], X[b]),
            "observed": float(stats.spearmanr(M[a][common], M[b][common]).statistic),
        })
    t = pd.DataFrame(rows)
    print(t.sort_values("w2").to_string(index=False,
                                        float_format=lambda v: "%.3f" % v))
    print()
    print("ordering: %s" % ("submitted fixed split" if old else "rotated"))
    for col, label in [("w2", "Gaussian"), ("w2_marginal", "marginal")]:
        r = stats.spearmanr(t[col], t.observed)
        print("Spearman(%s distance, observed transfer) = %+.3f, p = %.3f"
              % (label, r.statistic, r.pvalue))
    if not old:
        t.to_csv(os.path.join(ROT, "analysis_w2_transfer.csv"), index=False)
        print("wrote analysis_w2_transfer.csv")


if __name__ == "__main__":
    main()
