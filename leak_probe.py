"""How much of the pretrained-representation result was the PCA leak?

The revision changes two things at once in the representation arm: the projection
is now fitted inside each training fold, and the held-out group rotates. A drop
against the published number therefore has two possible causes, and Reviewer 2's
comment is specifically about the first.

This holds the folds fixed and varies only where the projection is fitted. Both
arms run the same rotation over the same groups with the same seeds and the same
detector roster; one fits PCA on the fold's training rows, the other on the whole
embedding matrix the way panns_embed.py and embed.py did. The difference between
them is the leak, with nothing else moving.

Usage:
  python leak_probe.py --dataset icbhi,esc50 --backbone panns --jobs 6
"""
import argparse
import os

import pandas as pd

import maad_bench as mb
import rotate_repr as rr


def global_pca_pre(n_components, random_state=0):
    """The leaky projection: fitted on every row, including the held-out ones.

    Reproduced here on purpose so the comparison is exact rather than argued.
    """
    def pre(X, tr, te):
        from sklearn.decomposition import PCA
        k = min(n_components, X.shape[1], len(X))
        p = PCA(n_components=k, random_state=random_state).fit(X)
        return p.transform(X[tr]), p.transform(X[te])
    return pre


def probe_one(name, backbone, pca, settings, jobs):
    df, emb, src = rr.load_embedding(name, backbone)
    groups = sorted(df["group"].unique().tolist())
    print("[%s/%s] %d clips x %d dims from %s; %d groups; pca=%d"
          % (name, backbone, emb.shape[0], emb.shape[1], os.path.basename(src),
             len(groups), pca), flush=True)
    out = {}
    for mode, pre in (("fold", mb.make_pca_pre(pca)),
                      ("global", global_pca_pre(pca))):
        if jobs > 1 and len(groups) > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=min(jobs, len(groups)))(
                delayed(mb.run_rotation)(df, emb, groups=[g], settings=settings,
                                         subsample_test=False,
                                         collect_scores_for=(), pre=pre)
                for g in groups)
            runs = pd.concat([r[0] for r in res], ignore_index=True)
        else:
            runs, _ = mb.run_rotation(df, emb, groups=groups, settings=settings,
                                      subsample_test=False,
                                      collect_scores_for=(), pre=pre)
        runs.insert(0, "dataset", name)
        out[mode] = mb.summarize_rotation(runs, name)
        print("[%s/%s] %s-fitted PCA done" % (name, backbone, mode), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="panns")
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--settings", default="primary")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    settings = tuple(s for s in args.settings.split(",") if s)
    rows = []
    for name in args.dataset.split(","):
        try:
            res = probe_one(name, args.backbone, args.pca, settings, args.jobs)
        except FileNotFoundError as e:
            print("[%s] skipped: %s" % (name, e), flush=True)
            continue
        f = res["fold"].set_index(["setting", "detector"]).auc_roc_mean
        g = res["global"].set_index(["setting", "detector"]).auc_roc_mean
        common = f.index.intersection(g.index)
        for setting in settings:
            fs = f[f.index.get_level_values(0) == setting]
            gs = g[g.index.get_level_values(0) == setting]
            if not len(fs) or not len(gs):
                continue
            rows.append({
                "dataset": name, "backbone": args.backbone, "setting": setting,
                "best_fold_fitted": float(fs.max()),
                "best_global_fitted": float(gs.max()),
                "leak_on_best": float(gs.max() - fs.max()),
                "mean_leak": float((g[common] - f[common]).mean()),
                "max_leak": float((g[common] - f[common]).max()),
                "n_detectors": int(len(common) / max(1, len(settings))),
            })
    if not rows:
        print("nothing to report")
        return
    R = pd.DataFrame(rows)
    p = os.path.join(mb.RESULTS_DIR, "leak_probe_%s.csv" % args.backbone)
    R.to_csv(p, index=False)
    print()
    print("A positive leak value means the projection fitted on every row scored")
    print("higher than the one fitted on the training fold alone.")
    print(R.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
