"""Group-rotated re-run of the pretrained-representation arms.

Two v1 defects are fixed here.

1. Leakage. panns_embed.py:57 and embed.py:116 called
   PCA(n_components=64).fit_transform on the complete embedding matrix and only
   split afterwards, justified in a comment as leakage-safe because PCA sees no
   labels. Label-blindness is not the issue: in an inductive protocol whose
   held-out group is population-disjoint by design, letting the test rows set
   the projection axes is leakage. mb.make_pca_pre fits the projection on the
   training rows of each fold.

2. An undisclosed protocol mismatch. embed.py monkey-patched mb.SEEDS to
   [0, 1, 2] and dropped LMDD; twfr.py did the same and also dropped KPCA, ABOD,
   COF and SOD. The handcrafted column of the same comparison table used five
   seeds and all 25 detectors. This driver uses the standard roster and seeds
   for every arm, so the columns are comparable.

No audio and no GPU are needed: the raw embedding matrices are already cached
under cache/<dataset>_<backbone>_<hash>.parquet at full width (PANNs 2048, AST
768, CLAP 512, TWFR 128), and the cache key is the clip list, which the protocol
change does not touch.

Usage:
  python rotate_repr.py --dataset all --backbone panns --jobs 8
  python rotate_repr.py --dataset esc50 --backbone twfr --pca 0
"""
import argparse
import glob
import json
import os
import time

import pandas as pd

import maad_bench as mb

ALL = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
BACKBONES = ["panns", "ast", "clap", "twfr"]


def load_embedding(name, backbone):
    """Load the cached raw embedding matrix and the row-aligned metadata."""
    pat = os.path.join(mb.CACHE_DIR, "%s_%s_*.parquet" % (name, backbone))
    hits = [p for p in sorted(glob.glob(pat)) if "features_n" not in p]
    if not hits:
        raise FileNotFoundError(
            "no cached %s embedding for %s (looked for %s). Regenerate it with "
            "panns_embed.py / embed.py / twfr.py, which needs the raw audio."
            % (backbone, name, pat))
    if len(hits) > 1:
        raise RuntimeError("%d cached %s matrices for %s; the clip list is "
                           "ambiguous: %s" % (len(hits), backbone, name, hits))
    emb = pd.read_parquet(hits[0])
    df = pd.read_parquet(os.path.join(mb.RESULTS_DIR, name, "metadata.parquet"))
    if len(emb) != len(df):
        raise ValueError("%s/%s: embedding has %d rows, metadata has %d"
                         % (name, backbone, len(emb), len(df)))
    return df, emb, hits[0]


def rotate_one(name, backbone, pca_dim, settings, jobs, collect):
    df, emb, src = load_embedding(name, backbone)
    groups = sorted(df["group"].unique().tolist())
    pre = mb.make_pca_pre(pca_dim) if pca_dim > 0 else None
    print("[%s/%s] %d clips x %d dims from %s; %d groups; pca=%s"
          % (name, backbone, emb.shape[0], emb.shape[1], os.path.basename(src),
             len(groups), pca_dim if pca_dim > 0 else "none"), flush=True)

    t0 = time.time()
    if jobs > 1 and len(groups) > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=min(jobs, len(groups)), verbose=5)(
            delayed(mb.run_rotation)(df, emb, groups=[g], settings=settings,
                                     subsample_test=False,
                                     collect_scores_for=collect, pre=pre)
            for g in groups)
        runs = pd.concat([p[0] for p in parts], ignore_index=True)
        cps = [p[1] for p in parts if p[1] is not None]
        clips = pd.concat(cps, ignore_index=True) if cps else None
    else:
        runs, clips = mb.run_rotation(df, emb, groups=groups, settings=settings,
                                      subsample_test=False,
                                      collect_scores_for=collect, pre=pre)
    elapsed = time.time() - t0

    runs.insert(0, "representation", backbone)
    runs.insert(0, "dataset", name)
    out = os.path.join(mb.RESULTS_DIR, name)
    runs.to_parquet(os.path.join(out, "rot_%s_runs.parquet" % backbone), index=False)

    if clips is not None:
        clips.insert(0, "representation", backbone)
        clips.insert(0, "dataset", name)
        clips.to_parquet(os.path.join(out, "rot_%s_clip_scores.parquet" % backbone),
                         index=False)

    summ = mb.summarize_rotation(runs, name)
    summ.insert(1, "representation", backbone)
    summ.to_csv(os.path.join(out, "rot_%s_metrics.csv" % backbone), index=False)

    with open(os.path.join(out, "rot_%s_config.json" % backbone), "w") as fh:
        json.dump({"representation": backbone,
                   "embedding_source": os.path.basename(src),
                   "raw_dims": int(emb.shape[1]),
                   "pca_dims": int(pca_dim) if pca_dim > 0 else None,
                   "pca_fitted": "per training fold",
                   "protocol": "leave-one-group-out",
                   "subsample_test": False,
                   "groups": groups, "seeds": mb.SEEDS,
                   "settings": list(settings),
                   "elapsed_sec": round(elapsed, 1)}, fh, indent=2)

    print("[%s/%s] done in %.1fs; best primary: %s"
          % (name, backbone, elapsed,
             summ[summ.setting == "primary"].iloc[0][["detector", "auc_roc_mean"]].to_dict()),
          flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=ALL + ["all"])
    ap.add_argument("--backbone", required=True, choices=BACKBONES + ["all"])
    ap.add_argument("--pca", type=int, default=64,
                    help="fold-local PCA width; 0 disables (TWFR is already 128-d "
                         "and had no PCA in v1)")
    ap.add_argument("--settings", default="primary,secondary")
    ap.add_argument("--collect-scores", default="primary")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    names = ALL if args.dataset == "all" else [args.dataset]
    backs = BACKBONES if args.backbone == "all" else [args.backbone]
    settings = tuple(s for s in args.settings.split(",") if s)
    collect = () if args.collect_scores == "none" else tuple(
        s for s in args.collect_scores.split(",") if s)

    summaries = []
    for b in backs:
        pca = 0 if b == "twfr" else args.pca
        for n in names:
            try:
                summaries.append(rotate_one(n, b, pca, settings, args.jobs, collect))
            except FileNotFoundError as e:
                print("[%s/%s] skipped: %s" % (n, b, e), flush=True)
            except Exception as e:
                import traceback
                print("[%s/%s] FAILED: %s: %s" % (n, b, type(e).__name__, e), flush=True)
                traceback.print_exc()
    if summaries:
        comb = pd.concat(summaries, ignore_index=True)
        p = os.path.join(mb.RESULTS_DIR, "rot_repr_metrics_combined.csv")
        comb.to_csv(p, index=False)
        print("\nwrote", p, flush=True)


if __name__ == "__main__":
    main()
