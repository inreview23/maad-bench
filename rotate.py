"""Group-rotated re-run driver for MAAD-Bench.

The v1 protocol held out one group fixed in advance per sub-dataset and never
rotated it, and it subsampled the held-out anomalies down to a 10% test
prevalence. That left three sub-datasets with fewer than ten test positives and
left the group axis with no variance, so the reported standard deviations
measured only the anomaly subsample and detector randomness under one fixed
split. This driver rotates every native group through the test role and keeps
every held-out anomaly.

It reads results/<ds>/features.parquet and metadata.parquet directly, the same
fast path contam_sweep.py and runtime.py use, so it needs no audio and no
feature extraction.

Usage (on the DGX, one BLAS thread per worker):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python rotate.py --dataset all --jobs 6

  python rotate.py --dataset esc50 --mode regress   # v1-protocol regression
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import maad_bench as mb

ALL = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]


def load_tables(name):
    """Load the released feature matrix and evaluation metadata for one dataset.

    features.parquet and metadata.parquet are row-aligned by position with no
    join key, which is the contract run_dataset() wrote them under.
    """
    d = os.path.join(mb.RESULTS_DIR, name)
    feat = pd.read_parquet(os.path.join(d, "features.parquet"))
    df = pd.read_parquet(os.path.join(d, "metadata.parquet"))
    if len(feat) != len(df):
        raise ValueError("%s: features has %d rows, metadata has %d"
                         % (name, len(feat), len(df)))
    return df, feat


def rotate_one(name, settings, jobs, collect):
    df, feat = load_tables(name)
    groups = sorted(df["group"].unique().tolist())
    print("[%s] %d clips, %d anomalies, %d native groups: %s"
          % (name, len(df), int(df["anomaly_label"].sum()), len(groups), groups), flush=True)

    t0 = time.time()
    if jobs > 1 and len(groups) > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=min(jobs, len(groups)), verbose=5)(
            delayed(mb.run_rotation)(df, feat, groups=[g], settings=settings,
                                     subsample_test=False,
                                     collect_scores_for=collect)
            for g in groups)
        runs = pd.concat([p[0] for p in parts], ignore_index=True)
        clip_parts = [p[1] for p in parts if p[1] is not None]
        clips = pd.concat(clip_parts, ignore_index=True) if clip_parts else None
    else:
        runs, clips = mb.run_rotation(df, feat, groups=groups, settings=settings,
                                      subsample_test=False, collect_scores_for=collect)
    elapsed = time.time() - t0

    runs.insert(0, "dataset", name)
    out = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    runs.to_parquet(os.path.join(out, "rot_runs.parquet"), index=False)

    if clips is not None:
        clips.insert(0, "dataset", name)
        # Attach the source path so a score row is traceable to a clip without
        # relying on positional alignment downstream.
        clips = clips.merge(df[["path"]].reset_index().rename(
            columns={"index": "row_index"}), on="row_index", how="left")
        clips.to_parquet(os.path.join(out, "rot_clip_scores.parquet"), index=False)

    summ = mb.summarize_rotation(runs, name)
    summ.to_csv(os.path.join(out, "rot_metrics.csv"), index=False)

    skipped = runs[runs["detector"] == "*"]
    meta = {"protocol": "leave-one-group-out",
            "subsample_test": False,
            "groups": groups,
            "settings": list(settings),
            "seeds": mb.SEEDS,
            "contamination_train_side": mb.CONTAMINATION,
            "n_features": int(feat.shape[1]),
            "n_skipped_single_class_folds": int(len(skipped)),
            "elapsed_sec": round(elapsed, 1)}
    with open(os.path.join(out, "rot_config.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("[%s] done in %.1fs; %d run rows, %d skipped folds, %d clip rows"
          % (name, elapsed, len(runs), len(skipped),
             0 if clips is None else len(clips)), flush=True)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["setting", "detector", "auc_roc_mean", "auc_roc_fold_sd",
                "auc_roc_seed_sd", "n_folds"]
        print(summ[summ.setting == "primary"][cols].head(8).to_string(index=False), flush=True)
    return summ


def regress_one(name):
    """Re-run the v1 protocol from the released features and diff against the
    published metrics.csv. This separates "the protocol changed" from "the code
    broke" before any rotation number is trusted."""
    df, feat = load_tables(name)
    published = pd.read_csv(os.path.join(mb.RESULTS_DIR, name, "metrics.csv"))
    got = mb.summarize(mb.run(df, feat, mb.DATASETS[name]["test_groups"]), name)
    j = published.merge(got, on=["dataset", "setting", "detector"],
                        suffixes=("_pub", "_new"))
    j["delta"] = (j["auc_roc_mean_new"] - j["auc_roc_mean_pub"]).abs()
    worst = j.sort_values("delta", ascending=False).head(5)
    print("[%s] regression against published metrics.csv: %d detectors matched, "
          "max |delta AUC| = %.6f" % (name, len(j), j["delta"].max()), flush=True)
    print(worst[["setting", "detector", "auc_roc_mean_pub",
                 "auc_roc_mean_new", "delta"]].to_string(index=False), flush=True)
    return j["delta"].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=ALL + ["all"])
    ap.add_argument("--mode", choices=["rotate", "regress"], default="rotate")
    ap.add_argument("--settings", default="primary,secondary")
    ap.add_argument("--collect-scores", default="primary",
                    help="comma-separated settings to persist per-clip scores for, "
                         "or 'none'")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    names = ALL if args.dataset == "all" else [args.dataset]
    settings = tuple(s for s in args.settings.split(",") if s)
    collect = () if args.collect_scores == "none" else tuple(
        s for s in args.collect_scores.split(",") if s)

    if args.mode == "regress":
        worst = {}
        for n in names:
            try:
                worst[n] = regress_one(n)
            except Exception as e:
                print("[%s] FAILED: %s: %s" % (n, type(e).__name__, e), flush=True)
        print("\nmax |delta AUC| per dataset:", json.dumps(
            {k: round(float(v), 6) for k, v in worst.items()}, indent=2), flush=True)
        return

    summaries = []
    for n in names:
        try:
            summaries.append(rotate_one(n, settings, args.jobs, collect))
        except Exception as e:
            import traceback
            print("[%s] FAILED: %s: %s" % (n, type(e).__name__, e), flush=True)
            traceback.print_exc()
    if summaries:
        comb = pd.concat(summaries, ignore_index=True)
        p = os.path.join(mb.RESULTS_DIR, "rot_metrics_combined.csv")
        comb.to_csv(p, index=False)
        print("\nwrote", p, flush=True)


if __name__ == "__main__":
    main()
