"""E7: Rotated log-mel autoencoder anchor for CREMA-D and CirCor.

Applies the exact MAAD-Bench DCASE-style autoencoder baseline unchanged:
- log-mel: 64 mels, hop 512, SR 22050
- frame context: 5 consecutive frames (dim = 320)
- architecture: Linear(320->128)->ReLU->Linear(128->32)->ReLU->Linear(32->8)->ReLU
               ->Linear(8->32)->ReLU->Linear(32->128)->ReLU->Linear(128->320)
- training: Adam(lr=1e-3), MSELoss, 40 epochs, batch size 1024
- scoring: mean per-frame reconstruction error per held-out clip
- seeds: mb.SEEDS (0, 1, 2, 3, 4)
- protocol: leave-one-group-out (LOGO), primary setting (normal-only training), subsample_test=False
- outputs: results/<name>/ae_rot_metrics.csv and ae_rot_scores.parquet

Usage:
  python e7_ae.py --stage prebuild --dataset all
  python e7_ae.py --stage estimate --dataset all
  python e7_ae.py --stage run --dataset circor_rec
  python e7_ae.py --stage run --dataset cremad
  python e7_ae.py --stage run --dataset all
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

# Set Spark worker environment before joblib imports
os.environ["LOKY_MAX_CPU_COUNT"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import maad_bench as mb
import ae_baseline
from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
# 1. Dataset Registration
# -----------------------------------------------------------------------------
def register_all():
    """Register cremad and circor_rec in mb.DATASETS and mb.LOADERS."""
    mb.DATASETS["cremad"] = dict(
        normal=["neutral"],
        anomaly=["angry", "fearful"],
        test_groups=[1001],
    )
    mb.DATASETS["circor_rec"] = dict(
        normal=["absent"],
        anomaly=["present"],
        test_groups=[0],
    )

    for ds in ["cremad", "circor_rec"]:
        mb.LOADERS[ds] = (lambda d=ds: pd.read_parquet(os.path.join(mb.RESULTS_DIR, d, "metadata.parquet")))

register_all()


# -----------------------------------------------------------------------------
# 2. Frame Extraction with explicit job cap
# -----------------------------------------------------------------------------
def build_frames(name, df, n_jobs=3):
    """Build or load cached log-mel window frames (exact parity with ae_baseline)."""
    paths = df["path"].astype(str).tolist()
    offs = df["offset"].tolist() if "offset" in df.columns else [0.0] * len(df)
    durs = df["duration"].tolist() if "duration" in df.columns else [None] * len(df)
    spec = [(p, (0.0 if pd.isna(o) else float(o)), (None if pd.isna(d) else float(d)))
            for p, o, d in zip(paths, offs, durs)]
    key = hashlib.md5(
        ("ae2|" + "\n".join("%s@%.3f+%s" % (p, o, d) for p, o, d in spec)).encode()
    ).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, "%s_ae_%s.npz" % (name, key))
    if os.path.exists(cache):
        print(f"[{name}] Loading cached log-mel windows from {cache} ...", flush=True)
        z = np.load(cache)
        return z["frames"], z["clip"]
    
    print(f"[{name}] Extracting log-mel windows for {len(df)} clips using n_jobs={n_jobs} ...", flush=True)
    t0 = time.time()
    per = Parallel(n_jobs=n_jobs, verbose=5)(delayed(ae_baseline.logmel_windows)(*a) for a in spec)
    clip = np.concatenate([np.full(len(a), i, dtype=np.int32) for i, a in enumerate(per)])
    frames = np.concatenate(per, axis=0)
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, frames=frames, clip=clip)
    t1 = time.time()
    print(f"[{name}] Extracted and cached {len(frames)} frames to {cache} in {t1-t0:.1f}s", flush=True)
    return frames, clip


# -----------------------------------------------------------------------------
# 3. Model Training & Scoring (identical to ae_baseline.py)
# -----------------------------------------------------------------------------
train_ae = ae_baseline.train_ae
frame_errors = ae_baseline.frame_errors


# -----------------------------------------------------------------------------
# 4. Runtime Estimation
# -----------------------------------------------------------------------------
def estimate_runtime(name, n_estimate_folds=1, settings=("primary",)):
    """Run 1 fold (5 seeds) on GPU to measure exact elapsed time and project total."""
    import torch
    df = mb.LOADERS[name]()
    frames, clip = build_frames(name, df, n_jobs=3)
    y = df["anomaly_label"].to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = list(pd.unique(df["group"]))
    print(f"\n=== [ESTIMATE: {name}] device={device} total_clips={len(df)} frames={frames.shape} groups={len(groups)} ===", flush=True)

    test_g = groups[0]
    fold_times = []
    
    for seed in mb.SEEDS:
        t0 = time.time()
        tr, te = mb.make_split(df, seed, settings[0], [test_g], subsample_test=False)
        trm = np.isin(clip, tr)
        mu = frames[trm].mean(0)
        sd = frames[trm].std(0) + 1e-8
        net = train_ae((frames[trm] - mu) / sd, frames.shape[1], device, seed)
        tem = np.isin(clip, te)
        errs = frame_errors(net, (frames[tem] - mu) / sd, device)
        te_clip = clip[tem]
        scores = np.array([errs[te_clip == ci].mean() for ci in te])
        yte = y[te]
        m = mb.metrics(yte, scores, mb.CONTAMINATION)
        t_seed = time.time() - t0
        fold_times.append(t_seed)
        print(f"  [estimate {name}] group={test_g} seed={seed}: AUC-ROC={m['auc_roc']:.4f} in {t_seed:.2f}s", flush=True)

    t_1_fold = sum(fold_times)
    total_seeds = len(groups) * len(mb.SEEDS) * len(settings)
    total_est_seconds = t_1_fold * len(groups) * len(settings)
    print(f"\n>>> [{name}] 1 fold ({len(mb.SEEDS)} seeds) took: {t_1_fold:.2f}s ({t_1_fold/60:.2f} min)")
    print(f">>> [{name}] Total {len(groups)} folds ({total_seeds} runs) estimated: {total_est_seconds:.1f}s ({total_est_seconds/60:.1f} min / {total_est_seconds/3600:.2f} hrs)")
    return t_1_fold, total_est_seconds


# -----------------------------------------------------------------------------
# 5. Full AE Execution
# -----------------------------------------------------------------------------
def run(name, settings=("primary",), n_jobs=3):
    """Execute complete rotated autoencoder anchor and write schema-compliant outputs."""
    import torch
    df = mb.LOADERS[name]()
    frames, clip = build_frames(name, df, n_jobs=n_jobs)
    y = df["anomaly_label"].to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = list(pd.unique(df["group"]))
    print(f"[{name}] Starting AE rotation: device={device} frames={frames.shape} groups={len(groups)} settings={settings}", flush=True)

    rec = []
    t_start = time.time()
    total_runs = len(settings) * len(groups) * len(mb.SEEDS)
    run_idx = 0

    for setting in settings:
        for fold_i, g in enumerate(groups):
            t_fold_start = time.time()
            fold_aucs = []
            for seed in mb.SEEDS:
                run_idx += 1
                t_seed_start = time.time()
                tr, te = mb.make_split(df, seed, setting, [g], subsample_test=False)
                if not mb.is_scorable(y[te]):
                    print(f"[{name}] fold {g} is single-class, skipped", flush=True)
                    continue

                trm = np.isin(clip, tr)
                mu = frames[trm].mean(0)
                sd = frames[trm].std(0) + 1e-8
                net = train_ae((frames[trm] - mu) / sd, frames.shape[1], device, seed)
                tem = np.isin(clip, te)
                errs = frame_errors(net, (frames[tem] - mu) / sd, device)
                te_clip = clip[tem]
                scores = np.array([errs[te_clip == ci].mean() for ci in te])
                yte = y[te]
                m = mb.metrics(yte, scores, mb.CONTAMINATION)
                m.update({
                    "detector": "AE_logmel",
                    "setting": setting,
                    "group": g,
                    "seed": seed,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_test_anom": int(yte.sum()),
                })
                rec.append(m)
                fold_aucs.append(m["auc_roc"])
                t_seed_elapsed = time.time() - t_seed_start

            t_fold_elapsed = time.time() - t_fold_start
            fold_mean_auc = np.mean(fold_aucs) if fold_aucs else np.nan
            print(f"[{name}] Fold {fold_i+1}/{len(groups)} (group {g}): mean_AUC={fold_mean_auc:.4f} "
                  f"(took {t_fold_elapsed:.1f}s, run {run_idx}/{total_runs}, elapsed {time.time()-t_start:.1f}s)", flush=True)

    sdf = pd.DataFrame(rec)
    out = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    scores_path = os.path.join(out, "ae_rot_scores.parquet")
    sdf.to_parquet(scores_path, index=False)
    print(f"[{name}] Wrote {scores_path} (shape: {sdf.shape})", flush=True)

    per_fold = sdf.groupby(["setting", "group"], as_index=False)["auc_roc"].mean()
    summ = per_fold.groupby("setting")["auc_roc"].agg(["mean", "std", "count"])
    summ.columns = ["auc_roc_mean", "auc_roc_fold_sd", "n_folds"]
    metrics_path = os.path.join(out, "ae_rot_metrics.csv")
    summ.to_csv(metrics_path)
    print(f"[{name}] Wrote {metrics_path}", flush=True)

    t_total = time.time() - t_start
    print(f"\n=== {name} AE baseline completed in {t_total:.1f}s ({t_total/60:.2f} min) ===", flush=True)
    print(summ.round(3).to_string(), flush=True)
    return sdf, summ


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="E7 AE Baseline Anchor for cremad and circor_rec")
    parser.add_argument("--stage", required=True, choices=["prebuild", "estimate", "run", "summary"])
    parser.add_argument("--dataset", default="all", choices=["all", "cremad", "circor_rec"])
    parser.add_argument("--settings", default="primary")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()

    register_all()
    datasets = ["circor_rec", "cremad"] if args.dataset == "all" else [args.dataset]
    settings = tuple(x for x in args.settings.split(",") if x)

    if args.stage == "prebuild":
        for ds in datasets:
            df = mb.LOADERS[ds]()
            build_frames(ds, df, n_jobs=args.jobs)

    elif args.stage == "estimate":
        for ds in datasets:
            estimate_runtime(ds, n_estimate_folds=1, settings=settings)

    elif args.stage == "run":
        for ds in datasets:
            run(ds, settings=settings, n_jobs=args.jobs)

    elif args.stage == "summary":
        print("\n=== E7 AE Anchor Summary ===", flush=True)
        for ds in ["circor_rec", "cremad"]:
            p = os.path.join(mb.RESULTS_DIR, ds, "ae_rot_metrics.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                print(f"\n--- {ds} ---")
                print(df.to_string(index=False))
            else:
                print(f"\n--- {ds} ---: Missing {p}")


if __name__ == "__main__":
    main()
