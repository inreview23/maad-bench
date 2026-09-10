"""MAAD-Bench DCASE-style autoencoder baseline (log-mel reconstruction).

Trains a dense autoencoder on log-mel frames (normal-only in the primary setting,
contaminated in the secondary), scores each test clip by mean per-frame
reconstruction error, and reports the same metrics as the tabular sweep. Reuses
maad_bench's loaders, splits, and metrics. Uses a GPU when available.

Usage:
  python ae_baseline.py --dataset esc50
  python ae_baseline.py --dataset all
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

N_MELS = 64
CONTEXT = 5          # consecutive frames per autoencoder input window
HOP = 512
EPOCHS = 40
BATCH = 1024
LR = 1e-3


def logmel_windows(path, offset=0.0, dur=None):
    import librosa
    y, sr = librosa.load(path, sr=mb.SR, mono=True, offset=offset, duration=dur)
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP))
    T = S.shape[1]
    if T < CONTEXT:
        S = np.concatenate([S, np.zeros((N_MELS, CONTEXT - T), dtype=S.dtype)], axis=1)
        T = CONTEXT
    return np.stack([S[:, t:t + CONTEXT].T.reshape(-1)
                     for t in range(T - CONTEXT + 1)]).astype(np.float32)


def build_frames(name, df):
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
        z = np.load(cache)
        return z["frames"], z["clip"]
    from joblib import Parallel, delayed
    print("extracting log-mel windows for %d clips ..." % len(df), flush=True)
    per = Parallel(n_jobs=-1, verbose=5)(delayed(logmel_windows)(*a) for a in spec)
    clip = np.concatenate([np.full(len(a), i, dtype=np.int32) for i, a in enumerate(per)])
    frames = np.concatenate(per, axis=0)
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, frames=frames, clip=clip)
    return frames, clip


def train_ae(Xtr, dim, device, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(),
        nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 32), nn.ReLU(),
        nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, dim)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    X = torch.tensor(Xtr, device=device)
    n = len(X)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            b = X[perm[i:i + BATCH]]
            opt.zero_grad()
            loss = loss_fn(net(b), b)
            loss.backward()
            opt.step()
    return net


def frame_errors(net, X, device):
    import torch
    net.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            b = torch.tensor(X[i:i + 4096], device=device)
            out[i:i + 4096] = ((net(b) - b) ** 2).mean(dim=1).cpu().numpy()
    return out


def run(name, settings=("primary", "secondary")):
    import torch
    cfg = mb.DATASETS[name]
    df = mb.LOADERS[name]()
    frames, clip = build_frames(name, df)
    y = df["anomaly_label"].to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[%s] device=%s frames=%s" % (name, device, frames.shape), flush=True)

    groups = list(pd.unique(df["group"]))
    rec = []
    for setting in settings:
        for g in groups:
            for seed in mb.SEEDS:
                tr, te = mb.make_split(df, seed, setting, [g],
                                       subsample_test=False)
                if not mb.is_scorable(y[te]):
                    print("[%s] fold %s is single-class, skipped" % (name, g),
                          flush=True)
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
                m.update({"detector": "AE_logmel", "setting": setting,
                          "group": g, "seed": seed,
                          "n_train": int(len(tr)), "n_test": int(len(te)),
                          "n_test_anom": int(yte.sum())})
                rec.append(m)
    sdf = pd.DataFrame(rec)
    out = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    sdf.to_parquet(os.path.join(out, "ae_rot_scores.parquet"), index=False)
    per_fold = sdf.groupby(["setting", "group"], as_index=False)["auc_roc"].mean()
    summ = per_fold.groupby("setting")["auc_roc"].agg(["mean", "std", "count"])
    summ.columns = ["auc_roc_mean", "auc_roc_fold_sd", "n_folds"]
    summ.to_csv(os.path.join(out, "ae_rot_metrics.csv"))
    print("\n=== %s AE baseline, %d folds (fold mean, between-fold SD) ==="
          % (name, len(groups)), flush=True)
    print(summ.round(3).to_string(), flush=True)
    return sdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="primary,secondary")
    ap.add_argument("--dataset", required=True,
                    choices=["esc50", "urbansound8k", "ravdess", "nsynth", "mimii",
                             "physionet", "icbhi", "all"])
    args = ap.parse_args()
    allnames = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii",
                "physionet", "icbhi"]
    names = allnames if args.dataset == "all" else [args.dataset]
    for n in names:
        try:
            run(n, tuple(x for x in args.settings.split(",") if x))
        except Exception as e:
            print("[%s] FAILED %s: %s" % (n, type(e).__name__, e), flush=True)


if __name__ == "__main__":
    main()
