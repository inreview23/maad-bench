"""DeltaAI shard of the E7 CREMA-D log-mel autoencoder anchor: e7_ae.run's loop over a subset of folds.

Usage: python dtai_ae_shard.py <shard> <n_shards>. Writes results/cremad/ae_rot_scores_shard<shard>.parquet.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import e7_ae
import maad_bench as mb

shard, nshard = int(sys.argv[1]), int(sys.argv[2])
name, setting = "cremad", "primary"
import torch
df = mb.LOADERS[name]()
frames, clip = e7_ae.build_frames(name, df)
y = df["anomaly_label"].to_numpy()
device = "cuda" if torch.cuda.is_available() else "cpu"
groups = list(pd.unique(df["group"]))[shard::nshard]
print(f"[{name}] shard {shard}/{nshard} device={device} folds={len(groups)}", flush=True)
rec = []
t0 = time.time()
for g in groups:
    for seed in mb.SEEDS:
        tr, te = mb.make_split(df, seed, setting, [g], subsample_test=False)
        if not mb.is_scorable(y[te]):
            print(f"[{name}] fold {g} is single-class, skipped", flush=True)
            continue
        trm = np.isin(clip, tr)
        mu = frames[trm].mean(0)
        sd = frames[trm].std(0) + 1e-8
        net = e7_ae.train_ae((frames[trm] - mu) / sd, frames.shape[1], device, seed)
        tem = np.isin(clip, te)
        errs = e7_ae.frame_errors(net, (frames[tem] - mu) / sd, device)
        te_clip = clip[tem]
        scores = np.array([errs[te_clip == ci].mean() for ci in te])
        m = mb.metrics(y[te], scores, mb.CONTAMINATION)
        m.update({"detector": "AE_logmel", "setting": setting, "group": g, "seed": seed,
                  "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_anom": int(y[te].sum())})
        rec.append(m)
    print(f"[{name}] shard {shard} group {g} done, elapsed {time.time() - t0:.0f}s", flush=True)
out = os.path.join(mb.RESULTS_DIR, name)
pd.DataFrame(rec).to_parquet(os.path.join(out, f"ae_rot_scores_shard{shard}.parquet"), index=False)
print(f"[{name}] shard {shard} wrote {len(rec)} runs", flush=True)
