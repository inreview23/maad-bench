"""E2: DCASE 2020 Task 2 official baseline, pipeline comparison, and new benchmark sub-datasets.

Steps:
1. Data discovery, validation, and loading.
2. Official baseline documentation and metadata.
3. Faithful PyTorch re-implementation of the official DCASE 2020 Task 2 baseline.
4. 74-dim tabular pipeline evaluation on the official split.
5. Paper AE anchor configuration on the official split.
6. Benchmark registration and LOGO rotation for dcase_toycar and dcase_toyconveyor.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "5"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import glob
import hashlib
import json
import re
import sys
import time

import joblib
from joblib import Parallel, delayed
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

# Import maad_bench
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

DATA_DIR = os.path.join(mb.HERE, "data")
DCASE_DIR = os.path.join(DATA_DIR, "dcase2020t2")
RESULTS_DIR = mb.RESULTS_DIR
CACHE_DIR = mb.CACHE_DIR

MACHINE_TYPES = ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]

# Published DCASE 2020 Task 2 Baseline Results
# Source 1: Koizumi et al. 2020 (arXiv:2006.05822), Table 1 (10-trial mean +/- SD in %)
# Source 2: Official repository (y-kawagu/dcase2020_task2_baseline), README.md Section 7
PUBLISHED_BASELINE = {
    "ToyCar": {
        1: {"auc": 0.828474, "pauc": 0.678514, "paper_auc": 0.8136, "paper_pauc": 0.6840},
        2: {"auc": 0.866307, "pauc": 0.762236, "paper_auc": 0.8597, "paper_pauc": 0.7772},
        3: {"auc": 0.639396, "pauc": 0.546659, "paper_auc": 0.6330, "paper_pauc": 0.5521},
        4: {"auc": 0.869477, "pauc": 0.701461, "paper_auc": 0.8445, "paper_pauc": 0.6897},
        "average": {"auc": 0.800914, "pauc": 0.672218, "paper_auc": 0.7877, "paper_pauc": 0.6758}
    },
    "ToyConveyor": {
        1: {"auc": 0.793497, "pauc": 0.650428, "paper_auc": 0.7807, "paper_pauc": 0.6425},
        2: {"auc": 0.642430, "pauc": 0.564437, "paper_auc": 0.6416, "paper_pauc": 0.5601},
        3: {"auc": 0.744579, "pauc": 0.604880, "paper_auc": 0.7535, "paper_pauc": 0.6103},
        "average": {"auc": 0.726835, "pauc": 0.606582, "paper_auc": 0.7253, "paper_pauc": 0.6043}
    },
    "fan": {
        0: {"auc": 0.539607, "pauc": 0.492435, "paper_auc": 0.5441, "paper_pauc": 0.4937},
        2: {"auc": 0.721866, "pauc": 0.554171, "paper_auc": 0.7340, "paper_pauc": 0.5481},
        4: {"auc": 0.622098, "pauc": 0.527072, "paper_auc": 0.6161, "paper_pauc": 0.5326},
        6: {"auc": 0.722770, "pauc": 0.529961, "paper_auc": 0.7392, "paper_pauc": 0.5235},
        "average": {"auc": 0.651585, "pauc": 0.525910, "paper_auc": 0.6583, "paper_pauc": 0.5245}
    },
    "pump": {
        0: {"auc": 0.670769, "pauc": 0.572690, "paper_auc": 0.6715, "paper_pauc": 0.5674},
        2: {"auc": 0.609369, "pauc": 0.580370, "paper_auc": 0.6153, "paper_pauc": 0.5810},
        4: {"auc": 0.888600, "pauc": 0.676842, "paper_auc": 0.8833, "paper_pauc": 0.6710},
        6: {"auc": 0.734902, "pauc": 0.570175, "paper_auc": 0.7455, "paper_pauc": 0.5802},
        "average": {"auc": 0.725910, "pauc": 0.600019, "paper_auc": 0.7289, "paper_pauc": 0.5999}
    },
    "slider": {
        0: {"auc": 0.963315, "pauc": 0.818451, "paper_auc": 0.9619, "paper_pauc": 0.8144},
        2: {"auc": 0.788090, "pauc": 0.628819, "paper_auc": 0.7897, "paper_pauc": 0.6368},
        4: {"auc": 0.946011, "pauc": 0.739503, "paper_auc": 0.9430, "paper_pauc": 0.7198},
        6: {"auc": 0.702247, "pauc": 0.490242, "paper_auc": 0.6959, "paper_pauc": 0.4902},
        "average": {"auc": 0.849916, "pauc": 0.669254, "paper_auc": 0.8476, "paper_pauc": 0.6653}
    },
    "valve": {
        0: {"auc": 0.687311, "pauc": 0.513490, "paper_auc": 0.6876, "paper_pauc": 0.5170},
        2: {"auc": 0.674083, "pauc": 0.512281, "paper_auc": 0.6818, "paper_pauc": 0.5183},
        4: {"auc": 0.744000, "pauc": 0.515351, "paper_auc": 0.7430, "paper_pauc": 0.5197},
        6: {"auc": 0.552833, "pauc": 0.482895, "paper_auc": 0.5390, "paper_pauc": 0.4843},
        "average": {"auc": 0.664557, "pauc": 0.506004, "paper_auc": 0.6628, "paper_pauc": 0.5098}
    }
}


def find_machine_dir(mtype):
    """Find directory containing train/ and test/ for given machine type."""
    candidates = [
        os.path.join(DCASE_DIR, "dev_data", mtype),
        os.path.join(DCASE_DIR, mtype),
        os.path.join(DCASE_DIR, "dev_data_%s" % mtype, mtype),
        os.path.join(DCASE_DIR, "dev_data_%s" % mtype),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "test")):
            return c
    # Fallback glob
    matches = glob.glob(os.path.join(DCASE_DIR, "**", mtype, "train"), recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError("Could not find train/test directories for machine type %s in %s" % (mtype, DCASE_DIR))


def scan_machine_files(mtype):
    """Scan and index all clips for a machine type into a structured DataFrame."""
    mdir = find_machine_dir(mtype)
    train_wavs = sorted(glob.glob(os.path.join(mdir, "train", "*.wav")))
    test_wavs = sorted(glob.glob(os.path.join(mdir, "test", "*.wav")))

    rows = []
    for split, wavs in [("train", train_wavs), ("test", test_wavs)]:
        for p in wavs:
            fname = os.path.basename(p)
            # Format: normal_id_01_00000000.wav or anomaly_id_01_00000000.wav
            m = re.match(r"(normal|anomaly)_id_(\d+)_\d+\.wav", fname)
            if not m:
                continue
            cond = m.group(1)
            mid = int(m.group(2))
            is_anom = int(cond == "anomaly")
            rows.append({
                "path": p,
                "filename": fname,
                "machine_type": mtype,
                "split": split,
                "condition": cond,
                "group": mid,
                "anomaly_label": is_anom,
                "category": cond
            })
    df = pd.DataFrame(rows)
    return df


# -------------------------------------------------------------------------
# Official DCASE 2020 Task 2 Baseline AE
# -------------------------------------------------------------------------
class DCASEBaselineAE(nn.Module):
    """Faithful PyTorch re-implementation of DCASE 2020 Task 2 baseline autoencoder.

    Architecture from keras_model.py:
    Dense(128) -> BatchNorm -> ReLU
    Dense(128) -> BatchNorm -> ReLU
    Dense(128) -> BatchNorm -> ReLU
    Dense(128) -> BatchNorm -> ReLU
    Dense(8)   -> BatchNorm -> ReLU
    Dense(128) -> BatchNorm -> ReLU
    Dense(128) -> BatchNorm -> ReLU
    Dense(128) -> BatchNorm -> ReLU
    Dense(128) -> BatchNorm -> ReLU
    Dense(inputDim) [linear output]
    """
    def __init__(self, input_dim=640):
        super().__init__()
        # PyTorch equivalent of Keras BatchNormalization: eps=1e-3, momentum=0.01
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.BatchNorm1d(8, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.net(x)


def extract_dcase_baseline_frames(wav_path, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    """Extract 5-frame concatenated log-mel feature vectors according to DCASE common.py."""
    y, sr = librosa.load(wav_path, sr=None, mono=False)
    if y.ndim > 1:
        y = y[0]
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel = (20.0 / power) * np.log10(mel + sys.float_info.epsilon)
    T = log_mel.shape[1]
    vector_array_size = T - frames + 1
    if vector_array_size < 1:
        return np.empty((0, n_mels * frames), dtype=np.float32)
    vector_array = np.zeros((vector_array_size, n_mels * frames), dtype=np.float32)
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel[:, t: t + vector_array_size].T
    return vector_array


def build_dcase_baseline_feature_cache(mtype, df, n_jobs=5):
    """Build or load log-mel frame cache for all clips in a machine type."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = df["path"].tolist()
    key = hashlib.md5(("dcase_base|" + "\n".join(paths)).encode()).hexdigest()[:10]
    cache_path = os.path.join(CACHE_DIR, "dcase_base_%s_%s.npz" % (mtype, key))
    if os.path.exists(cache_path):
        print("[%s] loading cached baseline log-mel features from %s ..." % (mtype, cache_path), flush=True)
        z = np.load(cache_path)
        return z["frames"], z["clip_idx"]

    print("[%s] extracting baseline log-mel features for %d clips (jobs=%d) ..." % (mtype, len(df), n_jobs), flush=True)
    with joblib.parallel_config(n_jobs=n_jobs):
        per = Parallel(n_jobs=n_jobs, verbose=5)(delayed(extract_dcase_baseline_frames)(p) for p in paths)
    clip_idx = np.concatenate([np.full(len(a), i, dtype=np.int32) for i, a in enumerate(per)])
    frames = np.concatenate(per, axis=0)
    np.savez(cache_path, frames=frames, clip_idx=clip_idx)
    print("[%s] cached %d frames (%s) to %s" % (mtype, len(frames), str(frames.shape), cache_path), flush=True)
    return frames, clip_idx


def train_baseline_ae(train_frames, device, seed=0, epochs=100, batch_size=512, lr=1e-3):
    """Train the DCASE baseline autoencoder.

    In Keras model.fit with validation_split=0.1, the last 10% is validation data
    and the first 90% is training data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_total = len(train_frames)
    n_train = int(0.9 * n_total)
    train_x = train_frames[:n_train]

    dim = train_frames.shape[1]
    model = DCASEBaselineAE(input_dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.tensor(train_x, device=device)
    n = len(X)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        X_ep = X[perm]
        for i in range(0, n, batch_size):
            b = X_ep[i:i + batch_size]
            optimizer.zero_grad()
            out = model(b)
            loss = loss_fn(out, b)
            loss.backward()
            optimizer.step()
    return model


def score_baseline_clips(model, frames, clip_idx, te_indices, device):
    """Calculate clip anomaly score = mean of frame MSE errors."""
    model.eval()
    tem = np.isin(clip_idx, te_indices)
    te_frames = frames[tem]
    te_clip = clip_idx[tem]

    out_errors = np.empty(len(te_frames), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(te_frames), 4096):
            b = torch.tensor(te_frames[i:i + 4096], device=device)
            out_errors[i:i + 4096] = ((model(b) - b) ** 2).mean(dim=1).cpu().numpy()

    df_err = pd.DataFrame({"clip": te_clip, "err": out_errors})
    means = df_err.groupby("clip")["err"].mean().to_dict()
    scores = np.array([means.get(ci, 0.0) for ci in te_indices])
    return scores


def run_step3_baseline(mtype, device, seeds=(0, 1, 2, 3, 4), n_jobs=5):
    """Step 3: Evaluate official DCASE baseline AE on dev train/test split.

    In the official DCASE 2020 Task 2 baseline (00_train.py / 01_test.py),
    a separate autoencoder is trained for each machine ID using only the
    normal training audio clips for that specific ID.
    """
    df = scan_machine_files(mtype)
    frames, clip_idx = build_dcase_baseline_feature_cache(mtype, df, n_jobs=n_jobs)

    groups = sorted(df[df["split"] == "test"]["group"].unique().tolist())
    print("\n[%s] Official Baseline Evaluation across groups %s" % (mtype, groups), flush=True)

    results = []
    for g in groups:
        # Official protocol: train AE only on normal training clips of this machine ID
        tr_mask = (df["split"] == "train") & (df["group"] == g) & (df["anomaly_label"] == 0)
        te_mask = (df["split"] == "test") & (df["group"] == g)

        tr_idx = np.where(tr_mask.to_numpy())[0]
        te_idx = np.where(te_mask.to_numpy())[0]
        y_test = df.loc[te_idx, "anomaly_label"].to_numpy()

        train_frames = frames[np.isin(clip_idx, tr_idx)]
        print("[%s Group %s] Training frames: %d, Test clips: %d (anom=%d)"
              % (mtype, g, len(train_frames), len(te_idx), int(y_test.sum())), flush=True)

        for seed in seeds:
            t0 = time.time()
            model = train_baseline_ae(train_frames, device, seed=seed)
            t_train = time.time() - t0

            scores = score_baseline_clips(model, frames, clip_idx, te_idx, device)
            auc = roc_auc_score(y_test, scores)
            pauc = roc_auc_score(y_test, scores, max_fpr=0.1)

            results.append({
                "machine_type": mtype,
                "group": g,
                "seed": seed,
                "auc": auc,
                "pauc": pauc,
                "n_test": len(y_test),
                "n_anom": int(y_test.sum()),
                "train_sec": round(t_train, 2)
            })
            print("[%s] Seed %d Group %s: AUC=%.4f, pAUC=%.4f (%.1fs)"
                  % (mtype, seed, g, auc, pauc, t_train), flush=True)

    res_df = pd.DataFrame(results)
    summ = res_df.groupby(["machine_type", "group"])[["auc", "pauc"]].agg(["mean", "std"])
    print("\n--- Summary for %s Baseline AE ---" % mtype, flush=True)
    print(summ.to_string(), flush=True)
    return res_df


# -------------------------------------------------------------------------
# Step 4: 74-dim Tabular Pipeline on Official Split
# -------------------------------------------------------------------------
def run_step4_tabular(mtype, seeds=(0, 1, 2, 3, 4), n_jobs=5):
    """Step 4: Run MAAD-Bench 74-dim tabular pipeline on official split per ID."""
    df = scan_machine_files(mtype)
    print("[%s] Extracting 74-dim tabular features for %d clips ..." % (mtype, len(df)), flush=True)

    # Use parallel_config to respect process limit
    with joblib.parallel_config(n_jobs=n_jobs):
        feat = mb.extract_features("dcase_%s" % mtype, df)

    groups = sorted(df[df["split"] == "test"]["group"].unique().tolist())
    print("[%s] Running 25 detectors on official split per ID (groups=%s) ..." % (mtype, groups), flush=True)

    X = feat.to_numpy(dtype=float)
    y = df["anomaly_label"].to_numpy()
    facs = mb.detector_factories()

    runs = []
    for g in groups:
        # Official split per ID:
        # Train on that ID's normal training clips
        tr_mask = (df["split"] == "train") & (df["group"] == g) & (df["anomaly_label"] == 0)
        te_mask = (df["split"] == "test") & (df["group"] == g)

        tr_idx = np.where(tr_mask.to_numpy())[0]
        te_idx = np.where(te_mask.to_numpy())[0]

        for seed in seeds:
            rows, _ = mb._sweep_split(
                X, y, tr_idx, te_idx, facs, seed, mb.CONTAMINATION,
                ctx={"machine_type": mtype, "group": g, "setting": "primary"}
            )
            runs.extend(rows)

    res_df = pd.DataFrame(runs)
    return res_df


# -------------------------------------------------------------------------
# Step 5: Paper AE Anchor Configuration on Official Split
# -------------------------------------------------------------------------
def extract_paper_ae_windows(path, n_mels=64, context=5, hop=512):
    """Paper AE anchor log-mel extraction (from ae_baseline.py)."""
    y, sr = librosa.load(path, sr=mb.SR, mono=True)
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop))
    T = S.shape[1]
    if T < context:
        S = np.concatenate([S, np.zeros((n_mels, context - T), dtype=S.dtype)], axis=1)
        T = context
    return np.stack([S[:, t:t + context].T.reshape(-1)
                     for t in range(T - context + 1)]).astype(np.float32)


def build_paper_ae_cache(mtype, df, n_jobs=5):
    """Build or load paper AE feature cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = df["path"].tolist()
    key = hashlib.md5(("paper_ae|" + "\n".join(paths)).encode()).hexdigest()[:10]
    cache_path = os.path.join(CACHE_DIR, "paper_ae_%s_%s.npz" % (mtype, key))
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["frames"], z["clip_idx"]

    print("[%s] extracting paper AE log-mel windows for %d clips ..." % (mtype, len(df)), flush=True)
    with joblib.parallel_config(n_jobs=n_jobs):
        per = Parallel(n_jobs=n_jobs, verbose=5)(delayed(extract_paper_ae_windows)(p) for p in paths)
    clip_idx = np.concatenate([np.full(len(a), i, dtype=np.int32) for i, a in enumerate(per)])
    frames = np.concatenate(per, axis=0)
    np.savez(cache_path, frames=frames, clip_idx=clip_idx)
    return frames, clip_idx


def train_paper_ae(Xtr, dim, device, seed, epochs=40, batch_size=1024, lr=1e-3):
    """Train paper AE (from ae_baseline.py)."""
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(),
        nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 32), nn.ReLU(),
        nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, dim)).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(Xtr, device=device)
    n = len(X)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        X_ep = X[perm]
        for i in range(0, n, batch_size):
            b = X_ep[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(b), b)
            loss.backward()
            opt.step()
    return net


def run_step5_paper_ae(mtype, device, seeds=(0, 1, 2, 3, 4), n_jobs=5):
    """Step 5: Run paper AE anchor configuration on official split."""
    df = scan_machine_files(mtype)
    frames, clip_idx = build_paper_ae_cache(mtype, df, n_jobs=n_jobs)
    groups = sorted(df[df["split"] == "test"]["group"].unique().tolist())
    print("\n[%s] Paper AE Anchor Evaluation on Official Split (groups=%s)" % (mtype, groups), flush=True)

    results = []
    for g in groups:
        tr_mask = (df["split"] == "train") & (df["group"] == g) & (df["anomaly_label"] == 0)
        te_mask = (df["split"] == "test") & (df["group"] == g)

        tr_idx = np.where(tr_mask.to_numpy())[0]
        te_idx = np.where(te_mask.to_numpy())[0]
        y_test = df.loc[te_idx, "anomaly_label"].to_numpy()

        trm = np.isin(clip_idx, tr_idx)
        tem = np.isin(clip_idx, te_idx)

        mu = frames[trm].mean(0)
        sd = frames[trm].std(0) + 1e-8
        norm_tr = (frames[trm] - mu) / sd
        norm_te = (frames[tem] - mu) / sd
        te_clip = clip_idx[tem]

        for seed in seeds:
            net = train_paper_ae(norm_tr, frames.shape[1], device, seed)
            net.eval()
            out_errors = np.empty(len(norm_te), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(norm_te), 4096):
                    b = torch.tensor(norm_te[i:i + 4096], device=device)
                    out_errors[i:i + 4096] = ((net(b) - b) ** 2).mean(dim=1).cpu().numpy()

            df_err = pd.DataFrame({"clip": te_clip, "err": out_errors})
            scores_dict = df_err.groupby("clip")["err"].mean().to_dict()
            scores = np.array([scores_dict[ci] for ci in te_idx])
            auc = roc_auc_score(y_test, scores)
            pauc = roc_auc_score(y_test, scores, max_fpr=0.1)

            results.append({
                "machine_type": mtype,
                "group": g,
                "seed": seed,
                "auc": auc,
                "pauc": pauc,
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
                "n_test_anom": int(y_test.sum())
            })
            print("[%s] Paper AE Seed %d Group %s: AUC=%.4f, pAUC=%.4f" % (mtype, seed, g, auc, pauc), flush=True)

    res_df = pd.DataFrame(results)
    return res_df


# -------------------------------------------------------------------------
# Step 6: Benchmark Sub-datasets (dcase_toycar and dcase_toyconveyor)
# -------------------------------------------------------------------------
def run_step6_subdataset(mtype, name, n_jobs=5):
    """Step 6: Register and run standard LOGO rotation on new sub-dataset."""
    print("\n===========================================================", flush=True)
    print("Step 6: Benchmark Sub-dataset %s (%s)" % (name, mtype), flush=True)
    print("===========================================================", flush=True)
    df = scan_machine_files(mtype)

    # Use all dev-set clips, organic label normal=0, anomaly=1, native groups=machine ID
    groups = sorted(df["group"].unique().tolist())
    print("[%s] Total clips: %d, Anomalies: %d, Native groups: %s"
          % (name, len(df), int(df["anomaly_label"].sum()), groups), flush=True)

    # 1. Extract 74-dim features
    with joblib.parallel_config(n_jobs=n_jobs):
        feat = mb.extract_features(name, df)

    # 2. Output directory
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # Save features.parquet and metadata.parquet
    feat.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    # metadata: path, group, category, anomaly_label
    meta_df = df[["path", "group", "category", "anomaly_label"]].copy()
    meta_df.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)

    # Save splits.parquet: row_index, group, dataset, test_fold
    splits_df = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "group": df["group"].values,
        "dataset": name,
        "test_fold": df["group"].values
    })
    splits_df.to_parquet(os.path.join(out_dir, "splits.parquet"), index=False)

    # 3. Run standard LOGO rotation (primary & secondary, 5 seeds, all 25 detectors)
    t0 = time.time()
    if n_jobs > 1 and len(groups) > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=min(n_jobs, len(groups)), verbose=5)(
            delayed(mb.run_rotation)(meta_df, feat, groups=[g],
                                     settings=("primary", "secondary"),
                                     subsample_test=False,
                                     collect_scores_for=("primary",))
            for g in groups)
        runs = pd.concat([p[0] for p in parts], ignore_index=True)
        clip_parts = [p[1] for p in parts if p[1] is not None]
        clips = pd.concat(clip_parts, ignore_index=True) if clip_parts else None
    else:
        runs, clips = mb.run_rotation(
            meta_df, feat,
            groups=groups,
            settings=("primary", "secondary"),
            subsample_test=False,
            collect_scores_for=("primary",)
        )
    elapsed = time.time() - t0

    runs.insert(0, "dataset", name)
    runs.to_parquet(os.path.join(out_dir, "rot_runs.parquet"), index=False)

    if clips is not None:
        clips.insert(0, "dataset", name)
        clips = clips.merge(meta_df[["path"]].reset_index().rename(
            columns={"index": "row_index"}), on="row_index", how="left")
        clips.to_parquet(os.path.join(out_dir, "rot_clip_scores.parquet"), index=False)

    summ = mb.summarize_rotation(runs, name)
    summ.to_csv(os.path.join(out_dir, "rot_metrics.csv"), index=False)

    skipped = runs[runs["detector"] == "*"]
    config_dict = {
        "protocol": "leave-one-group-out",
        "subsample_test": False,
        "groups": groups,
        "settings": ["primary", "secondary"],
        "seeds": mb.SEEDS,
        "contamination_train_side": mb.CONTAMINATION,
        "n_features": int(feat.shape[1]),
        "n_skipped_single_class_folds": int(len(skipped)),
        "elapsed_sec": round(elapsed, 1)
    }
    with open(os.path.join(out_dir, "rot_config.json"), "w") as fh:
        json.dump(config_dict, fh, indent=2)

    print("[%s] Saved all artifacts to %s in %.1fs" % (name, out_dir, elapsed), flush=True)
    cols = ["setting", "detector", "auc_roc_mean", "auc_roc_fold_sd", "auc_roc_seed_sd", "n_folds"]
    print("\nTop detectors on %s (primary):" % name, flush=True)
    print(summ[summ.setting == "primary"][cols].sort_values("auc_roc_mean", ascending=False).head(10).to_string(index=False), flush=True)
    return summ


# -------------------------------------------------------------------------
# Main CLI Driver
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="E2 DCASE Evaluation Driver")
    parser.add_argument("--step", default="all", choices=["3", "4", "5", "6", "all"])
    parser.add_argument("--machine-type", default="all",
                        choices=["all", "ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    seeds = tuple(int(s) for s in args.seeds.split(","))
    mtypes = MACHINE_TYPES if args.machine_type == "all" else [args.machine_type]

    out_records = os.path.join(mb.HERE, "records")
    os.makedirs(out_records, exist_ok=True)

    print("=================================================================", flush=True)
    print("E2 DCASE Driver: device=%s, jobs=%d, seeds=%s" % (device, args.jobs, seeds), flush=True)
    print("Machine Types: %s" % mtypes, flush=True)
    print("Steps: %s" % args.step, flush=True)
    print("=================================================================", flush=True)

    if args.step in ("3", "all"):
        print("\n>>> STEP 3: Official Baseline AE Re-implementation", flush=True)
        all_s3 = []
        for mt in mtypes:
            df_s3 = run_step3_baseline(mt, device, seeds=seeds, n_jobs=args.jobs)
            all_s3.append(df_s3)
        combined_s3 = pd.concat(all_s3, ignore_index=True)
        combined_s3.to_parquet(os.path.join(RESULTS_DIR, "e2_step3_baseline_runs.parquet"), index=False)
        print("\n>>> STEP 3 Completed. Saved to results/e2_step3_baseline_runs.parquet", flush=True)

    if args.step in ("4", "all"):
        print("\n>>> STEP 4: 74-dim Tabular Pipeline on Official Split", flush=True)
        all_s4 = []
        for mt in mtypes:
            df_s4 = run_step4_tabular(mt, seeds=seeds, n_jobs=args.jobs)
            all_s4.append(df_s4)
        combined_s4 = pd.concat(all_s4, ignore_index=True)
        combined_s4.to_parquet(os.path.join(RESULTS_DIR, "e2_step4_tabular_runs.parquet"), index=False)
        print("\n>>> STEP 4 Completed. Saved to results/e2_step4_tabular_runs.parquet", flush=True)

    if args.step in ("5", "all"):
        print("\n>>> STEP 5: Paper AE Anchor Configuration on Official Split", flush=True)
        all_s5 = []
        for mt in mtypes:
            df_s5 = run_step5_paper_ae(mt, device, seeds=seeds, n_jobs=args.jobs)
            all_s5.append(df_s5)
        combined_s5 = pd.concat(all_s5, ignore_index=True)
        combined_s5.to_parquet(os.path.join(RESULTS_DIR, "e2_step5_paper_ae_runs.parquet"), index=False)
        print("\n>>> STEP 5 Completed. Saved to results/e2_step5_paper_ae_runs.parquet", flush=True)

    if args.step in ("6", "all"):
        print("\n>>> STEP 6: New Benchmark Sub-datasets", flush=True)
        if "ToyCar" in mtypes:
            run_step6_subdataset("ToyCar", "dcase_toycar", n_jobs=args.jobs)
        if "ToyConveyor" in mtypes:
            run_step6_subdataset("ToyConveyor", "dcase_toyconveyor", n_jobs=args.jobs)
        print("\n>>> STEP 6 Completed.", flush=True)


if __name__ == "__main__":
    main()
