"""MAAD-Bench temporal-detector arm. Instead of collapsing each clip to one
74-dim mean/std vector, this keeps the per-frame feature SEQUENCE (37 features x
T frames) and runs PyOD's time-series detectors (ts_*) at the clip level. This
tests the paper's axis-2 claim: a fixed-length summary washes out transient
events, so detectors that look at temporal structure should help on transient
anomalies (glass break, siren, gunshot) and less on stationary textures.

Two families, mapped to two anomaly types:
  * reference-free, scored per clip on its own series (SpectralResidual,
    MatrixProfile): no training set used; clip score = max per-frame score.
    Sensitive to within-clip transients.
  * train-on-normal (TimeSeriesOD bridge over a PyOD base, LSTMAD): fit one
    detector on concatenated NORMAL training frames, then score each test clip.

Writes results/<name>/temporal_metrics.csv with the same schema as the tabular
metrics.csv (so make_table-style aggregation can read it). Run from anywhere with
the project python; paths are __file__-relative via maad_bench.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

WINDOW = 16                 # small, so short ICBHI cycles still yield windows
CAP_TRAIN_FRAMES = 120000   # bound the train-on-normal fit cost
ALLNAMES = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]


def frames_one(path, offset=0.0, duration=None):
    import librosa
    y, sr = librosa.load(path, sr=mb.SR, mono=True, offset=offset, duration=duration)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    parts = [
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mb.N_MFCC),
        librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
    ]
    T = min(p.shape[1] for p in parts)
    M = np.vstack([p[:, :T] for p in parts]).T  # (T, 37)
    return M.astype(np.float32)


def extract_frames(name):
    df = mb.LOADERS[name]()
    has_seg = "offset" in df.columns
    if has_seg:
        keysrc = "\n".join("%s|%s|%s" % (p, o, d)
                           for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(("frames|" + keysrc).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, "%s_frames_%s.npz" % (name, key))
    if os.path.exists(cache):
        z = np.load(cache)
        flat, off = z["flat"], z["offsets"]
        return df, [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    from joblib import Parallel, delayed
    print("[%s] extracting frame sequences for %d clips ..." % (name, len(df)), flush=True)
    if has_seg:
        mats = Parallel(n_jobs=-1, verbose=5)(
            delayed(frames_one)(p, o, d)
            for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        mats = Parallel(n_jobs=-1, verbose=5)(delayed(frames_one)(p) for p in df["path"])
    lengths = [m.shape[0] for m in mats]
    flat = np.vstack(mats).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    np.savez(cache, flat=flat, offsets=offsets)
    return df, mats


def _ts(modname, clsname):
    import importlib
    return getattr(importlib.import_module("pyod.models." + modname), clsname)


def detector_specs(which):
    SpectralResidual = _ts("ts_spectral_residual", "SpectralResidual")
    MatrixProfile = _ts("ts_matrix_profile", "MatrixProfile")
    TimeSeriesOD = _ts("ts_od", "TimeSeriesOD")
    specs = {
        "TempSR": ("selffit", "flux", lambda: SpectralResidual(score_window=3, channel_aggregation="max")),
        "TempMP": ("selffit", "flux", lambda: MatrixProfile(window_size=WINDOW, channel_aggregation="max")),
        "TempIF": ("trainnormal", "multi", lambda: TimeSeriesOD(detector="IForest", window_size=WINDOW,
                                                                step=4, score_aggregation="max")),
    }
    if "lstm" in which:
        LSTMAD = _ts("ts_lstm", "LSTMAD")
        specs["TempLSTM"] = ("trainnormal", "multi", lambda: LSTMAD(window_size=WINDOW, epochs=8))
    keep = [k for k in specs if k.lower().replace("temp", "") in which or which == ["all"]]
    return {k: specs[k] for k in (keep or list(specs))}


def clip_score(det, series):
    """Max per-position anomaly score over a clip's frame series; robust to the
    different lengths ts_* decision_function returns and to too-short clips."""
    if series.shape[0] < 4:
        return np.nan
    try:
        det.fit(series)
        s = np.asarray(det.decision_function(series), dtype=float).ravel()
        s = s[np.isfinite(s)]
        return float(s.max()) if s.size else np.nan
    except Exception:
        return np.nan


def score_with(det, series):
    if series.shape[0] < 4:
        return np.nan
    try:
        s = np.asarray(det.decision_function(series), dtype=float).ravel()
        s = s[np.isfinite(s)]
        return float(s.max()) if s.size else np.nan
    except Exception:
        return np.nan


def fill_nan(v):
    v = np.asarray(v, dtype=float)
    if np.isnan(v).any():
        med = np.nanmedian(v)
        v = np.where(np.isnan(v), med if np.isfinite(med) else 0.0, v)
    return v


def _flux(m):
    return np.r_[0.0, np.linalg.norm(np.diff(m, axis=0), axis=1)].reshape(-1, 1).astype(np.float32)


def _fit_scaler(mats, tr, seed, cap=CAP_TRAIN_FRAMES):
    """Per-channel standardization fitted on TRAINING clips only.

    v1 fitted this on np.vstack(mats), every clip in the dataset, and justified
    it in a comment as unsupervised. Label-blindness is not the question: the
    held-out group is population-disjoint by design, so letting its frames set
    the channel means and scales is leakage. The frame budget is capped the same
    way the detector fit is, which is ample for two moments per channel.
    """
    rng = np.random.RandomState(seed)
    chunks, total = [], 0
    for i in rng.permutation(tr):
        chunks.append(mats[i])
        total += mats[i].shape[0]
        if total >= cap:
            break
    return StandardScaler().fit(np.vstack(chunks))


def run(name, which, groups=None):
    """Group-rotated temporal arm with fold-local standardization."""
    df, mats = extract_frames(name)
    y = df["anomaly_label"].to_numpy()
    if groups is None:
        groups = sorted(df["group"].unique().tolist())
    specs = detector_specs(which)
    print("[%s] temporal detectors: %s (%d clips, %d groups)"
          % (name, ", ".join(specs), len(mats), len(groups)), flush=True)

    rec = []
    for g in groups:
        for setting in ("primary", "secondary"):
            for seed in mb.SEEDS:
                tr, te = mb.make_split(df, seed, setting, [g], subsample_test=False)
                ctx = {"setting": setting, "seed": seed, "group": g,
                       "n_train": len(tr), "n_test": len(te),
                       "n_test_anom": int(y[te].sum())}
                if not mb.is_scorable(y[te]):
                    rec.append(dict(mb.NAN_METRICS, detector="*",
                                    skipped="single_class_fold", **ctx))
                    continue

                scaler = _fit_scaler(mats, tr, seed)
                cache = {}

                def ser(kind, i):
                    key = (kind, i)
                    if key not in cache:
                        m = scaler.transform(mats[i]).astype(np.float32)
                        cache[key] = _flux(m) if kind == "flux" else m
                    return cache[key]

                for det_name, (mode, kind, build) in specs.items():
                    try:
                        if mode == "selffit":
                            s = fill_nan([clip_score(build(), ser(kind, i)) for i in te])
                        else:
                            rng = np.random.RandomState(seed)
                            chunks, total = [], 0
                            for i in rng.permutation(tr):
                                chunks.append(ser(kind, i))
                                total += chunks[-1].shape[0]
                                if total >= CAP_TRAIN_FRAMES:
                                    break
                            # TimeSeriesOD exposes no random_state and builds its
                            # base IForest with random_state=None, so the fit draws
                            # from the global NumPy RNG. Without this the arm is not
                            # reproducible: two runs of the same code and seeds
                            # differed by up to 0.009 AUC on TempIF while the
                            # deterministic TempSR agreed to four decimals.
                            np.random.seed(seed)
                            det = build()
                            det.fit(np.vstack(chunks))
                            s = fill_nan([score_with(det, ser(kind, i)) for i in te])
                        m = mb.metrics(y[te], s, mb.CONTAMINATION)
                    except Exception as e:
                        m = dict(mb.NAN_METRICS, error=type(e).__name__)
                    m.update({"detector": det_name}, **ctx)
                    rec.append(m)
        print("  group %s done" % g, flush=True)

    runs = pd.DataFrame(rec)
    runs.insert(0, "dataset", name)
    d = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    runs.to_parquet(os.path.join(d, "rot_temporal_runs.parquet"), index=False)
    out = mb.summarize_rotation(runs, name)
    out.to_csv(os.path.join(d, "rot_temporal_metrics.csv"), index=False)
    print("\n=== %s temporal, group-rotated (primary AUC-ROC) ===" % name, flush=True)
    print(out[out["setting"] == "primary"]
          [["detector", "auc_roc_mean", "auc_roc_fold_sd", "n_folds"]]
          .to_string(index=False), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=ALLNAMES + ["all"])
    ap.add_argument("--detectors", default="sr,if",
                    help="comma list from sr,mp,if,lstm or 'all'")
    args = ap.parse_args()
    which = [w.strip().lower() for w in args.detectors.split(",")]
    names = ALLNAMES if args.dataset == "all" else [args.dataset]
    for nm in names:
        try:
            run(nm, which)
        except Exception as e:
            print("[%s] FAILED %s: %s" % (nm, type(e).__name__, e), flush=True)


if __name__ == "__main__":
    main()
