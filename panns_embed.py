"""MAAD-Bench pretrained-embedding arm. Replaces the 74-dim handcrafted feature
vector with a 2048-dim PANNs CNN14 embedding (pretrained on AudioSet), then runs
the SAME PyOD detector sweep. This tests whether the "classical detectors are
competitive" finding is an artifact of the weak handcrafted features.

Requires: pip install panns_inference (pulls torchlibrosa and downloads CNN14
weights on first use). Run from the repository root on a GPU machine. CNN14
expects 32 kHz mono audio.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

PANNS_SR = 32000
ALLNAMES = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]


def embed_dataset(name, at):
    import librosa
    df = mb.LOADERS[name]()
    key = hashlib.md5(("panns|" + "\n".join(df["path"].astype(str))).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, "%s_panns_%s.parquet" % (name, key))
    if os.path.exists(cache):
        return df, pd.read_parquet(cache)
    embs = np.empty((len(df), 2048), dtype=np.float32)
    print("[%s] extracting PANNs embeddings for %d clips ..." % (name, len(df)), flush=True)
    for i, p in enumerate(df["path"]):
        y, _ = librosa.load(p, sr=PANNS_SR, mono=True)
        if len(y) < PANNS_SR:
            y = np.pad(y, (0, PANNS_SR - len(y)))
        _, emb = at.inference(y[None, :])
        embs[i] = emb[0]
        if i % 1000 == 0 and i:
            print("  %d/%d" % (i, len(df)), flush=True)
    feat = pd.DataFrame(embs, columns=["panns%d" % j for j in range(2048)])
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    feat.to_parquet(cache, index=False)
    return df, feat


def run(name, at):
    from sklearn.decomposition import PCA
    cfg = mb.DATASETS[name]
    df, feat = embed_dataset(name, at)
    print("[%s] panns embeddings %s" % (name, feat.shape), flush=True)
    # Reduce the 2048-dim embedding to a tractable, schema-comparable dimension.
    # Unsupervised PCA (no labels), so it is leakage-safe; the full 25-detector
    # sweep is intractable at 2048 dims (MCD covariance, KPCA kernel).
    k = min(64, feat.shape[1], feat.shape[0])
    red = PCA(n_components=k, random_state=0).fit_transform(feat.to_numpy(dtype=float))
    feat = pd.DataFrame(red, columns=["pc%d" % j for j in range(k)])
    print("[%s] reduced to %s via PCA" % (name, feat.shape), flush=True)
    scores = mb.run(df, feat, cfg["test_groups"])
    summ = mb.summarize(scores, name)
    out = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    scores.to_parquet(os.path.join(out, "panns_scores.parquet"), index=False)
    summ.to_csv(os.path.join(out, "panns_metrics.csv"), index=False)
    print("\n=== %s PANNs-embedding sweep (mean over seeds) ===" % name, flush=True)
    print(summ[summ["setting"] == "primary"][["detector", "auc_roc_mean"]].to_string(index=False),
          flush=True)


def main():
    from panns_inference import AudioTagging
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=ALLNAMES + ["all"])
    args = ap.parse_args()
    at = AudioTagging(checkpoint_path=None, device="cuda")
    names = ALLNAMES if args.dataset == "all" else [args.dataset]
    for n in names:
        try:
            run(n, at)
        except Exception as e:
            print("[%s] FAILED %s: %s" % (n, type(e).__name__, e), flush=True)


if __name__ == "__main__":
    main()
