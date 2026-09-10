"""MAAD-Bench multi-dataset runner (supersedes esc50_slice.py).

One feature schema, one group-safe split discipline, one detector sweep across
ESC-50, UrbanSound8K, and RAVDESS. Two settings per task: normal-only training
(primary) and contaminated ADBench-style training (secondary), over >= 5 seeds.
Writes leakage-safe artifacts (features / metadata / scores) and a metrics table
per dataset, plus a combined cross-dataset summary.

PRE-REGISTERED CLASS MAPPINGS (placeholder -- confirm with domain experts).
Split units: ESC-50 fold, UrbanSound8K fold (both group same source recording in
one fold), RAVDESS actor (actor-disjoint).

Usage:
  python maad_bench.py --dataset esc50
  python maad_bench.py --dataset ravdess
  python maad_bench.py --dataset urbansound8k
  python maad_bench.py --dataset all
  python maad_bench.py --dataset ravdess --stage features   # extract+cache only
"""
import argparse
import glob
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SR = 22050
N_MFCC = 20
CONTAMINATION = 0.10
SEEDS = [0, 1, 2, 3, 4]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CACHE_DIR = os.path.join(HERE, "cache")
RESULTS_DIR = os.path.join(HERE, "results")

# ------------------------- pre-registered mappings -------------------------
DATASETS = {
    "esc50": dict(
        normal=["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
                "water_drops", "wind", "pouring_water", "thunderstorm", "insects"],
        anomaly=["glass_breaking", "siren", "fireworks"],
        test_groups=[5],
    ),
    "urbansound8k": dict(
        normal=["air_conditioner", "engine_idling", "jackhammer", "drilling",
                "street_music", "children_playing"],
        anomaly=["gun_shot", "car_horn", "siren"],
        test_groups=[10],
    ),
    "ravdess": dict(
        normal=["neutral", "calm"],
        anomaly=["angry", "fearful"],
        test_groups=[21, 22, 23, 24],
    ),
    "nsynth": dict(
        normal=["acoustic"],
        anomaly=["electronic", "synthetic"],
        test_groups=[0],   # group = instrument id % 5; instruments stay disjoint
    ),
    "mimii": dict(
        normal=["normal"],
        anomaly=["abnormal"],
        # hold out pump id_06, the v1 fold; groups are (machine type, ID) pairs
        # because the four IDs recur across all four machine types
        test_groups=["pump-id_06"],
    ),
    "physionet": dict(
        normal=["normal"],
        anomaly=["abnormal"],
        test_groups=[5],   # hold out subset 'f'; PhysioNet subsets are population-disjoint
    ),
    "icbhi": dict(
        normal=["normal"],
        anomaly=["crackle", "wheeze", "both"],
        test_groups=[0],   # group = patient id % 5; patients stay disjoint
    ),
}


# --------------------------------- loaders ---------------------------------
def _download(url, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_esc50():
    url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    root = os.path.join(DATA_DIR, "ESC-50-master")
    if not os.path.exists(os.path.join(root, "meta", "esc50.csv")):
        zp = _download(url, os.path.join(DATA_DIR, "esc50.zip"))
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(DATA_DIR)
    meta = pd.read_csv(os.path.join(root, "meta", "esc50.csv"))
    cfg = DATASETS["esc50"]
    meta = meta[meta["category"].isin(set(cfg["normal"]) | set(cfg["anomaly"]))].copy()
    return pd.DataFrame({
        "path": meta["filename"].map(lambda f: os.path.join(root, "audio", f)),
        "group": meta["fold"].to_numpy(),
        "category": meta["category"].to_numpy(),
        "anomaly_label": meta["category"].isin(cfg["anomaly"]).astype(int).to_numpy(),
    })


def load_urbansound8k():
    root = os.path.join(DATA_DIR, "UrbanSound8K")
    csv = os.path.join(root, "metadata", "UrbanSound8K.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(
            "UrbanSound8K not found at %s. Download requires a form/mirror; place the "
            "extracted dataset there (metadata/UrbanSound8K.csv + audio/fold*/)." % root)
    meta = pd.read_csv(csv)
    cfg = DATASETS["urbansound8k"]
    meta = meta[meta["class"].isin(set(cfg["normal"]) | set(cfg["anomaly"]))].copy()
    return pd.DataFrame({
        "path": [os.path.join(root, "audio", "fold%d" % f, n)
                 for f, n in zip(meta["fold"], meta["slice_file_name"])],
        "group": meta["fold"].to_numpy(),
        "category": meta["class"].to_numpy(),
        "anomaly_label": meta["class"].isin(cfg["anomaly"]).astype(int).to_numpy(),
    })


def load_ravdess():
    url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    root = os.path.join(DATA_DIR, "RAVDESS")
    if not glob.glob(os.path.join(root, "Actor_*", "*.wav")):
        zp = _download(url, os.path.join(DATA_DIR, "ravdess.zip"))
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(root)
    emo = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
           "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
    cfg = DATASETS["ravdess"]
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "Actor_*", "*.wav"))):
        parts = os.path.splitext(os.path.basename(p))[0].split("-")
        if len(parts) != 7:
            continue
        cat = emo.get(parts[2])
        if cat not in set(cfg["normal"]) | set(cfg["anomaly"]):
            continue
        rows.append({"path": p, "group": int(parts[6]), "category": cat,
                     "anomaly_label": int(cat in cfg["anomaly"])})
    return pd.DataFrame(rows)


def load_nsynth():
    import tarfile
    root = os.path.join(DATA_DIR, "nsynth-valid")
    ex = os.path.join(root, "examples.json")
    if not os.path.exists(ex):
        tgz = os.path.join(DATA_DIR, "nsynth-valid.jsonwav.tar.gz")
        if not os.path.exists(tgz):
            raise FileNotFoundError("NSynth valid not found; expected %s" % ex)
        with tarfile.open(tgz) as tf:
            tf.extractall(DATA_DIR)
    with open(ex) as fh:
        meta = json.load(fh)
    src = {0: "acoustic", 1: "electronic", 2: "synthetic"}
    cfg = DATASETS["nsynth"]
    keep = set(cfg["normal"]) | set(cfg["anomaly"])
    rows = []
    for note, m in meta.items():
        cat = src.get(m["instrument_source"])
        if cat not in keep:
            continue
        rows.append({
            "path": os.path.join(root, "audio", note + ".wav"),
            "group": int(m["instrument"]) % 5,
            "category": cat,
            "anomaly_label": int(cat in cfg["anomaly"]),
        })
    return pd.DataFrame(rows)


def load_mimii():
    """MIMII at 6 dB, grouped by (machine type, machine ID).

    Defaults to the pump subset, which is what the paper reports and what the
    released feature matrices and embedding caches were built from. The group is
    the (type, ID) pair rather than the ID alone, because the same four IDs
    (00, 02, 04, 06) recur in every machine type: grouping on the ID would put a
    fan, a pump, a slider and a valve in one fold and call it a held-out machine.
    Widening the roster therefore changes the fold structure as well as the row
    count, which is why it is opt-in.
    """
    import re
    # The reported build is the 6 dB pump subset, so that is the default: a loader
    # that silently returned sixteen folds where the paper reports four would make
    # the tables irreproducible. Set MAAD_MIMII_TYPES to a comma-separated list
    # (or "all") to widen it; the pretrained-embedding caches are keyed on the
    # pump-only path list and would need regenerating from audio if you do.
    types = os.environ.get("MAAD_MIMII_TYPES", "pump")
    types = ["fan", "pump", "slider", "valve"] if types == "all" else \
        [t.strip() for t in types.split(",") if t.strip()]
    wavs = []
    for t in types:
        wavs += glob.glob(os.path.join(DATA_DIR, "mimii_%s" % t, "**", "id_*",
                                       "*", "*.wav"), recursive=True)
    if not wavs:
        raise FileNotFoundError(
            "MIMII machine types %s not found under %s. Extract 6_dB_<type>.zip "
            "into data/mimii_<type>/ (expects id_*/{normal,abnormal}/*.wav)."
            % (",".join(types), DATA_DIR))
    rows = []
    for p in sorted(wavs):
        cond = os.path.basename(os.path.dirname(p))
        if cond not in ("normal", "abnormal"):
            continue
        mt = re.search(r"mimii_([a-z]+)", p.replace("\\", "/"))
        mid = re.search(r"id_(\d+)", p)
        mtype = mt.group(1) if mt else "unknown"
        gid = mid.group(1) if mid else "??"
        is_abn = cond == "abnormal"
        rows.append({"path": p, "group": "%s-id_%s" % (mtype, gid),
                     "machine_type": mtype,
                     "category": "abnormal" if is_abn else "normal",
                     "anomaly_label": int(is_abn)})
    return pd.DataFrame(rows)


def load_physionet():
    # PhysioNet/CinC 2016 heart sound. Organic label: -1 normal, +1 abnormal.
    # Group by training subset (a..f), which are population-disjoint (Codex gate:
    # use the official population-disjoint split, no patient key needed).
    root = os.path.join(DATA_DIR, "physionet2016")
    refs = glob.glob(os.path.join(root, "**", "REFERENCE.csv"), recursive=True)
    if not refs:
        zp = os.path.join(DATA_DIR, "physionet2016_training.zip")
        if not os.path.exists(zp):
            raise FileNotFoundError("PhysioNet/CinC 2016 not found; expected %s" % zp)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(root)
        refs = glob.glob(os.path.join(root, "**", "REFERENCE.csv"), recursive=True)
    letters = "abcdefghij"
    rows = []
    for ref in sorted(refs):
        sub = os.path.dirname(ref)
        letter = os.path.basename(sub).split("-")[-1][:1]
        gid = letters.index(letter) if letter in letters else 0
        with open(ref) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                name, lab = parts[0].strip(), parts[1].strip()
                p = os.path.join(sub, name + ".wav")
                if not os.path.exists(p):
                    continue
                is_abn = (lab == "1")
                rows.append({"path": p, "group": gid,
                             "category": "abnormal" if is_abn else "normal",
                             "anomaly_label": int(is_abn)})
    return pd.DataFrame(rows)


def load_icbhi():
    # ICBHI 2017 respiratory sounds. Organic label: a respiratory cycle is
    # anomalous if it carries a crackle or a wheeze. Cycle-level samples via
    # (offset, duration) into each recording; grouped by patient (id % 5) so a
    # patient never spans train and test.
    root = os.path.join(DATA_DIR, "ICBHI", "ICBHI_final_database")
    rows = []
    for wav in sorted(glob.glob(os.path.join(root, "*.wav"))):
        base = os.path.basename(wav)[:-4]
        txt = os.path.join(root, base + ".txt")
        if not os.path.exists(txt):
            continue
        try:
            patient = int(base.split("_")[0])
        except ValueError:
            continue
        ann = pd.read_csv(txt, sep="\t", header=None,
                          names=["start", "end", "crackle", "wheeze"])
        for _, r in ann.iterrows():
            dur = float(r["end"]) - float(r["start"])
            if dur <= 0:
                continue
            crk, whz = int(r["crackle"]), int(r["wheeze"])
            cat = ("both" if crk and whz else "crackle" if crk
                   else "wheeze" if whz else "normal")
            rows.append({"path": wav, "offset": float(r["start"]), "duration": dur,
                         "group": patient % 5, "category": cat,
                         "anomaly_label": int(crk or whz)})
    return pd.DataFrame(rows)


LOADERS = {"esc50": load_esc50, "urbansound8k": load_urbansound8k,
           "ravdess": load_ravdess, "nsynth": load_nsynth, "mimii": load_mimii,
           "physionet": load_physionet, "icbhi": load_icbhi}


# --------------------------- feature extraction ---------------------------
def _summarize(name, arr):
    m, s = arr.mean(axis=1), arr.std(axis=1)
    out = {}
    for i in range(arr.shape[0]):
        out["%s%d_mean" % (name, i)] = float(m[i])
        out["%s%d_std" % (name, i)] = float(s[i])
    return out


def extract_one(path, offset=0.0, duration=None):
    import librosa
    y, sr = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    f = {}
    f.update(_summarize("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)))
    f.update(_summarize("chroma", librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0)))
    f.update(_summarize("cent", librosa.feature.spectral_centroid(y=y, sr=sr)))
    f.update(_summarize("bw", librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    f.update(_summarize("roll", librosa.feature.spectral_rolloff(y=y, sr=sr)))
    f.update(_summarize("zcr", librosa.feature.zero_crossing_rate(y)))
    f.update(_summarize("rms", librosa.feature.rms(y=y)))
    return f


def extract_features(name, df):
    import hashlib
    os.makedirs(CACHE_DIR, exist_ok=True)
    has_seg = "offset" in df.columns
    if has_seg:
        keysrc = "\n".join("%s|%s|%s" % (p, o, d)
                           for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(keysrc.encode()).hexdigest()[:10]
    cache = os.path.join(CACHE_DIR, "%s_feat_%s.parquet" % (name, key))
    if os.path.exists(cache):
        print("cached features:", cache, flush=True)
        return pd.read_parquet(cache)
    from joblib import Parallel, delayed
    print("extracting features for %d clips ..." % len(df), flush=True)
    if has_seg:
        rows = Parallel(n_jobs=-1, verbose=5)(
            delayed(extract_one)(p, o, d)
            for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        rows = Parallel(n_jobs=-1, verbose=5)(delayed(extract_one)(p) for p in df["path"])
    feat = pd.DataFrame(rows)
    feat.to_parquet(cache, index=False)
    return feat


# --------------------------------- splits ---------------------------------
def make_split(df, seed, setting, test_groups, subsample_test=True,
               contamination=None):
    """Group-disjoint split.

    subsample_test=True (default, the v1 protocol) draws test anomalies down to
    CONTAMINATION of the test set. That discards positives for no benefit under
    AUC-ROC, which is prevalence-invariant, and on small held-out groups it left
    as few as 5 positives. subsample_test=False keeps every anomaly in the
    held-out group and is the protocol used by run_rotation(). The training side
    is unaffected either way: CONTAMINATION still governs the secondary-setting
    injection below.
    """
    c = CONTAMINATION if contamination is None else float(contamination)
    rng = np.random.RandomState(seed)
    in_test = df["group"].isin(test_groups).to_numpy()
    is_anom = (df["anomaly_label"] == 1).to_numpy()
    idx = np.arange(len(df))

    test_normal = idx[in_test & ~is_anom]
    pool = idx[in_test & is_anom]
    if subsample_test:
        n_anom = max(1, int(round(c / (1 - c) * len(test_normal))))
        test_anom = rng.choice(pool, size=min(n_anom, len(pool)), replace=False)
    else:
        test_anom = pool
    test_idx = np.concatenate([test_normal, test_anom])

    train_normal = idx[~in_test & ~is_anom]
    if setting == "primary":
        train_idx = train_normal
    else:
        pool_tr = idx[~in_test & is_anom]
        n_tr = min(len(pool_tr),
                   max(1, int(round(c / (1 - c) * len(train_normal)))))
        train_idx = np.concatenate([train_normal, rng.choice(pool_tr, size=n_tr, replace=False)])
    return train_idx, test_idx


# -------------------------------- detectors --------------------------------
def detector_factories():
    """Build the PyOD detector roster. Constructor args are passed by signature
    introspection (contamination / random_state only when accepted), so detectors
    with incompatible APIs are skipped at build time rather than polluting results.
    """
    import importlib
    import inspect
    specs = [
        # proximity / density
        ("KNN", "pyod.models.knn", "KNN"),
        ("LOF", "pyod.models.lof", "LOF"),
        ("COF", "pyod.models.cof", "COF"),
        ("CBLOF", "pyod.models.cblof", "CBLOF"),
        ("ABOD", "pyod.models.abod", "ABOD"),
        ("SOD", "pyod.models.sod", "SOD"),
        ("KDE", "pyod.models.kde", "KDE"),
        ("Sampling", "pyod.models.sampling", "Sampling"),
        # linear / subspace
        ("PCA", "pyod.models.pca", "PCA"),
        ("KPCA", "pyod.models.kpca", "KPCA"),
        ("MCD", "pyod.models.mcd", "MCD"),
        ("OCSVM", "pyod.models.ocsvm", "OCSVM"),
        ("LMDD", "pyod.models.lmdd", "LMDD"),
        # probabilistic / statistical
        ("HBOS", "pyod.models.hbos", "HBOS"),
        ("COPOD", "pyod.models.copod", "COPOD"),
        ("ECOD", "pyod.models.ecod", "ECOD"),
        ("QMCD", "pyod.models.qmcd", "QMCD"),
        ("GMM", "pyod.models.gmm", "GMM"),
        # ensemble / tree
        ("INNE", "pyod.models.inne", "INNE"),
        ("IForest", "pyod.models.iforest", "IForest"),
        ("LODA", "pyod.models.loda", "LODA"),
        ("FeatureBagging", "pyod.models.feature_bagging", "FeatureBagging"),
        # neural
        ("AutoEncoder", "pyod.models.auto_encoder", "AutoEncoder"),
        ("VAE", "pyod.models.vae", "VAE"),
        ("DIF", "pyod.models.dif", "DIF"),
        ("LUNAR", "pyod.models.lunar", "LUNAR"),
    ]
    facs = {}
    for name, mod, cls in specs:
        try:
            klass = getattr(importlib.import_module(mod), cls)
            params = inspect.signature(klass.__init__).parameters
        except Exception:
            continue

        def mk(klass=klass, params=params, name=name):
            def build(c, rs):
                kw = {}
                if "contamination" in params:
                    kw["contamination"] = c
                if "random_state" in params:
                    kw["random_state"] = rs
                if name == "ABOD" and "method" in params:
                    kw["method"] = "fast"  # O(n^2) exact ABOD is too slow at scale
                return klass(**kw)
            return build

        b = mk()
        try:
            b(0.1, 0)  # test-construct; skip detectors whose API rejects these args
            facs[name] = b
        except Exception:
            pass
    return facs


NAN_METRICS = {"auc_roc": float("nan"), "avg_precision": float("nan"),
               "pauc_0.1": float("nan"), "f1_top_q": float("nan")}


def is_scorable(y):
    """A test fold needs both classes present for AUC-ROC to be defined."""
    y = np.asarray(y)
    return len(y) > 0 and 0 < int(y.sum()) < len(y)


def metrics(y, scores, contamination):
    """Four metrics for one (fold, seed, detector) cell.

    Returns NaNs plus a reason when the fold is single-class rather than letting
    roc_auc_score raise. Under the v1 protocol every fold was forced to 10%
    prevalence so this could not happen; rotating over every native group makes
    it reachable, and an uncaught raise there would blank all 25 detectors of
    that fold at once.

    f1_top_q flags a fixed budget of k = round(contamination * n_test) clips.
    Note that under the v1 protocol the test set was itself subsampled to that
    same prevalence, so k equalled the true positive count for six of seven
    sub-datasets and precision, recall and F1 coincided by construction. The
    realized prevalence is recorded alongside so that degeneracy is visible.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    y = np.asarray(y)
    if not is_scorable(y):
        return dict(NAN_METRICS, skipped="single_class_fold",
                    n_flagged=0, test_prevalence=float(y.mean()) if len(y) else float("nan"))
    out = {"auc_roc": float(roc_auc_score(y, scores)),
           "avg_precision": float(average_precision_score(y, scores))}
    try:
        out["pauc_0.1"] = float(roc_auc_score(y, scores, max_fpr=0.1))
    except Exception:
        out["pauc_0.1"] = float("nan")
    k = max(1, int(round(contamination * len(y))))
    thresh = np.sort(scores)[::-1][k - 1]
    out["f1_top_q"] = float(f1_score(y, (scores >= thresh).astype(int), zero_division=0))
    out["n_flagged"] = int(k)
    out["test_prevalence"] = float(y.mean())
    return out


def make_pca_pre(n_components=64, random_state=0):
    """Fold-local dimensionality reduction for the pretrained-embedding arms.

    panns_embed.py and embed.py called PCA(...).fit_transform on the complete
    embedding matrix and only split afterwards, justified in a comment as
    leakage-safe because PCA sees no labels. Label-blindness is not the issue:
    in an inductive protocol whose held-out group is population-disjoint by
    design, letting the test rows determine the projection axes is leakage. This
    fits the projection on the training rows of the current fold only.
    """
    def pre(X, tr, te):
        from sklearn.decomposition import PCA
        k = min(n_components, X.shape[1], len(tr))
        p = PCA(n_components=k, random_state=random_state).fit(X[tr])
        return p.transform(X[tr]), p.transform(X[te])
    return pre


def _sweep_split(X, y, tr, te, facs, seed, contamination, ctx=None, collect=False,
                 pre=None):
    """Fit and score every detector on one already-chosen split.

    Shared by run() and run_rotation() so the two protocols cannot drift apart.
    Any pre transform and the scaler are both fitted on the training rows only.
    When collect is set, the per-clip decision scores are kept instead of being
    discarded; the v1 code passed decision_function(...) straight into metrics()
    without binding it, which is why the released scores.parquet holds run-level
    metrics rather than the per-clip scores the paper claims to release.
    """
    from sklearn.preprocessing import StandardScaler
    ctx = ctx or {}
    Xtr_raw, Xte_raw = pre(X, tr, te) if pre is not None else (X[tr], X[te])
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte, yte = sc.transform(Xtr_raw), sc.transform(Xte_raw), y[te]
    rows, clips = [], []
    for name, make in facs.items():
        s = None
        try:
            # LODA takes no random_state and draws its random cuts from the
            # global NumPy generator inside fit, so without this its value
            # depends on how far that generator has advanced, which is to say
            # on the call order of whichever driver is running. It is the only
            # detector of the 25 that does this: measured across six
            # sub-datasets, running the secondary setting alone rather than
            # after the primary moved LODA by up to 0.018 and every other
            # detector by at most 3e-6.
            np.random.seed(seed)
            det = make(contamination, seed)
            det.fit(Xtr)
            s = np.asarray(det.decision_function(Xte), dtype=float)
            m = metrics(yte, s, contamination)
        except Exception as e:
            s = None
            m = dict(NAN_METRICS, error=type(e).__name__)
        m.update({"detector": name, "seed": seed, "n_train": len(tr),
                  "n_test": len(te), "n_test_anom": int(yte.sum())})
        m.update(ctx)
        rows.append(m)
        if collect and s is not None:
            clip = {"detector": name, "seed": seed,
                    "row_index": np.asarray(te, dtype=np.int32),
                    "anomaly_label": np.asarray(yte, dtype=np.int8),
                    "score": s.astype(np.float32)}
            clip.update(ctx)
            clips.append(pd.DataFrame(clip))
    return rows, clips


def run(df, feat, test_groups):
    """v1 protocol: one fixed held-out group, test anomalies subsampled to
    CONTAMINATION. Kept unchanged so the rotation results can be regressed
    against the published tables."""
    X = feat.to_numpy(dtype=float)
    y = df["anomaly_label"].to_numpy()
    facs = detector_factories()
    print("detectors:", ", ".join(facs), flush=True)
    rec = []
    for setting in ("primary", "secondary"):
        for seed in SEEDS:
            tr, te = make_split(df, seed, setting, test_groups)
            rows, _ = _sweep_split(X, y, tr, te, facs, seed, CONTAMINATION,
                                   ctx={"setting": setting})
            rec.extend(rows)
    return pd.DataFrame(rec)


def summarize(scores_df, dataset):
    agg = (scores_df.groupby(["setting", "detector"])
           [["auc_roc", "avg_precision", "pauc_0.1", "f1_top_q"]].agg(["mean", "std"]))
    agg.columns = ["%s_%s" % (a, b) for a, b in agg.columns]
    agg = agg.reset_index()
    agg.insert(0, "dataset", dataset)
    return agg.sort_values(["setting", "auc_roc_mean"], ascending=[True, False])


def run_rotation(df, feat, groups=None, settings=("primary", "secondary"),
                 subsample_test=False, collect_scores_for=("primary",), pre=None,
                 contamination=None):
    """Leave-one-group-out over every native group.

    The v1 protocol held out one pre-registered group per sub-dataset and never
    rotated it, so the seeds shared a training set and the group axis carried no
    variance at all. Rotating shows how much the choice of held-out group was
    doing: on PhysioNet a single HBOS fit moves between 0.162 and 0.839 across
    the six subsets.

    Returns (runs_df, clip_df). runs_df carries one row per
    (group, setting, seed, detector); clip_df carries one row per test clip for
    the settings named in collect_scores_for, or None when nothing was collected.
    """
    c = CONTAMINATION if contamination is None else float(contamination)
    X = feat.to_numpy(dtype=float)
    y = df["anomaly_label"].to_numpy()
    if groups is None:
        groups = sorted(df["group"].unique().tolist())
    facs = detector_factories()
    print("rotation: %d groups x %d detectors x %d seeds x %d settings"
          % (len(groups), len(facs), len(SEEDS), len(settings)), flush=True)
    rec, clips = [], []
    for g in groups:
        for setting in settings:
            for seed in SEEDS:
                tr, te = make_split(df, seed, setting, [g],
                                    subsample_test=subsample_test,
                                    contamination=c)
                ctx = {"setting": setting, "group": g}
                if not is_scorable(y[te]):
                    rec.append(dict(NAN_METRICS, detector="*", seed=seed,
                                    n_train=len(tr), n_test=len(te),
                                    n_test_anom=int(y[te].sum()),
                                    skipped="single_class_fold", **ctx))
                    print("  skip group=%s %s seed=%d: single-class test fold "
                          "(n_test=%d, n_anom=%d)"
                          % (g, setting, seed, len(te), int(y[te].sum())), flush=True)
                    continue
                rows, cl = _sweep_split(X, y, tr, te, facs, seed, c,
                                        ctx=ctx, collect=setting in collect_scores_for,
                                        pre=pre)
                rec.extend(rows)
                clips.extend(cl)
        print("  group %s done" % g, flush=True)
    clip_df = pd.concat(clips, ignore_index=True) if clips else None
    return pd.DataFrame(rec), clip_df


def summarize_rotation(runs_df, dataset):
    """Aggregate seeds within a fold first, then folds.

    Emits the between-fold SD alongside the within-fold seed SD. The published
    tables reported only the second, which measured the anomaly subsample and
    detector randomness under one fixed split; the first is the uncertainty that
    matters for a claim about a new task.
    """
    METS = ["auc_roc", "avg_precision", "pauc_0.1", "f1_top_q"]
    d = runs_df[runs_df["detector"] != "*"].copy()
    per_fold = (d.groupby(["setting", "detector", "group"])[METS]
                .agg(["mean", "std"]))
    per_fold.columns = ["%s_%s" % (a, b) for a, b in per_fold.columns]
    per_fold = per_fold.reset_index()

    # With subsample_test=False the test side is fixed by the group, so n_test
    # and n_test_anom are constant across seeds; first() is exact, and this
    # avoids groupby.apply's pandas-version-dependent include_groups argument.
    counts = (d.groupby(["setting", "detector", "group"])[["n_test", "n_test_anom"]]
              .first().reset_index())
    counts["pair_weight"] = ((counts["n_test"] - counts["n_test_anom"])
                             * counts["n_test_anom"])
    per_fold = per_fold.merge(counts[["setting", "detector", "group", "pair_weight"]],
                              on=["setting", "detector", "group"], how="left")

    out = []
    for (setting, det), t in per_fold.groupby(["setting", "detector"]):
        row = {"dataset": dataset, "setting": setting, "detector": det,
               "n_folds": int(t["group"].nunique())}
        for m in METS:
            fold_means = t["%s_mean" % m]
            row["%s_mean" % m] = float(fold_means.mean())
            row["%s_fold_sd" % m] = float(fold_means.std())
            row["%s_seed_sd" % m] = float(t["%s_std" % m].mean())
        w = t["pair_weight"].to_numpy(dtype=float)
        a = t["auc_roc_mean"].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
        row["auc_roc_pairweighted"] = (float((a[ok] * w[ok]).sum() / w[ok].sum())
                                       if ok.any() else float("nan"))
        out.append(row)
    agg = pd.DataFrame(out)
    return agg.sort_values(["setting", "auc_roc_mean"], ascending=[True, False])


def run_dataset(name):
    cfg = DATASETS[name]
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    df = LOADERS[name]()
    # group-safety: no group straddles the train/test line (it cannot, groups are atomic)
    print("[%s] %d clips, %d anomalies, groups=%d, test_groups=%s" % (
        name, len(df), int(df["anomaly_label"].sum()), df["group"].nunique(), cfg["test_groups"]),
        flush=True)
    feat = extract_features(name, df)
    feat.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    df[["path", "group", "category", "anomaly_label"]].to_parquet(
        os.path.join(out_dir, "metadata.parquet"), index=False)
    scores = run(df, feat, cfg["test_groups"])
    scores.to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)
    summ = summarize(scores, name)
    summ.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({**cfg, "contamination": CONTAMINATION, "seeds": SEEDS,
                   "n_features": feat.shape[1]}, fh, indent=2)
    print("\n=== %s summary (mean over seeds) ===" % name, flush=True)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summ.to_string(index=False), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["esc50", "urbansound8k", "ravdess", "nsynth", "mimii",
                             "physionet", "icbhi", "all"])
    ap.add_argument("--stage", choices=["all", "features"], default="all")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ALL = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
    names = ALL if args.dataset == "all" else [args.dataset]
    summaries = []
    for name in names:
        try:
            if args.stage == "features":
                df = LOADERS[name]()
                extract_features(name, df)
                print("[%s] features done" % name, flush=True)
            else:
                summaries.append(run_dataset(name))
        except Exception as e:
            print("[%s] FAILED: %s: %s" % (name, type(e).__name__, e), flush=True)
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "metrics_combined.csv"), index=False)
        print("\nwrote combined summary to", os.path.join(RESULTS_DIR, "metrics_combined.csv"), flush=True)


if __name__ == "__main__":
    main()
