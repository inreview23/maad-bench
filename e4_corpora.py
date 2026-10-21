"""E4 Corpora: CREMA-D (emotional speech) and CirCor DigiScope (heart sound).

Unit E4 adds two new corpora with native groups to MAAD-Bench:
1. CREMA-D (designated-class, Cao et al. 2014, IEEE Trans. Affective Computing):
   - Normal = neutral (NEU); Anomaly = angry (ANG) and fearful (FEA).
   - Groups: 91 native actors (leave-one-actor-out, runtime estimate ~2.53h at 5 workers < 3h).
   - License: Open Database License (ODbL) v1.0.

2. CirCor DigiScope Phonocardiogram Dataset (organic-label, Oliveira et al. 2022, IEEE JBHI):
   - PhysioNet v1.0.3, open access.
   - Normal = murmur Absent; Anomaly = murmur Present; Unknown excluded (68 patients, 156 clips).
   - Groups: 5 patient buckets via patient_id % 5 (since 2 screening campaigns < 4 groups).
     Every patient and all their auscultation locations remain wholly inside one group.
   - Clip length handling: full clip without truncation (same as PhysioNet 2016 loader).
   - License: Open Data Commons Attribution License v1.0 (ODC-By v1.0).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import maad_bench as mb

# ----------------- Mappings Fixed in Advance -----------------
mb.DATASETS["cremad"] = dict(
    normal=["neutral"],
    anomaly=["angry", "fearful"],
    test_groups=[1001],
)

mb.DATASETS["circor"] = dict(
    normal=["absent"],
    anomaly=["present"],
    test_groups=[0],
)

# ----------------- Loaders -----------------
def load_cremad(bucketed=False):
    """Load CREMA-D AudioWAV files.
    
    File naming: <ActorID>_<Sentence>_<Emotion>_<Level>.wav
    Normal: neutral (NEU)
    Anomaly: angry (ANG), fearful (FEA)
    Other emotions (HAP, SAD, DIS) are excluded.
    """
    root = os.path.join(mb.DATA_DIR, "cremad", "AudioWAV")
    all_wavs = sorted(glob.glob(os.path.join(root, "*.wav")))
    if not all_wavs:
        raise FileNotFoundError(f"CREMA-D AudioWAV not found in {root}")
    
    emo_map = {"NEU": "neutral", "ANG": "angry", "FEA": "fearful"}
    rows = []
    for p in all_wavs:
        fname = os.path.basename(p)
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 4:
            continue
        aid, sent, emo, lev = parts
        if emo not in emo_map:
            continue
        actor_id = int(aid)
        cat = emo_map[emo]
        anom = 0 if emo == "NEU" else 1
        rows.append({
            "path": p,
            "actor_id": actor_id,
            "category": cat,
            "anomaly_label": anom
        })
    df = pd.DataFrame(rows)
    
    actors = sorted(df["actor_id"].unique())
    if bucketed:
        # 10 actor buckets by actor ID order, fixed rule, every actor wholly in one bucket
        actor_to_bucket = {aid: (i * 10 // len(actors)) for i, aid in enumerate(actors)}
        df["group"] = df["actor_id"].map(actor_to_bucket)
    else:
        # Full native leave-one-actor-out (91 groups)
        df["group"] = df["actor_id"]
    
    return df[["path", "group", "category", "anomaly_label"]]


def load_circor():
    """Load CirCor DigiScope heart-sound recordings.
    
    Organic label: murmur Present (anomaly) vs Absent (normal).
    Unknown murmur patients are excluded.
    Each patient has recordings at several auscultation locations (AV, MV, PV, TV).
    Group assignment: patient_id % 5 (5 patient buckets), keeping every patient wholly in one group.
    """
    root = os.path.join(mb.DATA_DIR, "circor")
    csv_path = os.path.join(root, "training_data.csv")
    wav_dir = os.path.join(root, "training_data")
    if not os.path.exists(csv_path) or not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"CirCor training data not found in {root}")
    
    cdf = pd.read_csv(csv_path)
    # Exclude Unknown
    unknown_mask = cdf["Murmur"] == "Unknown"
    valid_cdf = cdf[~unknown_mask].copy()
    
    patient_map = dict(zip(valid_cdf["Patient ID"], valid_cdf["Murmur"]))
    patient_to_group = {pid: (pid % 5) for pid in valid_cdf["Patient ID"]}
    
    all_wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    rows = []
    for p in all_wavs:
        fname = os.path.basename(p)
        pid = int(fname.split("_")[0])
        if pid not in patient_map:
            continue
        murmur = patient_map[pid]
        anom = 1 if murmur == "Present" else 0
        cat = "present" if anom == 1 else "absent"
        rows.append({
            "path": p,
            "group": patient_to_group[pid],
            "category": cat,
            "anomaly_label": anom
        })
    df = pd.DataFrame(rows)
    return df[["path", "group", "category", "anomaly_label"]]


mb.LOADERS["cremad"] = load_cremad
mb.LOADERS["circor"] = load_circor

# ----------------- Feature Extraction & Splits Helper -----------------
def extract_features_parallel(name, df, n_jobs=5):
    import hashlib
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(keysrc.encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, "%s_feat_%s.parquet" % (name, key))
    if os.path.exists(cache):
        print(f"[{name}] cached features found: {cache}", flush=True)
        return pd.read_parquet(cache)
    
    print(f"[{name}] extracting features for {len(df)} clips (n_jobs={n_jobs}) ...", flush=True)
    t0 = time.time()
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(mb.extract_one)(p) for p in df["path"])
    feat = pd.DataFrame(rows)
    feat.to_parquet(cache, index=False)
    print(f"[{name}] feature extraction done in {time.time()-t0:.1f}s, shape: {feat.shape}", flush=True)
    return feat


def write_splits(ds, md, out_dir):
    sp = md[["group"]].reset_index().rename(columns={"index": "row_index"})
    sp["dataset"] = ds
    sp["test_fold"] = sp["group"]
    p = os.path.join(out_dir, "splits.parquet")
    sp.to_parquet(p, index=False)
    print(f"[{ds}] wrote {p} ({len(sp)} rows)", flush=True)


# ----------------- Rotation Runner -----------------
def rotate_dataset(name, jobs=5, settings=("primary", "secondary"), collect=("primary",)):
    """Run standard rotation over all groups and write standard artifact schema."""
    out_dir = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    
    md_path = os.path.join(out_dir, "metadata.parquet")
    feat_path = os.path.join(out_dir, "features.parquet")
    
    if os.path.exists(md_path) and os.path.exists(feat_path):
        df = pd.read_parquet(md_path)
        feat = pd.read_parquet(feat_path)
    else:
        if name == "cremad":
            df = load_cremad()
        elif name == "circor":
            df = load_circor()
        else:
            df = mb.LOADERS[name]()
        df.to_parquet(md_path, index=False)
        feat = extract_features_parallel(name, df, n_jobs=jobs)
        feat.to_parquet(feat_path, index=False)
        write_splits(name, df, out_dir)
    
    groups = sorted(df["group"].unique().tolist())
    print(f"[{name}] Starting rotation: {len(df)} clips, {int(df['anomaly_label'].sum())} anomalies, {len(groups)} groups: {groups[:5]}... (jobs={jobs})", flush=True)
    
    t0 = time.time()
    if jobs > 1 and len(groups) > 1:
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
    runs.to_parquet(os.path.join(out_dir, "rot_runs.parquet"), index=False)
    
    if clips is not None:
        clips.insert(0, "dataset", name)
        clips = clips.merge(df[["path"]].reset_index().rename(columns={"index": "row_index"}),
                            on="row_index", how="left")
        clips.to_parquet(os.path.join(out_dir, "rot_clip_scores.parquet"), index=False)
    
    summ = mb.summarize_rotation(runs, name)
    summ.to_csv(os.path.join(out_dir, "rot_metrics.csv"), index=False)
    
    skipped = runs[runs["detector"] == "*"]
    meta = {
        "protocol": "leave-one-group-out",
        "subsample_test": False,
        "groups": groups,
        "settings": list(settings),
        "seeds": mb.SEEDS,
        "contamination_train_side": mb.CONTAMINATION,
        "n_features": int(feat.shape[1]),
        "n_skipped_single_class_folds": int(len(skipped)),
        "elapsed_sec": round(elapsed, 1)
    }
    with open(os.path.join(out_dir, "rot_config.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    
    print(f"[{name}] rotation complete in {elapsed:.1f}s; {len(runs)} run rows, {len(skipped)} skipped folds, {0 if clips is None else len(clips)} clip rows", flush=True)
    return summ

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cremad", "circor", "both"])
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    
    datasets = ["circor", "cremad"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        print(f"\n================ Running rotation for {ds} ================", flush=True)
        rotate_dataset(ds, jobs=args.jobs)
