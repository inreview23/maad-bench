"""E6: Pretrained-representation arms (AST, PANNs) and Autoencoder anchor
for cremad, circor_rec, dcase_toycar, and dcase_toyconveyor.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import maad_bench as mb
import rotate_repr
import embed
import panns_embed
import ae_baseline

# -----------------------------------------------------------------------------
# 1. Dataset Registration
# -----------------------------------------------------------------------------
def register_all():
    """Register cremad, circor, circor_rec, dcase_toycar, dcase_toyconveyor."""
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
    mb.DATASETS["circor_rec"] = dict(
        normal=["absent"],
        anomaly=["present"],
        test_groups=[0],
    )
    mb.DATASETS["dcase_toycar"] = dict(
        normal=["normal"],
        anomaly=["anomaly"],
        test_groups=[1],
    )
    mb.DATASETS["dcase_toyconveyor"] = dict(
        normal=["normal"],
        anomaly=["anomaly"],
        test_groups=[1],
    )

    for ds in ["cremad", "circor", "circor_rec", "dcase_toycar", "dcase_toyconveyor"]:
        mb.LOADERS[ds] = (lambda d=ds: pd.read_parquet(os.path.join(mb.RESULTS_DIR, d, "metadata.parquet")))

register_all()

# -----------------------------------------------------------------------------
# 2. Embedding Extraction
# -----------------------------------------------------------------------------
def extract_ast_embeddings(name, df, batch=16):
    """Extract AST embeddings on GPU and cache to cache/<name>_ast_<key>.parquet."""
    import librosa
    keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(("ast|" + keysrc).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, f"{name}_ast_{key}.parquet")
    if os.path.exists(cache):
        print(f"[{name}/ast] cached embeddings found: {cache}", flush=True)
        return pd.read_parquet(cache)

    embed_batch, sr, dim, _ = embed.build_ast()
    embs = np.empty((len(df), dim), dtype=np.float32)
    bi, bw = [], []

    def flush():
        if bw:
            e = embed_batch(bw)
            for j, idx in enumerate(bi):
                embs[idx] = e[j]
        bi.clear()
        bw.clear()

    print(f"[{name}/ast] extracting embeddings for {len(df)} clips (batch={batch}) ...", flush=True)
    t0 = time.time()
    for i in range(len(df)):
        p = df["path"].iloc[i]
        y, _ = librosa.load(p, sr=sr, mono=True)
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)))
        bi.append(i)
        bw.append(y)
        if len(bw) >= batch:
            flush()
        if i % 1000 == 0 and i:
            print(f"  {i}/{len(df)} ({time.time()-t0:.1f}s)", flush=True)
    flush()
    feat = pd.DataFrame(embs, columns=[f"e{j}" for j in range(dim)])
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    feat.to_parquet(cache, index=False)
    print(f"[{name}/ast] saved {feat.shape} to {cache} in {time.time()-t0:.1f}s", flush=True)
    return feat


def extract_panns_embeddings(name, df):
    """Extract PANNs CNN14 embeddings on GPU and cache to cache/<name>_panns_<key>.parquet."""
    import librosa
    from panns_inference import AudioTagging

    keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(("panns|" + keysrc).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, f"{name}_panns_{key}.parquet")
    if os.path.exists(cache):
        print(f"[{name}/panns] cached embeddings found: {cache}", flush=True)
        return pd.read_parquet(cache)

    at = AudioTagging(checkpoint_path=None, device="cuda")
    embs = np.empty((len(df), 2048), dtype=np.float32)
    print(f"[{name}/panns] extracting embeddings for {len(df)} clips ...", flush=True)
    t0 = time.time()
    for i, p in enumerate(df["path"]):
        y, _ = librosa.load(p, sr=panns_embed.PANNS_SR, mono=True)
        if len(y) < panns_embed.PANNS_SR:
            y = np.pad(y, (0, panns_embed.PANNS_SR - len(y)))
        _, emb = at.inference(y[None, :])
        embs[i] = emb[0]
        if i % 1000 == 0 and i:
            print(f"  {i}/{len(df)} ({time.time()-t0:.1f}s)", flush=True)
    feat = pd.DataFrame(embs, columns=[f"panns{j}" for j in range(2048)])
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    feat.to_parquet(cache, index=False)
    print(f"[{name}/panns] saved {feat.shape} to {cache} in {time.time()-t0:.1f}s", flush=True)
    return feat


def subset_circor_rec_embeddings(backbone):
    """Subset circor embeddings by path to create circor_rec embeddings."""
    df_rec = pd.read_parquet(os.path.join(mb.RESULTS_DIR, "circor_rec", "metadata.parquet"))
    df_circ = pd.read_parquet(os.path.join(mb.RESULTS_DIR, "circor", "metadata.parquet"))

    pat = os.path.join(mb.CACHE_DIR, f"circor_{backbone}_*.parquet")
    hits = [p for p in sorted(glob.glob(pat)) if "features_n" not in p and "circor_rec" not in p]
    if not hits:
        raise FileNotFoundError(f"No circor_{backbone} cache found to subset!")
    emb_circ = pd.read_parquet(hits[0])

    path_to_idx = {p: i for i, p in enumerate(df_circ["path"])}
    indices = [path_to_idx[p] for p in df_rec["path"]]
    emb_rec = emb_circ.iloc[indices].reset_index(drop=True)

    keysrc = "\n".join(df_rec["path"].astype(str))
    key = hashlib.md5((f"{backbone}|" + keysrc).encode()).hexdigest()[:10]
    out_cache = os.path.join(mb.CACHE_DIR, f"circor_rec_{backbone}_{key}.parquet")
    emb_rec.to_parquet(out_cache, index=False)
    print(f"[circor_rec/{backbone}] created from circor subset: {emb_rec.shape} -> {out_cache}", flush=True)
    return emb_rec


# -----------------------------------------------------------------------------
# 3. AE Anchor for DCASE Machine Sound Datasets
# -----------------------------------------------------------------------------
def prepare_dcase_ae_cache():
    """Ensure ae_baseline.build_frames finds pre-extracted log-mel windows."""
    for ds, mtype in [("dcase_toycar", "ToyCar"), ("dcase_toyconveyor", "ToyConveyor")]:
        df = pd.read_parquet(os.path.join(mb.RESULTS_DIR, ds, "metadata.parquet"))
        paths = df["path"].astype(str).tolist()
        spec = [(p, 0.0, None) for p in paths]
        key = hashlib.md5(
            ("ae2|" + "\n".join("%s@%.3f+%s" % (p, o, d) for p, o, d in spec)).encode()
        ).hexdigest()[:10]
        target_cache = os.path.join(mb.CACHE_DIR, f"{ds}_ae_{key}.npz")
        if os.path.exists(target_cache):
            print(f"[{ds}/ae] cache already exists: {target_cache}", flush=True)
            continue

        pat = os.path.join(mb.CACHE_DIR, f"paper_ae_{mtype}_*.npz")
        hits = sorted(glob.glob(pat))
        if hits:
            print(f"[{ds}/ae] Converting {hits[0]} -> {target_cache} ...", flush=True)
            z = np.load(hits[0])
            np.savez(target_cache, frames=z["frames"], clip=z["clip_idx"])
            print(f"[{ds}/ae] Wrote {target_cache}", flush=True)
        else:
            print(f"[{ds}/ae] Pre-building frames from raw audio ...", flush=True)
            ae_baseline.build_frames(ds, df)


def run_ae(name, settings=("primary",)):
    """Run log-mel autoencoder anchor on GPU (settings=('primary',))."""
    return ae_baseline.run(name, settings=settings)


# -----------------------------------------------------------------------------
# 4. Detector Rotation
# -----------------------------------------------------------------------------
def run_rotation(name, backbone, pca_dim=64, jobs=4, settings=("primary", "secondary"),
                 collect=("primary",)):
    """Run rotated detector sweep using rotate_repr.rotate_one."""
    return rotate_repr.rotate_one(name, backbone, pca_dim=pca_dim, settings=settings,
                                  jobs=jobs, collect=collect)


def benchmark_cremad_one_fold(backbone, pca_dim=64):
    """Benchmark 1 fold of CREMA-D to estimate full 91-fold wall-clock runtime."""
    df, emb, src = rotate_repr.load_embedding("cremad", backbone)
    groups = sorted(df["group"].unique().tolist())
    pre = mb.make_pca_pre(pca_dim) if pca_dim > 0 else None
    g0 = groups[0]
    print(f"[cremad/{backbone}] Benchmarking 1 fold (group={g0}) out of {len(groups)} ...", flush=True)
    t0 = time.time()
    runs, _ = mb.run_rotation(df, emb, groups=[g0], settings=("primary", "secondary"),
                              subsample_test=False, collect_scores_for=("primary",), pre=pre)
    dt = time.time() - t0
    est_total_4workers = (dt * len(groups)) / 4.0
    print(f"[cremad/{backbone}] 1 fold completed in {dt:.1f}s. "
          f"Projected {len(groups)} folds with 4 workers: {est_total_4workers:.1f}s ({est_total_4workers/3600:.2f}h)", flush=True)
    return dt, est_total_4workers


# -----------------------------------------------------------------------------
# 5. Tabulation
# -----------------------------------------------------------------------------
def tabulate_all():
    """Tabulate comparison across handcrafted, AST, PANNs, and AE anchor."""
    datasets = ["cremad", "circor_rec", "dcase_toycar", "dcase_toyconveyor"]
    records = []

    for ds in datasets:
        row = {"subdataset": ds}
        
        # 1. Handcrafted
        h_path = os.path.join(mb.RESULTS_DIR, ds, "rot_metrics.csv")
        if os.path.exists(h_path):
            hdf = pd.read_csv(h_path)
            for setg in ["primary", "secondary"]:
                sub = hdf[hdf["setting"] == setg]
                if not sub.empty:
                    best_r = sub.loc[sub["auc_roc_mean"].idxmax()]
                    row[f"hand_{setg}_best_det"] = best_r["detector"]
                    row[f"hand_{setg}_best_auc"] = best_r["auc_roc_mean"]
                    row[f"hand_{setg}_med_auc"] = sub["auc_roc_mean"].median()

        # 2. AST
        ast_path = os.path.join(mb.RESULTS_DIR, ds, "rot_ast_metrics.csv")
        if os.path.exists(ast_path):
            adf = pd.read_csv(ast_path)
            for setg in ["primary", "secondary"]:
                sub = adf[adf["setting"] == setg]
                if not sub.empty:
                    best_r = sub.loc[sub["auc_roc_mean"].idxmax()]
                    row[f"ast_{setg}_best_det"] = best_r["detector"]
                    row[f"ast_{setg}_best_auc"] = best_r["auc_roc_mean"]
                    row[f"ast_{setg}_med_auc"] = sub["auc_roc_mean"].median()

        # 3. PANNs
        panns_path = os.path.join(mb.RESULTS_DIR, ds, "rot_panns_metrics.csv")
        if os.path.exists(panns_path):
            pdf = pd.read_csv(panns_path)
            for setg in ["primary", "secondary"]:
                sub = pdf[pdf["setting"] == setg]
                if not sub.empty:
                    best_r = sub.loc[sub["auc_roc_mean"].idxmax()]
                    row[f"panns_{setg}_best_det"] = best_r["detector"]
                    row[f"panns_{setg}_best_auc"] = best_r["auc_roc_mean"]
                    row[f"panns_{setg}_med_auc"] = sub["auc_roc_mean"].median()

        # 4. AE Anchor (for dcase)
        ae_path = os.path.join(mb.RESULTS_DIR, ds, "ae_rot_metrics.csv")
        if os.path.exists(ae_path):
            aedf = pd.read_csv(ae_path)
            aedf = aedf[aedf["setting"] == "primary"] if "setting" in aedf.columns else aedf
            if not aedf.empty:
                row["ae_anchor_primary_auc"] = aedf.iloc[0]["auc_roc_mean"]
                row["ae_anchor_primary_sd"] = aedf.iloc[0]["auc_roc_fold_sd"]
        
        records.append(row)

    res_df = pd.DataFrame(records)
    print("\n=== E6 Summary Table ===", flush=True)
    print(res_df.to_string(), flush=True)
    return res_df


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="E6 Pretrained Representation Arms Driver")
    parser.add_argument("--stage", required=True,
                        choices=["extract_ast", "extract_panns", "ae", "rotate",
                                 "benchmark_cremad", "rotate_cremad", "tabulate", "all_gpu"])
    parser.add_argument("--dataset", default="all",
                        choices=["all", "cremad", "circor", "circor_rec", "dcase_toycar", "dcase_toyconveyor"])
    parser.add_argument("--backbone", default="all", choices=["all", "ast", "panns"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    register_all()
    os.makedirs(os.path.join(mb.HERE, "logs"), exist_ok=True)

    if args.stage == "extract_ast":
        datasets = ["circor", "cremad", "dcase_toycar", "dcase_toyconveyor"] if args.dataset == "all" else [args.dataset]
        for ds in datasets:
            if ds == "circor_rec":
                subset_circor_rec_embeddings("ast")
            else:
                df = mb.LOADERS[ds]()
                extract_ast_embeddings(ds, df, batch=args.batch)
                if ds == "circor":
                    subset_circor_rec_embeddings("ast")

    elif args.stage == "extract_panns":
        datasets = ["circor", "cremad", "dcase_toycar", "dcase_toyconveyor"] if args.dataset == "all" else [args.dataset]
        for ds in datasets:
            if ds == "circor_rec":
                subset_circor_rec_embeddings("panns")
            else:
                df = mb.LOADERS[ds]()
                extract_panns_embeddings(ds, df)
                if ds == "circor":
                    subset_circor_rec_embeddings("panns")

    elif args.stage == "ae":
        prepare_dcase_ae_cache()
        datasets = ["dcase_toycar", "dcase_toyconveyor"] if args.dataset == "all" else [args.dataset]
        for ds in datasets:
            if ds in ["dcase_toycar", "dcase_toyconveyor"]:
                run_ae(ds, settings=("primary",))

    elif args.stage == "all_gpu":
        print("=== Running all GPU stages: AST, PANNs, AE ===", flush=True)
        # AST
        for ds in ["circor", "cremad", "dcase_toycar", "dcase_toyconveyor"]:
            df = mb.LOADERS[ds]()
            extract_ast_embeddings(ds, df, batch=args.batch)
            if ds == "circor":
                subset_circor_rec_embeddings("ast")
        # PANNs
        for ds in ["circor", "cremad", "dcase_toycar", "dcase_toyconveyor"]:
            df = mb.LOADERS[ds]()
            extract_panns_embeddings(ds, df)
            if ds == "circor":
                subset_circor_rec_embeddings("panns")
        # AE
        prepare_dcase_ae_cache()
        for ds in ["dcase_toycar", "dcase_toyconveyor"]:
            run_ae(ds, settings=("primary",))

    elif args.stage == "rotate":
        datasets = ["circor_rec", "dcase_toycar", "dcase_toyconveyor"] if args.dataset == "all" else [args.dataset]
        backbones = ["ast", "panns"] if args.backbone == "all" else [args.backbone]
        for b in backbones:
            for ds in datasets:
                print(f"\n>>> Running rotation for {ds} with {b} (jobs={args.jobs}) ...", flush=True)
                run_rotation(ds, b, pca_dim=64, jobs=args.jobs, settings=("primary", "secondary"), collect=("primary",))

    elif args.stage == "benchmark_cremad":
        backbones = ["ast", "panns"] if args.backbone == "all" else [args.backbone]
        for b in backbones:
            benchmark_cremad_one_fold(b, pca_dim=64)

    elif args.stage == "rotate_cremad":
        backbones = ["ast", "panns"] if args.backbone == "all" else [args.backbone]
        for b in backbones:
            print(f"\n>>> Running full CREMA-D rotation for {b} (91 folds, jobs={args.jobs}) ...", flush=True)
            run_rotation("cremad", b, pca_dim=64, jobs=args.jobs, settings=("primary", "secondary"), collect=("primary",))

    elif args.stage == "tabulate":
        tabulate_all()


if __name__ == "__main__":
    main()
