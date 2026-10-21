"""E1: MIMII fan, slider and valve as three new organic-label sub-datasets.

Implements loaders, feature extraction, group rotation, representation arms (AST, PANNs),
log-mel autoencoder anchor, and comparative analysis for MIMII machine types.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import maad_bench as mb
import rotate
import rotate_repr
import embed
import panns_embed
import ae_baseline

MIMII_TYPES = ["fan", "slider", "valve"]
MIMII_DATASETS = ["mimii_fan", "mimii_slider", "mimii_valve"]
ALL_MIMII_DATASETS = ["mimii", "mimii_fan", "mimii_slider", "mimii_valve"]


def load_mimii_machine(mtype):
    """Load one MIMII 6 dB machine type with machine ID as native group.

    Ordering: normal files sorted by path, then abnormal files sorted by path.
    This exactly replicates the row order and schema of results/mimii/ (pump).
    Groups: 0, 2, 4, 6 (integer machine IDs).
    """
    root = os.path.realpath(os.path.join(mb.DATA_DIR, f"mimii_{mtype}"))
    norm_files = sorted(glob.glob(os.path.join(root, "**", "normal", "*.wav"), recursive=True))
    abn_files = sorted(glob.glob(os.path.join(root, "**", "abnormal", "*.wav"), recursive=True))
    if not norm_files or not abn_files:
        raise FileNotFoundError(f"MIMII {mtype} wavs not found under {root}")
    rows = []
    for p in norm_files:
        m = re.search(r"id_(\d+)", p)
        gid = int(m.group(1)) if m else 0
        rows.append({"path": p, "group": gid, "category": "normal", "anomaly_label": 0})
    for p in abn_files:
        m = re.search(r"id_(\d+)", p)
        gid = int(m.group(1)) if m else 0
        rows.append({"path": p, "group": gid, "category": "abnormal", "anomaly_label": 1})
    return pd.DataFrame(rows)


def register_all():
    """Register all MIMII machine types in mb.DATASETS and mb.LOADERS."""
    for t in ["fan", "slider", "valve", "pump"]:
        ds_name = "mimii" if t == "pump" else f"mimii_{t}"
        mb.DATASETS[ds_name] = dict(
            normal=["normal"],
            anomaly=["abnormal"],
            test_groups=[6],
        )
        mb.LOADERS[ds_name] = (lambda m=t: load_mimii_machine(m))
    mb.DATASETS["e1_parity_pump"] = dict(
        normal=["normal"],
        anomaly=["abnormal"],
        test_groups=[6],
    )
    mb.LOADERS["e1_parity_pump"] = (lambda: load_mimii_machine("pump"))


def write_splits(name, out_dir):
    """Write splits.parquet under results/<name>/."""
    md_path = os.path.join(out_dir, "metadata.parquet")
    md = pd.read_parquet(md_path)
    sp = md[["group"]].reset_index().rename(columns={"index": "row_index"})
    sp["dataset"] = name
    sp["test_fold"] = sp["group"]
    sp.to_parquet(os.path.join(out_dir, "splits.parquet"), index=False)


def prepare_features_and_metadata(name, mtype):
    """Extract and cache features, write metadata, splits, and config."""
    out_dir = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    df = load_mimii_machine(mtype)
    print(f"[{name}] {len(df)} clips, {int(df['anomaly_label'].sum())} anomalies, groups={sorted(df['group'].unique())}", flush=True)

    feat = mb.extract_features(name, df)
    feat.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    df[["path", "group", "category", "anomaly_label"]].to_parquet(
        os.path.join(out_dir, "metadata.parquet"), index=False)
    write_splits(name, out_dir)

    cfg = mb.DATASETS[name]
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({**cfg, "contamination": mb.CONTAMINATION, "seeds": mb.SEEDS,
                   "n_features": feat.shape[1]}, fh, indent=2)
    return df, feat


def run_rotation_only(name, jobs=4, settings=("primary", "secondary"), collect=("primary",)):
    """Run rotation using rotate.rotate_one assuming features & metadata exist."""
    return rotate.rotate_one(name, settings=settings, jobs=jobs, collect=collect)


def run_features_and_rotation(name, mtype, jobs=4, settings=("primary", "secondary"),
                              collect=("primary",)):
    """Extract features and run rotation for one sub-dataset."""
    prepare_features_and_metadata(name, mtype)
    summ = rotate.rotate_one(name, settings=settings, jobs=jobs, collect=collect)
    return summ


def extract_ast_embeddings(name, df, batch=16):
    """Extract AST embeddings on GPU and cache to cache/<name>_ast_<key>.parquet."""
    import librosa
    keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(("ast|" + keysrc).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, f"{name}_ast_{key}.parquet")
    if os.path.exists(cache):
        print(f"[{name}/ast] cached embeddings {cache}", flush=True)
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

    print(f"[{name}/ast] extracting embeddings for {len(df)} clips ...", flush=True)
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
            print(f"  {i}/{len(df)}", flush=True)
    flush()
    feat = pd.DataFrame(embs, columns=[f"e{j}" for j in range(dim)])
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    feat.to_parquet(cache, index=False)
    print(f"[{name}/ast] saved embeddings to {cache}", flush=True)
    return feat


def extract_panns_embeddings(name, df):
    """Extract PANNs CNN14 embeddings on GPU and cache to cache/<name>_panns_<key>.parquet."""
    import librosa
    from panns_inference import AudioTagging

    keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(("panns|" + keysrc).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, f"{name}_panns_{key}.parquet")
    if os.path.exists(cache):
        print(f"[{name}/panns] cached embeddings {cache}", flush=True)
        return pd.read_parquet(cache)

    at = AudioTagging(checkpoint_path=None, device="cuda")
    embs = np.empty((len(df), 2048), dtype=np.float32)
    print(f"[{name}/panns] extracting embeddings for {len(df)} clips ...", flush=True)
    for i, p in enumerate(df["path"]):
        y, _ = librosa.load(p, sr=panns_embed.PANNS_SR, mono=True)
        if len(y) < panns_embed.PANNS_SR:
            y = np.pad(y, (0, panns_embed.PANNS_SR - len(y)))
        _, emb = at.inference(y[None, :])
        embs[i] = emb[0]
        if i % 1000 == 0 and i:
            print(f"  {i}/{len(df)}", flush=True)
    feat = pd.DataFrame(embs, columns=[f"panns{j}" for j in range(2048)])
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    feat.to_parquet(cache, index=False)
    print(f"[{name}/panns] saved embeddings to {cache}", flush=True)
    return feat


def run_representation(name, backbone, pca_dim=64, jobs=4, settings=("primary", "secondary"),
                       collect=("primary",)):
    """Run rotated representation arm (PCA-64 per fold training rows)."""
    return rotate_repr.rotate_one(name, backbone, pca_dim=pca_dim, settings=settings,
                                  jobs=jobs, collect=collect)


def prebuild_ae_frames(name, mtype):
    """Pre-build log-mel frames on CPU so GPU run only trains networks."""
    df = load_mimii_machine(mtype)
    frames, clip = ae_baseline.build_frames(name, df)
    print(f"[{name}/ae_frames] built {len(frames)} frames for {len(df)} clips", flush=True)
    return frames, clip


def run_ae(name, settings=("primary",)):
    """Run log-mel autoencoder anchor on GPU (settings=('primary',))."""
    return ae_baseline.run(name, settings=settings)


def run_parity():
    """Step 2: Parity check between results/e1_parity_pump/ and results/mimii/."""
    register_all()
    print("=== Running Step 2: Parity check for pump ===", flush=True)
    out_dir = os.path.join(mb.RESULTS_DIR, "e1_parity_pump")
    os.makedirs(out_dir, exist_ok=True)

    df = load_mimii_machine("pump")
    df_mimii = pd.read_parquet(os.path.join(mb.RESULTS_DIR, "mimii", "metadata.parquet"))
    assert df.equals(df_mimii), "Metadata DataFrame does not match results/mimii/metadata.parquet!"
    print("Metadata parity: PASSED (identical)", flush=True)

    keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(keysrc.encode()).hexdigest()[:10]
    orig_cache = os.path.join(mb.CACHE_DIR, f"mimii_feat_{key}.parquet")
    parity_cache = os.path.join(mb.CACHE_DIR, f"e1_parity_pump_feat_{key}.parquet")
    if os.path.exists(orig_cache) and not os.path.exists(parity_cache):
        os.symlink(orig_cache, parity_cache)

    feat = mb.extract_features("e1_parity_pump", df)
    feat.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    df[["path", "group", "category", "anomaly_label"]].to_parquet(
        os.path.join(out_dir, "metadata.parquet"), index=False)
    write_splits("e1_parity_pump", out_dir)

    feat_mimii = pd.read_parquet(os.path.join(mb.RESULTS_DIR, "mimii", "features.parquet"))
    feat_diff = (feat - feat_mimii).abs().to_numpy().max()
    print(f"Feature matrix parity: max absolute difference = {feat_diff:.2e}", flush=True)
    assert feat_diff == 0.0, f"Feature matrix mismatch: {feat_diff}"

    cfg = mb.DATASETS["e1_parity_pump"]
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({**cfg, "contamination": mb.CONTAMINATION, "seeds": mb.SEEDS,
                   "n_features": feat.shape[1]}, fh, indent=2)

    # Run rotation on e1_parity_pump
    summ = rotate.rotate_one("e1_parity_pump", settings=("primary", "secondary"), jobs=4,
                             collect=("primary",))

    # Compare rot_metrics.csv
    new_metrics = pd.read_csv(os.path.join(out_dir, "rot_metrics.csv"))
    orig_metrics = pd.read_csv(os.path.join(mb.RESULTS_DIR, "mimii", "rot_metrics.csv"))

    merged = orig_metrics.merge(new_metrics, on=["setting", "detector"], suffixes=("_orig", "_parity"))
    max_auc_diff = (merged["auc_roc_mean_orig"] - merged["auc_roc_mean_parity"]).abs().max()
    max_ap_diff = (merged["avg_precision_mean_orig"] - merged["avg_precision_mean_parity"]).abs().max()
    print(f"Rotation metrics parity across {len(merged)} detector rows:", flush=True)
    print(f"  Max |delta AUC-ROC mean|: {max_auc_diff:.6e}", flush=True)
    print(f"  Max |delta AP mean|:      {max_ap_diff:.6e}", flush=True)

    worst_auc = merged.sort_values(by="auc_roc_mean_orig", ascending=False)[
        ["setting", "detector", "auc_roc_mean_orig", "auc_roc_mean_parity"]
    ].head(10)
    print("Top 10 detector comparison:\n", worst_auc.to_string(index=False), flush=True)
    return max_auc_diff


def df_to_markdown(df, index=False):
    """Simple markdown table formatter without external dependencies."""
    if index:
        df = df.reset_index()
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        formatted_vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                formatted_vals.append(f"{val:.3f}")
            else:
                formatted_vals.append(str(val))
        lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(lines)


def tabulate_all():
    """Step 6: Tabulate results and compute Spearman rank correlations."""
    print("\n" + "=" * 80)
    print("STEP 6: COMPREHENSIVE TABULATION FOR ALL FOUR MIMII TYPES")
    print("=" * 80 + "\n")

    summary_rows = []
    primary_tables = {}

    for ds in ALL_MIMII_DATASETS:
        d = os.path.join(mb.RESULTS_DIR, ds)
        rot_file = os.path.join(d, "rot_metrics.csv")
        if not os.path.exists(rot_file):
            print(f"Warning: {rot_file} not found, skipping {ds}")
            continue

        rot = pd.read_csv(rot_file)
        prim = rot[rot["setting"] == "primary"].sort_values("auc_roc_mean", ascending=False)
        sec = rot[rot["setting"] == "secondary"].sort_values("auc_roc_mean", ascending=False)

        best_prim = prim.iloc[0]
        best_sec = sec.iloc[0]
        primary_tables[ds] = prim

        # AST
        ast_file = os.path.join(d, "rot_ast_metrics.csv")
        if os.path.exists(ast_file):
            ast_df = pd.read_csv(ast_file)
            ast_prim = ast_df[ast_df["setting"] == "primary"].sort_values("auc_roc_mean", ascending=False).iloc[0]
            ast_str = f"{ast_prim['detector']} ({ast_prim['auc_roc_mean']:.3f} ± {ast_prim['auc_roc_fold_sd']:.3f})"
        else:
            ast_str = "N/A"

        # PANNs
        panns_file = os.path.join(d, "rot_panns_metrics.csv")
        if os.path.exists(panns_file):
            panns_df = pd.read_csv(panns_file)
            panns_prim = panns_df[panns_df["setting"] == "primary"].sort_values("auc_roc_mean", ascending=False).iloc[0]
            panns_str = f"{panns_prim['detector']} ({panns_prim['auc_roc_mean']:.3f} ± {panns_prim['auc_roc_fold_sd']:.3f})"
        else:
            panns_str = "N/A"

        # AE anchor
        ae_file = os.path.join(d, "ae_rot_metrics.csv")
        if os.path.exists(ae_file):
            ae_df = pd.read_csv(ae_file)
            ae_row = ae_df[ae_df["setting"] == "primary"].iloc[0] if "setting" in ae_df.columns else ae_df.iloc[0]
            ae_str = f"{ae_row['auc_roc_mean']:.3f} ± {ae_row['auc_roc_fold_sd']:.3f}"
        else:
            ae_str = "N/A"

        summary_rows.append({
            "Sub-dataset": ds,
            "Primary Winner": f"{best_prim['detector']} ({best_prim['auc_roc_mean']:.3f} ± {best_prim['auc_roc_fold_sd']:.3f})",
            "Secondary Winner": f"{best_sec['detector']} ({best_sec['auc_roc_mean']:.3f} ± {best_sec['auc_roc_fold_sd']:.3f})",
            "AST Best (primary)": ast_str,
            "PANNs Best (primary)": panns_str,
            "AE Anchor (primary)": ae_str,
        })

    summary_df = pd.DataFrame(summary_rows)
    print("### High-Level Summary Across MIMII Machine Types")
    print(df_to_markdown(summary_df, index=False))
    print("\n")

    # Full 25-detector tables in the format of rot_metrics.csv
    cols_to_show = ["detector", "auc_roc_mean", "auc_roc_fold_sd", "auc_roc_seed_sd",
                    "avg_precision_mean", "avg_precision_fold_sd", "pauc_0.1_mean",
                    "f1_top_q_mean", "auc_roc_pairweighted"]
    for ds in ["mimii_fan", "mimii_slider", "mimii_valve"]:
        if ds in primary_tables:
            print(f"### {ds} Primary 25-Detector Table (Format of rot_metrics.csv)")
            t = primary_tables[ds][cols_to_show].copy()
            print(df_to_markdown(t, index=False))
            print("\n")

    # Spearman rank correlation across 25 detectors
    print("### Spearman Rank Correlation Matrix of 25-Detector Primary Orderings")
    available_ds = [ds for ds in ALL_MIMII_DATASETS if ds in primary_tables]
    corr_matrix = pd.DataFrame(index=available_ds, columns=available_ds, dtype=float)
    pval_matrix = pd.DataFrame(index=available_ds, columns=available_ds, dtype=float)

    for d1 in available_ds:
        for d2 in available_ds:
            s1 = primary_tables[d1].set_index("detector")["auc_roc_mean"]
            s2 = primary_tables[d2].set_index("detector")["auc_roc_mean"]
            common = s1.index.intersection(s2.index)
            r, p = spearmanr(s1.loc[common], s2.loc[common])
            corr_matrix.loc[d1, d2] = r
            pval_matrix.loc[d1, d2] = p

    print(df_to_markdown(corr_matrix, index=True))
    print("\nPairwise Details:")
    for i, d1 in enumerate(available_ds):
        for d2 in available_ds[i + 1:]:
            r = corr_matrix.loc[d1, d2]
            p = pval_matrix.loc[d1, d2]
            print(f"  {d1} vs {d2}: Spearman rho = {r:.3f} (p = {p:.3e})")
    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["parity", "prepare_features", "rotate_handcrafted",
                                 "features_and_rotation", "extract_ast", "extract_panns",
                                 "rotate_ast", "rotate_panns", "prebuild_ae", "ae",
                                 "run_all_gpu", "run_all_rotations", "summary"])
    parser.add_argument("--dataset", choices=MIMII_DATASETS + ["all", "pump"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    register_all()
    os.environ["LOKY_MAX_CPU_COUNT"] = "5"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    if args.stage == "parity":
        diff = run_parity()
        print(f"Parity check completed with max AUC diff: {diff:.6e}")
        return

    datasets = MIMII_DATASETS if args.dataset == "all" else [args.dataset]

    if args.stage == "prepare_features":
        for ds in datasets:
            mtype = ds.replace("mimii_", "")
            prepare_features_and_metadata(ds, mtype)

    elif args.stage == "rotate_handcrafted":
        for ds in datasets:
            print(f"=== Running handcrafted rotation for {ds} ===", flush=True)
            run_rotation_only(ds, jobs=args.jobs)

    elif args.stage == "features_and_rotation":
        for ds in datasets:
            mtype = ds.replace("mimii_", "")
            print(f"=== Running features and rotation for {ds} ===", flush=True)
            run_features_and_rotation(ds, mtype, jobs=args.jobs)

    elif args.stage == "extract_ast":
        for ds in datasets:
            mtype = ds.replace("mimii_", "")
            df = load_mimii_machine(mtype)
            extract_ast_embeddings(ds, df)

    elif args.stage == "extract_panns":
        for ds in datasets:
            mtype = ds.replace("mimii_", "")
            df = load_mimii_machine(mtype)
            extract_panns_embeddings(ds, df)

    elif args.stage == "prebuild_ae":
        for ds in datasets:
            mtype = ds.replace("mimii_", "")
            prebuild_ae_frames(ds, mtype)

    elif args.stage == "rotate_ast":
        for ds in datasets:
            print(f"=== Running AST rotated detectors for {ds} ===", flush=True)
            run_representation(ds, "ast", pca_dim=64, jobs=args.jobs)

    elif args.stage == "rotate_panns":
        for ds in datasets:
            print(f"=== Running PANNs rotated detectors for {ds} ===", flush=True)
            run_representation(ds, "panns", pca_dim=64, jobs=args.jobs)

    elif args.stage == "ae":
        for ds in datasets:
            print(f"=== Running AE baseline anchor for {ds} ===", flush=True)
            run_ae(ds, settings=("primary",))

    elif args.stage == "run_all_gpu":
        # Run all GPU extraction and AE training under gpu-launch
        for ds in datasets:
            mtype = ds.replace("mimii_", "")
            df = load_mimii_machine(mtype)
            print(f"[{ds}] Starting AST extraction...", flush=True)
            extract_ast_embeddings(ds, df)
            print(f"[{ds}] Starting PANNs extraction...", flush=True)
            extract_panns_embeddings(ds, df)
            print(f"[{ds}] Starting AE baseline training...", flush=True)
            run_ae(ds, settings=("primary",))
            print(f"[{ds}] All GPU tasks completed!", flush=True)

    elif args.stage == "run_all_rotations":
        for ds in datasets:
            print(f"=== Running all rotations for {ds} ===", flush=True)
            run_rotation_only(ds, jobs=args.jobs)
            run_representation(ds, "ast", pca_dim=64, jobs=args.jobs)
            run_representation(ds, "panns", pca_dim=64, jobs=args.jobs)

    elif args.stage == "summary":
        tabulate_all()


if __name__ == "__main__":
    main()
