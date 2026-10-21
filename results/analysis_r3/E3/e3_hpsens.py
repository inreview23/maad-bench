"""MAAD-Bench Hyperparameter Sensitivity Grid & Runner (UNIT E3).

Evaluates whether detector defaults drive the ranking and reliability conclusions
on the seven current sub-datasets (esc50, urbansound8k, ravdess, nsynth, mimii, physionet, icbhi).

Grid Fixed in Advance:
- AutoEncoder: epochs {10 (default), 50, 100}
- VAE: epochs {30 (default), 100}
- GMM: n_components {1 (default), 2, 4, 8} x covariance_type {full (default), diag}
- KNN: n_neighbors {5 (default), 10, 20, 50}
- LOF: n_neighbors {20 (default), 10, 50}
- COF: n_neighbors {20 (default), 10, 50}
- IForest: n_estimators {100 (default), 500}
- OCSVM: gamma {auto (pyod default), scale} x nu {0.5 (default), 0.1}
- DIF: n_estimators {6 (default), 12} (larger-training variant from pyod signature)
- LUNAR: n_epochs {200 (default), 400} (larger-training variant from pyod signature)

"Reasonable" Non-Default Config Fixed in Advance:
- AutoEncoder: epochs=100
- VAE: epochs=100
- GMM: nc=4, covariance_type='diag'
- KNN: n_neighbors=10
- LOF: n_neighbors=10
- COF: n_neighbors=10
- IForest: n_estimators=500
- OCSVM: gamma='scale', nu=0.1
- DIF: n_estimators=12
- LUNAR: epochs=400
- All other 15 detectors: default
"""
import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure import from working directory
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import maad_bench as mb

DATASETS = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
RESULTS_DIR = os.path.join(HERE, "results")
HPSENS_DIR = os.path.join(RESULTS_DIR, "hpsens")

# Reasonable config fixed in advance for tuned detectors
REASONABLE_CONFIG = {
    "AutoEncoder": "epochs=100",
    "VAE": "epochs=100",
    "GMM": "nc=4_cov=diag",
    "KNN": "k=10",
    "LOF": "k=10",
    "COF": "k=10",
    "IForest": "n_est=500",
    "OCSVM": "gamma=scale_nu=0.1",
    "DIF": "n_est=12",
    "LUNAR": "epochs=400",
}


def build_grid_definitions():
    """Return explicit grid definitions with metadata."""
    grid = {
        "AutoEncoder": {
            "type": "neural",
            "configs": {
                "epochs=10": {"epoch_num": 10, "is_default": True},
                "epochs=50": {"epoch_num": 50, "is_default": False},
                "epochs=100": {"epoch_num": 100, "is_default": False},
            },
            "seeds": [0, 1, 2, 3, 4],
        },
        "VAE": {
            "type": "neural",
            "configs": {
                "epochs=30": {"epoch_num": 30, "is_default": True},
                "epochs=100": {"epoch_num": 100, "is_default": False},
            },
            "seeds": [0, 1, 2, 3, 4],
        },
        "GMM": {
            "type": "cpu",
            "configs": {
                "nc=1_cov=full": {"n_components": 1, "covariance_type": "full", "is_default": True, "seeds": [0]},
                "nc=1_cov=diag": {"n_components": 1, "covariance_type": "diag", "is_default": False, "seeds": [0]},
                "nc=2_cov=full": {"n_components": 2, "covariance_type": "full", "is_default": False, "seeds": [0, 1, 2, 3, 4]},
                "nc=2_cov=diag": {"n_components": 2, "covariance_type": "diag", "is_default": False, "seeds": [0, 1, 2, 3, 4]},
                "nc=4_cov=full": {"n_components": 4, "covariance_type": "full", "is_default": False, "seeds": [0, 1, 2, 3, 4]},
                "nc=4_cov=diag": {"n_components": 4, "covariance_type": "diag", "is_default": False, "seeds": [0, 1, 2, 3, 4]},
                "nc=8_cov=full": {"n_components": 8, "covariance_type": "full", "is_default": False, "seeds": [0, 1, 2, 3, 4]},
                "nc=8_cov=diag": {"n_components": 8, "covariance_type": "diag", "is_default": False, "seeds": [0, 1, 2, 3, 4]},
            },
        },
        "KNN": {
            "type": "cpu",
            "configs": {
                "k=5": {"n_neighbors": 5, "is_default": True},
                "k=10": {"n_neighbors": 10, "is_default": False},
                "k=20": {"n_neighbors": 20, "is_default": False},
                "k=50": {"n_neighbors": 50, "is_default": False},
            },
            "seeds": [0],
        },
        "LOF": {
            "type": "cpu",
            "configs": {
                "k=20": {"n_neighbors": 20, "is_default": True},
                "k=10": {"n_neighbors": 10, "is_default": False},
                "k=50": {"n_neighbors": 50, "is_default": False},
            },
            "seeds": [0],
        },
        "COF": {
            "type": "cpu",
            "configs": {
                "k=20": {"n_neighbors": 20, "is_default": True},
                "k=10": {"n_neighbors": 10, "is_default": False},
                "k=50": {"n_neighbors": 50, "is_default": False},
            },
            "seeds": [0],
        },
        "IForest": {
            "type": "cpu",
            "configs": {
                "n_est=100": {"n_estimators": 100, "is_default": True},
                "n_est=500": {"n_estimators": 500, "is_default": False},
            },
            "seeds": [0, 1, 2, 3, 4],
        },
        "OCSVM": {
            "type": "cpu",
            "configs": {
                "gamma=auto_nu=0.5": {"gamma": "auto", "nu": 0.5, "is_default": True},
                "gamma=auto_nu=0.1": {"gamma": "auto", "nu": 0.1, "is_default": False},
                "gamma=scale_nu=0.5": {"gamma": "scale", "nu": 0.5, "is_default": False},
                "gamma=scale_nu=0.1": {"gamma": "scale", "nu": 0.1, "is_default": False},
            },
            "seeds": [0],
        },
        "DIF": {
            "type": "neural",
            "configs": {
                "n_est=6": {"n_estimators": 6, "is_default": True},
                "n_est=12": {"n_estimators": 12, "is_default": False},
            },
            "seeds": [0, 1, 2, 3, 4],
        },
        "LUNAR": {
            "type": "neural",
            "configs": {
                "epochs=200": {"n_epochs": 200, "is_default": True},
                "epochs=400": {"n_epochs": 400, "is_default": False},
            },
            "seeds": [0, 1, 2, 3, 4],
        },
    }
    return grid


def instantiate_detector(detector_name: str, config_kwargs: dict, contamination: float, seed: int):
    """Instantiate a detector given its name, config arguments, contamination, and seed."""
    kw = {k: v for k, v in config_kwargs.items() if k not in ("is_default", "seeds")}
    if detector_name == "AutoEncoder":
        from pyod.models.auto_encoder import AutoEncoder
        return AutoEncoder(contamination=contamination, random_state=seed, verbose=0, **kw)
    elif detector_name == "VAE":
        from pyod.models.vae import VAE
        return VAE(contamination=contamination, random_state=seed, verbose=0, **kw)
    elif detector_name == "GMM":
        from pyod.models.gmm import GMM
        return GMM(contamination=contamination, random_state=seed, **kw)
    elif detector_name == "KNN":
        from pyod.models.knn import KNN
        return KNN(contamination=contamination, **kw)
    elif detector_name == "LOF":
        from pyod.models.lof import LOF
        return LOF(contamination=contamination, **kw)
    elif detector_name == "COF":
        from pyod.models.cof import COF
        return COF(contamination=contamination, **kw)
    elif detector_name == "IForest":
        from pyod.models.iforest import IForest
        return IForest(contamination=contamination, random_state=seed, **kw)
    elif detector_name == "OCSVM":
        from pyod.models.ocsvm import OCSVM
        return OCSVM(contamination=contamination, **kw)
    elif detector_name == "DIF":
        from pyod.models.dif import DIF
        return DIF(contamination=contamination, random_state=seed, **kw)
    elif detector_name == "LUNAR":
        from pyod.models.lunar import LUNAR
        return LUNAR(contamination=contamination, verbose=0, **kw)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def load_dataset_data(dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List]:
    """Load metadata, features, and unique groups."""
    d = os.path.join(RESULTS_DIR, dataset_name)
    feat_df = pd.read_parquet(os.path.join(d, "features.parquet"))
    meta_df = pd.read_parquet(os.path.join(d, "metadata.parquet"))
    X = feat_df.to_numpy(dtype=float)
    y = meta_df["anomaly_label"].to_numpy()
    groups = sorted(meta_df["group"].unique().tolist())
    return meta_df, X, y, groups


def fit_and_eval_cell(Xtr, Xte, yte, detector_name, cfg_name, cfg_kwargs, seed, contamination):
    """Fit detector on Xtr and evaluate on Xte, yte."""
    try:
        np.random.seed(seed)
        det = instantiate_detector(detector_name, cfg_kwargs, contamination, seed)
        det.fit(Xtr)
        s = np.asarray(det.decision_function(Xte), dtype=float)
        m = mb.metrics(yte, s, contamination)
        error = None
    except Exception as e:
        m = dict(mb.NAN_METRICS)
        error = type(e).__name__
    finally:
        del det
        gc.collect()

    res = {
        "detector": detector_name,
        "config": cfg_name,
        "seed": seed,
        "auc_roc": m.get("auc_roc", float("nan")),
        "avg_precision": m.get("avg_precision", float("nan")),
        "pauc_0.1": m.get("pauc_0.1", float("nan")),
        "f1_top_q": m.get("f1_top_q", float("nan")),
        "n_flagged": m.get("n_flagged", 0),
        "test_prevalence": m.get("test_prevalence", float("nan")),
        "error": error,
    }
    return res


def run_single_group_cpu(ds_name: str, group, grid_subset: dict):
    """Run CPU models for a single fold."""
    meta_df, X, y, _ = load_dataset_data(ds_name)
    tr, te = mb.make_split(meta_df, 0, "primary", [group], subsample_test=False, contamination=mb.CONTAMINATION)
    yte = y[te]
    if not mb.is_scorable(yte):
        return []

    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr])
    Xte = sc.transform(X[te])

    rows = []
    for det_name, det_spec in grid_subset.items():
        configs = det_spec["configs"]
        for cfg_name, cfg_kwargs in configs.items():
            seeds = cfg_kwargs.get("seeds", det_spec.get("seeds", [0]))
            for seed in seeds:
                res = fit_and_eval_cell(Xtr, Xte, yte, det_name, cfg_name, cfg_kwargs, seed, mb.CONTAMINATION)
                res.update({
                    "dataset": ds_name,
                    "group": group,
                    "fold": group,
                    "setting": "primary",
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_test_anom": int(yte.sum()),
                    "is_default": cfg_kwargs.get("is_default", False),
                })
                rows.append(res)
    return rows


def run_cpu_grid(workers: int = 4):
    """Run all CPU configurations across all datasets and folds."""
    os.makedirs(HPSENS_DIR, exist_ok=True)
    out_path = os.path.join(HPSENS_DIR, "cpu_runs.parquet")
    grid = build_grid_definitions()
    cpu_grid = {k: v for k, v in grid.items() if v["type"] == "cpu"}

    tasks = []
    for ds in DATASETS:
        meta_df, _, _, groups = load_dataset_data(ds)
        for g in groups:
            tasks.append((ds, g))

    print(f"[CPU] Running {len(tasks)} fold tasks across {len(DATASETS)} datasets with {workers} workers...", flush=True)
    t0 = time.time()

    from joblib import Parallel, delayed
    results_nested = Parallel(n_jobs=workers, verbose=10)(
        delayed(run_single_group_cpu)(ds, g, cpu_grid) for ds, g in tasks
    )

    all_rows = [r for sublist in results_nested for r in sublist]
    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"[CPU] Completed in {elapsed:.1f}s ({len(df)} rows saved to {out_path})", flush=True)
    return df


def run_single_group_neural(ds_name: str, group, neural_grid: dict, include_defaults: bool = False):
    """Run neural configurations for a single fold."""
    import torch
    meta_df, X, y, _ = load_dataset_data(ds_name)
    tr, te = mb.make_split(meta_df, 0, "primary", [group], subsample_test=False, contamination=mb.CONTAMINATION)
    yte = y[te]
    if not mb.is_scorable(yte):
        return []

    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr])
    Xte = sc.transform(X[te])

    rows = []
    for det_name, det_spec in neural_grid.items():
        for cfg_name, cfg_kwargs in det_spec["configs"].items():
            is_def = cfg_kwargs.get("is_default", False)
            if is_def and not include_defaults:
                continue
            seeds = det_spec["seeds"]
            for s in seeds:
                res = fit_and_eval_cell(Xtr, Xte, yte, det_name, cfg_name, cfg_kwargs, s, mb.CONTAMINATION)
                res.update({
                    "dataset": ds_name,
                    "group": group,
                    "fold": group,
                    "setting": "primary",
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_test_anom": int(yte.sum()),
                    "is_default": is_def,
                })
                rows.append(res)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return rows


def run_neural_grid(workers: int = 3, include_defaults: bool = False):
    """Run neural configurations (AutoEncoder, VAE, DIF, LUNAR) under gpu-launch."""
    import torch
    from joblib import Parallel, delayed
    print(f"[Neural] CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[Neural] Device Name: {torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(HPSENS_DIR, exist_ok=True)
    out_path = os.path.join(HPSENS_DIR, "neural_runs.parquet")

    existing_df = None
    completed_folds = set()
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_parquet(out_path)
            counts = existing_df.groupby(["dataset", "fold"]).size()
            for (d, f), count in counts.items():
                if count >= 25:
                    completed_folds.add((str(d), str(f)))
            print(f"[Neural] Found existing output with {len(existing_df)} rows ({len(completed_folds)} completed folds).", flush=True)
        except Exception as e:
            print(f"[Neural] Error reading existing parquet: {e}", flush=True)
            existing_df = None

    grid = build_grid_definitions()
    neural_grid = {k: v for k, v in grid.items() if v["type"] == "neural"}

    all_rows = [] if existing_df is None else existing_df.to_dict("records")

    priority_order = ["ravdess", "mimii", "physionet", "icbhi", "nsynth", "urbansound8k", "esc50"]
    tasks = []
    for ds in priority_order:
        meta_df, _, _, groups = load_dataset_data(ds)
        for g in groups:
            if (str(ds), str(g)) not in completed_folds:
                tasks.append((ds, g))

    print(f"[Neural] {len(tasks)} remaining fold tasks to run with {workers} workers...", flush=True)
    t0 = time.time()

    gen = Parallel(n_jobs=workers, return_as="generator")(
        delayed(run_single_group_neural)(ds, g, neural_grid, include_defaults) for ds, g in tasks
    )

    completed = 0
    for fold_rows in gen:
        if fold_rows:
            all_rows.extend(fold_rows)
            completed += 1
            df = pd.DataFrame(all_rows)
            df.to_parquet(out_path, index=False)
            ds_f = fold_rows[0]["dataset"]
            g_f = fold_rows[0]["fold"]
            print(f"[Neural] [{completed}/{len(tasks)}] Finished {ds_f} fold {g_f} ({len(fold_rows)} rows). Total rows: {len(df)}. Elapsed: {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    final_df = pd.DataFrame(all_rows)
    final_df.to_parquet(out_path, index=False)
    print(f"[Neural] Completed in {elapsed:.1f}s ({len(final_df)} rows saved to {out_path})", flush=True)
    return final_df


def assemble_all():
    """Assemble CPU runs, Neural runs, and baseline rot_runs into hpsens_runs.parquet."""
    os.makedirs(HPSENS_DIR, exist_ok=True)
    final_out = os.path.join(HPSENS_DIR, "hpsens_runs.parquet")

    grid = build_grid_definitions()

    # 1. Load CPU runs
    cpu_path = os.path.join(HPSENS_DIR, "cpu_runs.parquet")
    if not os.path.exists(cpu_path):
        raise FileNotFoundError(f"Missing {cpu_path}; run with --mode cpu first.")
    cpu_df = pd.read_parquet(cpu_path)
    print(f"Loaded {len(cpu_df)} CPU rows.")

    # 2. Load Neural runs
    neural_path = os.path.join(HPSENS_DIR, "neural_runs.parquet")
    neural_df = pd.DataFrame()
    if os.path.exists(neural_path):
        neural_df = pd.read_parquet(neural_path)
        print(f"Loaded {len(neural_df)} Neural rows.")

    # 3. Load baseline rot_runs for default configs and un-tuned detectors
    base_rows = []
    default_config_names = {
        "AutoEncoder": "epochs=10",
        "VAE": "epochs=30",
        "DIF": "n_est=6",
        "LUNAR": "epochs=200",
        "GMM": "nc=1_cov=full",
        "KNN": "k=5",
        "LOF": "k=20",
        "COF": "k=20",
        "IForest": "n_est=100",
        "OCSVM": "gamma=auto_nu=0.5",
    }

    for ds in DATASETS:
        rot_path = os.path.join(RESULTS_DIR, ds, "rot_runs.parquet")
        if not os.path.exists(rot_path):
            continue
        rot = pd.read_parquet(rot_path)
        rot_primary = rot[(rot["setting"] == "primary") & (rot["detector"] != "*")].copy()

        for _, r in rot_primary.iterrows():
            det = r["detector"]
            seed = int(r["seed"])
            is_tuned = det in grid

            if is_tuned:
                def_cfg_name = default_config_names[det]
                # For deterministic detectors, keep seed 0 only
                if det in ["KNN", "LOF", "COF", "OCSVM", "GMM"] and seed > 0:
                    continue
                row = dict(r)
                row["config"] = def_cfg_name
                row["fold"] = r["group"]
                row["is_default"] = True
                base_rows.append(row)
            else:
                # Un-tuned detector
                # Keep seed 0 for deterministic detectors, seeds 0..4 for stochastic
                row = dict(r)
                row["config"] = "default"
                row["fold"] = r["group"]
                row["is_default"] = True
                base_rows.append(row)

    base_df = pd.DataFrame(base_rows)
    print(f"Loaded {len(base_df)} baseline/default rows from rot_runs.")

    # Combine everything
    # Check for duplicates between cpu_df / neural_df and base_df
    combined = []
    seen = set()

    for df_part in [cpu_df, neural_df, base_df]:
        if df_part is None or df_part.empty:
            continue
        for _, r in df_part.iterrows():
            key = (str(r["dataset"]), str(r["fold"]), str(r["detector"]), str(r["config"]), int(r["seed"]))
            if key not in seen:
                seen.add(key)
                combined.append(dict(r))

    total_df = pd.DataFrame(combined)

    # Standardize column types
    for col in ["n_flagged", "n_train", "n_test", "n_test_anom", "seed"]:
        if col in total_df.columns:
            total_df[col] = pd.to_numeric(total_df[col], errors='coerce').astype('Int64')

    total_df.to_parquet(final_out, index=False)
    print(f"\nSuccessfully assembled {len(total_df)} rows into {final_out}.")

    # Summary validation
    print("\nSummary per dataset:")
    summ = total_df.groupby("dataset").agg(
        n_rows=("auc_roc", "count"),
        n_detectors=("detector", "nunique"),
        n_configs=("config", "nunique"),
        n_folds=("fold", "nunique"),
    )
    print(summ.to_string())

    return total_df


def main():
    ap = argparse.ArgumentParser(description="MAAD-Bench Hyperparameter Sensitivity Grid & Runner")
    ap.add_argument("--mode", choices=["cpu", "neural", "assemble", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4, help="Max worker processes for CPU grid")
    ap.add_argument("--include-defaults", action="store_true", help="Also re-run default configs in neural grid")
    args = ap.parse_args()

    t_start = time.time()
    if args.mode == "cpu":
        run_cpu_grid(workers=args.workers)
    elif args.mode == "neural":
        run_neural_grid(workers=args.workers, include_defaults=args.include_defaults)
    elif args.mode == "assemble":
        assemble_all()
    elif args.mode == "all":
        print("=== Running CPU grid ===")
        run_cpu_grid(workers=args.workers)
        print("=== Running Neural grid ===")
        run_neural_grid(workers=args.workers, include_defaults=args.include_defaults)
        print("=== Assembling all results ===")
        assemble_all()
    print(f"Done in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
