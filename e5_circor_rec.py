"""Unit E5: CirCor with recording-level murmur labels and campaign/location confound check.

Protocol:
1. Rule fixed in code before running:
   - Absent patients: every recording is normal (anomaly_label = 0).
   - Present patients: recording is anomalous (anomaly_label = 1) if its location (from file name)
     is listed in that patient's "Murmur locations", and excluded otherwise.
   - Unknown patients: excluded as before (156 clips).
   - Groups: patient id mod 5 (5 patient buckets), keeping every patient wholly in one group.
   - Every file name parsed with regex r"^(\\d+)_([A-Za-z]+)(?:_(\\d+))?\\.wav$".
2. Register circor_rec. Features taken from results/circor/features.parquet by path alignment.
   Sample of 20 clips verified for exact identity with re-extraction.
   Standard artifact set written to results/circor_rec/.
   Standard rotation: primary & secondary settings, 5 seeds, 25 detectors, primary per-clip scores.
   nice -n 10, at most 4 workers.
3. Campaign & location confound check:
   - Evaluated on circor_rec and patient-level circor.
   - Primary per-clip scores and loudness baseline (rms0_mean absolute z-score against fold training normals).
   - Per-fold AUC within CC2014 and CC2015 for all 25 detectors + baseline.
   - Anomaly prevalence by campaign and by location.
   - Fold-mean AUC by auscultation location (AV, MV, PV, TV, Phc) for top 5 detectors.
   - Counts of detectors above 0.5 within each campaign.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import maad_bench as mb

# ----------------- Registration -----------------
mb.DATASETS["circor_rec"] = dict(
    normal=["absent"],
    anomaly=["present"],
    test_groups=[0],
)


def load_circor_rec(verbose=True):
    """Load CirCor with recording-level murmur labels.
    
    Absent patients: all recordings normal (category="absent", anomaly_label=0).
    Present patients: recording anomalous (category="present", anomaly_label=1)
      if its auscultation location is in patient's Murmur locations, excluded otherwise.
    Unknown patients: excluded.
    Groups: patient_id % 5.
    """
    root = os.path.join(mb.DATA_DIR, "circor")
    csv_path = os.path.join(root, "training_data.csv")
    wav_dir = os.path.join(root, "training_data")
    if not os.path.exists(csv_path) or not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"CirCor data not found in {root}")

    cdf = pd.read_csv(csv_path)
    patient_info = {}
    for _, row in cdf.iterrows():
        pid = int(row["Patient ID"])
        murmur = row["Murmur"]
        locs_str = str(row["Murmur locations"]) if pd.notna(row["Murmur locations"]) else ""
        loc_set = set(locs_str.split("+")) if locs_str else set()
        campaign = row["Campaign"]
        patient_info[pid] = {
            "murmur": murmur,
            "murmur_locations": loc_set,
            "campaign": campaign,
            "group": pid % 5
        }

    all_wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    pattern = re.compile(r"^(\d+)_([A-Za-z]+)(?:_(\d+))?\.wav$")

    parsed_count = 0
    counts = {"normal": 0, "anomalous": 0, "excluded_unknown": 0, "excluded_present_nonloc": 0}
    per_group = {g: {"normal": 0, "anomalous": 0, "excluded_unknown": 0, "excluded_present_nonloc": 0} for g in range(5)}
    
    rows = []
    for p in all_wavs:
        fname = os.path.basename(p)
        m = pattern.match(fname)
        if not m:
            raise ValueError(f"Failed to parse filename: {fname}")
        parsed_count += 1
        pid = int(m.group(1))
        loc = m.group(2)

        info = patient_info[pid]
        g = info["group"]
        murmur = info["murmur"]

        if murmur == "Unknown":
            counts["excluded_unknown"] += 1
            per_group[g]["excluded_unknown"] += 1
            continue
        elif murmur == "Absent":
            counts["normal"] += 1
            per_group[g]["normal"] += 1
            rows.append({
                "path": os.path.abspath(p),
                "group": g,
                "category": "absent",
                "anomaly_label": 0,
                "pid": pid,
                "loc": loc,
                "campaign": info["campaign"]
            })
        elif murmur == "Present":
            if loc in info["murmur_locations"]:
                counts["anomalous"] += 1
                per_group[g]["anomalous"] += 1
                rows.append({
                    "path": os.path.abspath(p),
                    "group": g,
                    "category": "present",
                    "anomaly_label": 1,
                    "pid": pid,
                    "loc": loc,
                    "campaign": info["campaign"]
                })
            else:
                counts["excluded_present_nonloc"] += 1
                per_group[g]["excluded_present_nonloc"] += 1
                continue

    if verbose:
        print(f"[circor_rec] Checked {parsed_count} / {len(all_wavs)} files. All parsed successfully.")
        print(f"[circor_rec] Counts: normal={counts['normal']}, anomalous={counts['anomalous']}, "
              f"excluded_unknown={counts['excluded_unknown']}, excluded_present_nonloc={counts['excluded_present_nonloc']}")
        print(f"[circor_rec] Total included={len(rows)}, Total excluded={counts['excluded_unknown'] + counts['excluded_present_nonloc']}")
        for g in range(5):
            pg = per_group[g]
            tot_ex = pg['excluded_unknown'] + pg['excluded_present_nonloc']
            print(f"  Group {g}: normal={pg['normal']}, anomalous={pg['anomalous']}, excluded={tot_ex} "
                  f"(unknown={pg['excluded_unknown']}, present_nonloc={pg['excluded_present_nonloc']}), total_files={pg['normal'] + pg['anomalous'] + tot_ex}")

    df = pd.DataFrame(rows)
    return df


mb.LOADERS["circor_rec"] = lambda: load_circor_rec(verbose=False)[["path", "group", "category", "anomaly_label"]]


# ----------------- Feature Preparation & Parity -----------------
def prepare_circor_rec_artifacts():
    out_dir = os.path.join(mb.RESULTS_DIR, "circor_rec")
    os.makedirs(out_dir, exist_ok=True)

    df_full = load_circor_rec(verbose=True)
    df = df_full[["path", "group", "category", "anomaly_label"]]

    circor_md_path = os.path.join(mb.RESULTS_DIR, "circor", "metadata.parquet")
    circor_feat_path = os.path.join(mb.RESULTS_DIR, "circor", "features.parquet")
    if not os.path.exists(circor_md_path) or not os.path.exists(circor_feat_path):
        raise FileNotFoundError("circor metadata or features parquet missing under results/circor/")

    circor_md = pd.read_parquet(circor_md_path)
    circor_feat = pd.read_parquet(circor_feat_path)

    # Parity check on 20 clips
    print("[circor_rec] Running 20-clip feature re-extraction parity check...")
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(df), size=20, replace=False)
    sample_df = df.iloc[sample_idx]
    
    # Get corresponding rows in circor_feat
    circor_path_to_idx = {p: i for i, p in enumerate(circor_md["path"])}
    sample_cached_feat = circor_feat.iloc[[circor_path_to_idx[p] for p in sample_df["path"]]].reset_index(drop=True)

    re_extracted = []
    for p in sample_df["path"]:
        re_extracted.append(mb.extract_one(p))
    re_feat = pd.DataFrame(re_extracted)

    diff = np.max(np.abs(sample_cached_feat.values - re_feat.values))
    print(f"[circor_rec] Max absolute difference on 20 clips: {diff:.6e}")
    if diff > 1e-5:
        raise ValueError(f"Feature parity check failed! Diff: {diff}")
    print("[circor_rec] Parity check PASSED: identical features.")

    # Match features by path
    rec_indices = [circor_path_to_idx[p] for p in df["path"]]
    feat = circor_feat.iloc[rec_indices].reset_index(drop=True)

    # Save artifacts
    md_out = os.path.join(out_dir, "metadata.parquet")
    feat_out = os.path.join(out_dir, "features.parquet")
    sp_out = os.path.join(out_dir, "splits.parquet")

    df.to_parquet(md_out, index=False)
    feat.to_parquet(feat_out, index=False)

    sp = df[["group"]].reset_index().rename(columns={"index": "row_index"})
    sp["dataset"] = "circor_rec"
    sp["test_fold"] = sp["group"]
    sp.to_parquet(sp_out, index=False)
    print(f"[circor_rec] Saved metadata ({len(df)}), features ({feat.shape}), splits ({len(sp)}) to {out_dir}")
    return df, feat


# ----------------- Rotation Runner -----------------
def run_circor_rec_rotation(jobs=4):
    out_dir = os.path.join(mb.RESULTS_DIR, "circor_rec")
    df_path = os.path.join(out_dir, "metadata.parquet")
    feat_path = os.path.join(out_dir, "features.parquet")

    if not os.path.exists(df_path) or not os.path.exists(feat_path):
        df, feat = prepare_circor_rec_artifacts()
    else:
        df = pd.read_parquet(df_path)
        feat = pd.read_parquet(feat_path)

    groups = sorted(df["group"].unique().tolist())
    settings = ("primary", "secondary")
    collect = ("primary",)

    print(f"[circor_rec] Starting rotation: {len(df)} clips, {int(df['anomaly_label'].sum())} anomalies, "
          f"{len(groups)} groups: {groups} (jobs={jobs})", flush=True)

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

    runs.insert(0, "dataset", "circor_rec")
    runs.to_parquet(os.path.join(out_dir, "rot_runs.parquet"), index=False)

    if clips is not None:
        clips.insert(0, "dataset", "circor_rec")
        clips = clips.merge(df[["path"]].reset_index().rename(columns={"index": "row_index"}),
                            on="row_index", how="left")
        clips.to_parquet(os.path.join(out_dir, "rot_clip_scores.parquet"), index=False)

    summ = mb.summarize_rotation(runs, "circor_rec")
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

    print(f"[circor_rec] Rotation complete in {elapsed:.1f}s; {len(runs)} run rows, {len(skipped)} skipped folds, "
          f"{0 if clips is None else len(clips)} clip rows", flush=True)
    return summ


# ----------------- Confound Analysis -----------------
def analyze_confounds_for_dataset(ds_name):
    """Run campaign and location confound analysis for a circor dataset (circor or circor_rec)."""
    ds_dir = os.path.join(mb.RESULTS_DIR, ds_name)
    meta = pd.read_parquet(os.path.join(ds_dir, "metadata.parquet"))
    feat = pd.read_parquet(os.path.join(ds_dir, "features.parquet"))
    rcs = pd.read_parquet(os.path.join(ds_dir, "rot_clip_scores.parquet"))
    rcs = rcs[rcs["setting"] == "primary"].copy()

    # Patient and campaign info
    csv_path = os.path.join(mb.DATA_DIR, "circor", "training_data.csv")
    df_csv = pd.read_csv(csv_path)
    camp_map = dict(zip(df_csv["Patient ID"], df_csv["Campaign"]))
    pattern = re.compile(r"^(\d+)_([A-Za-z]+)(?:_(\d+))?\.wav$")

    meta["pid"] = [int(pattern.match(os.path.basename(p)).group(1)) for p in meta["path"]]
    meta["loc"] = [pattern.match(os.path.basename(p)).group(2) for p in meta["path"]]
    meta["campaign"] = [camp_map[pid] for pid in meta["pid"]]

    path_to_info = dict(zip(meta["path"], zip(meta["loc"], meta["campaign"])))
    rcs["loc"] = [path_to_info[p][0] for p in rcs["path"]]
    rcs["campaign"] = [path_to_info[p][1] for p in rcs["path"]]

    # Loudness baseline: rms0_mean absolute z against each fold's training normals
    df_feat_meta = pd.concat([meta, feat[["rms0_mean"]]], axis=1)
    folds = sorted(meta["group"].unique())
    seeds = sorted(rcs["seed"].unique())

    rms0_rows = []
    for f in folds:
        train_mask = (df_feat_meta["group"] != f) & (df_feat_meta["anomaly_label"] == 0)
        mu = df_feat_meta.loc[train_mask, "rms0_mean"].mean()
        sigma = df_feat_meta.loc[train_mask, "rms0_mean"].std()

        test_rows = df_feat_meta[df_feat_meta["group"] == f].copy()
        test_rows["score"] = (test_rows["rms0_mean"] - mu).abs() / sigma
        test_rows["detector"] = "rms0_mean_abs_z"
        test_rows["setting"] = "primary"
        test_rows["dataset"] = ds_name
        for s in seeds:
            t_s = test_rows.copy()
            t_s["seed"] = s
            rms0_rows.append(t_s[["dataset", "detector", "seed", "anomaly_label", "score", "setting", "group", "path", "loc", "campaign"]])

    df_rms0 = pd.concat(rms0_rows, ignore_index=True)
    all_scores = pd.concat([rcs, df_rms0], ignore_index=True)

    # 1. Campaign-stratified AUC per fold and fold-mean
    campaigns = ["CC2014", "CC2015"]
    camp_results = []
    fold_details = []

    for det in sorted(all_scores["detector"].unique()):
        sub = all_scores[all_scores["detector"] == det]
        fold_pooled = []
        fold_camps = {c: [] for c in campaigns}

        for f in folds:
            s_pooled = []
            s_camps = {c: [] for c in campaigns}
            for s in seeds:
                sub_fs = sub[(sub["group"] == f) & (sub["seed"] == s)]
                s_pooled.append(roc_auc_score(sub_fs["anomaly_label"], sub_fs["score"]))
                for c in campaigns:
                    sub_fsc = sub_fs[sub_fs["campaign"] == c]
                    if len(sub_fsc["anomaly_label"].unique()) == 2:
                        s_camps[c].append(roc_auc_score(sub_fsc["anomaly_label"], sub_fsc["score"]))

            auc_pooled_f = float(np.mean(s_pooled))
            fold_pooled.append(auc_pooled_f)
            auc_cc2014_f = float(np.mean(s_camps["CC2014"])) if s_camps["CC2014"] else np.nan
            auc_cc2015_f = float(np.mean(s_camps["CC2015"])) if s_camps["CC2015"] else np.nan
            fold_camps["CC2014"].append(auc_cc2014_f)
            fold_camps["CC2015"].append(auc_cc2015_f)

            fold_details.append({
                "dataset": ds_name,
                "detector": det,
                "group": f,
                "pooled_auc": auc_pooled_f,
                "CC2014_auc": auc_cc2014_f,
                "CC2015_auc": auc_cc2015_f
            })

        camp_results.append({
            "dataset": ds_name,
            "detector": det,
            "pooled_auc": float(np.mean(fold_pooled)),
            "CC2014_auc": float(np.nanmean(fold_camps["CC2014"])),
            "CC2015_auc": float(np.nanmean(fold_camps["CC2015"])),
            "CC2014_n_folds": int(np.sum(~np.isnan(fold_camps["CC2014"]))),
            "CC2015_n_folds": int(np.sum(~np.isnan(fold_camps["CC2015"])))
        })

    camp_df = pd.DataFrame(camp_results).sort_values("pooled_auc", ascending=False).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_details)

    # 2. Location-stratified AUC for top 5 detectors
    # Exclude rms0 baseline from detector ranking for top 5
    pyod_ranked = camp_df[camp_df["detector"] != "rms0_mean_abs_z"].sort_values("pooled_auc", ascending=False)
    top5_dets = pyod_ranked["detector"].head(5).tolist()

    locations = ["AV", "MV", "PV", "TV", "Phc"]
    loc_results = []
    for det in top5_dets:
        sub = rcs[rcs["detector"] == det]
        row = {"dataset": ds_name, "detector": det}
        for loc in locations:
            fold_aucs = []
            for f in folds:
                s_aucs = []
                for s in seeds:
                    sub_fsl = sub[(sub["group"] == f) & (sub["seed"] == s) & (sub["loc"] == loc)]
                    if len(sub_fsl["anomaly_label"].unique()) == 2:
                        s_aucs.append(roc_auc_score(sub_fsl["anomaly_label"], sub_fsl["score"]))
                if s_aucs:
                    fold_aucs.append(np.mean(s_aucs))
            row[loc] = float(np.mean(fold_aucs)) if fold_aucs else np.nan
            row[f"{loc}_n_folds"] = len(fold_aucs)
        loc_results.append(row)
    loc_df = pd.DataFrame(loc_results)

    # 3. Prevalence table
    prev_camp = meta.groupby("campaign")["anomaly_label"].agg(
        n_total="count", n_anom="sum", prevalence="mean"
    ).reset_index().rename(columns={"campaign": "stratum"})
    prev_camp["stratum_type"] = "campaign"

    prev_loc = meta.groupby("loc")["anomaly_label"].agg(
        n_total="count", n_anom="sum", prevalence="mean"
    ).reset_index().rename(columns={"loc": "stratum"})
    prev_loc["stratum_type"] = "location"

    prev_df = pd.concat([prev_camp, prev_loc], ignore_index=True)
    prev_df.insert(0, "dataset", ds_name)

    return camp_df, fold_df, loc_df, prev_df


def run_all_confounds():
    print("\n================ Running Confound Analysis for circor and circor_rec ================")
    out_dir = os.path.join(mb.RESULTS_DIR, "circor_rec")
    
    rec_camp, rec_folds, rec_loc, rec_prev = analyze_confounds_for_dataset("circor_rec")
    circor_camp, circor_folds, circor_loc, circor_prev = analyze_confounds_for_dataset("circor")

    # Save CSVs
    rec_camp.to_csv(os.path.join(out_dir, "e5_confound_campaign_circor_rec.csv"), index=False)
    rec_folds.to_csv(os.path.join(out_dir, "e5_confound_folds_circor_rec.csv"), index=False)
    circor_camp.to_csv(os.path.join(out_dir, "e5_confound_campaign_circor.csv"), index=False)
    circor_folds.to_csv(os.path.join(out_dir, "e5_confound_folds_circor.csv"), index=False)

    comb_loc = pd.concat([circor_loc, rec_loc], ignore_index=True)
    comb_loc.to_csv(os.path.join(out_dir, "e5_confound_location_top5.csv"), index=False)

    comb_prev = pd.concat([circor_prev, rec_prev], ignore_index=True)
    comb_prev.to_csv(os.path.join(out_dir, "e5_confound_prevalence.csv"), index=False)

    # Comparison table between patient-level and recording-level
    c_m = pd.read_csv(os.path.join(mb.RESULTS_DIR, "circor", "rot_metrics.csv"))
    r_m = pd.read_csv(os.path.join(mb.RESULTS_DIR, "circor_rec", "rot_metrics.csv"))

    c_prim = c_m[c_m["setting"] == "primary"][["detector", "auc_roc_mean", "auc_roc_fold_sd", "auc_roc_pairweighted"]].rename(
        columns={"auc_roc_mean": "circor_auc_mean", "auc_roc_fold_sd": "circor_auc_sd", "auc_roc_pairweighted": "circor_pairweighted"}
    )
    r_prim = r_m[r_m["setting"] == "primary"][["detector", "auc_roc_mean", "auc_roc_fold_sd", "auc_roc_pairweighted"]].rename(
        columns={"auc_roc_mean": "circor_rec_auc_mean", "auc_roc_fold_sd": "circor_rec_auc_sd", "auc_roc_pairweighted": "circor_rec_pairweighted"}
    )
    comp = c_prim.merge(r_prim, on="detector")
    comp["diff_rec_minus_pat"] = comp["circor_rec_auc_mean"] - comp["circor_auc_mean"]
    comp = comp.sort_values("circor_rec_auc_mean", ascending=False).reset_index(drop=True)
    comp.to_csv(os.path.join(out_dir, "e5_comparison_circor_vs_rec.csv"), index=False)

    print("\n=== CONFOUND ANALYSIS COMPLETE ===")
    print("\n--- CirCor Rec Campaign Summary ---")
    print(rec_camp.to_string(index=False))
    print(f"\ncircor_rec PyOD detectors > 0.5: CC2014={(rec_camp[rec_camp['detector'] != 'rms0_mean_abs_z']['CC2014_auc'] > 0.5).sum()}/25, "
          f"CC2015={(rec_camp[rec_camp['detector'] != 'rms0_mean_abs_z']['CC2015_auc'] > 0.5).sum()}/25, "
          f"Pooled={(rec_camp[rec_camp['detector'] != 'rms0_mean_abs_z']['pooled_auc'] > 0.5).sum()}/25")

    print("\n--- CirCor (Patient-level) Campaign Summary ---")
    print(circor_camp.to_string(index=False))
    print(f"\ncircor PyOD detectors > 0.5: CC2014={(circor_camp[circor_camp['detector'] != 'rms0_mean_abs_z']['CC2014_auc'] > 0.5).sum()}/25, "
          f"CC2015={(circor_camp[circor_camp['detector'] != 'rms0_mean_abs_z']['CC2015_auc'] > 0.5).sum()}/25, "
          f"Pooled={(circor_camp[circor_camp['detector'] != 'rms0_mean_abs_z']['pooled_auc'] > 0.5).sum()}/25")

    print("\n--- Top 5 Detectors by Location ---")
    print(comb_loc.to_string(index=False))

    print("\n--- Anomaly Prevalence by Campaign & Location ---")
    print(comb_prev.to_string(index=False))

    print("\n--- Primary Setting Comparison: Patient-level vs Recording-level ---")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["all", "prepare", "rotate", "confound"], default="all")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    if args.step in ("all", "prepare"):
        prepare_circor_rec_artifacts()
    if args.step in ("all", "rotate"):
        run_circor_rec_rotation(jobs=args.jobs)
    if args.step in ("all", "confound"):
        run_all_confounds()
