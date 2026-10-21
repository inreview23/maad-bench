import os
import json
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.expanduser("~/projects/maad-bench-r3/results")

# Published Baseline numbers from Koizumi et al. 2020 (arXiv:2006.05822) Table 2 (page 3)
PUBLISHED = {
    "ToyCar": {
        1: {"auc": 0.8139, "pauc": 0.6842},
        2: {"auc": 0.8603, "pauc": 0.7770},
        3: {"auc": 0.6331, "pauc": 0.5524},
        4: {"auc": 0.8450, "pauc": 0.6904},
        "mean": {"auc": 0.7877, "pauc": 0.6758}
    },
    "ToyConveyor": {
        1: {"auc": 0.7806, "pauc": 0.6416},
        2: {"auc": 0.6420, "pauc": 0.5599},
        3: {"auc": 0.7533, "pauc": 0.6103},
        "mean": {"auc": 0.7253, "pauc": 0.6043}
    },
    "fan": {
        0: {"auc": 0.5441, "pauc": 0.4939},
        2: {"auc": 0.7340, "pauc": 0.5484},
        4: {"auc": 0.6161, "pauc": 0.5332},
        6: {"auc": 0.7380, "pauc": 0.5234},
        "mean": {"auc": 0.6581, "pauc": 0.5244}
    },
    "pump": {
        0: {"auc": 0.6713, "pauc": 0.5670},
        2: {"auc": 0.6150, "pauc": 0.5810},
        4: {"auc": 0.8833, "pauc": 0.6711},
        6: {"auc": 0.7460, "pauc": 0.5796},
        "mean": {"auc": 0.7289, "pauc": 0.6163}
    },
    "slider": {
        0: {"auc": 0.9615, "pauc": 0.8139},
        2: {"auc": 0.7897, "pauc": 0.6368},
        4: {"auc": 0.9430, "pauc": 0.7198},
        6: {"auc": 0.6959, "pauc": 0.4902},
        "mean": {"auc": 0.8476, "pauc": 0.6653}
    },
    "valve": {
        0: {"auc": 0.6876, "pauc": 0.5170},
        2: {"auc": 0.6818, "pauc": 0.5183},
        4: {"auc": 0.7430, "pauc": 0.5197},
        6: {"auc": 0.5390, "pauc": 0.4843},
        "mean": {"auc": 0.6628, "pauc": 0.5098}
    }
}

print("Loading step 3, 4, 5 parquets...")
df_s3 = pd.read_parquet(os.path.join(RESULTS_DIR, "e2_step3_baseline_runs.parquet"))
df_s4 = pd.read_parquet(os.path.join(RESULTS_DIR, "e2_step4_tabular_runs.parquet"))
df_s5 = pd.read_parquet(os.path.join(RESULTS_DIR, "e2_step5_paper_ae_runs.parquet"))

print("\n=== STEP 3: BASELINE AE ===")
s3_rows = []
for mtype in ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]:
    sub = df_s3[df_s3["machine_type"] == mtype]
    groups = sorted(sub["group"].unique())
    our_aucs, our_paucs, pub_aucs, pub_paucs = [], [], [], []
    for g in groups:
        gsub = sub[sub["group"] == g]
        our_auc = float(gsub["auc"].mean())
        our_pauc = float(gsub["pauc"].mean())
        pub = PUBLISHED[mtype][g]
        pub_auc = float(pub["auc"])
        pub_pauc = float(pub["pauc"])
        delta_auc = our_auc - pub_auc
        delta_pauc = our_pauc - pub_pauc
        our_aucs.append(our_auc)
        our_paucs.append(our_pauc)
        pub_aucs.append(pub_auc)
        pub_paucs.append(pub_pauc)
        s3_rows.append({
            "machine_type": mtype,
            "group": str(g),
            "published_auc": pub_auc,
            "published_pauc": pub_pauc,
            "our_baseline_auc": our_auc,
            "our_baseline_pauc": our_pauc,
            "auc_delta": delta_auc,
            "pauc_delta": delta_pauc,
            "is_average": False
        })
    mean_our_auc = float(np.mean(our_aucs))
    mean_our_pauc = float(np.mean(our_paucs))
    mean_pub_auc = float(np.mean(pub_aucs))
    mean_pub_pauc = float(np.mean(pub_paucs))
    s3_rows.append({
        "machine_type": mtype,
        "group": "Average",
        "published_auc": mean_pub_auc,
        "published_pauc": mean_pub_pauc,
        "our_baseline_auc": mean_our_auc,
        "our_baseline_pauc": mean_our_pauc,
        "auc_delta": mean_our_auc - mean_pub_auc,
        "pauc_delta": mean_our_pauc - mean_pub_pauc,
        "is_average": True
    })

df_s3_summary = pd.DataFrame(s3_rows)
df_s3_summary.to_csv(os.path.join(RESULTS_DIR, "step3_baseline_results.csv"), index=False)
print("Saved step3_baseline_results.csv")

# Find largest per-ID deviation in Step 3
id_rows = [r for r in s3_rows if not r["is_average"]]
max_dev_row = max(id_rows, key=lambda r: abs(r["auc_delta"]))
print("Step 3 largest per-ID deviation: %s ID %s (delta=%+.4f, our=%.4f, pub=%.4f)"
      % (max_dev_row["machine_type"], max_dev_row["group"], max_dev_row["auc_delta"],
         max_dev_row["our_baseline_auc"], max_dev_row["published_auc"]))

print("\n=== STEP 4: 74-DIM TABULAR PIPELINE ===")
s4_rows = []
beat_best_count = 0
beat_median_count = 0

for mtype in ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]:
    sub = df_s4[df_s4["machine_type"] == mtype]
    # Average across seeds and groups for each detector
    det_group_means = sub.groupby(["detector", "group"])[["auc_roc", "pauc_0.1"]].mean().reset_index()
    det_means = det_group_means.groupby("detector")[["auc_roc", "pauc_0.1"]].mean()

    best_det = det_means["auc_roc"].idxmax()
    best_auc = float(det_means.loc[best_det, "auc_roc"])
    best_pauc = float(det_means.loc[best_det, "pauc_0.1"])
    roster_median_auc = float(det_means["auc_roc"].median())
    roster_median_pauc = float(det_means["pauc_0.1"].median())

    # Step 3 baseline for this machine type
    sub_ae = df_s3[df_s3["machine_type"] == mtype]
    base_ae_auc = float(sub_ae.groupby("group")["auc"].mean().mean())
    base_ae_pauc = float(sub_ae.groupby("group")["pauc"].mean().mean())

    # Unrounded comparisons
    best_beats = bool(best_auc > base_ae_auc)
    median_beats = bool(roster_median_auc > base_ae_auc)
    if best_beats:
        beat_best_count += 1
    if median_beats:
        beat_median_count += 1

    s4_rows.append({
        "machine_type": mtype,
        "best_detector": best_det,
        "best_auc": best_auc,
        "best_pauc": best_pauc,
        "roster_median_auc": roster_median_auc,
        "roster_median_pauc": roster_median_pauc,
        "baseline_ae_auc": base_ae_auc,
        "baseline_ae_pauc": base_ae_pauc,
        "best_vs_baseline_delta": best_auc - base_ae_auc,
        "median_vs_baseline_delta": roster_median_auc - base_ae_auc,
        "best_beats_baseline": best_beats,
        "median_beats_baseline": median_beats
    })

df_s4_summary = pd.DataFrame(s4_rows)
df_s4_summary.to_csv(os.path.join(RESULTS_DIR, "step4_tabular_results.csv"), index=False)
print("Saved step4_tabular_results.csv")
print("Step 4: Best detector beats baseline on %d/6 types" % beat_best_count)
print("Step 4: Median detector beats baseline on %d/6 types" % beat_median_count)
for r in s4_rows:
    print("  %s: Best=%s (AUC=%.4f, pAUC=%.4f) vs BaseAE=%.4f -> delta=%+.4f (beats=%s); Median=%.4f (beats=%s)"
          % (r["machine_type"], r["best_detector"], r["best_auc"], r["best_pauc"],
             r["baseline_ae_auc"], r["best_vs_baseline_delta"], r["best_beats_baseline"],
             r["roster_median_auc"], r["median_beats_baseline"]))

print("\n=== STEP 5: PAPER AE ANCHOR VS PROTOCOL ===")
s5_rows = []
for mtype in ["pump", "ToyCar", "ToyConveyor", "fan", "slider", "valve"]:
    sub_p = df_s5[df_s5["machine_type"] == mtype]
    p_auc = float(sub_p.groupby("group")["auc"].mean().mean())
    p_pauc = float(sub_p.groupby("group")["pauc"].mean().mean())

    sub_b = df_s3[df_s3["machine_type"] == mtype]
    b_auc = float(sub_b.groupby("group")["auc"].mean().mean())
    b_pauc = float(sub_b.groupby("group")["pauc"].mean().mean())

    logo_auc = 0.541 if mtype == "pump" else np.nan
    protocol_gap = p_auc - logo_auc if not np.isnan(logo_auc) else np.nan
    model_gap = b_auc - p_auc

    s5_rows.append({
        "machine_type": mtype,
        "paper_ae_official_auc": p_auc,
        "paper_ae_official_pauc": p_pauc,
        "baseline_ae_official_auc": b_auc,
        "baseline_ae_official_pauc": b_pauc,
        "paper_ae_logo_auc": logo_auc,
        "protocol_gap": protocol_gap,
        "model_gap": model_gap
    })
    print("  %s: PaperAE_Official=%.4f, BaseAE_Official=%.4f, LOGO=%.3f, ProtocolGap=%s, ModelGap=%+.4f"
          % (mtype, p_auc, b_auc, logo_auc,
             ("%.4f" % protocol_gap if not np.isnan(protocol_gap) else "N/A"),
             model_gap))

df_s5_summary = pd.DataFrame(s5_rows)
df_s5_summary.to_csv(os.path.join(RESULTS_DIR, "step5_paper_ae_results.csv"), index=False)
print("Saved step5_paper_ae_results.csv")

print("\n=== STEP 6: BENCHMARK SUB-DATASETS ===")
s6_rows = []
for name in ["dcase_toycar", "dcase_toyconveyor"]:
    d = os.path.join(RESULTS_DIR, name)
    m = pd.read_csv(os.path.join(d, "rot_metrics.csv"))
    runs = pd.read_parquet(os.path.join(d, "rot_runs.parquet"))
    meta = pd.read_parquet(os.path.join(d, "metadata.parquet"))
    with open(os.path.join(d, "rot_config.json")) as f:
        cfg = json.load(f)

    prim = m[m["setting"] == "primary"].sort_values("auc_roc_mean", ascending=False)
    sec = m[m["setting"] == "secondary"].sort_values("auc_roc_mean", ascending=False)

    best_p = prim.iloc[0]
    best_s = sec.iloc[0]
    med_p = float(prim["auc_roc_mean"].median())

    n_errors = int(len(runs[runs["detector"] == "*"]))
    n_detectors = int(runs[runs["detector"] != "*"]["detector"].nunique())
    seeds = sorted(runs["seed"].unique().tolist())
    settings = sorted(runs["setting"].unique().tolist())
    folds = sorted(runs["group"].unique().tolist())

    s6_rows.append({
        "subdataset": name,
        "n_clips": len(meta),
        "n_anomalies": int(meta["anomaly_label"].sum()),
        "folds": str(folds),
        "n_folds": len(folds),
        "settings": str(settings),
        "seeds": str(seeds),
        "n_detectors": n_detectors,
        "n_errors": n_errors,
        "primary_winner": best_p["detector"],
        "primary_winner_auc": float(best_p["auc_roc_mean"]),
        "primary_winner_fold_sd": float(best_p["auc_roc_fold_sd"]),
        "secondary_winner": best_s["detector"],
        "secondary_winner_auc": float(best_s["auc_roc_mean"]),
        "secondary_winner_fold_sd": float(best_s["auc_roc_fold_sd"]),
        "roster_median_auc_primary": med_p
    })
    print("  %s: N=%d (anom=%d), folds=%s, settings=%s, seeds=%s, detectors=%d, errors=%d"
          % (name, len(meta), int(meta["anomaly_label"].sum()), folds, settings, seeds, n_detectors, n_errors))
    print("    Primary Winner: %s (AUC=%.4f ± %.4f fold_sd)"
          % (best_p["detector"], best_p["auc_roc_mean"], best_p["auc_roc_fold_sd"]))
    print("    Secondary Winner: %s (AUC=%.4f ± %.4f fold_sd)"
          % (best_s["detector"], best_s["auc_roc_mean"], best_s["auc_roc_fold_sd"]))
    print("    Roster Median AUC (Primary): %.4f" % med_p)

df_s6_summary = pd.DataFrame(s6_rows)
df_s6_summary.to_csv(os.path.join(RESULTS_DIR, "step6_summary.csv"), index=False)
print("Saved step6_summary.csv")
