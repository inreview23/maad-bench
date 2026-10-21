"""Tabulate and verify all E2 experiment results.

Generates the exact markdown tables for:
- Step 1 & 2: Dataset metadata & official published baseline numbers
- Step 3: Re-implemented baseline AE vs published numbers (AUC, pAUC, delta)
- Step 4: 74-dim tabular detectors vs official baseline AE (best detector, median, AE)
- Step 5: Paper AE anchor on official split vs LOGO split (0.541 gap attribution)
- Step 6: dcase_toycar and dcase_toyconveyor benchmark results & artifact schema
"""

import json
import os
import sys
import numpy as np
import pandas as pd

# Published DCASE 2020 Baseline Numbers
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


def format_step3_table(results_dir):
    p = os.path.join(results_dir, "e2_step3_baseline_runs.parquet")
    if not os.path.exists(p):
        return "Step 3 results parquet not found at %s" % p
    df = pd.read_parquet(p)
    lines = []
    lines.append("| Machine Type | Machine ID | Published AUC | Published pAUC | Our Baseline AUC (mean ± sd) | Our Baseline pAUC (mean ± sd) | AUC Delta |")
    lines.append("|---|---|---|---|---|---|---|")

    for mtype in ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]:
        sub = df[df["machine_type"] == mtype]
        groups = sorted(sub["group"].unique())
        auc_means, pauc_means = [], []
        pub_auc_means, pub_pauc_means = [], []

        for g in groups:
            gsub = sub[sub["group"] == g]
            our_auc = gsub["auc"].mean()
            our_auc_sd = gsub["auc"].std()
            our_pauc = gsub["pauc"].mean()
            our_pauc_sd = gsub["pauc"].std()

            pub = PUBLISHED_BASELINE[mtype].get(g, {})
            pub_auc = pub.get("paper_auc", pub.get("auc", np.nan))
            pub_pauc = pub.get("paper_pauc", pub.get("pauc", np.nan))
            delta = our_auc - pub_auc

            auc_means.append(our_auc)
            pauc_means.append(our_pauc)
            pub_auc_means.append(pub_auc)
            pub_pauc_means.append(pub_pauc)

            lines.append(f"| {mtype} | ID {g:02d} | {pub_auc:.3f} | {pub_pauc:.3f} | {our_auc:.3f} ± {our_auc_sd:.3f} | {our_pauc:.3f} ± {our_pauc_sd:.3f} | {delta:+.3f} |")

        # Average row
        mean_our_auc = np.mean(auc_means)
        mean_our_pauc = np.mean(pauc_means)
        mean_pub_auc = np.mean(pub_auc_means)
        mean_pub_pauc = np.mean(pub_pauc_means)
        mean_delta = mean_our_auc - mean_pub_auc
        lines.append(f"| **{mtype}** | **Average** | **{mean_pub_auc:.3f}** | **{mean_pub_pauc:.3f}** | **{mean_our_auc:.3f}** | **{mean_our_pauc:.3f}** | **{mean_delta:+.3f}** |")

    return "\n".join(lines)


def format_step4_table(results_dir):
    p = os.path.join(results_dir, "e2_step4_tabular_runs.parquet")
    p_ae = os.path.join(results_dir, "e2_step3_baseline_runs.parquet")
    if not os.path.exists(p):
        return "Step 4 results parquet not found at %s" % p
    df = pd.read_parquet(p)
    df_ae = pd.read_parquet(p_ae) if os.path.exists(p_ae) else None

    lines = []
    lines.append("| Machine Type | Best 74-dim Detector | Best Detector AUC | Roster Median AUC | Official Baseline AE AUC | Best vs Baseline Delta |")
    lines.append("|---|---|---|---|---|---|")

    for mtype in ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]:
        sub = df[df["machine_type"] == mtype]
        # Average across seeds and groups for each detector
        det_group_means = sub.groupby(["detector", "group"])["auc_roc"].mean().reset_index()
        det_means = det_group_means.groupby("detector")["auc_roc"].mean()

        best_det = det_means.idxmax()
        best_auc = det_means.max()
        roster_median = det_means.median()

        if df_ae is not None:
            ae_sub = df_ae[df_ae["machine_type"] == mtype]
            ae_auc = ae_sub.groupby("group")["auc"].mean().mean()
        else:
            ae_auc = PUBLISHED_BASELINE[mtype]["average"].get("paper_auc", PUBLISHED_BASELINE[mtype]["average"]["auc"])

        diff = best_auc - ae_auc
        lines.append(f"| {mtype} | {best_det} | {best_auc:.3f} | {roster_median:.3f} | {ae_auc:.3f} | {diff:+.3f} |")

    return "\n".join(lines)


def format_step5_table(results_dir):
    p_paper = os.path.join(results_dir, "e2_step5_paper_ae_runs.parquet")
    p_base = os.path.join(results_dir, "e2_step3_baseline_runs.parquet")
    if not os.path.exists(p_paper):
        return "Step 5 results parquet not found at %s" % p_paper

    df_paper = pd.read_parquet(p_paper)
    df_base = pd.read_parquet(p_base) if os.path.exists(p_base) else None

    # Published LOGO anchor numbers for MIMII pump (from ae_rot_metrics.csv)
    # pump LOGO anchor = 0.541
    lines = []
    lines.append("| Machine Type | Paper AE Anchor (Official Split) | Official Baseline AE (Official Split) | Paper AE Anchor (LOGO Split) | Protocol Gap | Model Gap |")
    lines.append("|---|---|---|---|---|---|")

    for mtype in ["pump", "ToyCar", "ToyConveyor", "fan", "slider", "valve"]:
        sub_p = df_paper[df_paper["machine_type"] == mtype]
        p_auc = sub_p.groupby("group")["auc"].mean().mean()

        if df_base is not None:
            sub_b = df_base[df_base["machine_type"] == mtype]
            b_auc = sub_b.groupby("group")["auc"].mean().mean()
        else:
            b_auc = PUBLISHED_BASELINE[mtype]["average"].get("paper_auc", PUBLISHED_BASELINE[mtype]["average"]["auc"])

        logo_auc = 0.541 if mtype == "pump" else np.nan
        if not np.isnan(logo_auc):
            protocol_gap = p_auc - logo_auc
            model_gap = b_auc - p_auc
            lines.append(f"| **{mtype}** | **{p_auc:.3f}** | **{b_auc:.3f}** | **{logo_auc:.3f}** | **{protocol_gap:+.3f}** | **{model_gap:+.3f}** |")
        else:
            model_gap = b_auc - p_auc
            lines.append(f"| {mtype} | {p_auc:.3f} | {b_auc:.3f} | N/A | N/A | {model_gap:+.3f} |")

    return "\n".join(lines)


def format_step6_table(results_dir):
    lines = []
    lines.append("| Sub-dataset | Native Groups | N Clips | N Anomalies | Primary AUC Winner (mean ± fold_sd) | Secondary AUC Winner (mean ± fold_sd) | Roster Median AUC (Primary) |")
    lines.append("|---|---|---|---|---|---|---|")

    for name in ["dcase_toycar", "dcase_toyconveyor"]:
        d = os.path.join(results_dir, name)
        m_path = os.path.join(d, "rot_metrics.csv")
        meta_path = os.path.join(d, "metadata.parquet")
        if not os.path.exists(m_path) or not os.path.exists(meta_path):
            lines.append(f"| {name} | Pending / Incomplete | - | - | - | - | - |")
            continue

        meta = pd.read_parquet(meta_path)
        m = pd.read_csv(m_path)

        n_clips = len(meta)
        n_anom = int(meta["anomaly_label"].sum())
        groups = sorted(meta["group"].unique().tolist())

        prim = m[m["setting"] == "primary"].sort_values("auc_roc_mean", ascending=False)
        sec = m[m["setting"] == "secondary"].sort_values("auc_roc_mean", ascending=False)

        best_prim = prim.iloc[0]
        best_sec = sec.iloc[0]
        med_prim = prim["auc_roc_mean"].median()

        lines.append(f"| {name} | {groups} | {n_clips} | {n_anom} | {best_prim['detector']} ({best_prim['auc_roc_mean']:.3f} ± {best_prim['auc_roc_fold_sd']:.3f}) | {best_sec['detector']} ({best_sec['auc_roc_mean']:.3f} ± {best_sec['auc_roc_fold_sd']:.3f}) | {med_prim:.3f} |")

    return "\n".join(lines)


if __name__ == "__main__":
    rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/projects/maad-bench-r3/results")
    print("=== Step 3 ===")
    print(format_step3_table(rdir))
    print("\n=== Step 4 ===")
    print(format_step4_table(rdir))
    print("\n=== Step 5 ===")
    print(format_step5_table(rdir))
    print("\n=== Step 6 ===")
    print(format_step6_table(rdir))
