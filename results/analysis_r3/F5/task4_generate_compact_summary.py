#!/usr/bin/env python3
"""Task 4: Generate Compact Summary Table and Confounds Synthesis.
Creates:
1. Compact markdown and CSV tables synthesizing ICBHI device confounds and PhysioNet source confounds.
2. A unified synthesis suitable for the paper's confounds paragraph.
"""

import os
import pandas as pd

OUT_DIR = "results/analysis_r3/F5"

def main():
    # Load previously generated CSVs
    icbhi_strat = pd.read_csv(os.path.join(OUT_DIR, "icbhi_full_roster_device_stratified.csv"))
    icbhi_mech = pd.read_csv(os.path.join(OUT_DIR, "icbhi_mechanism_score_offsets.csv"))
    physio_src = pd.read_csv(os.path.join(OUT_DIR, "physionet_source_breakdown.csv"))
    physio_corr = pd.read_csv(os.path.join(OUT_DIR, "physionet_source_correlations.csv"))
    
    # 1. ICBHI Compact Summary
    # Extract row counts and prevalences from icbhi_mech (using LOF row)
    icbhi_dev_stats = icbhi_mech[icbhi_mech["detector"] == "LOF"][["device", "n_total", "n_norm", "n_anom", "prevalence"]].copy()
    
    # Key detectors to showcase: LOF (best pooled), Sampling (median), KDE (best on LittC2SE), CBLOF, rms0 baseline
    key_dets = ["LOF", "Sampling", "KDE", "CBLOF", "rms0_mean_abs_z"]
    
    # Extract AUCs for key detectors
    auc_cols = ["AKGC417L", "Litt3200", "LittC2SE", "Meditron", "pooled"]
    sub_auc = icbhi_strat[icbhi_strat["detector"].isin(key_dets)].set_index("detector")[auc_cols].T
    
    # 2. Build Compact Table for ICBHI
    icbhi_summary_rows = []
    # Order: AKGC417L, Litt3200, LittC2SE, Meditron, Pooled
    dev_order = ["AKGC417L", "Litt3200", "LittC2SE", "Meditron"]
    for d in dev_order:
        st = icbhi_dev_stats[icbhi_dev_stats["device"] == d].iloc[0]
        # Get mean baseline offsets for COPOD and LOF
        copod_mean = icbhi_mech[(icbhi_mech["detector"] == "COPOD") & (icbhi_mech["device"] == d)]["mean_score_overall"].values[0]
        lof_mean = icbhi_mech[(icbhi_mech["detector"] == "LOF") & (icbhi_mech["device"] == d)]["mean_score_overall"].values[0]
        
        row = {
            "Stratum / Source": f"ICBHI: {d}",
            "N": int(st["n_total"]),
            "Prevalence": f"{st['prevalence']:.3f}",
            "Baseline Offset": f"COPOD {copod_mean:.1f}, LOF {lof_mean:.3f}",
            "AUC LOF": f"{icbhi_strat.loc[icbhi_strat['detector']=='LOF', d].values[0]:.3f}",
            "AUC Sampling": f"{icbhi_strat.loc[icbhi_strat['detector']=='Sampling', d].values[0]:.3f}",
            "AUC KDE": f"{icbhi_strat.loc[icbhi_strat['detector']=='KDE', d].values[0]:.3f}",
            "AUC Baseline (rms0)": f"{icbhi_strat.loc[icbhi_strat['detector']=='rms0_mean_abs_z', d].values[0]:.3f}",
            "Detectors < 0.5": f"{sum(icbhi_strat[d] < 0.5)} / 26"
        }
        icbhi_summary_rows.append(row)
        
    # Pooled ICBHI row
    st_pool = icbhi_dev_stats[icbhi_dev_stats["device"] == "POOLED (All)"].iloc[0]
    copod_pool = icbhi_mech[(icbhi_mech["detector"] == "COPOD") & (icbhi_mech["device"] == "POOLED (All)")]["mean_score_overall"].values[0]
    lof_pool = icbhi_mech[(icbhi_mech["detector"] == "LOF") & (icbhi_mech["device"] == "POOLED (All)")]["mean_score_overall"].values[0]
    icbhi_summary_rows.append({
        "Stratum / Source": "ICBHI: Pooled (LOGO)",
        "N": int(st_pool["n_total"]),
        "Prevalence": f"{st_pool['prevalence']:.3f}",
        "Baseline Offset": f"COPOD {copod_pool:.1f}, LOF {lof_pool:.3f}",
        "AUC LOF": f"{icbhi_strat.loc[icbhi_strat['detector']=='LOF', 'pooled'].values[0]:.3f}",
        "AUC Sampling": f"{icbhi_strat.loc[icbhi_strat['detector']=='Sampling', 'pooled'].values[0]:.3f}",
        "AUC KDE": f"{icbhi_strat.loc[icbhi_strat['detector']=='KDE', 'pooled'].values[0]:.3f}",
        "AUC Baseline (rms0)": f"{icbhi_strat.loc[icbhi_strat['detector']=='rms0_mean_abs_z', 'pooled'].values[0]:.3f}",
        "Detectors < 0.5": f"{sum(icbhi_strat['pooled'] < 0.5)} / 26"
    })
    
    # Device-Balanced Equal-Weighted row
    icbhi_summary_rows.append({
        "Stratum / Source": "ICBHI: Device-Balanced (Equal)",
        "N": "6898",
        "Prevalence": "---",
        "Baseline Offset": "Equal 4-device mean",
        "AUC LOF": f"{icbhi_strat.loc[icbhi_strat['detector']=='LOF', 'dev_bal_equal_weighted'].values[0]:.3f}",
        "AUC Sampling": f"{icbhi_strat.loc[icbhi_strat['detector']=='Sampling', 'dev_bal_equal_weighted'].values[0]:.3f}",
        "AUC KDE": f"{icbhi_strat.loc[icbhi_strat['detector']=='KDE', 'dev_bal_equal_weighted'].values[0]:.3f}",
        "AUC Baseline (rms0)": f"{icbhi_strat.loc[icbhi_strat['detector']=='rms0_mean_abs_z', 'dev_bal_equal_weighted'].values[0]:.3f}",
        "Detectors < 0.5": f"{sum(icbhi_strat['dev_bal_equal_weighted'] < 0.5)} / 26"
    })
    
    # Device-Balanced Pair-Weighted row
    icbhi_summary_rows.append({
        "Stratum / Source": "ICBHI: Device-Balanced (Pair-Wt)",
        "N": "6898",
        "Prevalence": "---",
        "Baseline Offset": "89.6% AKGC417L weight",
        "AUC LOF": f"{icbhi_strat.loc[icbhi_strat['detector']=='LOF', 'dev_bal_pair_weighted'].values[0]:.3f}",
        "AUC Sampling": f"{icbhi_strat.loc[icbhi_strat['detector']=='Sampling', 'dev_bal_pair_weighted'].values[0]:.3f}",
        "AUC KDE": f"{icbhi_strat.loc[icbhi_strat['detector']=='KDE', 'dev_bal_pair_weighted'].values[0]:.3f}",
        "AUC Baseline (rms0)": f"{icbhi_strat.loc[icbhi_strat['detector']=='rms0_mean_abs_z', 'dev_bal_pair_weighted'].values[0]:.3f}",
        "Detectors < 0.5": f"{sum(icbhi_strat['dev_bal_pair_weighted'] < 0.5)} / 26"
    })

    df_compact_icbhi = pd.DataFrame(icbhi_summary_rows)
    df_compact_icbhi.to_csv(os.path.join(OUT_DIR, "compact_summary_icbhi.csv"), index=False)
    
    # 3. Build Compact Table for PhysioNet
    physio_summary_rows = []
    for _, r in physio_src.iterrows():
        physio_summary_rows.append({
            "Source": f"PhysioNet: {r['source']}",
            "N": int(r["n"]),
            "Prevalence": f"{r['prevalence']:.3f}",
            "AUC SOD": f"{r['auc_SOD']:.3f}",
            "AUC ECOD": f"{r['auc_ECOD']:.3f}",
            "AUC GMM": f"{r['auc_GMM']:.3f}",
            "AUC COPOD": f"{r['auc_COPOD']:.3f}",
            "AUC QMCD": f"{r['auc_QMCD']:.3f}"
        })
        
    # Mean row
    physio_summary_rows.append({
        "Source": "PhysioNet: Fold Mean",
        "N": int(physio_src["n"].sum()),
        "Prevalence": f"{physio_src['n_anom'].sum()/physio_src['n'].sum():.3f}",
        "AUC SOD": f"{physio_src['auc_SOD'].mean():.3f}",
        "AUC ECOD": f"{physio_src['auc_ECOD'].mean():.3f}",
        "AUC GMM": f"{physio_src['auc_GMM'].mean():.3f}",
        "AUC COPOD": f"{physio_src['auc_COPOD'].mean():.3f}",
        "AUC QMCD": f"{physio_src['auc_QMCD'].mean():.3f}"
    })
    
    # Correlation row with N
    sod_r_n = physio_corr.loc[physio_corr['detector']=='SOD', 'pearson_r_n'].values[0]
    ecod_r_n = physio_corr.loc[physio_corr['detector']=='ECOD', 'pearson_r_n'].values[0]
    gmm_r_n = physio_corr.loc[physio_corr['detector']=='GMM', 'pearson_r_n'].values[0]
    copod_r_n = physio_corr.loc[physio_corr['detector']=='COPOD', 'pearson_r_n'].values[0]
    qmcd_r_n = physio_corr.loc[physio_corr['detector']=='QMCD', 'pearson_r_n'].values[0]
    physio_summary_rows.append({
        "Source": "Corr with N (r, p<0.05*)",
        "N": "---",
        "Prevalence": "---",
        "AUC SOD": f"r={sod_r_n:+.3f}*",
        "AUC ECOD": f"r={ecod_r_n:+.3f}",
        "AUC GMM": f"r={gmm_r_n:+.3f}",
        "AUC COPOD": f"r={copod_r_n:+.3f}",
        "AUC QMCD": f"r={qmcd_r_n:+.3f}*"
    })
    
    df_compact_physio = pd.DataFrame(physio_summary_rows)
    df_compact_physio.to_csv(os.path.join(OUT_DIR, "compact_summary_physionet.csv"), index=False)
    
    print("\n=== COMPACT ICBHI SUMMARY ===")
    print(df_compact_icbhi.to_string(index=False))
    
    print("\n=== COMPACT PHYSIONET SUMMARY ===")
    print(df_compact_physio.to_string(index=False))

if __name__ == "__main__":
    main()
