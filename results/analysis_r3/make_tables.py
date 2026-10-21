"""Build the MAAD-Bench TST tables and figures from the verified analysis outputs.

Every number in the generated .tex files comes from a CSV written by the analysis
drivers (out/R1F, out/R2F, out/E*), so no value is typed by hand.
Usage: python make_tables.py <maad-r3 scratch root>
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = Path(sys.argv[1])
R1, R2 = S / "out/R1F", S / "out/R2F"
DATA = S / "snap2/results"
TAB, FIG = S / "paper-sensors/tables", S / "paper-sensors/figures"

ORDER = ["esc50", "urbansound8k", "ravdess", "nsynth", "cremad",
         "mimii", "mimii_fan", "mimii_slider", "mimii_valve", "dcase_toycar", "dcase_toyconveyor",
         "physionet", "icbhi", "circor_rec"]
DESIGNATED = ORDER[:5]
NAME = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "ravdess": "RAVDESS", "nsynth": "NSynth",
        "cremad": "CREMA-D", "mimii": "MIMII pump", "mimii_fan": "MIMII fan", "mimii_slider": "MIMII slider",
        "mimii_valve": "MIMII valve", "dcase_toycar": "ToyCar", "dcase_toyconveyor": "ToyConveyor",
        "physionet": "PhysioNet", "icbhi": "ICBHI", "circor_rec": "CirCor"}
FAMILY = {"esc50": "env", "urbansound8k": "env", "ravdess": "speech", "cremad": "speech",
          "mimii": "machine", "mimii_fan": "machine", "mimii_slider": "machine", "mimii_valve": "machine",
          "dcase_toycar": "machine", "dcase_toyconveyor": "machine", "physionet": "heart", "circor_rec": "heart",
          "nsynth": "music", "icbhi": "lung"}

# Static corpus facts (sampling rate and clip length from the corpus documentation, checked
# on the files for the new sub-datasets; group description from each loader).
STATIC = {
    "esc50": ("Environmental", "44.1~kHz", "5~s", "10\\,/\\,3", "5 folds"),
    "urbansound8k": ("Urban", "varies", "$\\le$4~s", "6\\,/\\,3", "10 folds"),
    "ravdess": ("Emotional speech", "48~kHz", "3--4~s", "2\\,/\\,2", "24 actors"),
    "nsynth": ("Musical timbre", "16~kHz", "4~s", "1\\,/\\,2", "5 instrument buckets"),
    "cremad": ("Emotional speech", "16~kHz", "1--5~s", "1\\,/\\,2", "91 actors"),
    "mimii": ("Machine (pump)", "16~kHz", "10~s", "1\\,/\\,1", "4 machine IDs"),
    "mimii_fan": ("Machine (fan)", "16~kHz", "10~s", "1\\,/\\,1", "4 machine IDs"),
    "mimii_slider": ("Machine (slide rail)", "16~kHz", "10~s", "1\\,/\\,1", "4 machine IDs"),
    "mimii_valve": ("Machine (valve)", "16~kHz", "10~s", "1\\,/\\,1", "4 machine IDs"),
    "dcase_toycar": ("Toy machine (car)", "16~kHz", "11~s", "1\\,/\\,1", "4 machine IDs"),
    "dcase_toyconveyor": ("Toy machine (conveyor)", "16~kHz", "10~s", "1\\,/\\,1", "3 machine IDs"),
    "physionet": ("Heart sound", "$\\sim$2~kHz", "variable", "1\\,/\\,1", "6 source databases"),
    "icbhi": ("Respiratory sound", "4--44~kHz", "breath cycle", "1\\,/\\,3", "5 patient buckets"),
    "circor_rec": ("Heart sound (pediatric)", "4~kHz", "5--65~s", "1\\,/\\,1", "5 patient buckets"),
}


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def th(n):
    return f"{n:,}".replace(",", "{,}")


def mathminus(text):
    return re.sub(r"(?<=[\s\[(}])-(?=\d\.\d)", "$-$", text)


def write(name, body):
    (TAB / name).write_text(mathminus(body))
    print("wrote", name)


def rule_after(ds):
    return ds in ("cremad",)


def tab_datasets():
    rows = []
    for ds in ORDER:
        md = pd.read_parquet(DATA / ds / "metadata.parquet")
        dom, sr, ln, cl, grp = STATIC[ds]
        n, a = len(md), int(md.anomaly_label.sum())
        rows.append(f"{NAME[ds]} & {dom} & {sr} & {ln} & {cl} & {th(n)}\\,/\\,{th(a)} & {grp} \\\\")
        if rule_after(ds):
            rows.append("\\midrule")
    body = "\n".join(rows)
    write("tab_datasets.tex", body + "\n")


def tab_pertask():
    mt = pd.read_csv(R1 / "main_table.csv")
    mt = mt[mt.setting == "primary"].set_index("dataset")
    nl = pd.read_csv(R1 / "null_max25.csv").set_index("dataset")
    sg = pd.read_csv(R1 / "sig_tests_summary.csv")
    sg = sg[sg.setting == "primary"].set_index("dataset")
    tb = pd.read_csv(R1 / "trivial_baselines.csv")
    tb = tb[tb.setting == "primary"]
    loud = tb[tb.baseline == "rms0_abs_z"].set_index("dataset")
    pw = pd.read_csv(R1 / "pair_weighted_top3.csv")
    pw = pw[pw["rank"] == 1].set_index("dataset")
    rows, csv = [], []
    for ds in ORDER:
        m, n, s, l = mt.loc[ds], nl.loc[ds], sg.loc[ds], loud.loc[ds]
        best, null = m.best_auc, n.corr_95pct_max
        above_null = best > null
        loud_vs_med = l.auc_mean > m.median_auc
        best_cell = f"{m.best_detector} {f3(best)}"
        if not above_null:
            best_cell = "\\cellcolor{coral!35}" + best_cell
        loud_cell = f"{f3(l.auc_mean)} ({int(l.detectors_beaten)})"
        if loud_vs_med:
            loud_cell = "\\cellcolor{mint}" + loud_cell
        rows.append(f"{NAME[ds]} & {int(m.folds)} & {f3(m.anom_fraction)} & {best_cell} & {f3(m.median_auc)} & "
                    f"{loud_cell} & {f3(null)} & {int(s.nb_above_holm)} & {int(s.nb_below_holm)} \\\\")
        if rule_after(ds):
            rows.append("\\midrule")
        csv.append(dict(dataset=ds, folds=int(m.folds), anom_fraction=m.anom_fraction, best=m.best_detector,
                        best_auc=best, median=m.median_auc, loud=l.auc_mean, loud_beats=int(l.detectors_beaten),
                        null95_corr=null, null95_indep=n.indep_95pct_max, best_above_null=above_null,
                        loud_above_median=loud_vs_med, nb_above=int(s.nb_above_holm), nb_below=int(s.nb_below_holm),
                        best_pairweighted=pw.loc[ds].auc_pair_weighted))
    write("tab_pertask.tex", "\n".join(rows) + "\n")
    pd.DataFrame(csv).to_csv(TAB / "pertask_numbers.csv", index=False)


def tab_reliability():
    rel = pd.read_csv(R2 / "item1_reliability_all14.csv").set_index("dataset")
    fw = pd.read_csv(R2 / "item4_fold_winners_summary.csv")
    fw = fw[fw.setting == "primary"].set_index("dataset")
    rows = []
    for ds in ORDER:
        r, w = rel.loc[ds], fw.loc[ds]
        rho = f"{r.mean_pairwise_rho:+.2f} [{r.mp_ci_lo:+.2f}, {r.mp_ci_hi:+.2f}]"
        p = r.p_two_sided
        pc = "$<$0.001" if p < 0.001 else f3(p)
        cell = rho if p < 0.05 else "\\cellcolor{coral!35}" + rho
        sb = f2(r.pairwise_sb) if r.mean_pairwise_rho >= 0 else "n/a"
        rows.append(f"{NAME[ds]} & {int(r.k_folds)} & {cell} & {pc} & {sb} & "
                    f"{w.n_differs}/{w.n_folds} & {w.obs_mean_rank:.1f} & {w.null_mean_rank:.1f} \\\\")
        if rule_after(ds):
            rows.append("\\midrule")
    write("tab_reliability.tex", "\n".join(rows) + "\n")


def tab_agreement():
    fam = pd.read_csv(R2 / "item2_family_summaries.csv")
    label = {"Machine sound": "Machine sound (6 tasks)", "Speech emotion": "Emotional speech (2)",
             "Heart sound": "Heart sound (2)", "Environmental sound": "Environmental sound (2)",
             "ALL Within-Family Pairs": "All within-family pairs", "ALL Between-Family Pairs": "All between-family pairs",
             "All 91 Sub-dataset Pairs": "All pairs"}
    rows = []
    # Pairs with a ceiling per row (n_c), from the pair table with the family map of Section 3.5.
    pr = pd.read_csv(R2 / "item2_cross_task_pairs_all.csv")
    fa, fb = pr.dataset_a.map(FAMILY), pr.dataset_b.map(FAMILY)
    named = {"env", "speech", "machine", "heart"}
    sel = {"Machine sound": (fa == "machine") & (fb == "machine"), "Speech emotion": (fa == "speech") & (fb == "speech"),
           "Heart sound": (fa == "heart") & (fb == "heart"), "Environmental sound": (fa == "env") & (fb == "env"),
           "ALL Within-Family Pairs": (fa == fb) & fa.isin(named),
           "ALL Between-Family Pairs": (fa != fb) & fa.isin(named) & fb.isin(named),
           "All 91 Sub-dataset Pairs": pd.Series(True, index=pr.index)}
    for _, r in fam.iterrows():
        if r["name"] not in label:
            continue
        q = pr[sel[r["name"]]]
        assert len(q) == int(r.n_pairs), (r["name"], len(q), r.n_pairs)
        nc = int(q.ceiling.notna().sum())
        ceil = f2(r.mean_ceiling) if not np.isnan(r.mean_ceiling) else "--"
        gap = f2(r.mean_gap) if not np.isnan(r.mean_gap) else "--"
        rows.append(f"{label[r['name']]} & {int(r.n_pairs)} & {r.mean_observed_rho:+.2f} & "
                    f"[{r.min_observed_rho:+.2f}, {r.max_observed_rho:+.2f}] & {nc} & {ceil} & {gap} \\\\")
        if r["name"] in ("Environmental sound",):
            rows.append("\\midrule")
    # Regime-level rows: designated-class vs organic-label (same pair table, same columns).
    da, db = pr.dataset_a.isin(DESIGNATED), pr.dataset_b.isin(DESIGNATED)
    reg = [("Within designated-class", da & db), ("Within organic-label", ~da & ~db), ("Between regimes", da != db)]
    rows.append("\\midrule")
    for lab, m in reg:
        q = pr[m]
        c = q[q.ceiling.notna()]
        rows.append(f"{lab} & {len(q)} & {q.observed_rho.mean():+.2f} & [{q.observed_rho.min():+.2f}, "
                    f"{q.observed_rho.max():+.2f}] & {len(c)} & {f2(c.ceiling.mean())} & {f2((c.ceiling - c.observed_rho).mean())} \\\\")
    write("tab_agreement.tex", "\n".join(rows) + "\n")


def tab_singlecol():
    tb = pd.read_csv(R1 / "trivial_baselines.csv")
    tb = tb[tb.setting == "primary"]
    cols = [("rms0_abs_z", False), ("l2_std_norm", False), ("honest_selector", False), ("oracle_best_col", True)]
    rows = []
    for ds in ORDER:
        cells = []
        for b, named in cols:
            r = tb[(tb.dataset == ds) & (tb.baseline == b)].iloc[0]
            c = f"{f3(r.auc_mean)} ({int(r.detectors_beaten)})"
            if named:
                c = "\\texttt{" + str(r.detail).replace("_", "\\_") + "} " + c
            if int(r.detectors_beaten) >= 13:
                c = "\\cellcolor{mint}" + c
            cells.append(c)
        rows.append(f"{NAME[ds]} & " + " & ".join(cells) + " \\\\")
        if rule_after(ds):
            rows.append("\\midrule")
    write("tab_singlecol.tex", "\n".join(rows) + "\n")


def tab_matrix(setting):
    num = pd.read_csv(R1 / f"appendix_auc_matrix_{setting}_numeric.csv", index_col=0)
    blocks = {"designated": DESIGNATED, "organic": ORDER[5:]}
    for bname, cols in blocks.items():
        lines = []
        means = pd.DataFrame({ds: num[f"{ds}_mean"] for ds in cols})
        for det in sorted(num.index, key=str.lower):
            cells = []
            for ds in cols:
                m, sd = num.loc[det, f"{ds}_mean"], num.loc[det, f"{ds}_sd"]
                c = f"{m:.3f}$\\pm${sd:.3f}"
                if m == means[ds].max():
                    c = "\\cellcolor{mint}\\textbf{" + c + "}"
                cells.append(c)
            lines.append(f"{det} & " + " & ".join(cells) + " \\\\")
        write(f"tab_matrix_{setting}_{bname}.tex", "\n".join(lines) + "\n")


def fig_transfer():
    pr = pd.read_csv(R2 / "item2_cross_task_pairs_all.csv")
    pr = pr[pr.ceiling.notna()]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    within = pr.category == "within_family"
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.scatter(pr.ceiling[~within], pr.observed_rho[~within], s=12, color="#C9C9C9", edgecolor="#777777",
               lw=0.4, label="other pairs")
    ax.scatter(pr.ceiling[within], pr.observed_rho[within], s=16, color="#ED8D5A", edgecolor="#9a4a1f",
               lw=0.4, label="same family")
    for _, r in pr[within].iterrows():
        if r.pair in ("esc50 / urbansound8k", "urbansound8k / esc50", "ravdess / cremad", "cremad / ravdess"):
            a, b = r.pair.split(" / ")
            spot = (0.55, 0.88) if a in ("esc50", "urbansound8k") else (0.62, -0.25)
            ax.annotate(f"{NAME[a]} / {NAME[b]}", (r.ceiling, r.observed_rho), fontsize=6, xytext=spot,
                        textcoords="data", ha="right", va="center",
                        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"Reliability ceiling $\sqrt{r_A r_B}$", fontsize=8)
    ax.set_ylabel("Observed rank correlation", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_transfer.pdf")
    print("wrote fig_transfer.pdf", len(pr), "pairs with a positive ceiling")


def fig_cd():
    rk = pd.read_csv(R2 / "item3_detector_ranks_all_datasets.csv").sort_values("avg_rank")
    cd = pd.read_csv(R2 / "item3_cd_summary.csv").set_index("regime").loc["all_datasets"]
    rejected = set(str(cd.rejected_dets_holm).split(", "))
    fig, ax = plt.subplots(figsize=(5.45, 2.5))
    lo, hi = 8, 19
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.80
    ax.plot([lo, hi], [y0, y0], color="black", lw=0.9)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y0, y0 + 0.03], color="black", lw=0.7)
        ax.text(t, y0 + 0.06, str(t), ha="center", va="bottom", fontsize=7)
    ax.plot([lo, lo + cd.nemenyi_cd], [0.97, 0.97], color="#2b5d8a", lw=1.6)
    ax.text(lo + cd.nemenyi_cd / 2, 0.985, f"Nemenyi CD = {cd.nemenyi_cd:.2f}", ha="center", va="bottom", fontsize=7)
    ax.plot([rk.avg_rank.min(), rk.avg_rank.max()], [y0 - 0.05, y0 - 0.05], color="#555555", lw=2.4, alpha=0.6)
    n = len(rk)
    left, right = rk.iloc[: (n + 1) // 2], rk.iloc[(n + 1) // 2:]
    step = 0.68 / max(len(left), len(right))
    for i, (_, r) in enumerate(left.iterrows()):
        yy = y0 - 0.10 - i * step
        col = "#ED8D5A" if r.detector in rejected else "black"
        ax.plot([r.avg_rank, r.avg_rank, lo - 0.1], [y0, yy, yy], color="#888888", lw=0.5)
        ax.text(lo - 0.15, yy, f"{r.detector} ({r.avg_rank:.1f})", ha="right", va="center", fontsize=6.5, color=col)
    for i, (_, r) in enumerate(right.iloc[::-1].iterrows()):
        yy = y0 - 0.10 - i * step
        col = "#c4561f" if r.detector in rejected else "black"
        ax.plot([r.avg_rank, r.avg_rank, hi + 0.1], [y0, yy, yy], color="#888888", lw=0.5)
        ax.text(hi + 0.15, yy, f"({r.avg_rank:.1f}) {r.detector}", ha="left", va="center", fontsize=6.5, color=col)
    fig.subplots_adjust(left=0.17, right=0.84, top=0.97, bottom=0.02)
    fig.savefig(FIG / "fig_cd.pdf")
    print("wrote fig_cd.pdf; Friedman chi2", round(cd.chi2, 2), "p", round(cd.p_friedman, 4), "rejected", rejected)


def repr_numbers():
    """Best and median fold-mean AUC-ROC per representation (primary setting)."""
    rows = []
    for ds in ORDER:
        rec = {"dataset": ds}
        hc = pd.read_csv(DATA / ds / "rot_metrics.csv")
        sources = {"hand": hc}
        for rep in ("ast", "panns"):
            # CREMA-D's embedding rotations ran on DeltaAI (cremad_dtai) with the Spark embeddings and the same
            # call; cremadx is the Spark copy of the same run, used only if the DeltaAI output is missing.
            f = DATA / ds / f"rot_{rep}_metrics.csv"
            if ds == "cremad":
                for alt in ("cremad_dtai", "cremadx"):
                    if (DATA / alt / f"rot_{rep}_metrics.csv").exists():
                        f = DATA / alt / f"rot_{rep}_metrics.csv"
                        break
            sources[rep] = pd.read_csv(f) if f.exists() else None
        for rep, df in sources.items():
            if df is None:
                rec.update({f"{rep}_best": np.nan, f"{rep}_det": "", f"{rep}_median": np.nan})
                continue
            p = df[df.setting == "primary"].set_index("detector").auc_roc_mean
            rec.update({f"{rep}_best": p.max(), f"{rep}_det": p.idxmax(), f"{rep}_median": p.median(),
                        f"{rep}_n": len(p)})
        ae = DATA / ds / "ae_rot_metrics.csv"
        if ds == "cremad" and not ae.exists():
            # CREMA-D's anchor ran on DeltaAI in four fold shards, merged with e7_ae.run's summary code.
            ae = DATA / "cremad_dtai_ae" / "ae_rot_metrics.csv"
        rec["ae"] = pd.read_csv(ae).query("setting == 'primary'").auc_roc_mean.iloc[0] if ae.exists() else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "repr_numbers.csv", index=False)
    return df


def tab_repr():
    df = repr_numbers()
    lines = []
    for _, r in df.iterrows():
        cells = []
        best = np.nanmax([r.hand_best, r.ast_best, r.panns_best])
        for rep in ("hand", "ast", "panns"):
            if np.isnan(r[f"{rep}_best"]):
                cells += ["--", "--"]
                continue
            c = f"{r[f'{rep}_best']:.3f} ({r[f'{rep}_det']})"
            if r[f"{rep}_best"] == best:
                c = "\\cellcolor{mint}" + c
            cells += [c, f3(r[f"{rep}_median"])]
        ae = f3(r.ae) if not np.isnan(r.ae) else "--"
        lines.append(f"{NAME[r.dataset]} & " + " & ".join(cells) + f" & {ae} \\\\")
        if rule_after(r.dataset):
            lines.append("\\midrule")
    write("tab_repr.tex", "\n".join(lines) + "\n")


def fig_repr():
    df = pd.read_csv(TAB / "repr_numbers.csv")
    x = np.arange(len(df))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    cols = {"hand": ("#C9C9C9", "Handcrafted 74"), "ast": ("#ED8D5A", "AST 64"), "panns": ("#BFDFD2", "PANNs 64")}
    for i, rep in enumerate(("hand", "ast", "panns")):
        ax.bar(x + (i - 1) * w, df[f"{rep}_best"], w, color=cols[rep][0], edgecolor="#555555", lw=0.4,
               label=cols[rep][1])
        ax.scatter(x + (i - 1) * w, df[f"{rep}_median"], marker="_", color="black", s=30, lw=1.0, zorder=3)
    ax.axhline(0.5, color="#888888", lw=0.6, ls="--")
    ax.axvline(4.5, color="#888888", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[d] for d in df.dataset], rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUC-ROC (primary)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig_repr.pdf")
    print("wrote fig_repr.pdf")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "repr":
    TAB.mkdir(parents=True, exist_ok=True)
    tab_repr()
    fig_repr()
    sys.exit(0)

if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tab_datasets()
    tab_pertask()
    tab_reliability()
    tab_agreement()
    tab_singlecol()
    tab_matrix("primary")
    tab_matrix("secondary")
    fig_transfer()
    fig_cd()
