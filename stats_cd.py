"""Cross-dataset detector statistics for MAAD-Bench: Friedman, Nemenyi, and a
cluster-aware permutation test.

The v1 version of this file pivoted the per-seed scores on ["dataset", "seed"],
producing 35 blocks from 7 sub-datasets and a Friedman statistic of 188.1 with
p below 1e-26. The five seeds within a sub-dataset were repeated measurements on
one fixed split, not independent blocks; in the primary setting they shared a
training set outright. Averaging over folds and seeds and blocking on the
sub-dataset is the honest version, and it costs most of that significance.

Two further points this file now makes explicit, both of which the v1 analysis
elided:

  - Rejecting equal ranks says detectors differ somewhere. It does not say that
    no single detector wins, and it does not license a pairwise ordering. The
    Nemenyi critical difference is reported alongside so the reader can see how
    little 7 blocks and 25 treatments resolve.
  - The distinct-winner count is reported against its exchangeable-detector
    null, because six winners across seven sub-datasets is what interchangeable
    detectors produce and is therefore not evidence of anything.

This is a thin front end over analysis.py, which holds the implementations.

Usage:
  python stats_cd.py                     # handcrafted, primary
  python stats_cd.py --repr panns --setting secondary
"""
import argparse

from analysis import (dataset_matrix, fold_table, friedman_block, load_runs,
                      permutation_block, winner_null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="auc_roc")
    ap.add_argument("--repr", default="handcrafted")
    ap.add_argument("--setting", default="primary")
    ap.add_argument("--perms", type=int, default=20000)
    args = ap.parse_args()

    runs = load_runs(args.repr, args.metric)
    mat = dataset_matrix(fold_table(runs, args.metric, args.setting))

    fr = friedman_block(mat)
    if fr is None:
        raise SystemExit("not enough blocks or detectors for an omnibus test")

    print("=== %s / %s: %d sub-dataset blocks, %d detectors ==="
          % (args.repr, args.setting, fr["n_blocks"], fr["n_detectors"]))
    print("Friedman chi2=%.2f, p=%.4g" % (fr["chi2"], fr["p"]))
    pm = permutation_block(mat, n_perm=args.perms)
    print("Cluster-aware permutation (sub-dataset as the resampling unit): "
          "p=%.4g over %d permutations" % (pm["p"], pm["n_perm"]))
    print("Nemenyi critical difference (alpha=0.05): %.2f rank units"
          % fr["critical_difference"])
    print("  largest observed gap: %.2f (%s %.2f to %s %.2f)"
          % (fr["largest_gap"], fr["best_detector"], fr["best_avg_rank"],
             fr["worst_detector"], fr["worst_avg_rank"]))
    print("  detectors within CD of the best: %d/%d; significantly separated "
          "pairs: %d" % (fr["n_within_cd_of_best"], fr["n_detectors"],
                         fr["n_separated_pairs"]))
    if fr["n_separated_pairs"] == 0:
        print("  -> the post-hoc resolves no pairwise difference. This is a power "
              "limit of %d blocks against %d treatments, not evidence that the "
              "detectors are equivalent." % (fr["n_blocks"], fr["n_detectors"]))

    print("\naverage ranks (1 = best):")
    for det, r in fr["avg_ranks"].items():
        print("  %-14s %5.2f" % (det, r))

    wn = winner_null(mat)
    print("\ndistinct winners: %d over %d sub-datasets; exchangeable-detector "
          "null expects %.2f, P(>= observed) = %.3f"
          % (wn["observed_distinct_winners"], wn["n_blocks"],
             wn["expected_under_null"], wn["p_at_least_observed"]))
    if wn["p_at_least_observed"] > 0.1:
        print("  -> the winner count carries no evidence against exchangeable "
              "detectors and should not be cited as support for rank instability.")


if __name__ == "__main__":
    main()
