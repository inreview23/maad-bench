import glob, os
import pandas as pd
out = "results/cremad"
parts = sorted(glob.glob(os.path.join(out, "ae_rot_scores_shard*.parquet")))
sdf = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
sdf.to_parquet(os.path.join(out, "ae_rot_scores.parquet"), index=False)
per_fold = sdf.groupby(["setting", "group"], as_index=False)["auc_roc"].mean()
summ = per_fold.groupby("setting")["auc_roc"].agg(["mean", "std", "count"])
summ.to_csv(os.path.join(out, "ae_rot_metrics.csv"))
print(len(parts), sdf.shape, sdf.group.nunique(), sdf.seed.nunique())
print(summ)
