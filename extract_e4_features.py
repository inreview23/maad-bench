import os
import time
import pandas as pd
from joblib import Parallel, delayed
import maad_bench as mb
from test_loaders import load_cremad, load_circor

def extract_features_parallel(name, df, n_jobs=5):
    import hashlib
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    has_seg = "offset" in df.columns
    if has_seg:
        keysrc = "\n".join("%s|%s|%s" % (p, o, d)
                           for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(keysrc.encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, "%s_feat_%s.parquet" % (name, key))
    if os.path.exists(cache):
        print(f"[{name}] cached features found: {cache}", flush=True)
        return pd.read_parquet(cache)
    
    print(f"[{name}] extracting features for {len(df)} clips (n_jobs={n_jobs}) ...", flush=True)
    t0 = time.time()
    if has_seg:
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(mb.extract_one)(p, o, d)
            for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(mb.extract_one)(p) for p in df["path"])
    feat = pd.DataFrame(rows)
    feat.to_parquet(cache, index=False)
    print(f"[{name}] feature extraction done in {time.time()-t0:.1f}s, shape: {feat.shape}", flush=True)
    return feat

def write_splits(ds, md, out_dir):
    sp = md[["group"]].reset_index().rename(columns={"index": "row_index"})
    sp["dataset"] = ds
    sp["test_fold"] = sp["group"]
    p = os.path.join(out_dir, "splits.parquet")
    sp.to_parquet(p, index=False)
    print(f"[{ds}] wrote {p} ({len(sp)} rows)", flush=True)

# 1. CREMA-D
print("=== Preparing CREMA-D ===", flush=True)
crema_dir = os.path.join(mb.RESULTS_DIR, "cremad")
os.makedirs(crema_dir, exist_ok=True)
df_crema = load_cremad(bucketed=False)
df_crema.to_parquet(os.path.join(crema_dir, "metadata.parquet"), index=False)
feat_crema = extract_features_parallel("cremad", df_crema, n_jobs=5)
feat_crema.to_parquet(os.path.join(crema_dir, "features.parquet"), index=False)
write_splits("cremad", df_crema, crema_dir)

# 2. CirCor
print("=== Preparing CirCor ===", flush=True)
circor_dir = os.path.join(mb.RESULTS_DIR, "circor")
os.makedirs(circor_dir, exist_ok=True)
df_circor, unk_pts, unk_clips = load_circor()
df_circor.to_parquet(os.path.join(circor_dir, "metadata.parquet"), index=False)
feat_circor = extract_features_parallel("circor", df_circor, n_jobs=5)
feat_circor.to_parquet(os.path.join(circor_dir, "features.parquet"), index=False)
write_splits("circor", df_circor, circor_dir)

print("=== Feature extraction complete for both corpora! ===", flush=True)
