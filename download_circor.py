import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

DEST_DIR = os.path.expanduser("~/projects/maad-bench-r3/data/circor/training_data")
os.makedirs(DEST_DIR, exist_ok=True)

print("Listing keys from S3...", flush=True)
url = "https://physionet-open.s3.amazonaws.com/?prefix=circor-heart-sound/1.0.3/training_data/"
keys = []
marker = ""
while True:
    req_url = url + ("&marker=" + marker if marker else "")
    with urllib.request.urlopen(req_url) as resp:
        tree = ET.fromstring(resp.read())
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    contents = tree.findall("s3:Contents", ns)
    if not contents:
        break
    for c in contents:
        keys.append(c.find("s3:Key", ns).text)
    is_trunc = tree.find("s3:IsTruncated", ns)
    if is_trunc is not None and is_trunc.text == "true":
        marker = keys[-1]
    else:
        break

print(f"Total keys to download: {len(keys)}", flush=True)

def download_one(key):
    fname = os.path.basename(key)
    dest_path = os.path.join(DEST_DIR, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True, 0
    file_url = f"https://physionet-open.s3.amazonaws.com/{key}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(dest_path, "wb") as fh:
                fh.write(data)
            return True, len(data)
        except Exception as e:
            if attempt == 2:
                print(f"Failed {fname}: {e}", file=sys.stderr, flush=True)
                return False, 0
            time.sleep(1)

t0 = time.time()
n_done = 0
total_bytes = 0
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = {executor.submit(download_one, k): k for k in keys}
    for f in as_completed(futures):
        ok, nbytes = f.result()
        if ok:
            n_done += 1
            total_bytes += nbytes
        if n_done % 1000 == 0 or n_done == len(keys):
            print(f"Progress: {n_done}/{len(keys)} files ({total_bytes / (1024*1024):.1f} MB) in {time.time()-t0:.1f}s", flush=True)

print(f"CirCor download complete! {n_done}/{len(keys)} files in {time.time()-t0:.1f}s.", flush=True)
