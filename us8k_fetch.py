"""Fetch UrbanSound8K directly from Zenodo (soundata's downloader is broken in
this version: it references urllib.request without importing it). The archive
extracts to <data>/UrbanSound8K/{metadata,audio}, which maad_bench expects.
"""
import os
import tarfile
import urllib.request

# Derive the data root from this file, the way maad_bench.py does, so every
# checkout downloads to its own visible data directory.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
tgz = os.path.join(DATA, "UrbanSound8K.tar.gz")

if not os.path.exists(tgz):
    print("downloading UrbanSound8K (~6 GB) ...", flush=True)
    urllib.request.urlretrieve(URL, tgz)

csv = os.path.join(DATA, "UrbanSound8K", "metadata", "UrbanSound8K.csv")
if not os.path.exists(csv):
    print("extracting ...", flush=True)
    with tarfile.open(tgz) as tf:
        tf.extractall(DATA)

print("EXISTS_CSV", os.path.exists(csv), csv)
print("DOWNLOAD_DONE")
