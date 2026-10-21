#!/usr/bin/env python3
"""Download DCASE 2020 Task 2 Development Dataset from Zenodo in parallel, verify MD5, and unzip."""

import concurrent.futures
import hashlib
import os
import subprocess
import sys
import time

FILES = [
    {
        "name": "dev_data_valve.zip",
        "size": 1013925627,
        "md5": "34d2672f55bb041589ef79c10dc89934",
        "url": "https://zenodo.org/api/records/3678171/files/dev_data_valve.zip/content"
    },
    {
        "name": "dev_data_slider.zip",
        "size": 1002998966,
        "md5": "da24a757719f0d94d5aa2d646bbfdc86",
        "url": "https://zenodo.org/api/records/3678171/files/dev_data_slider.zip/content"
    },
    {
        "name": "dev_data_pump.zip",
        "size": 1031279015,
        "md5": "90e7091ef722b7238a7f1009365779cd",
        "url": "https://zenodo.org/api/records/3678171/files/dev_data_pump.zip/content"
    },
    {
        "name": "dev_data_fan.zip",
        "size": 1354774772,
        "md5": "649bdfc06263ae7a838963f43b6641e6",
        "url": "https://zenodo.org/api/records/3678171/files/dev_data_fan.zip/content"
    },
    {
        "name": "dev_data_ToyCar.zip",
        "size": 1816443231,
        "md5": "4dec75ca8d9f666aa9e4c1894a740501",
        "url": "https://zenodo.org/api/records/3678171/files/dev_data_ToyCar.zip/content"
    },
    {
        "name": "dev_data_ToyConveyor.zip",
        "size": 1868564070,
        "md5": "03b6aa1bfd09a39d53af0ba39f71fa91",
        "url": "https://zenodo.org/api/records/3678171/files/dev_data_ToyConveyor.zip/content"
    },
]

DEST_DIR = os.path.expanduser("~/projects/maad-bench-r3/data/dcase2020t2")


def check_md5(filepath, expected_md5):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest() == expected_md5


def download_and_verify(item):
    name = item["name"]
    dest_path = os.path.join(DEST_DIR, name)
    expected_md5 = item["md5"]

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == item["size"]:
        print(f"[{name}] File exists with correct size. Checking MD5...", flush=True)
        if check_md5(dest_path, expected_md5):
            print(f"[{name}] MD5 matches ({expected_md5}). Skipping download.", flush=True)
            return True
        else:
            print(f"[{name}] MD5 mismatch. Re-downloading...", flush=True)

    cmd = [
        "curl", "-C", "-", "-L", "-s",
        "--retry", "10",
        "--retry-delay", "5",
        "-o", dest_path,
        item["url"]
    ]
    print(f"[{name}] Starting download...", flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[{name}] Download failed with code {ret.returncode}", flush=True)
        return False
    dt = time.time() - t0
    print(f"[{name}] Downloaded in {dt:.1f}s. Verifying MD5...", flush=True)
    if not check_md5(dest_path, expected_md5):
        print(f"[{name}] FATAL: MD5 mismatch after download!", flush=True)
        return False
    print(f"[{name}] MD5 verified successfully ({expected_md5}).", flush=True)
    return True


def unzip_file(item):
    name = item["name"]
    dest_path = os.path.join(DEST_DIR, name)
    print(f"[{name}] Unzipping...", flush=True)
    t0 = time.time()
    ret = subprocess.run(["unzip", "-o", "-q", dest_path, "-d", DEST_DIR])
    if ret.returncode != 0:
        print(f"[{name}] Unzip failed with code {ret.returncode}", flush=True)
        return False
    print(f"[{name}] Unzipped in {time.time() - t0:.1f}s", flush=True)
    return True


def process_item(item):
    if download_and_verify(item):
        return unzip_file(item)
    return False


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Starting parallel download/extract for {len(FILES)} files in {DEST_DIR}...", flush=True)
    start_total = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(process_item, item): item["name"] for item in FILES}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                res = future.result()
                print(f"[{name}] Done: success={res}", flush=True)
                if not res:
                    print(f"[{name}] FAILED!", flush=True)
            except Exception as e:
                print(f"[{name}] Exception: {e}", flush=True)

    print(f"All downloads and unzips finished in {time.time() - start_total:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
