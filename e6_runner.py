"""Master runner for E6: Pretrained representation arms and Autoencoder anchor.

Orchestrates GPU worker (AST -> PANNs -> AE) and CPU worker (rotations on circor_rec,
dcase_toycar, dcase_toyconveyor, plus CREMA-D 1-fold benchmark & full rotation).
"""
import glob
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import maad_bench as mb

PY = sys.executable


def run_cmd(cmd_list, log_path):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running: {' '.join(cmd_list)}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n=== CMD: {' '.join(cmd_list)} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.flush()
        ret = subprocess.call(cmd_list, stdout=f, stderr=subprocess.STDOUT)
    if ret != 0:
        print(f"ERROR: Command failed with code {ret}: {' '.join(cmd_list)}", flush=True)
    return ret


def check_embedding_exists(ds, backbone):
    """Check if cache/<ds>_<backbone>_*.parquet exists."""
    pat = os.path.join(mb.CACHE_DIR, f"{ds}_{backbone}_*.parquet")
    hits = [p for p in glob.glob(pat) if "features_n" not in p]
    if hits:
        return hits[0]
    return None


def gpu_worker():
    log = os.path.join(HERE, "logs", "e6_gpu_worker.log")
    print("Starting E6 GPU worker...", flush=True)

    # 1. AST extraction
    cmd_ast = ["gpu-launch", "--no-check", "--no-mutex", "--", PY,
               os.path.join(HERE, "e6_repr.py"), "--stage", "extract_ast", "--dataset", "all"]
    run_cmd(cmd_ast, log)

    # 2. PANNs extraction
    cmd_panns = ["gpu-launch", "--no-check", "--no-mutex", "--", PY,
                 os.path.join(HERE, "e6_repr.py"), "--stage", "extract_panns", "--dataset", "all"]
    run_cmd(cmd_panns, log)

    # 3. AE baseline training for DCASE
    cmd_ae = ["gpu-launch", "--no-check", "--no-mutex", "--", PY,
              os.path.join(HERE, "e6_repr.py"), "--stage", "ae", "--dataset", "all"]
    run_cmd(cmd_ae, log)

    print("E6 GPU worker finished all stages!", flush=True)


def cpu_worker():
    log = os.path.join(HERE, "logs", "e6_cpu_repr_worker.log")
    print("Starting E6 CPU representation worker...", flush=True)

    # Order of rotations:
    # 1. AST rotations
    for ds in ["circor_rec", "dcase_toycar", "dcase_toyconveyor"]:
        print(f"Waiting for {ds} AST embedding...", flush=True)
        while not check_embedding_exists(ds, "ast"):
            time.sleep(10)
        print(f"{ds} AST embedding found! Starting rotation (jobs=4)...", flush=True)
        cmd_rot = [PY, os.path.join(HERE, "e6_repr.py"),
                   "--stage", "rotate", "--dataset", ds, "--backbone", "ast", "--jobs", "4"]
        run_cmd(cmd_rot, log)

    # 2. PANNs rotations
    for ds in ["circor_rec", "dcase_toycar", "dcase_toyconveyor"]:
        print(f"Waiting for {ds} PANNs embedding...", flush=True)
        while not check_embedding_exists(ds, "panns"):
            time.sleep(10)
        print(f"{ds} PANNs embedding found! Starting rotation (jobs=4)...", flush=True)
        cmd_rot = [PY, os.path.join(HERE, "e6_repr.py"),
                   "--stage", "rotate", "--dataset", ds, "--backbone", "panns", "--jobs", "4"]
        run_cmd(cmd_rot, log)

    # 3. CREMA-D 1-fold benchmark
    print("Waiting for cremad AST & PANNs embeddings...", flush=True)
    while not (check_embedding_exists("cremad", "ast") and check_embedding_exists("cremad", "panns")):
        time.sleep(10)
    print("cremad embeddings ready! Running 1-fold benchmark...", flush=True)
    cmd_bench = [PY, os.path.join(HERE, "e6_repr.py"),
                 "--stage", "benchmark_cremad", "--jobs", "4"]
    run_cmd(cmd_bench, log)

    # 4. CREMA-D full rotation
    print("Starting full CREMA-D rotations (91 folds, jobs=4)...", flush=True)
    cmd_crem_ast = [PY, os.path.join(HERE, "e6_repr.py"),
                    "--stage", "rotate_cremad", "--backbone", "ast", "--jobs", "4"]
    run_cmd(cmd_crem_ast, log)

    cmd_crem_panns = [PY, os.path.join(HERE, "e6_repr.py"),
                      "--stage", "rotate_cremad", "--backbone", "panns", "--jobs", "4"]
    run_cmd(cmd_crem_panns, log)

    print("E6 CPU representation worker finished all tasks!", flush=True)


def main():
    t_gpu = threading.Thread(target=gpu_worker)
    t_cpu = threading.Thread(target=cpu_worker)

    t_gpu.start()
    t_cpu.start()

    t_gpu.join()
    t_cpu.join()

    print("All tasks completed! Running tabulation...", flush=True)
    log_tab = os.path.join(HERE, "logs", "e6_tabulate.log")
    cmd_tab = [PY, os.path.join(HERE, "e6_repr.py"), "--stage", "tabulate"]
    run_cmd(cmd_tab, log_tab)


if __name__ == "__main__":
    main()
