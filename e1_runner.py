"""Master runner for E1 GPU and representation rotation tasks.

Launches GPU worker (AST extraction -> PANNs extraction -> AE training) and
CPU worker (AST rotation -> PANNs rotation) concurrently.
"""
import glob
import os
import subprocess
import sys
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


def gpu_worker():
    log = os.path.join(HERE, "logs", "e1_gpu_worker.log")
    print("Starting GPU worker...", flush=True)

    # 1. AST extraction
    cmd_ast = ["gpu-launch", "--no-check", "--no-mutex", "--", PY,
               os.path.join(HERE, "e1_mimii_types.py"), "--stage", "extract_ast", "--dataset", "all"]
    run_cmd(cmd_ast, log)

    # 2. PANNs extraction
    cmd_panns = ["gpu-launch", "--no-check", "--no-mutex", "--", PY,
                 os.path.join(HERE, "e1_mimii_types.py"), "--stage", "extract_panns", "--dataset", "all"]
    run_cmd(cmd_panns, log)

    # 3. AE baseline training
    cmd_ae = ["gpu-launch", "--no-check", "--no-mutex", "--", PY,
              os.path.join(HERE, "e1_mimii_types.py"), "--stage", "ae", "--dataset", "all"]
    run_cmd(cmd_ae, log)
    print("GPU worker finished all stages!", flush=True)


def check_embeddings_exist(backbone):
    for ds in ["mimii_fan", "mimii_slider", "mimii_valve"]:
        hits = glob.glob(os.path.join(mb.CACHE_DIR, f"{ds}_{backbone}_*.parquet"))
        if not hits:
            return False
    return True


def cpu_worker():
    log = os.path.join(HERE, "logs", "e1_cpu_repr_worker.log")
    print("Starting CPU representation worker...", flush=True)

    # Wait for AST embeddings
    print("Waiting for AST embeddings to be extracted...", flush=True)
    while not check_embeddings_exist("ast"):
        time.sleep(15)
    print("AST embeddings found! Starting AST rotations (jobs=4)...", flush=True)
    cmd_ast_rot = [PY, os.path.join(HERE, "e1_mimii_types.py"),
                   "--stage", "rotate_ast", "--dataset", "all", "--jobs", "4"]
    run_cmd(cmd_ast_rot, log)

    # Wait for PANNs embeddings
    print("Waiting for PANNs embeddings to be extracted...", flush=True)
    while not check_embeddings_exist("panns"):
        time.sleep(15)
    print("PANNs embeddings found! Starting PANNs rotations (jobs=4)...", flush=True)
    cmd_panns_rot = [PY, os.path.join(HERE, "e1_mimii_types.py"),
                     "--stage", "rotate_panns", "--dataset", "all", "--jobs", "4"]
    run_cmd(cmd_panns_rot, log)

    print("CPU representation worker finished all rotations!", flush=True)


def main():
    import threading

    t_gpu = threading.Thread(target=gpu_worker)
    t_cpu = threading.Thread(target=cpu_worker)

    t_gpu.start()
    t_cpu.start()

    t_gpu.join()
    t_cpu.join()

    print("All GPU and CPU tasks completed! Running final summary...", flush=True)
    log_sum = os.path.join(HERE, "logs", "e1_summary.log")
    cmd_sum = [PY, os.path.join(HERE, "e1_mimii_types.py"), "--stage", "summary"]
    run_cmd(cmd_sum, log_sum)


if __name__ == "__main__":
    main()
