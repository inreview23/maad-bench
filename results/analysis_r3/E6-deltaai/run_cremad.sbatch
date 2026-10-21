#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --job-name=maad-cremad
#SBATCH --output=./logs/%x-%j.out
cd .
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
./env/bin/python run_cremad.py $1 91
