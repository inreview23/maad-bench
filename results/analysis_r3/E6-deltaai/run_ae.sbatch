#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --job-name=maad-cremad-ae
#SBATCH --output=./logs/%x-%j.out
cd .
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i env/bin/python dtai_ae_shard.py $i 4 > logs/ae_shard$i.log 2>&1 &
done
wait
echo all shards done
