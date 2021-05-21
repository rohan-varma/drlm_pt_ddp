#!/bin/bash

#SBATCH --job-name=ddp_gpu_sbatch

#SBATCH --partition=train

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#NOT SURE if neededl#SewoieBATCH --gpus-per-task=1

#SBATCH --time=1:00:00

srun --label test.sh
