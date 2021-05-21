#!/bin/bash

#export MASTER_PORT=29500
#export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
#export MASTER_ADDR="localhost"
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}


MASTER_ADDR="train-dy-p4d24xlarge-1" MASTER_PORT=29501 python dlrm_s_pytorch.py --use-gpu --print-time --dist-backend nccl --nepochs 2 --data-size 1024 --mini-batch-size 8 --debug-mode --arch-sparse-feature-size 2

exit 0
echo "Running with master addr "$MASTER_ADDR 
python dlrm_s_pytorch.py --use-gpu --print-time --dist-backend nccl --nepochs 2 --data-size 1024 --mini-batch-size 8 --debug-mode --arch-sparse-feature-size 2
