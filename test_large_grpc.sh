#!/bin/bash
export MASTER_PORT=29501
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 2 | tail -1)
export GRPC_MASTER_PORT=29502
export GRPC_MASTER_ADDR=$(scontrol show hostname ${SLURM_NODE_LIST} | head -n 1)

echo "using master addr"$MASTER_ADDR
echo "using part "$SLURM_PARTITION
echo "GRPC master "$GRPC_MASTER_ADDR
# TODO - minibatch size must be at least the no of local gpu?
echo "running python script"
echo "world size "$SLURM_NTASKS
echo "node name "$SLURMD_NODENAME
echo "rank? "$SLURM_PROCID
python dlrm_s_pytorch.py --use-gpu --print-time --dist-backend nccl --nepochs 2 --data-size 2048 --mini-batch-size ${MINI_BATCH_SIZE} --debug-mode --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 2000-1024-1024-512-256-64 --arch-mlp-top 4096-4096-1 --node-world-size ${SLURM_NTASKS} --rank ${SLURM_PROCID} --grpc True
