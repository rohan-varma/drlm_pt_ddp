#!/bin/bash
export MASTER_PORT=29501
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "using master addr"$MASTER_ADDR
echo "using part "$SLURM_PARTITION
# TODO - minibatch size must be at least the no of local gpu?
echo "running python script"
echo "world size "$SLURM_NTASKS
echo "node name "$SLURMD_NODENAME
echo "rank? "$SLURM_PROCID
echo "embedding size "$EMBEDDING_DIM_SIZE
export NITERS=512
z=$((NITERS*MINI_BATCH_SIZE))
echo $z
python dlrm_s_pytorch.py --use-gpu --print-time --dist-backend nccl --nepochs 1 --data-size ${z} --mini-batch-size ${MINI_BATCH_SIZE} --debug-mode --arch-sparse-feature-size ${EMBEDDING_DIM_SIZE} --arch-embedding-size 28000000-28000000-28000000-28000000-28000000 --arch-mlp-bot 2000-1024-1024-512-256-${EMBEDDING_DIM_SIZE} --arch-mlp-top 4096-4096-1 --node-world-size ${SLURM_NTASKS} --rank ${SLURM_PROCID}
