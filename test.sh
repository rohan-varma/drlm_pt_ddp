export MASTER_PORT=29501
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "using master addr"$MASTER_ADDR
python dlrm_s_pytorch.py --use-gpu --print-time --dist-backend nccl --nepochs 2 --data-size 1024 --mini-batch-size 8 --debug-mode --arch-sparse-feature-size 2
