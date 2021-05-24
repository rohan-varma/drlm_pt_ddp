from common import identity, identity_script, heavy, heavy_script

from torch.distributed import rpc
import torch

import os

def run(addr="localhost", port="29500"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    assert torch.cuda.device_count() == 1

    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port

    options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads = 256,
            device_maps={"client" : {0 : 0}}
    )
    rpc.init_rpc(
        "server",
        rank=0,
        world_size=2,
        rpc_backend_options=options
    )

    rpc.shutdown()
