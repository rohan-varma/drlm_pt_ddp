try:
    import benchmark_pb2, benchmark_pb2_grpc
except:
    from . import benchmark_pb2, benchmark_pb2_grpc

import pickle
from concurrent import futures
import torch
from torch import Tensor
import torch.nn as nn
import grpc
from threading import Lock
import os
import numpy as np

from .common import (
    identity,
    identity_script,
    heavy,
    heavy_script,
    identity_cuda,
    identity_script_cuda,
    heavy_cuda,
    heavy_script_cuda,
    stamp_time,
    compute_delay,
    embedding,
)

funcs = {
    "identity" : identity,
    "identity_script" : identity_script,
    "heavy" : heavy,
    "heavy_script" : heavy_script,
    "identity_cuda" : identity_cuda,
    "identity_script_cuda" : identity_script_cuda,
    "heavy_cuda" : heavy_cuda,
    "heavy_script_cuda" : heavy_script_cuda,
    "stamp_time" : stamp_time,
    "compute_delay" : compute_delay,
    "embedding": embedding,
}

MAX_MESSAGE_LENGTH = 10000 * 10000 * 10

class Server(benchmark_pb2_grpc.GRPCBenchmarkServicer):
    def __init__(self, server_address):
        self.server_address = server_address
        self.future = futures.Future()
        self.embedding = None
        self.emb_l = None
        self.lock = Lock()
    
    def create_dlrm_embedding(self, size, n, m):
        pass
        # assert self.emb_l is None
        # n = 2
        # self.emb_l = nn.ModuleList()
        # for _ in range(size):
        #     pass
        #     # EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
        #     # # initialize embeddings
        #     # # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
        #     # W = np.random.uniform(
        #     #     low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
        #     # ).astype(np.float32)
        #     # # approach 1
        #     # EE.weight.data = torch.tensor(
        #     #     W,
        #     #     requires_grad=True,
        #     #     device=0,
        #     # )
        #     # self.emb_l.append(EE)
        
        # return self.emb_l

    def embedding(self, x: Tensor) -> Tensor:
        assert self.embedding is not None
        return self.embedding(x)

    def meta_run(self, request, context):
        with self.lock:
            req_data = pickle.loads(request.data)
        name = req_data[0]
        if name == "dlrm_embedding_lookup_async":
            with self.lock:
                name, k, sparse_index_group_batch, sparse_offset_group_batch, per_sample_weights, cuda = req_data
                if sparse_index_group_batch is not None:
                    sparse_index_group_batch = sparse_index_group_batch.cpu()
                if sparse_offset_group_batch is not None:
                    sparse_offset_group_batch = sparse_offset_group_batch.cpu()
                if per_sample_weights is not None:
                    per_sample_weights = per_sample_weights.cpu()
                E = self.emb_l[k]
                V = E(
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                return benchmark_pb2.Response(data=pickle.dumps(V))
        elif name == "dlrm_embedding_send_grads":
            name, grad_tensors, cuda = req_data
            # Apply gradients and mock step
            i = 0
            for emb, grad_tensor in zip(self.emb_l, grad_tensors):
                # print(f"type of grad {type(self.emb_l[i].weight.grad)} vs {type(grad_tensor)}")
                # grad_tensor = grad_tensor.to(self.emb_l[i].weight.dtype)
                # grad_tensor = grad_tensor.to(0)
                grad_tensor = torch.ones_like(self.emb_l[i].weight)
                self.emb_l[i].weight.grad = grad_tensor
                i += 1
            # Apply step
            for emb in self.emb_l:
                with torch.no_grad():
                    if emb.weight.grad is not None:
                        emb.weight -= 0.001 * emb.weight.grad
            
            return benchmark_pb2.Response(data=pickle.dumps(torch.tensor([1])))
        else:
            # try:
            #     name, tensor, cuda = pickle.loads(request.data)
            # except:
            #     name, size, n, m, cuda = pickle.loads(request.data)
            if name == "create_embedding":
                name, tensor, cuda = pickle.loads(request.data)
                self.embedding = torch.nn.Embedding(10, 10)
                return benchmark_pb2.Response(data=pickle.dumps(identity(tensor)))
            elif name == "create_dlrm_embedding":
                name, size, n, m, cuda = pickle.loads(request.data)
                assert self.emb_l is None
                # n = 2
                # return benchmark_pb2.Response(data=pickle.dumps(identity(torch.ones(1))))
                self.emb_l = nn.ModuleList()
                for _ in range(size):
                    # pass
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                    # initialize embeddings
                    # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                    W = np.random.uniform(
                        low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                    ).astype(np.float32)
                    # approach 1
                    EE.weight.data = torch.tensor(
                        W,
                        requires_grad=True,
                        device="cpu",
                    )
                    self.emb_l.append(EE)
                # self.create_dlrm_embedding(size=size,n=n,m=m)
                return benchmark_pb2.Response(data=pickle.dumps(identity(torch.ones(1))))
                # if self.emb_l is None:
                #     self.emb_l = torch.nn.ModuleList()
                #     self.emb_l.append(tensor)
                # else:
                #     self.emb_l.append(tensor)
                # self.emb_l = tensor
                #if cuda:
                    #for emb in self.emb_l:
                    #    emb = emb.cuda(0)

                print(f"Server: created embedding {self.emb_l}")
                return benchmark_pb2.Response(data=pickle.dumps(identity(torch.ones(1))))
            elif name == "embedding_lookup":
                name, tensor, cuda = pickle.loads(request.data)
                if cuda:
                    tensor = tensor.cuda(0)
                return benchmark_pb2.Response(data=pickle.dumps(self.embedding(tensor)))
            else:
                name, tensor, cuda = pickle.loads(request.data)
                if cuda:
                    tensor = tensor.cuda(0)
                return benchmark_pb2.Response(data=pickle.dumps(funcs[name](tensor)))

    def terminate(self, request, context):
        print("Server got terminate")
        self.future.set_result(0)
        return benchmark_pb2.EmptyMessage()

    def run(self):
        server = grpc.server(
            futures.ThreadPoolExecutor(),
            options = [
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
            ]
        )

        benchmark_pb2_grpc.add_GRPCBenchmarkServicer_to_server(self, server)

        server.add_insecure_port(self.server_address)
        server.start()
        print("Servre blocking on result...")
        self.future.result()
        print("Server unblocked.")

def run(expected_dev_count, addr="localhost", port="29500"):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # assert torch.cuda.device_count() == 1
    # assert torch.cuda.device_count() == expected_dev_count

    server = Server(f"{addr}:{port}")
    server.run()
