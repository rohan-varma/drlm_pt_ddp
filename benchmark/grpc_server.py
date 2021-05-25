try:
    import benchmark_pb2, benchmark_pb2_grpc
except:
    from . import benchmark_pb2, benchmark_pb2_grpc

import pickle
from concurrent import futures
import torch
from torch import Tensor
import grpc
import os

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

    def embedding(self, x: Tensor) -> Tensor:
        assert self.embedding is not None
        return self.embedding(x)

    def meta_run(self, request, context):
        req_data = pickle.loads(request.data)
        name = req_data[0]
        if name == "dlrm_embedding_lookup_async":
            name, k, sparse_index_group_batch, sparse_offset_group_batch, per_sample_weights, cuda = req_data
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
                self.emb_l[i].weight.grad = grad_tensor
                i += 1
            # Apply step
            for emb in self.emb_l:
                with torch.no_grad():
                    emb.weight -= 0.001 * emb.weight.grad
            
            return benchmark_pb2.Response(data=pickle.dumps(torch.tensor([1])))
        else:
            name, tensor, cuda = pickle.loads(request.data)
            if name == "create_embedding":
                self.embedding = torch.nn.Embedding(10, 10)
                return benchmark_pb2.Response(data=pickle.dumps(identity(tensor)))
            elif name == "create_dlrm_embedding":
                self.emb_l = tensor
                #if cuda:
                    #for emb in self.emb_l:
                    #    emb = emb.cuda(0)

                print(f"Server: created embedding {self.emb_l}")
                return benchmark_pb2.Response(data=pickle.dumps(identity(torch.ones(1))))
            elif name == "embedding_lookup":
                if cuda:
                    tensor = tensor.cuda(0)
                return benchmark_pb2.Response(data=pickle.dumps(self.embedding(tensor)))
            else:
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

def run(addr="localhost", port="29500"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    assert torch.cuda.device_count() == 1

    server = Server(f"{addr}:{port}")
    server.run()
