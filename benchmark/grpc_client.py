from . import benchmark_pb2, benchmark_pb2_grpc

import pickle
import torch
import grpc
import os
import threading
import time
from functools import partial
from statistics import stdev

from concurrent import futures


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
    NUM_RPC,
)

MAX_MESSAGE_LENGTH = 10000 * 10000 * 10


def get_all_results(futs, cuda):
    cpu_tensors = [pickle.loads(f.result().data) for f in futs]
    if cuda:
        cuda_tensors = [t.cuda(0) for t in cpu_tensors]
        return cuda_tensors
    return cpu_tensors


class Client:
    def __init__(self, server_address):
        self.stubs = []
        for _ in range(NUM_RPC):
            channel = grpc.insecure_channel(
                server_address,
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ]
            )
            self.stubs.append(benchmark_pb2_grpc.GRPCBenchmarkStub(channel))



    def embedding_lookup(self, *, name=None, tensor=None, cuda=False, out_file=None):
        futs = []
        assert name == "embedding"
        data = pickle.dumps((name, tensor, cuda))
        request = benchmark_pb2.Request(data=data)
        futs.append(self.stubs[0].meta_run.future(request))
        cpu_tensors = get_all_results(futs, cuda)
        # TODO - need to move back to device.
        return cpu_tensors

    def create_embedding(self, *, name=None, tensor=None, cuda=False, out_file=None):
        futs = []
        assert name == "create_embedding"
        data = pickle.dumps((name, tensor, cuda))
        req = benchmark_pb2.Request(data=data)
        futs.append(self.stubs[0].meta_run.future(req))
        cpu_tensors = get_all_results(futs, cuda)
        return cpu_tensors

    def create_dlrm_embedding(self, *, name=None, emb=None, cuda=False):
       print("Creating embedding")
       assert name == "create_dlrm_embedding"
       print("About to pickle an ENORMOUS embedding bag")
       print(emb)
       data = pickle.dumps((name, emb, cuda))
       print("Done pickling")
       req = benchmark_pb2.Request(data=data)
       futs = []
       futs.append(self.stubs[0].meta_run.future(req))
       print("Sent request, waiting")
       cpu_tensors = get_all_results(futs, cuda)
       print(f"Done with request, got {cpu_tensors}")
       return cpu_tensors
    
    def dlrm_embedding_lookup_async(self, *, name=None, k, sparse_index_group_batch, sparse_offset_group_batch, per_sample_weights, cuda):
        assert name == "dlrm_embedding_lookup_async"
        data = pickle.dumps((name, k, sparse_index_group_batch, sparse_offset_group_batch, per_sample_weights, cuda))
        req = benchmark_pb2.Request(data=data)
        futs = []
        futs.append(self.stubs[0].meta_run.future(req))
        cpu_tensors = get_all_results(futs, cuda)
        return cpu_tensors
    
    def dlrm_embedding_send_grads(self, *, name=None, grad_tensors, cuda):
        assert name == "dlrm_embedding_send_grads"
        data = pickle.dumps((name, grad_tensors, cuda))
        # print(" --- dumped data")
        req = benchmark_pb2.Request(data=data)
        # print(" -- created req")
        futs = []
        futs.append(self.stubs[0].meta_run.future(req))
        # print(" --- sending req ---")
        cpu_tensors = get_all_results(futs, cuda)
        return cpu_tensors 


    def measure(self, *, name=None, tensor=None, cuda=False, out_file=None):
        # warmup
        futs = []
        for i in range(NUM_RPC):
            data = pickle.dumps((name, tensor, cuda))
            request = benchmark_pb2.Request(data=data)
            futs.append(self.stubs[i].meta_run.future(request))

        get_all_results(futs, cuda)

        # warmup done
        timestamps = {}

        states = {
            "lock": threading.Lock(),
            "future": futures.Future(),
            "pending": NUM_RPC
        }
        def mark_complete_cpu(index, cuda, fut):
            tensor = pickle.loads(fut.result().data)
            if cuda:
                tensor.cuda(0)
            timestamps[index]["tok"] = stamp_time(cuda)

            with states["lock"]:
                states["pending"] -= 1
                if states["pending"] == 0:
                    states["future"].set_result(0)

        start = time.time()
        futs = []
        for index in range(NUM_RPC):
            timestamps[index] = {}
            timestamps[index]["tik"] = stamp_time(cuda)

            data = pickle.dumps((name, tensor, cuda))
            request = benchmark_pb2.Request(data=data)
            fut = self.stubs[index].meta_run.future(request)
            futs.append(fut)

            fut.add_done_callback(partial(mark_complete_cpu, index, cuda))

        states["future"].result()

        delays = []
        for index in range(len(timestamps)):
            delays.append(compute_delay(timestamps[index], cuda))

        end = time.time()

        mean = sum(delays)/len(delays)
        stdv = stdev(delays)
        total = end - start
        name = f"{name}_{'cuda' if cuda else 'cpu'}"
        print(f"{name}: mean = {mean}, stdev = {stdv}, total = {total}", flush=True)
        if out_file:
            out_file.write(f"{name}, {mean}, {stdv}, {total}\n")
        return mean, stdv, total

    def terminate(self):
        self.stubs[0].terminate(benchmark_pb2.EmptyMessage())

def run(addr="localhost", port="29500"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    assert torch.cuda.device_count() == 1

    client = Client(f"{addr}:{port}")

    for size in [100]:
    #for size in [100, 1000]:
        print(f"======= size = {size} =====")
        f = open(f"logs/single_grpc_{size}.log", "w")

        tensor = torch.ones(size, size)
        embedding_idx = torch.tensor([1])
        client.measure(
                name="embedding",
                tensor=embedding_idx,
                cuda=False,
                )

        f.close()

    print("Sending term signal")
    client.terminate()
