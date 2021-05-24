import torch.multiprocessing as mp
import pt_rpc_client
import pt_rpc_server
import grpc_client
import grpc_server

def main():
    ctx = mp.get_context('spawn')


    print("#######==== gRPC ===#######")
    server_proc = ctx.Process(target=grpc_server.run)
    server_proc.start()
    import time ; time.sleep(2)
    client_proc = ctx.Process(target=grpc_client.run)
    client_proc.start()
    for p in [client_proc, server_proc]:
        p.join()

if __name__ == "__main__":
    main()
