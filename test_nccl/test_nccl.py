import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, world_size):
    """Esegue la comunicazione tra i nodi."""
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    tensor = torch.tensor([rank], dtype=torch.float32).cuda()
    
    if rank == 0:
        # Nodo 0 invia il tensor al Nodo 1
        dist.send(tensor, dst=1)
        print(f"Nodo {rank} ha inviato {tensor.item()} al nodo 1")
    elif rank == 1:
        # Nodo 1 riceve il tensor dal Nodo 0
        dist.recv(tensor, src=0)
        print(f"Nodo {rank} ha ricevuto {tensor.item()} dal nodo 0")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    run(rank, world_size)
