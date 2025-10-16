# torchrun --nproc-per-node=4 dtensor_partial_example.py
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
import torch.distributed as dist

dist.init_process_group(backend="nccl")

world_size = dist.get_world_size()
rank = dist.get_rank()
torch.cuda.set_device(rank)
print(f"Running example on {rank=} in a world with {world_size=}")

device_mesh = init_device_mesh("cuda", (world_size,))


def shard_in_shard_out():
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    
    a = distribute_tensor(a, device_mesh=device_mesh, placements=[Shard(0)])  # row-wise sharding
    b = distribute_tensor(b, device_mesh=device_mesh, placements=[Shard(0)])  # row-wise sharding
    
    print(f"[Rank {rank}][Input] a-> {a}, shape: {a.shape}")
    print(f"[Rank {rank}][Input] b-> {b}, shape: {b.shape}")
    
    c = a + b
    print(f"[Rank {rank}][Output] c=a+b placements is (Shard(0),) same as a and b, with data -> {c}, shape: {c.shape}")


def shard_in_partial_out():
    a = torch.randn(2, 8)
    b = torch.randn(8, 2)
    
    a = distribute_tensor(a, device_mesh=device_mesh, placements=[Shard(1)])  # column-wise sharding
    b = distribute_tensor(b, device_mesh=device_mesh, placements=[Shard(0)])  # row-wise sharding
    
    print(f"[Rank {rank}][Input] a-> {a}, shape: {a.shape}")
    print(f"[Rank {rank}][Input] b-> {b}, shape: {b.shape}")
    
    c = torch.matmul(a, b)
    print(f"[Rank {rank}][Output] c=mm(a, b) placements is (Partial(sum),) with data -> {c}, shape: {c.shape}")


def partial_and_shard_in_shard_out():
    a = torch.randn(2, 8)
    b = torch.randn(8, 2)
    
    a = distribute_tensor(a, device_mesh=device_mesh, placements=[Shard(1)])  # column-wise sharding
    b = distribute_tensor(b, device_mesh=device_mesh, placements=[Shard(0)])  # row-wise sharding
    
    c = torch.matmul(a, b)
    print(f"[Rank {rank}][Input] c=mm(a, b) placements is (Partial(sum),) with data -> {c}, shape: {c.shape}")
    
    d = torch.randn(2, 2)
    d = distribute_tensor(d, device_mesh=device_mesh, placements=[Shard(0)])  # row-wise sharding
    print(f"[Rank {rank}][Input] d-> {d}, shape: {d.shape}")
    
    e = c + d
    print(f"[Rank {rank}][Output] e=c+d placements is [Shard(0)] same as d -> {e}, shape: {e.shape}")


if __name__ == "__main__":
    # shard_in_shard_out()
    # shard_in_partial_out()
    partial_and_shard_in_shard_out()
    dist.destroy_process_group()



