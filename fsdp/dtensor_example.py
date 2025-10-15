# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
import torch.distributed as dist

dist.init_process_group(backend="nccl")

world_size = dist.get_world_size()
rank = dist.get_rank()
torch.cuda.set_device(rank)
print(f"Running example on {rank=} in a world with {world_size=}")

# if we want to do row-wise sharding
rowwise_placement=[Shard(0)]
# if we want to do col-wise sharding
colwise_placement=[Shard(1)]

def distribute_tensor_example():
    print("======================================= distribute_tensor example =======================================")
    # construct a device mesh with available devices (multi-host or single host)
    device_mesh = init_device_mesh("cuda", (world_size,))
    # big_tensor = torch.randn(888, 12)
    big_tensor = torch.randn(4, 4)
    print(f"[Rank {rank}] big_tensor: {big_tensor}")
    # distributed tensor returned will be sharded across the dimension specified in placements
    # 以rank0 的 big_tensor为准，进行切分并分发到其他rank。其他rank的big_tensor都被忽略
    rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)
    print(f"[Rank {rank}] rowwise_tensor shape: {rowwise_tensor.shape}, type: {type(rowwise_tensor)}, to_local shape: {rowwise_tensor.to_local().shape}, type: {type(rowwise_tensor.to_local())}")
    
    print(f"[Rank {rank}] rowwise_tensor: {rowwise_tensor}")
    
    # if we want to do replication across a certain device list
    replica_placement = [Replicate()]
    # distributed tensor will be replicated to all four GPUs.
    # 以 rank0 的 big_tensor为准，进行复制并分发到其他rank。其他rank的big_tensor都被忽略
    replica_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=replica_placement)
    print(f"[Rank {rank}] replica_tensor shape: {replica_tensor.shape}, type: {type(replica_tensor)}, to_local shape: {replica_tensor.to_local().shape}, type: {type(replica_tensor.to_local())}")
    print(f"[Rank {rank}] replica_tensor: {replica_tensor}")
    
    # if we want to distributed a tensor with both replication and sharding
    device_mesh = init_device_mesh("cuda", (2, 2))
    # replicate across the first dimension of device mesh, then sharding on the second dimension of device mesh
    spec=[Replicate(), Shard(0)]
    partial_replica = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=spec)
    print(f"[Rank {rank}] partial_replica shape: {partial_replica.shape}, type: {type(partial_replica)}, to_local shape: {partial_replica.to_local().shape}, type: {type(partial_replica.to_local())}")
    print(f"[Rank {rank}] partial_replica: {partial_replica}")


def dtensor_from_local_example():
    print("======================================= DTensor.from_local example =======================================")
    device_mesh = init_device_mesh("cuda", (2, 2))
    # create a DistributedTensor that shards on dim 0, from a local torch.Tensor
    local_tensor = torch.randn((2, 2), requires_grad=True)
    print(f"[Rank {rank}] local_tensor shape: {local_tensor.shape}, data: {local_tensor}")
    # todo: device_mesh 形状是 [2,2], 而 rowwise_placement 是 [Shard(0)], 为什么能正常运行？
    #  是默认将 placement 扩展到 device_mesh 的形状吗？扩展为 [Shard(0), Replicate()] ? 从结果的形状为 (4, 2) 来看，确实是按这样扩展的
    rowwise_tensor = DTensor.from_local(local_tensor, device_mesh, rowwise_placement, run_check=True)
    
    # global_tensor = torch.randn((16, 8), requires_grad=True)
    # print(f"[Rank {rank}] global_tensor shape: {global_tensor.shape}, data: {global_tensor}")
    # # 报错：  ValueError: `placements` must have the same length as `device_mesh.ndim`! Found placements length: 1, and device_mesh.ndim: 2.
    # rowwise_tensor = distribute_tensor(global_tensor, device_mesh=device_mesh, placements=rowwise_placement)  
    
    print(f"[Rank {rank}] rowwise_tensor-> placements: {rowwise_tensor.placements}, shape: {rowwise_tensor.shape}, type: {type(rowwise_tensor)}, to_local shape: {rowwise_tensor.to_local().shape}, type: {type(rowwise_tensor.to_local())}")
    print(f"[Rank {rank}] rowwise_tensor-> to_local(): {rowwise_tensor.to_local()}")

    # 按from_local的API说明， Replicate() 维度的rank0数据会被广播到其他rank
    # 但是，从结果来看，Replicate() 维度的rank0数据没有被广播到其他rank
    # 这是bug？
    full_rowwise_tensor = rowwise_tensor.full_tensor()
    print(f"[Rank {rank}] full_rowwise_tensor-> shape: {full_rowwise_tensor.shape}, type: {type(full_rowwise_tensor)}")
    print(f"[Rank {rank}] full_rowwise_tensor: {full_rowwise_tensor}")

    exit()
    
    
def dtensor_redistribute_example():
    print("======================================= DTensor.redistribute example =======================================")
    device_mesh = init_device_mesh("cuda", (2, 2))
    spec=[Shard(0), Replicate()]
    big_tensor = torch.randn((2, 2))
    print(f"[Rank {rank}] big_tensor-> shape: {big_tensor.shape}, data: {big_tensor}")
    # 以 rank0 的 big_tensor为准，进行切分并分发到其他rank。其他rank的big_tensor都被忽略
    rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=spec)
    print(f"[Rank {rank}] rowwise_tensor-> {rowwise_tensor}, shape: {rowwise_tensor.shape}")

    # reshard the current row-wise tensor to a colwise tensor or replicate tensor
    colwise_tensor = rowwise_tensor.redistribute(device_mesh, [Shard(1), Replicate()])
    print(f"[Rank {rank}] colwise_tensor -> {colwise_tensor}, shape: {colwise_tensor.shape}")

    replica_tensor = rowwise_tensor.redistribute(device_mesh, [Replicate(), Replicate()])
    print(f"[Rank {rank}] replica_tensor -> {replica_tensor}, shape: {replica_tensor.shape}")
    

if __name__ == "__main__":
    # distribute_tensor_example()
    # dtensor_from_local_example()
    dtensor_redistribute_example()

    dist.destroy_process_group()