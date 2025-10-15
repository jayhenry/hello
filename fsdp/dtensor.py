# to run this file (i.e. dtensor_example.py):
# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
import torch.distributed as dist

# Create a mesh topology with the available devices:
# 1. We can directly create the mesh using elastic launcher, (recommended)
# 2. If using mp.spawn, one need to initialize the world process_group first and set device
#   i.e. torch.distributed.init_process_group(backend="nccl", world_size=world_size)

mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))
big_tensor = torch.randn(2, 8)
# A. big_tensor 各个rank不相同
print(f"[rank {dist.get_rank()}] big_tensor.T: {big_tensor.T}")
# Shard this tensor over the mesh by sharding `big_tensor`'s 1th dimension over the 0th dimension of `mesh`.
# B. 在distribute_tensor时，以 rank0 的big_tensor为准，进行切分并分发到其他rank。其他rank的big_tensor都被忽略
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=1)])

print(f"[rank {dist.get_rank()}] big_tensor shape: {big_tensor.shape}, type: {type(big_tensor)}")
print(f"[rank {dist.get_rank()}] my_dtensor.device_mesh: {my_dtensor.device_mesh}")
print(f"[rank {dist.get_rank()}] my_dtensor.placements: {my_dtensor.placements}")
print(f"[rank {dist.get_rank()}] my_dtensor shape: {my_dtensor.shape}, type: {type(my_dtensor)}")
print(f"[rank {dist.get_rank()}] my_dtensor.to_local() shape: {my_dtensor.to_local().shape}, type: {type(my_dtensor.to_local())}")
# C. 各个rank的 my_dtensor.to_local() 是 rank0的big_tensor的切片
print(f"[rank {dist.get_rank()}] my_dtensor.to_local().T: {my_dtensor.to_local().T}")
