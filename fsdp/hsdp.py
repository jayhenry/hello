import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


# HSDP: MeshShape(2, 4)
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_replicate", "dp_shard"))
model = ToyModel()
rank = dist.get_rank()
# print(f"[rank {rank}] before fully_shard model: {model}")
print(f"[rank {rank}] before fully_shard model.net1.weight: {model.net1.weight}")

model = fully_shard(
    model, mesh=mesh_2d
)

print(f"[rank {rank}] after fully_shard model: {model}")
print(f"[rank {rank}] after fully_shard model.net1.weight: {model.net1.weight}")

model(torch.randn(10, 10))


dist.destroy_process_group()
