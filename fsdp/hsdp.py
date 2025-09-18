import torch
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard as FSDP


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
model = FSDP(
    ToyModel(), mesh=mesh_2d
)

print(model)

model(torch.randn(10, 10))
