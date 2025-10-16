# Ref: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html#combine-tensor-parallel-with-fully-sharded-data-parallel-together
import torch
from torch.distributed.tensor import DTensor
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
import torch.nn.functional as F

# FSDP + EP: MeshShape(4, 2)
mesh_2d = init_device_mesh("cuda", (4, 2), mesh_dim_names=("fsdp", "ep"))
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")


class MyEpLinear(nn.Linear):
    def forward(self, input):
        assert isinstance(self.weight, DTensor)
        print(f"[rank {rank}][Fwd] MyEpLinear weight: {self.weight}")
        weight = self.weight.to_local()  # Shard(dim=0)
        return F.linear(input, weight, self.bias)

class MyLinear(nn.Linear):
    def forward(self, input):
        assert isinstance(self.weight, DTensor)
        print(f"[rank {rank}][Fwd] MyLinear weight: {self.weight}")
        weight = self.weight.to_local()  # Replicate()
        return F.linear(input, weight, self.bias)

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = MyEpLinear(2, 8, bias=False)
        self.relu = nn.ReLU()
        self.net2 = MyLinear(4, 3, bias=False)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x

model = ToyModel()
model.to(device)
# print(f"[rank {rank}] before fully_shard model: {model}")
# print(f"[rank {rank}] before fully_shard model.net1.weight: {model.net1.weight}")

ep_mesh = mesh_2d["ep"]
fsdp_mesh = mesh_2d["fsdp"]

# 处理MoE参数(用 net1.weight 表示)，在 ep_mesh 上进行sharding
module1 = model.net1
for name1, param1 in module1.named_parameters(recurse=False):
    dist_param1 = nn.Parameter(distribute_tensor(param1, device_mesh=ep_mesh, placements=[Shard(0)]))
    module1.register_parameter(name1, dist_param1)
    # print(f"[rank {rank}][MoE] {name1} raw param: {param1},\n dist param: {dist_param1}")
    # now dist_param1 is DTensor with ep_mesh and placements=(Shard(dim=0),)
    #  dist param: DTensor(local_tensor=tensor([[-0.1230,  0.3088,  0.1842,  0.0310],
    #         [ 0.1005, -0.1026, -0.4433, -0.1732],
    #         [-0.3405,  0.1323,  0.4023,  0.3453],
    #         [-0.3115,  0.0284,  0.4527,  0.0554]], device='cuda:5'), device_mesh=DeviceMesh('cuda', [4, 5], mesh_dim_names=('ep',)), placements=(Shard(dim=0),))


##################################################################
# distribute2_tensor = distribute_tensor(model.net1.weight, device_mesh=fsdp_mesh, placements=[Shard(0)])
# ValueError: Cannot distribute a DTensor with device mesh DeviceMesh('cuda', [6, 7], mesh_dim_names=('ep',)) to a different device mesh DeviceMesh('cuda', [0, 2, 4, 6], mesh_dim_names=('fsdp',))
##################################################################

# 处理非MoE参数 (用 net2.weight 表示)，在 ep_mesh 上进行 replicate
module2 = model.net2
for name2, param2 in module2.named_parameters(recurse=False):
    dist_param2 = nn.Parameter(distribute_tensor(param2, device_mesh=ep_mesh, placements=[Replicate()]))
    module2.register_parameter(name2, dist_param2)
    # print(f"[rank {rank}][Non-MoE] {name2} raw param: {param2},\n dist param: {dist_param2}")
    # now dist_param2 is DTensor with ep_mesh and placements=(Replicate(),)
    # dist param: DTensor(local_tensor=tensor([[ 0.3183,  0.1901,  0.3320, -0.2140, -0.2410,  0.3289, -0.2777,  0.2281],
    #     [ 0.1805, -0.1688, -0.3374, -0.1553,  0.0519, -0.1474,  0.1524, -0.0589],
    #     [ 0.1560, -0.1998, -0.1996, -0.3189, -0.2846, -0.1965, -0.0708,  0.0765],
    #     [ 0.2816, -0.2186,  0.1313,  0.1901, -0.2588, -0.1393,  0.3088,  0.2919],
    #     [ 0.1996,  0.1892,  0.1189, -0.1393,  0.0617,  0.1334,  0.3219,  0.1917],
    #     [-0.0677, -0.0111,  0.1356, -0.3216,  0.2357, -0.1655,  0.2474,  0.2672],
    #     [ 0.1562, -0.2295,  0.1073,  0.1657,  0.3436, -0.0150,  0.1547,  0.3528],
    #     [-0.2675,  0.0052, -0.3036,  0.0027, -0.3214, -0.1046, -0.2485, -0.1646]],
    #    device='cuda:2'), device_mesh=DeviceMesh('cuda', [2, 3], mesh_dim_names=('ep',)), placements=(Replicate(),))


##################################################################
# distribute2_tensor = distribute_tensor(model.net2.weight, device_mesh=fsdp_mesh, placements=[Shard(0)])
# ValueError: Cannot distribute a DTensor with device mesh DeviceMesh('cuda', [6, 7], mesh_dim_names=('ep',)) to a different device mesh DeviceMesh('cuda', [0, 2, 4, 6], mesh_dim_names=('fsdp',)).
# print(f"[rank {rank}] distribute2_tensor: {distribute2_tensor}")
##################################################################

model = fully_shard(
    model, mesh=fsdp_mesh
)

# print(f"[rank {rank}][MoE][After fully_shard] model.net1.weight param: {model.net1.weight}")
# now weight is DTensor with 2d_mesh and placements=(_StridedShard(dim=0, sf=2), Shard(dim=0))
""" Output:
[rank 0][MoE][After fully_shard] weight param: DTensor(local_tensor=tensor([[ 0.3154,  0.3833,  0.1895, -0.3052]], device='cuda:0'), device_mesh=DeviceMesh('cuda', [[0, 1], [2, 3], [4, 5], [6, 7]], mesh_dim_names=('fsdp', 'ep')), placements=(_StridedShard(dim=0, sf=2), Shard(dim=0)))
"""
from torch.distributed.tensor.placement_types import _StridedShard
# More about _StridedShard: https://dev-discuss.pytorch.org/t/dtensor-status-design-and-looking-forward/2749#p-4726-shard-placement-order-26

# print(f"[rank {rank}][Non-MoE][After fully_shard] model.net2.weight param: {model.net2.weight}")
# now weight is DTensor with 2d_mesh and placements=(Shard(dim=0), Replicate())
"""Output:
[rank 7][Non-MoE][After fully_shard] model.net2.weight param: DTensor(local_tensor=tensor([[-0.0844, -0.0397,  0.3031, -0.2915, -0.1712,  0.0448,  0.2613,  0.1839],
        [ 0.3296, -0.3431,  0.1960, -0.0560, -0.2104, -0.0991, -0.1308,  0.2763]],
       device='cuda:7'), device_mesh=DeviceMesh('cuda', [[0, 1], [2, 3], [4, 5], [6, 7]], mesh_dim_names=('fsdp', 'ep')), placements=(Shard(dim=0), Replicate()))
"""

# print(f"[rank {rank}] after fully_shard model: {model}")
# print(f"[rank {rank}] after fully_shard model.net1.weight: {model.net1.weight}")

# Forward pass
output = model(torch.randn(10, 2))
target = torch.randn(10, 3).to(device)
loss = F.mse_loss(output, target)
# Backward pass
loss.backward()

# Grad reduce in fsdp_mesh is done automatically by FSDP
# Grad reduce in ep_mesh need to be done manually
# 在 mesh(fsdp, ep)时，非MOE参数DTensor的placements为 (Shard(0), Replicate())。
# 其中 fsdp维度 torch fsdp会自动对梯度做ReduceScatter(op.MEAN)，而 ep维度在这里手动做Reduce(op.MEAN) 

for name, param in model.named_parameters(recurse=True):
    if not param.requires_grad:
        continue
    if rank == 0:
        print(f"[rank {rank}][Grad] {name} grad: {param.grad}")
    
    if 'net1' in name:  # MoE参数
        param.grad.div_(ep_mesh.size())
    elif 'net2' in name:  # Non-MoE参数
        assert isinstance(param.grad, DTensor)
        grad = param.grad.to_local()
        dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=ep_mesh.get_group(0))



dist.destroy_process_group()
