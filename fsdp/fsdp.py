import os
import sys

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.checkpoint.state_dict import set_model_state_dict, get_model_state_dict, StateDictOptions


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 10)

        layer_num = 8
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(layer_num)])

    def forward(self, x):
        x =  self.net2(self.relu(self.net1(x)))
        for layer in self.layers:
            x = layer(x)
        return x


dist.init_process_group(backend="nccl")

checkpoint_fname = f"checkpoints/model_state_dict.pt"
local_rank = rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
# device = torch.device(f"cuda:{rank}")
device = torch.cuda.current_device()
print(f"Rank {rank}, Device: {device}")

# FSDP: MeshShape(8)
# mesh = init_device_mesh("cuda", (8, ), mesh_dim_names=("dp_shard", ))

############################## build model ######################
with torch.device("meta"):
    model = ToyModel()

mesh = init_device_mesh("cuda", (world_size, 1), mesh_dim_names=("fsdp", "ep"))
fsdp_mesh = mesh["fsdp"]
fsdp_kwargs = {
    "mesh": fsdp_mesh,
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
}

############################## shard model ######################
for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)


############################ init or load model from meta to actual device ######################

def load_model(model: nn.Module, checkpoint_fname: str):
    # mmap=True reduces CPU memory usage
    full_sd = torch.load(
        checkpoint_fname,
        mmap=True,
        weights_only=True,
        map_location='cpu',
    )
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # `assign=True` since we cannot call `copy_` on meta tensor
    model.load_state_dict(sharded_sd, assign=True)


def load_model_dcp(model: nn.Module, checkpoint_fname: str):
    # With broadcast_from_rank0=True, we can load the full state dict only on rank 0 to avoid peaking CPU memory. 
    # DCP will shard tensors and broadcast them to other ranks.
    full_sd = {}
    if rank == 0:
        full_sd = torch.load(
            checkpoint_fname,
            mmap=True,
            weights_only=True,
            map_location='cpu',
        )
    set_model_state_dict(
        model=model,
        model_state_dict=full_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
        ),
    )

load = False
if load and os.path.exists(checkpoint_fname):
    # load_model(model, checkpoint_fname)
    print(f"Loading meta device model from {checkpoint_fname} using dcp")
    load_model_dcp(model, checkpoint_fname)
else:
    print(f"Skip Loading model, put meta to actual device by to_empty()")
    model.to_empty(device=device)
    # maybe initialize model weight here if needed


############################## check model ######################
assert isinstance(model, FSDPModule)
if rank == 0:
    print(model)

for param in model.parameters():
    assert isinstance(param, DTensor)
    assert param.placements == (Shard(0),)
    # inspect sharded parameters with param.to_local()

# sharded parameters are float32
for param in model.parameters():
    assert param.dtype == torch.float32

# unsharded parameters are bfloat16
cur = model.net1
for name, param in cur.named_parameters(recurse=False):
    if rank == 0:
        print(f"before unshard cur_model param: {name} has dtype {param.dtype}, has type {type(param)}")

model.unshard()
for name, param in cur.named_parameters(recurse=False):
    param: torch.nn.Parameter
    if rank == 0:
        print(f"unshard cur_model param: {name} has dtype {param.dtype}, has type {type(param)}")
    assert param.dtype == torch.bfloat16

model.reshard()
for name, param in cur.named_parameters(recurse=False):
    if rank == 0:
        print(f"after reshard cur_model param: {name} has dtype {param.dtype}, has type {type(param)}")

sys.exit()

############################## optimizer ######################
# Note the optimizer is constructed after applying fully_shard. Both model and optimizer state dicts are represented in DTensor.
# optimizer states are in float32
optim = torch.optim.Adam(model.parameters(), lr=1e-2)

epochs = 10

###################### Implicit Prefetching ######################
"""
for i in range(epochs):
    if rank == 0:
        print(f"Epoch {i}, Rank {rank}")
    x = torch.randn(10, 10)
    loss = model(x).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()
"""
####################################################################


###################### Explicit Prefetching ######################
# """
num_to_forward_prefetch = 2
for i, layer in enumerate(model.layers):
    if i >= len(model.layers) - num_to_forward_prefetch:
        break
    layers_to_prefetch = [
        model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
    ]
    layer.set_modules_to_forward_prefetch(layers_to_prefetch)

num_to_backward_prefetch = 2
for i, layer in enumerate(model.layers):
    if i < num_to_backward_prefetch:
        continue
    layers_to_prefetch = [
        model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
    ]
    layer.set_modules_to_backward_prefetch(layers_to_prefetch)

for i in range(epochs):
    if rank == 0:
        print(f"Epoch {i}, Rank {rank}")
    # trigger 1st all-gather earlier
    # this overlaps all-gather with any computation before model(x)
    model.unshard()
    x = torch.randn(10, 10)
    loss = model(x).sum()
    loss.backward()
    # For gradient clipping, torch.nn.utils.clip_grad_norm_ works for DTensor parameters. 
    # Tensor ops will be dispatched correctly inside DTensor to communicate partial tensors across ranks to preserve the single device semantic.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optim.step()
    optim.zero_grad()
# """

####################################################################


###################### Save Checkpoint ######################

def save_model(model: nn.Module, checkpoint_fname: str):
    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        if torch.distributed.get_rank() == 0:
            print(f"All Gather {param_name} 's sharded_param of shape {sharded_param.shape} to get full_param of shape {full_param.shape}, and put it to cpu")
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    if rank == 0:
        print(f"Saving checkpoint to {checkpoint_fname}")
        torch.save(cpu_state_dict, checkpoint_fname)

def save_model_dcp(model: nn.Module, checkpoint_fname: str):
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
    )
    if rank == 0:
        print(f"Saving checkpoint to {checkpoint_fname} using dcp")
        torch.save(model_state_dict, checkpoint_fname)

# save_model(model, checkpoint_fname)
save_model_dcp(model, checkpoint_fname)

####################################################################


dist.destroy_process_group()
