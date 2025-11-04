# ref: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
import os
import sys
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

CHECKPOINT_DIR = "checkpoint"


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        if dist.get_rank() == 0:
            print(f"[RANK {dist.get_rank()}] model_state_dict: {model_state_dict}")
            print(f"[RANK {dist.get_rank()}] optimizer_state_dict: {optimizer_state_dict}")

        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        if dist.get_rank() == 0:
            print(f"[RANK {dist.get_rank()}] loading state_dict['model']: {state_dict['model']}")
            print(f"[RANK {dist.get_rank()}] loading state_dict['optim']: {state_dict['optim']}")
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16, bias=False)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = fully_shard(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16, device="cuda")).sum().backward()
    optimizer.step()

    state_dict = { "app": AppState(model, optimizer) }
    dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)

    cleanup()


def run_fsdp_checkpoint_load_example(rank, world_size):
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = fully_shard(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    """
    Please note that you will have to call model.state_dict() prior to loading and pass it to DCP’s load_state_dict() API. 
    This is fundamentally different from torch.load(), as torch.load() simply requires the path to the checkpoint prior for loading. 
    The reason that we need the state_dict prior to loading is:
    - DCP uses the pre-allocated storage from model state_dict to load from the checkpoint directory. During loading, the state_dict passed in will be updated in place.
    - DCP requires the sharding information from the model prior to loading to support resharding.
    """
    state_dict = { "app": AppState(model, optimizer)}
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=CHECKPOINT_DIR,
    )

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    arg1 = sys.argv[1]
    if arg1 == "save":
        print(f"Running fsdp checkpoint saving example on {world_size} devices.")
        mp.spawn(
            run_fsdp_checkpoint_save_example,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
    elif arg1 == "load":
        print(f"Running fsdp checkpoint loading example on {world_size} devices.")
        mp.spawn(
            run_fsdp_checkpoint_load_example,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
    else:
        print(f"Invalid argument: {arg1}")
        print(f"Usage: python {sys.argv[0]} <save|load>")
        exit(1)

