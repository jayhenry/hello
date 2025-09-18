import os
import socket
from datetime import datetime
from functools import partial

import torch
from torch.autograd.profiler import record_function
from torch._C._profiler import _ExperimentalConfig


N = 10
input = torch.randn(10, 4096, 4096, requires_grad=True).to("cuda")
weight = torch.randn(4096, 4096).to("cuda")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # In this example with wait=1, warmup=1, active=3, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    with_stack=True,
    with_modules=True,
    # on_trace_ready=partial(trace_handler, dir="save"),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./my_tb3"),
    # ref: https://github.com/pytorch/kineto/issues/798
    #  https://github.com/pytorch/pytorch/issues/100253
    #  https://stackoverflow.com/questions/76171274/empty-stacks-from-torch-profiler
    #  (这个只有问题没解决) https://discuss.pytorch.org/t/pytorch-profilers-with-stack-is-not-working/177092
    experimental_config=_ExperimentalConfig(verbose=True)
) as p:
    for iter in range(N):
        with record_function("forward"):
            out = input @ weight
        loss = out.mean()
        with record_function("backward"):
            loss.backward()
        # send a signal to the profiler that the next iteration has started
        p.step()

print(p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
print("这里是分割线--------------------------------")
print(p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=20))
