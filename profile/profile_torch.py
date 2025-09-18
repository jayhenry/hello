import os
import socket
from datetime import datetime
from functools import partial

import torch
from torch.autograd.profiler import record_function


# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof: torch.profiler.profile, dir=""):
    os.makedirs(dir, exist_ok=True)

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=10, max_name_column_width=200
        )
    )

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    trace_file_prefix = os.path.join(dir, f"profiler_trace_{host_name}_{timestamp}")
    memory_tl_file_prefix = os.path.join(
        dir, f"profiler_memory_timeline_{host_name}_{timestamp}"
    )
    kernel_times_file_prefix = os.path.join(
        dir, f"profiler_kernel_times_{host_name}_{timestamp}"
    )

    # Construct the trace file.
    prof.export_chrome_trace(f"{trace_file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{memory_tl_file_prefix}.html")

    # Construct the kernel times file.
    with open(f"{kernel_times_file_prefix}.txt", "w") as f:
        f.write(
            prof.key_averages(group_by_stack_n=5).table(
                sort_by="cuda_time_total", row_limit=-1
            )
        )


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
    with_stack=True,
    on_trace_ready=partial(trace_handler, dir="save"),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
) as p:
    for iter in range(N):
        with record_function("forward"):
            out = input @ weight
        loss = out.mean()
        with record_function("backward"):
            loss.backward()
        # send a signal to the profiler that the next iteration has started
        p.step()
