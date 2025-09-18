import os
import socket
from datetime import datetime
from functools import partial

import torch
from torch.autograd.profiler import record_function


def profile_mem_v1():
    # Keep a max of 100,000 alloc/free events in the recorded history
    # leading up to the snapshot.
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

    input = torch.randn(10, 4096, 4096, requires_grad=True).to("cuda")
    weight = torch.randn(4096, 4096).to("cuda")
    out = input @ weight

    torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


def profile_mem_v2():
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
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        # on_trace_ready=partial(trace_handler, dir="save"),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
    ) as p:

        input = torch.randn(10, 4096, 4096, requires_grad=True).to("cuda")
        weight = torch.randn(4096, 4096).to("cuda")
        out = input @ weight
        p.step()
    
    prof = p
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=10, max_name_column_width=200
        )
    )

    print(
        prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=-1)
    )

    prof.export_memory_timeline(f"my_mem_timeline.html")



# profile_mem_v1()

profile_mem_v2()