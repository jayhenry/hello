# ref: https://docs.pytorch.org/docs/main/torch.compiler_custom_backends.html#registering-custom-backends
import torch
from typing import List

torch.cuda.set_device(0)

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("[Entrypoint] custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Reset since we are using a different backend.
torch._dynamo.reset()

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    # If fullgraph is True, the graph will break with the following error:
    #
    # torch._dynamo.exc.Unsupported: Data-dependent branching
    #   Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
    #   Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
    #   Hint: Use `torch.cond` to express dynamic control flow.
    if b.sum() < 0:
        b = b * -1
    return x * b

opt_bar = torch.compile(bar, backend=custom_backend, fullgraph=False)
inp1 = torch.randn(10).cuda(0)
inp2 = torch.randn(10).cuda(0)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)
