# ref: https://docs.pytorch.org/docs/main/torch.compiler_custom_backends.html#registering-custom-backends
import torch
from typing import List

torch.cuda.set_device(0)

# 1. Custom backend to print fx graph，但是不能支持反向传播
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("[Entrypoint] custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# 2. 支持反向传播的方法
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("[Entrypoint] my compiler called with FX graph:".center(100,"="))
    gm.graph.print_tabular()
    print("code is:",gm.code)
    return make_boxed_func(gm.forward)

my_backend = aot_autograd(fw_compiler=my_compiler)  # bw_compiler=my_compiler


# Reset since we are using a different backend.
torch._dynamo.reset()

def bar(a, b):
    x = torch.abs(a)
    return x * b

# opt_bar = torch.compile(bar, backend=custom_backend, fullgraph=False)
opt_bar = torch.compile(bar, backend=my_backend, fullgraph=False)

inp1 = torch.randn(10, requires_grad=True).cuda(0)
inp2 = torch.randn(10, requires_grad=True).cuda(0)
res = opt_bar(inp1, inp2)
res.sum().backward()
