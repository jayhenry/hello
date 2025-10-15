# ref: https://zhuanlan.zhihu.com/p/21053905491
# 从2.4版本开始，PyTorch开始支持通过torch.library的方式往torch.compile中注册新operator
# 自定义op实现
import torch
from torch import Tensor
from typing import List

# 定义operator的前向函数，可以用python/C++/CUDA实现，python下用torch.library.custom_op定义，C++/CUDA下用TORCH_LIBRARY_IMPL定义；
@torch.library.custom_op("mylib::fused_mul_add", mutates_args={})
def fused_mul_add(x: Tensor, y: Tensor, z: float) -> Tensor:
    return x * y + z

# 通过register_fake定义operator的FakeTensor生成函数，主要定义结果的shape、device信息；
@fused_mul_add.register_fake
def _(x, y, z):
    torch._check(x.device == y.device)
    torch._check(x.shape == y.shape)
    torch._check(x.dtype == y.dtype)
    return torch.empty_like(x)

def _backward(ctx, grad):
    x, y = ctx.saved_tensors
    grad_x, grad_y = None, None
    if ctx.needs_input_grad[0]:
        grad_x = grad * y
        # grad_x = my_mul(grad, y)
    if ctx.needs_input_grad[1]:
        grad_y = grad * x
        # grad_y = my_mul(grad, x)
    return grad_x, grad_y, None, None

def _setup_context(ctx, inputs, output):
    x, y, z = inputs
    save_x, save_y = None, None
    if ctx.needs_input_grad[0]:
        save_y = y
    if ctx.needs_input_grad[1]:
        save_x = x
    ctx.save_for_backward(save_x, save_y)

# 通过torch.library.register_autograd定义operator的反向传播函数；
torch.library.register_autograd("mylib::fused_mul_add", _backward, setup_context=_setup_context)

# PyTorch代码
import torch

def my_func(x, y, z):
    x = x + 1
    return torch.ops.mylib.fused_mul_add(x, y, z)

# 通过自定义backend来打印fx graph
# 观察Dynamo捕获的前反向计算图，可以发现:
# 1. 前向计算图的target直接就是mylib.fused_mul_add，即自定义op被Dynamo正常捕获，同时由于反传就定义为两个乘法，因此反向计算图会有两个aten.mul操作。
# 2. 对于用户自定义的op，torch.compile不会进行decompose，也就不会拆解到PrimTorch规定的op上，Inductor会归为ExternKernel直接走fallback，不会参与Inductor Scheduler的算子融合等，从Dynamo到Inductor到执行全程作为一个独立的个体存在。给用户自定义提供了一个基础的接口，如果想对自定义算子用上Inductor的优化，则需要修改Inductor里的lowering函数、各种pass等函数，涉及到的代码修改就比较多了。
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("[Entrypoint] my compiler called with FX graph:".center(100,"="))
    gm.graph.print_tabular()
    print("code is:",gm.code)
    return gm.forward

my_backend = aot_autograd(fw_compiler=my_compiler)  # bw_compiler=my_compiler


# 入口函数
def main():
    x = torch.randn((2, 3), device='cuda', dtype=torch.float, requires_grad=True)
    y = torch.randn((2, 3), device='cuda', dtype=torch.float, requires_grad=True)
    z = 0.5
    # compiled_func = torch.compile(my_func, backend="inductor")
    compiled_func = torch.compile(my_func, backend=my_backend)
    data = compiled_func(x, y, z)
    data.mean().backward()

if __name__ == '__main__':
    main()
