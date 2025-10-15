import torch
import warnings

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    print(f"device_cap: {device_cap}")
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )

torch.cuda.set_device(0)
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


def method1():
    opt_foo1 = torch.compile(foo)
    print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

def method2():
    t1 = torch.randn(10, 10).cuda()
    t2 = torch.randn(10, 10).cuda()
    print(opt_foo2(t1, t2))

if __name__ == "__main__":
    # method1()
    method2()
