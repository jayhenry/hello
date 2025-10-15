import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}")  # Prints tensor layout at compile time
    cute.printf("tensor: {}", tensor)         # Prints tensor values at runtime


# Example1: Static layout
print("Example1:".center(50, "="))
a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
# a_pack = a
print("compile time:")
compiled_func = cute.compile(foo, a_pack)
print("run time:")
compiled_func(a_pack)

print("run time with wrong shape:")
b = torch.tensor([11, 12, 13, 14, 15], dtype=torch.uint16)
b_pack = from_dlpack(b)
compiled_func(b_pack)  # ❌ This results in an unexpected result at runtime due to type mismatch


# Example2: Dynamic layout
print("Example2:".center(50, "="))
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
print("compile time with dynamic layout:")
compiled_func = cute.compile(foo, a)
print("run time with dynamic layout:")
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
print("run time with dynamic layout and different shape:")
compiled_func(b)  # Reuse the same compiled function for different shape


# Example3: static and dynamic arguments in if statement
@cute.jit
def foo2(tensor, x: cutlass.Constexpr[int]):
    print(cute.size(tensor))  # Prints 3 for the 1st call
                              # Prints ? for the 2nd call
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
print("Example3:".center(50, "="))
print("call with static layout:")
foo2(from_dlpack(a), 3)   # First call with static layout

print("call with dynamic layout:")
b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo2(b, 3)                # Second call with dynamic layout


# Example4: static and dynamic arguments in for loop
@cute.jit
def foo3(tensor, x: cutlass.Constexpr[int]):
    for i in range(cute.size(tensor)):
        cute.printf("tensor[{}]: {}", i, tensor[i])

print("Example4:".center(50, "="))
a = torch.tensor([1, 2, 3], dtype=torch.uint16)
print("call with static layout")
foo3(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
print("call with dynamic layout")
foo3(b, 3)                # Second call with dynamic layout