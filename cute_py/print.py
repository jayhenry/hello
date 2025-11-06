import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass.cute.runtime import from_dlpack


# Example 1: Print example
@cute.jit
def print_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    """
    Demonstrates different printing methods in CuTe and how they handle static vs dynamic values.

    This example shows:
    1. How Python's `print` function works with static values at compile time but can't show dynamic values
    2. How `cute.printf` can display both static and dynamic values at runtime
    3. The difference between types in static vs dynamic contexts
    4. How layouts are represented in both printing methods

    Args:
        a: A dynamic Int32 value that will be determined at runtime
        b: A static (compile-time constant) integer value
    """
    # Use Python `print` to print static information
    print(">>>", b)  # => 2
    # `a` is dynamic value
    print(">>>", a)  # => ?

    # Use `cute.printf` to print dynamic information
    cute.printf(">?? {}", a)  # => 8
    cute.printf(">?? {}", b)  # => 2

    print(">>>", type(a))  # => <class 'cutlass.Int32'>
    print(">>>", type(b))  # => <class 'int'>

    layout = cute.make_layout((a, b))
    print(">>>", layout)            # => (?,2):(1,?)
    cute.printf(">?? {}", layout)   # => (8,2):(1,8)


def example1_run1():
    print_example(cutlass.Int32(8), 2)

def example1_run2():
    print_example_compiled = cute.compile(print_example, cutlass.Int32(8), 2)
    print_example_compiled(cutlass.Int32(8))


# Example 2: Print tensor
@cute.jit
def print_tensor_basic(x : cute.Tensor):
    # Print the tensor
    print("Basic output:")
    cute.print_tensor(x)
    
@cute.jit
def print_tensor_verbose(x : cute.Tensor):
    # Print the tensor with verbose mode
    print("Verbose output:")
    cute.print_tensor(x, verbose=True)

@cute.jit
def print_tensor_slice(x : cute.Tensor, coord : tuple):
    # slice a 2D tensor from the 3D tensor
    sliced_data = cute.slice_(x, coord)
    y = cute.make_fragment(sliced_data.layout, sliced_data.element_type)
    # Convert to TensorSSA format by loading the sliced data into the fragment
    y.store(sliced_data.load())
    print("Slice output:")
    cute.print_tensor(y)

def tensor_print_example1():
    shape = (4, 3, 2)
    
    # Creates [0,...,23] and reshape to (4, 3, 2)
    data = np.arange(24, dtype=np.float32).reshape(*shape) 
      
    print_tensor_basic(from_dlpack(data))
    print_tensor_verbose(from_dlpack(data))


def tensor_print_example2():
    shape = (4, 3)
    
    # Creates [0,...,11] and reshape to (4, 3)
    data = np.arange(12, dtype=np.float32).reshape(*shape) 
      
    print_tensor_basic(from_dlpack(data))
    print_tensor_verbose(from_dlpack(data))


def tensor_print_example3():
    shape = (4, 3)
    
    # Creates [0,...,11] and reshape to (4, 3)
    data = np.arange(12, dtype=np.float32).reshape(*shape) 
      
    print("print slice with (None, 0)")
    print_tensor_slice(from_dlpack(data), (None, 0))
    print("print slice with (1, None)")
    print_tensor_slice(from_dlpack(data), (1, None))


# Example 3: Print tensor on GPU
@cute.kernel
def print_tensor_gpu(src: cute.Tensor):
    print("compile time:", src)
    cute.print_tensor(src)

@cute.jit
def print_tensor_host(src: cute.Tensor):
    print_tensor_gpu(src).launch(grid=(1,1,1), block=(2,1,1))


def tensor_print_example4():
    a = torch.randn(4, 3, device="cuda")
    # cutlass.cuda.initialize_cuda_context()
    print_tensor_host(from_dlpack(a))


if __name__ == "__main__":
    # Example 1
    # example1_run1()
    # example1_run2()

    # Example 2
    # tensor_print_example1()
    # tensor_print_example2()
    # tensor_print_example3()

    # Example 3
    tensor_print_example4()
