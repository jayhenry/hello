import torch

from cutlass.torch import dtype as torch_dtype
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.runtime as cute_rt
import cutlass
import cutlass.cute as cute

@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5, 1))
    tensor = cute.make_tensor(ptr, layout)
    # tensor.fill(1)
    cute.printf("tensor with layout (8,5):(5,1) = {}", tensor)
    cute.print_tensor(tensor)

    layout2 = cute.make_layout((8, 5), stride=(1, 8))
    tensor2 = cute.make_tensor(ptr, layout2)
    cute.printf("tensor with layout (8,5):(1,8) = ")
    cute.print_tensor(tensor2)

    layout3 = cute.make_layout((8, 5), stride=(1, 2))
    tensor3 = cute.make_tensor(ptr, layout3)
    cute.printf("tensor with layout (8,5):(1,2) = ")
    cute.print_tensor(tensor3)


def example1():
    # a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
    a = torch.arange(0, 8*5, dtype=torch_dtype(cutlass.Float32))
    ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())
    
    create_tensor_from_ptr(ptr_a)


@cute.jit
def tensor_access_item(a: cute.Tensor):
    # access data using linear index
    # TODO: Below we create a coordinate tensor with a column-major layout? how to create a row-major coordinate tensor?
    linear_idx2coord = cute.make_identity_tensor(a.layout.shape)  # column-major coordinate tensor
    """
    ref: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.make_identity_tensor
# Create a simple 1D coord tensor
tensor = make_identity_tensor(6)  # [0,1,2,3,4,5]

# Create a 2D coord tensor
tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

# Create hierarchical coord tensor
tensor = make_identity_tensor(((2,1),3))
# [((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)]
    """
    cute.printf("identity tensor = {}", linear_idx2coord)
    # cute.print_tensor(cute.make_identity_tensor(a.layout.shape))
    cute.print_tensor(a, verbose=False)
    cute.printf("a[2] = {} (equivalent to a[{}])", a[2],
                linear_idx2coord[2])
    cute.printf("a[9] = {} (equivalent to a[{}])", a[9],
                linear_idx2coord[9])

    # access data using n-d coordinates, following two are equivalent
    cute.printf("a[2,0] = {}", a[2, 0])
    cute.printf("a[2,4] = {}", a[2, 4])
    cute.printf("a[(2,4)] = {}", a[(2, 4)])

    # assign value to tensor@(2,4)
    a[2,3] = 100.0
    a[2,4] = 101.0
    cute.printf("a[2,3] = {}", a[2,3])
    cute.printf("a[(2,4)] = {}", a[(2,4)])


def example2():
    # Create a tensor with sequential data using torch
    data = torch.arange(0, 8*5, dtype=torch.float32).reshape(8, 5)
    tensor_access_item(from_dlpack(data))
    
    print("After tensor_access_item, data =", data)


if __name__ == "__main__":
    example1()
    # example2()