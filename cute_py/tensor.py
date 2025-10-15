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
    tensor.fill(1)
    cute.print_tensor(tensor)


def example1():
    a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
    ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())
    
    create_tensor_from_ptr(ptr_a)


@cute.jit
def tensor_access_item(a: cute.Tensor):
    # access data using linear index
    # TODO: Below we create a coordinate tensor with a column-major layout? how to create a row-major coordinate tensor?
    cute.printf("identity tensor = {}", cute.make_identity_tensor(a.layout.shape))
    # cute.print_tensor(cute.make_identity_tensor(a.layout.shape))
    cute.print_tensor(a, verbose=True)
    cute.printf("a[2] = {} (equivalent to a[{}])", a[2],
                cute.make_identity_tensor(a.layout.shape)[2])
    cute.printf("a[9] = {} (equivalent to a[{}])", a[9],
                cute.make_identity_tensor(a.layout.shape)[9])

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
    
    print(data)


if __name__ == "__main__":
    # example1()
    example2()