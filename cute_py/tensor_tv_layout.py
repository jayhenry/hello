# https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html#thread-value-partitioning
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


@cute.jit
def tensor_tv_layout(a: cute.Tensor):
    # Tensor: (M4,N8)
    cute.printf("a = {}", a)
    cute.print_tensor(a)
    # Construct a TV-layout that maps 8 thread indices and 4 value indices
    # to 1D coordinates within a 4x8 tensor
    # tv_layout: (T8,V4) 
    tv_layout = cute.make_layout(((2,4), (2,2)), stride=((8,1), (4,16)))
    print(f"tv_layout = {tv_layout}")
    tv_a = cute.composition(a, tv_layout)
    cute.printf("tv_a = {}", tv_a)
    cute.print_tensor(tv_a)

    for thread_idx in range(8):
        cute.printf("thread_idx = {}, tv_a slice = {}", thread_idx, tv_a[(thread_idx, None)])

def tv_layout_example1():
    a = torch.arange(32).reshape(8,4).transpose(0,1)
    print(f"torch tensor = {a}")
    tensor_tv_layout(from_dlpack(a))


@cute.jit
def test_make_layout_tv():
    # TODO:
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")  # (16, 256)
    print(f"thr_layout: {thr_layout}")
    print(f"val_layout: {val_layout}")
    print(f"TV Layout: {tv_layout}")  # ((32,4),(8,4)):((128,4),(16,1))


@cute.jit
def test_right_inverse():
    # layout = cute.make_layout((4,8), stride=(8,1))
    layout = cute.make_layout((4,8))
    print(f"layout = {layout}")
    inverse_layout = cute.right_inverse(layout)
    print(f"inverse_layout = {inverse_layout}")


if __name__ == "__main__":
    # tv_layout_example1()
    # test_right_inverse()
    test_make_layout_tv()
