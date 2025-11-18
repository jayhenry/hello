# https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html#thread-value-partitioning
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


@cute.jit
def tensor_tv_layout(a: cute.Tensor):
    # Tensor: (M4,N8)
    cute.printf("a = ")
    cute.print_tensor(a)
    # Construct a TV-layout that maps 8 thread indices and 4 value indices
    # to 1D coordinates within a 4x8 tensor
    # tv_layout: (T8,V4) 
    tv_layout = cute.make_layout(((2,4), (2,2)), stride=((8,1), (4,16)))
    print(f"tv_layout = {tv_layout}")
    tv_a = cute.composition(a, tv_layout)
    cute.printf("tv_a = ")
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
    #         8                     8  共32组，共256个元素
    #        --------------------------
    #     4  | t0 | t1 |    |    |    |
    #        --------------------------
    #        |    |    |    |    |    |
    #        --------------------------
    #        |    |    |    |    |    |
    #        --------------------------
    #     4  |    |    |    |    |    |
    #        --------------------------
    #   共4组，共16个元素
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))  # 上面外层tile的 4组 x 32组
    val_layout = cute.make_layout((4, 8), stride=(8, 1))  # 上面内层元素的 4 x 8
    print(f"thr_layout: {thr_layout}")
    print(f"val_layout: {val_layout}")
    # TODO: make_layout_tv anatomy
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")  # (16, 256) = (4*4, 32*8)
    print(f"TV Layout: {tv_layout}")  # ((32,4),(8,4)):((128,4),(16,1))


@cute.jit
def test_make_layout_tv2():
    # TODO:
    #         8                     8  共32组，共256个元素
    #        --------------------------
    #     4  | t0 | t1 |    |    |    |
    #        --------------------------
    #        |    |    |    |    |    |
    #        --------------------------
    #        |    |    |    |    |    |
    #        --------------------------
    #     4  |    |    |    |    |    |
    #        --------------------------
    #   共4组，共16个元素
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))  # 上面外层tile的 4组 x 32组
    val_layout = cute.make_ordered_layout((4, 8), order=(1, 0))  # 上面内层元素的 4 x 8
    print(f"thr_layout: {thr_layout}")
    print(f"val_layout: {val_layout}")
    # TODO: make_layout_tv anatomy
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")  # (16, 256) = (4*4, 32*8)
    print(f"TV Layout: {tv_layout}")  # ((32,4),(8,4)):((128,4),(16,1))
    # Why modes of thread domain of TV Layout looks swapped especially when tensor is row major?
    # We may notice that TV Layout in above example is ((32,4),(8,4)):((128,4),(16,1)). 
    # However, on visualization, thread indices are arrange as shape (4,32) rather than (32,4) of TV Layout.
    # This is a commonly asked question by developers from both internal teams and community.
    # It's important to keep in mind that TV Layout maps (thread_index, value_index) to (row_index, column_index) of 
    # logical domain (TileM, TileN). However, visualization shows inverse mapping of 
    # logical domain (TileM, TileN) to (thread_domain, value_domain), because this is more intuitive for human developer.
    # That's why the shape of domain of TV Layout doesn't necessarily match logical view.


@cute.jit
def test_right_inverse():
    # TODO:
    # layout = cute.make_layout((4,8), stride=(8,1))
    layout = cute.make_layout((4,8))
    print(f"layout = {layout}")
    inverse_layout = cute.right_inverse(layout)
    print(f"inverse_layout = {inverse_layout}")


if __name__ == "__main__":
    # tv_layout_example1()
    # test_right_inverse()
    # test_make_layout_tv()
    test_make_layout_tv2()
