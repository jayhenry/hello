import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import numpy as np
import torch

@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    """
    Load data from memory and store the result to memory.

    :param res: The destination tensor to store the result.
    :param a: The source tensor to be loaded.
    :param b: The source tensor to be loaded.
    """
    a_vec = a.load()
    print(f"a_vec: {a_vec}")      # prints `a_vec: vector<12xf32> o (3, 4)`
    b_vec = b.load()
    print(f"b_vec: {b_vec}")      # prints `b_vec: vector<12xf32> o (3, 4)`
    res.store(a_vec + b_vec)
    cute.print_tensor(res)


def example1():
    a = np.ones(12).reshape((3, 4)).astype(np.float32)
    b = np.ones(12).reshape((3, 4)).astype(np.float32)
    c = np.zeros(12).reshape((3, 4)).astype(np.float32)
    load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))


@cute.jit
def apply_slice(src: cute.Tensor, dst: cute.Tensor, indices: cutlass.Constexpr):
    """
    Apply slice operation on the src tensor and store the result to the dst tensor.

    :param src: The source tensor to be sliced.
    :param dst: The destination tensor to store the result.
    :param indices: The indices to slice the source tensor.
    """
    src_vec = src.load()
    dst_vec = src_vec[indices]
    print(f"{src_vec} -> {dst_vec}")
    if cutlass.const_expr(isinstance(dst_vec, cute.TensorSSA)):
        print(f"dst_vec is a TensorSSA")
        dst.store(dst_vec)
        cute.print_tensor(dst)
    else:
        print(f"dst_vec is not a TensorSSA")
        dst[0] = dst_vec
        cute.print_tensor(dst)

def slice_1():
    src_shape = (4, 2, 3)
    dst_shape = (4, 3)
    indices = (None, 1, None)

    """
    a:
    [[[ 0.  1.  2.]
      [ 3.  4.  5.]]

     [[ 6.  7.  8.]
      [ 9. 10. 11.]]

     [[12. 13. 14.]
      [15. 16. 17.]]

     [[18. 19. 20.]
      [21. 22. 23.]]]
    """
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)


def slice_2():
    src_shape = (4, 2, 3)
    dst_shape = (1,)
    indices = 10
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)


@cute.jit
def binary_op_1(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()
    b_vec = b.load()

    add_res = a_vec + b_vec
    cute.print_tensor(add_res)          # prints [3.000000, 3.000000, 3.000000]

    sub_res = a_vec - b_vec
    cute.print_tensor(sub_res)          # prints [-1.000000, -1.000000, -1.000000]

    mul_res = a_vec * b_vec
    cute.print_tensor(mul_res)          # prints [2.000000, 2.000000, 2.000000]

    div_res = a_vec / b_vec
    cute.print_tensor(div_res)          # prints [0.500000, 0.500000, 0.500000]

    floor_div_res = a_vec // b_vec
    cute.print_tensor(floor_div_res)              # prints [0.000000, 0.000000, 0.000000]

    mod_res = a_vec % b_vec
    cute.print_tensor(mod_res)          # prints [1.000000, 1.000000, 1.000000]
    # Both below are ok. But res=mod_res is wrong.
    res.store(mod_res)
    # res[None] = mod_res


def ssa_binary():
    a = np.empty((3,), dtype=np.float32)
    a.fill(1.0)
    b = np.empty((3,), dtype=np.float32)
    b.fill(2.0)
    res = np.empty((3,), dtype=np.float32)
    binary_op_1(from_dlpack(res), from_dlpack(a), from_dlpack(b))
    print("res: ", res)


@cute.jit
def unary_op_1(res: cute.Tensor, a: cute.Tensor):
    a_vec = a.load()

    sqrt_res = cute.math.sqrt(a_vec)
    cute.print_tensor(sqrt_res)   # prints [2.000000, 2.000000, 2.000000]

    sin_res = cute.math.sin(a_vec)
    res.store(sin_res)
    cute.print_tensor(sin_res)    # prints [-0.756802, -0.756802, -0.756802]

    exp2_res = cute.math.exp2(a_vec)
    cute.print_tensor(exp2_res)   # prints [16.000000, 16.000000, 16.000000]


def ssa_unary():
    a = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    res = np.empty((3,), dtype=np.float32)
    unary_op_1(from_dlpack(res), from_dlpack(a))
    print("res: ", res)


@cute.jit
def reduction_op(a: cute.Tensor):
    """
    Apply reduction operation on the src tensor.

    :param src: The source tensor to be reduced.
    """
    a_vec = a.load()
    red_res = a_vec.reduce(
        cute.ReductionOp.ADD,
        0.0,
        reduction_profile=0
    )
    cute.printf(red_res)              # prints 21.000000

    red_res = a_vec.reduce(
        cute.ReductionOp.ADD,
        0.0,
        reduction_profile=(None, 1)
    )
    cute.print_tensor(red_res)        # prints [6.000000, 15.000000]

    red_res = a_vec.reduce(
        cute.ReductionOp.ADD,
        1.0,
        reduction_profile=(1, None)
    )
    cute.print_tensor(red_res)        # prints [6.000000, 8.000000, 10.000000]


def ssa_reduction():
   a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
   reduction_op(from_dlpack(a))


import cutlass
import cutlass.cute as cute


@cute.jit
def broadcast_examples():
    a = cute.make_fragment((1,3), dtype=cutlass.Float32)
    a[0] = 0.0
    a[1] = 1.0
    a[2] = 2.0
    a_val = a.load()
    cute.print_tensor(a_val.broadcast_to((4, 3)))
    # tensor(raw_ptr(0x00007ffe26625740: f32, rmem, align<32>) o (4,3):(1,4), data=
    #    [[ 0.000000,  1.000000,  2.000000, ],
    #     [ 0.000000,  1.000000,  2.000000, ],
    #     [ 0.000000,  1.000000,  2.000000, ],
    #     [ 0.000000,  1.000000,  2.000000, ]])

    c = cute.make_fragment((4,1), dtype=cutlass.Float32)
    c[0] = 0.0
    c[1] = 1.0
    c[2] = 2.0
    c[3] = 3.0
    cute.print_tensor(a.load() + c.load())
    # tensor(raw_ptr(0x00007ffe26625780: f32, rmem, align<32>) o (4,3):(1,4), data=
    #        [[ 0.000000,  1.000000,  2.000000, ],
    #         [ 1.000000,  2.000000,  3.000000, ],
    #         [ 2.000000,  3.000000,  4.000000, ],
    #         [ 3.000000,  4.000000,  5.000000, ]])



if __name__ == "__main__":
    # example1()
    # slice_1()
    # slice_2()
    ssa_binary()
    # ssa_unary()
    # ssa_reduction()
    # broadcast_examples()