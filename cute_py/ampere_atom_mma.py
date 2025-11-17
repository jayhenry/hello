import argparse
import math
from typing import Tuple, Type

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

mma_inst_shape = (16, 8, 16)
mmaM, mmaN, mmaK = mma_inst_shape

mT, nT, kT = 2, 3, 5
cta_tiler = (16*mT, 8*nT, 16*kT)  # mma_inst_shape  # (128, 128, 32)
bM, bN, bK = cta_tiler

atom_layout_mnk = (1,1,1)  # (2, 2, 1)
atom_lay_M, atom_lay_N, atom_lay_K = atom_layout_mnk
num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32  # 1 * 1 * 1 * 32 = 32

# mnkl = (1024, 1024, 1024, 1)
# mnkl = (16, 8, 16, 1)
mnkl = (16*mT, 8*nT, 16*kT, 1)
# mnkl = (128, 128, 32, 1)
# mnkl = (128*2, 128*2, 32*2, 1)
M, N, K, L = mnkl

c_dtype = ab_dtype = cutlass.Float16
acc_dtype = c_dtype  # TODO: cutlass.Float32

a_major = "k"
b_major = "n"
c_major = "n"

# Create and permute tensor A/B/C
def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else: (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_tensor = (
        torch.empty(*shape, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=cutlass_torch.dtype(dtype))
        .permute(permute_order)
        .cuda()
    )
    assert l == 1, "l must be 1"
    torch_tensor = torch_tensor.squeeze(-1)  # (mode0, mode1, l) -> (mode0, mode1)
    # assume input is 16B aligned
    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=16)
        # .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
        # .mark_compact_shape_dynamic(
        #     mode=(1 if not is_mode0_major else 0),
        #     # https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#mark-the-tensor-s-layout-as-dynamic-with-mark-compact-shape-dynamic
        #     stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
        #     divisibility=(128 // dtype.width),
        # )
    )
    return cute_tensor, torch_tensor

mA, a_torch = create_and_permute_tensor(L, M, K, a_major == "m", ab_dtype)
mB, b_torch = create_and_permute_tensor(L, N, K, b_major == "n", ab_dtype)
mC, c_torch = create_and_permute_tensor(L, M, N, c_major == "m", c_dtype)
print(f"mA: {mA}")
print(f"mB: {mB}")
print(f"mC: {mC}")


@cute.jit
def call_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):

    # ///////////////////////////////////////////////////////////////////////////////
    # Tiled MMA
    # ///////////////////////////////////////////////////////////////////////////////

    # Creates a mma atom with 16x8x16 shape for MNK
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
        ab_dtype, acc_dtype, mma_inst_shape
    )
    print(f"mma_atom: {mma_atom}")

    # TODO: visualize tiled_mma like `print_latex(tiled_mma)` in https://github.com/NVIDIA/cutlass/discussions/1345
    # ///////////////////////////////////////////////////////////////////////////////
    # 关于 permutation_mnk 的解释:
    # From cute作者 Cecka：
    #  https://github.com/NVIDIA/cutlass/discussions/1345
    # 
    # From ampere_tensorop_gemm.py:
    #   见行内注释
    # 
    # From sgemm.py:
    #   Here, the MMA layout is set so that each thread copies four
    # consecutive elements from shared memory to registers.
    # `permutation_tiler_M/N` maps the elements handled by each thread
    # to the permuted element in the tensor.
    # For increasing indices in the tensor, the thread ID that reads it is:
    #   - (without permutation) ==>
    #      0 1 2 ... 15 0 1 2 ... 15 0 1 2 ... 15 0 1 2 ... 15 ......
    #   - (with permutation) ==>
    #      0 0 0 0 1 1 1 1 2 2 2 2 ... 15 15 15 15 0 0 0 0 1 1 1 1 ......
    # 
    # From cute之简单GEMM实现 https://zhuanlan.zhihu.com/p/667521327 :
    # 我们知道SM80的Tensor Core执行是warp level的，也就是说这个MMA_Atom是32个线程。
    # 我们对MMA_Atom能力通过增加线程的方式进行M、N方向的重复，
    # 同时我们让B矩阵C矩阵使用更多寄存器在N方向扩展2次，得到main函数中的MMA类型。
    # 这样，我们便可以得到TiledMMA需要32x2x2 = 128线程，其能处理的矩阵的大小: 
    # M = 16 x 2 x 1 = 32, N = 8 x 2 x 2 = 32, K = 16 x 1 x 1 = 16, 
    # 即TiledMMA能处理的MNK为32x32x16。
    # ///////////////////////////////////////////////////////////////////////////////
    permutation_mnk = (
        atom_layout_mnk[0] * mma_inst_shape[0],        
        atom_layout_mnk[1] * mma_inst_shape[1] ,       
        atom_layout_mnk[2] * mma_inst_shape[2],        
    )
    print(f"permutation_mnk: {permutation_mnk}")  

    # Created a tiled mma that tiles the atom according to specified layout.
    # For a 2x2x1 atom layout, the mma atom is duplicated 4 times, twice
    # across M and twice across N
    tC = cute.make_layout(atom_layout_mnk)  # (2, 2, 1)
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        tC,
        permutation_mnk=permutation_mnk,
    )
    # cute.print_latex(tiled_mma)
    print(f"tiled_mma: {tiled_mma}")
    # tiled_mma: Tiled MMA
    #   Thr Layout VMNK: (32,1,1,1):(1,0,0,0)
    #   Permutation MNK: (16:1,8:1,16:1)
    # MMA Atom
    #   ThrID:           32:1
    #   Shape MNK:       (16,8,16)
    #   TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
    #   TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
    #   TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))

    ref_tiled_mma = cute.make_tiled_mma(mma_atom)  # same with tiled_mma
    print(f"ref_tiled_mma: {ref_tiled_mma}")
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
    # ref_tiled_mma: Tiled MMA
    #   Thr Layout VMNK: (32,1,1,1):(1,0,0,0)
    #   Permutation MNK: (_,_,_)
    # MMA Atom
    #   ThrID:           32:1
    #   Shape MNK:       (16,8,16)
    #   TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
    #   TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
    #   TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))

    # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, 1)
    grid_dim = cute.ceil_div(mC.shape, (bM, bN)) + (1,)
    print(f"grid_dim: {grid_dim}")  # (1, 1, 1)

    kernel_fn(
            mA,
            mB,
            mC,
            # tiled_mma,
            ref_tiled_mma,
        ).launch(
            grid=grid_dim,
            block=[num_threads, 1, 1],
        )


@cute.kernel
def kernel_fn(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
):
    # Thread index, block index
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    tiler_coord = (bidx, bidy, None)

    # ///////////////////////////////////////////////////////////////////////////////
    # Get the appropriate tiles for this thread block.
    # gA: (BLK_M, BLK_N, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
    # ///////////////////////////////////////////////////////////////////////////////

    # Applying a tiler and then slicing out that tile by indexing into the remainder mode is common and has been 
    # wrapped into its own function inner_partition(Tensor, Tiler, Coord). 
    # You'll often see local_tile(Tensor, Tiler, Coord) which is just another name for inner_partition.
    # We call this an inner-partition because it keeps the inner “tile” mode.
    # ref: https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html#inner-and-outer-partitioning
    gA = cute.local_tile(mA[None, None], tiler=cta_tiler, coord=tiler_coord, proj=(1, None, 1),)
    gB = cute.local_tile(mB[None, None], tiler=cta_tiler, coord=tiler_coord, proj=(None, 1, 1),)
    gC = cute.local_tile(mC[None, None], tiler=cta_tiler, coord=tiler_coord, proj=(1, 1, None),)
    print(f"gA: {gA}")  # tensor<ptr<f16, gmem, align<16>> o (16,16,1):(16,1,0)>
    print(f"gB: {gB}")  # tensor<ptr<f16, gmem, align<16>> o (8,16,1):(1,8,0)>
    print(f"gC: {gC}")  # tensor<ptr<f16, gmem, align<16>> o (16,8):(8,1)>
    # gA(kTileM, kTileK, num_tile_k)
    # gB(kTileN, kTileK, num_tile_k)
    # gC(kTileM, kTileN) 

    # ///////////////////////////////////////////////////////////////////////////////
    # Tile MMA compute thread partitions and allocate accumulators
    # ///////////////////////////////////////////////////////////////////////////////
    # illustration: https://zhuanlan.zhihu.com/p/675308830

    # print(f"tiled_mma: {tiled_mma}")
    # tiled_mma: Tiled MMA
    #   Thr Layout VMNK: (32,1,1,1):(1,0,0,0)
    #   Permutation MNK: (_,_,_)
    # MMA Atom
    #   ThrID:           32:1
    #   Shape MNK:       (16,8,16)
    #   TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
    #   TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
    #   TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
    thr_mma = tiled_mma.get_slice(tidx)
    print(f"thr_mma: {thr_mma}")
    # thr_mma: Tiled MMA
    #   Thr Layout VMNK: (32,1,1,1):(1,0,0,0)
    #   Permutation MNK: (_,_,_)
    # MMA Atom
    #   ThrID:           32:1
    #   Shape MNK:       (16,8,16)
    #   TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
    #   TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
    #   TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
    # MMA_M、MMA_K表示(kTileM, kTileK)按照TiledMMA能力去划分的时候，M方向和K方向需要重复多少次才能完成该矩阵乘法，即M K方向需要循环多少遍TildMMA才能完成计算
    tAgA = thr_mma.partition_A(gA)  # (MMA=(2,2,2), MMA_M=1, MMA_K=1, num_tile_k)
    print(f"tAgA: {tAgA}")  # tAgA: tensor<ptr<f16, gmem, align<4>> o ((2,2,2),1,1,1):((1,128,8),0,0,0)>
    tBgB = thr_mma.partition_B(gB)  # (MMA=(2,2), MMA_N=1, MMA_K=1, num_tile_k)
    print(f"tBgB: {tBgB}")  # tBgB: tensor<ptr<f16, gmem> o ((2,2),1,1,1):((8,64),0,0,0)>
    tCgC = thr_mma.partition_C(gC)  # (MMA=(2,2), MMA_M=1, MMA_N=1)
    print(f"tCgC: {tCgC}")  # tCgC: tensor<ptr<f16, gmem, align<4>> o ((2,2),1,1):((1,64),0,0)>
    tArA = tiled_mma.make_fragment_A(tAgA[None, None, None, 0])  # (MMA, MMA_M, MMA_K)
    print(f"tArA: {tArA}")  # tArA: tensor<ptr<f16, rmem, align<16>> o ((2,2,2),1,1):((1,2,4),0,0)>
    tBrB = tiled_mma.make_fragment_B(tBgB[None, None, None, 0])  # (MMA, MMA_N, MMA_K)
    print(f"tBrB: {tBrB}")  # tBrB: tensor<ptr<f16, rmem, align<8>> o ((2,2),1,1):((1,2),0,0)>
    tCrC = tiled_mma.make_fragment_C(tCgC)                       # (MMA, MMA_M, MMA_N)
    print(f"tCrC: {tCrC}")  # tCrC: tensor<ptr<f16, rmem, align<8>> o ((2,2),1,1):((1,2),0,0)>
    # Clear the accumulator
    tCrC.fill(0.0)

    num_tile_k = gA.shape[2]

    # global <-> register copy
    load_copy_atom= cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    store_copy_atom= cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    for tile_k_index in range(num_tile_k):
        cute.copy(load_copy_atom, tAgA[None, None, None, tile_k_index], tArA)
        cute.copy(load_copy_atom, tBgB[None, None, None, tile_k_index], tBrB)
        # 因为mma是warp level的，而warp中32个线程从硬件上就同步执行，所以这里不需要在copy后做同步。

        # Thread-level register gemm for k_block
        # 下面会多次调用mma指令，取决于 block size / tiledMMA size
        cute.gemm(
            tiled_mma,
            tCrC,  # d = a*b + c
            tArA,  # a
            tBrB,  # b 
            tCrC,  # c
        )
    
    cute.copy(store_copy_atom, tCrC, tCgC)



def main():
    print("Compiling kernel with cute.compile ...")
    compiled_kernel = cute.compile(call_kernel, mA, mB, mC)

    print("Executing Atom Demo kernel...")
    compiled_kernel(mA, mB, mC)
    print("Kernel executed successfully!")

    torch.cuda.synchronize()
    print("Verifying results...")
    ref = torch.einsum("mk,nk->mn", a_torch, b_torch)
    torch.testing.assert_close(c_torch.cpu(), ref.cpu(), atol=1e-03, rtol=1e-05)
    print("Results verified successfully!")


if __name__ == "__main__":
    main()