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

cta_tiler = (128, 128, 32)
bM, bN, bK = cta_tiler

atom_layout_mnk = (2, 2, 1)
atom_lay_M, atom_lay_N, atom_lay_K = atom_layout_mnk
num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32  # 2 * 2 * 1 * 32 = 128

mma_inst_shape = (16, 8, 16)
mmaM, mmaN, mmaK = mma_inst_shape
# mnkl = (1024, 1024, 1024, 1)
# mnkl = (16, 8, 16, 1)
mnkl = (128, 128, 32, 1)
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

    # ///////////////////////////////////////////////////////////////////////////////
    # 关于 permutation_mnk 的解释:
    # From ampere_tensorop_gemm.py:
    #   见行内注释
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
    # ///////////////////////////////////////////////////////////////////////////////
    permutation_mnk = (
        atom_layout_mnk[0] * mma_inst_shape[0],  # 2*16 = 32
        # if atom layout's N-mode is 1, to leverage the largest coalesced
        # shared memory -> register copy, set the tiled mma's N mode to 16
        atom_layout_mnk[1] * mma_inst_shape[1] * 2,  # 2*8*2 = 32
        atom_layout_mnk[2] * mma_inst_shape[2],  # 1*16 = 16
    )

    # Created a tiled mma that tiles the atom according to specified layout.
    # For a 2x2x1 atom layout, the mma atom is duplicated 4 times, twice
    # across M and twice across N
    tC = cute.make_layout(atom_layout_mnk)  # (2, 2, 1)
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        tC,
        permutation_mnk=permutation_mnk,
    )

    # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, 1)
    grid_dim = cute.ceil_div(mC.shape, (bM, bN)) + (1,)

    kernel_fn(
            mA,
            mB,
            mC,
            tiled_mma,
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
    # cta_tiler: (128, 128, 32)
    gA = cute.local_tile(mA[None, None], tiler=cta_tiler, coord=tiler_coord, proj=(1, None, 1),)
    gB = cute.local_tile(mB[None, None], tiler=cta_tiler, coord=tiler_coord, proj=(None, 1, 1),)
    gC = cute.local_tile(mC[None, None], tiler=cta_tiler, coord=tiler_coord, proj=(1, 1, None),)
    # gA(kTileM, kTileK, num_tile_k)
    # gB(kTileN, kTileK, num_tile_k)
    # gC(kTileM, kTileN) 

    # ///////////////////////////////////////////////////////////////////////////////
    # Tile MMA compute thread partitions and allocate accumulators
    # ///////////////////////////////////////////////////////////////////////////////
    thr_mma = tiled_mma.get_slice(tidx)
    tAgA = thr_mma.partition_A(gA)  # (MMA, MMA_M, MMA_K, num_tile_k)
    tBgB = thr_mma.partition_B(gB)  # (MMA, MMA_N, MMA_K, num_tile_k)
    tCgC = thr_mma.partition_C(gC)  # (MMA, MMA_M, MMA_N)
    tArA = tiled_mma.make_fragment_A(tAgA[None, None, None, 0])  # (MMA, MMA_M, MMA_K)
    tBrB = tiled_mma.make_fragment_B(tBgB[None, None, None, 0])  # (MMA, MMA_N, MMA_K)
    tCrC = tiled_mma.make_fragment_C(tCgC)                       # (MMA, MMA_M, MMA_N)
    # Clear the accumulator
    tCrC.fill(0.0)

    num_tile_k = gA.shape[2]

    # global <-> register copy
    copy_atom= cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)

    for tile_k_index in range(num_tile_k):
        cute.copy(copy_atom, tAgA[None, None, None, tile_k_index], tArA)
        cute.copy(copy_atom, tBgB[None, None, None, tile_k_index], tBrB)

        # Thread-level register gemm for k_block
        cute.gemm(
            tiled_mma,
            tCrC,  # d = a*b + c
            tArA,  # a
            tBrB,  # b 
            tCrC,  # c
        )
    
    cute.copy(copy_atom, tCrC, tCgC)



def main():
    print("Compiling kernel with cute.compile ...")
    compiled_kernel = cute.compile(call_kernel, mA, mB, mC)

    print("Executing Copy Atom Demo kernel...")
    compiled_kernel(mA, mB, mC)
    print("Kernel executed successfully!")


if __name__ == "__main__":
    main()