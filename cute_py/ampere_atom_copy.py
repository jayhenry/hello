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

mT, nT, kT = 1, 1, 1
cta_tiler = (128, 128, 32)  # (16*mT, 8*nT, 16*kT)  # mma_inst_shape  # (128, 128, 32)
bM, bN, bK = cta_tiler

atom_layout_mnk = (2,2,1)  # (1,1,1)  # (2, 2, 1)
atom_lay_M, atom_lay_N, atom_lay_K = atom_layout_mnk
num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32  # 128  # 1 * 1 * 1 * 32 = 32

# mnkl = (1024, 1024, 1024, 1)
# mnkl = (16, 8, 16, 1)
# mnkl = (16*mT, 8*nT, 16*kT, 1)
mnkl = (128, 128, 32, 1)
# mnkl = (128*2, 128*2, 32*2, 1)
M, N, K, L = mnkl

c_dtype = ab_dtype = cutlass.Float16
acc_dtype = c_dtype  # TODO: cutlass.Float32

a_major = "k"
b_major = "n"
c_major = "n"

ab_copy_bits = copy_bits = 128

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
    # assume input is 128B aligned (required for async copy with 128-bit ops)
    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=128)
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
    # Shared memory layout:
    # ///////////////////////////////////////////////////////////////////////////////
    # sA_layout = self._make_smem_layout_AB(
    #     mA.element_type,
    #     self.a_major_mode,
    #     ab_copy_bits,
    #     #        M                 K
    #     (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
    # )
    sA_layout = cute.make_ordered_layout((bM, bK), order=(1, 0))  # (128, 32)
    sB_layout = cute.make_ordered_layout((bN, bK), order=(0, 1))  # (128, 32)
    smem_size = max(
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout),
            0,  # cute.size_in_bytes(mC.element_type, sC_layout),
    )

    # ///////////////////////////////////////////////////////////////////////////////
    # Tiled copy:
    # The majorness of tA/tB/tC follows the majorness of gA/gB/gC,
    # enabling merged accesses to global memory for faster data
    # transfer between global and shared memory.
    # ///////////////////////////////////////////////////////////////////////////////
    dtype = mA.element_type
    atom_async_copy = cute.make_copy_atom(
        # https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=ab_copy_bits,
    )
    # Create thread layouts for tiled copy from the copy atom where the
    # thread layout simply follows the leading dimension of the tensor
    # a_major_mode = utils.LayoutEnum.from_tensor(mA)  # row major
    # tiled_copy_A = _make_gmem_tiled_copy_AB(
        # atom_async_copy, mA.element_type, a_major_mode, ab_copy_bits
    # )
    copy_elems = copy_bits // dtype.width  # 128 // 16 = 8
    shape_dim_1 = cute.size(bK) // copy_elems  # 32 // 8 = 4
    # thread layout for copy
    thread_layout = cute.make_layout((num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1))  # (128 // 4, 4) = (32, 4)
    # Value layout for copy
    value_layout = cute.make_layout((1, copy_elems))  # (1, 8)
    tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy, thread_layout, value_layout)
    tiled_copy_B = tiled_copy_A


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
        atom_layout_mnk[1] * mma_inst_shape[1] * 2,       
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
            tiled_mma,
            # ref_tiled_mma,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
        ).launch(
            grid=grid_dim,
            block=[num_threads, 1, 1],
            smem=smem_size,
        )


@cute.kernel
def kernel_fn(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
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
    print(f"gA: {gA}")  # gA: tensor<ptr<f16, gmem, align<128>> o (128,32,1):(32,1,0)>
    print(f"gB: {gB}")  # gB: tensor<ptr<f16, gmem, align<128>> o (128,32,1):(1,128,0)>
    print(f"gC: {gC}")  # gC: tensor<ptr<f16, gmem, align<128>> o (128,128):(128,1)>
    # gA(kTileM, kTileK, num_tile_k)
    # gB(kTileN, kTileK, num_tile_k)
    # gC(kTileM, kTileN) 

    # ///////////////////////////////////////////////////////////////////////////////
    # Create shared memory buffers
    # sA:   (BLK_M, BLK_K, PIPE)       , sB:   (BLK_N, BLK_K, PIPE)
    # /////////////////////////////////////////////////////////////////////////////// 
    # Shared memory buffer
    smem = cutlass.utils.SmemAllocator()

    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)  # (kTileM, kTileK)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)  # (kTileN, kTileK)
    print(f"sA: {sA}")
    # sA: tensor<ptr<f16, smem, align<1024>> o (128,32):(32,1)>

    # ///////////////////////////////////////////////////////////////////////////////
    # By tiled copy, get the appropriate fragments for this thread.
    # tAgA: (CPY, CPY_M, CPY_K, k)     , tBgB: (CPY, CPY_N, CPY_K, k)
    # tAsA: (CPY, CPY_M, CPY_K, PIPE)  , tBsB: (CPY, CPY_N, CPY_K, PIPE)
    # /////////////////////////////////////////////////////////////////////////////// 
    print(f"tiled_copy_A: {tiled_copy_A}")
    # tiled_copy_A: Tiled Copy
    #   Tiler MN:        (32:1,32:1)
    #   TV Layout tiled: ((4,32),8):((256,1),32)
    # Copy Atom
    #   ThrID:           1:0
    #   TV Layout Src:   (1,8):(0,1)
    #   TV Layout Dst:   (1,8):(0,1)
    #   Value type:      f16
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    print(f"thr_copy_A: {thr_copy_A}")  
    # thr_copy_A: Tiled Copy
    #   Tiler MN:        (32:1,32:1)
    #   TV Layout tiled: ((4,32),8):((256,1),32)
    # Copy Atom
    #   ThrID:           1:0
    #   TV Layout Src:   (1,8):(0,1)
    #   TV Layout Dst:   (1,8):(0,1)
    #   Value type:      f16
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA_copy = thr_copy_A.partition_S(gA)  # // (CPY, CPY_M=128/32=4, CPY_K=32/32=1, num_tile_k)
    tAsA_copy = thr_copy_A.partition_D(sA)  # // (CPY, CPY_M, CPY_K)
    tBgB_copy = thr_copy_B.partition_S(gB)
    tBsB_copy = thr_copy_B.partition_D(sB)
    print(f"tAgA_copy: {tAgA_copy}")  # tAgA_copy: tensor<ptr<f16, gmem, align<16>> o ((8,1),4,1,1):((1,0),1024,0,0)>
    print(f"tAsA_copy: {tAsA_copy}")  # tAsA_copy: tensor<ptr<f16, smem> o ((8,1),4,1):((128,0),32,0)>


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
    print(f"tAgA: {tAgA}")  # tAgA: tensor<ptr<f16, gmem, align<4>> o ((2,2,2),4,2,1):((1,256,8),1024,16,0)>
    tBgB = thr_mma.partition_B(gB)  # (MMA=(2,2), MMA_N=1, MMA_K=1, num_tile_k)
    print(f"tBgB: {tBgB}")  # 
    tCgC = thr_mma.partition_C(gC)  # (MMA=(2,2), MMA_M=1, MMA_N=1)
    print(f"tCgC: {tCgC}")  # 
    tArA = tiled_mma.make_fragment_A(tAgA[None, None, None, 0])  # (MMA, MMA_M, MMA_K)
    print(f"tArA: {tArA}")  # tArA: tensor<ptr<f16, rmem, align<16>> o ((2,2,2),4,2):((1,2,4),16,8)>
    tBrB = tiled_mma.make_fragment_B(tBgB[None, None, None, 0])  # (MMA, MMA_N, MMA_K)
    print(f"tBrB: {tBrB}")  # 
    tCrC = tiled_mma.make_fragment_C(tCgC)                       # (MMA, MMA_M, MMA_N)
    print(f"tCrC: {tCrC}")  # 
    # Clear the accumulator
    tCrC.fill(0.0)

    # ///////////////////////////////////////////////////////////////////////////////
    # SharedMem to Register Copy Atom 
    # ///////////////////////////////////////////////////////////////////////////////
    # Create the copy atoms for the copy from shared memory to register
    atom_copy_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            False, 4  # atomM=16, atomK=16, so num_matrices=4 (8x8 * 4)
        ),
        mA.element_type,
    )
    atom_copy_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            True, 4
        ),
        mB.element_type,
    )
    # Creates the tiled copy so that it matches the thread-value layout
    # expected by the tiled mma
    # 也可以使用 make_tiled_copy_tv 来创建 tiled copy，见 https://github.com/NTT123/cute-viz/blob/main/examples/ldmatrix_copy_example.py
    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
    print(f"tiled_copy_s2r_A: {tiled_copy_s2r_A}")
    # tiled_copy_s2r_A: Tiled Copy
    #   Tiler MN:        (32:1,16:1)
    #   TV Layout tiled: ((4,8,2,2),((2,2,2),(1,1))):((64,1,16,0),((32,8,256),(0,0)))
    # Copy Atom
    #   ThrID:           32:1
    #   TV Layout Src:   (32,8):(8,1)
    #   TV Layout Dst:   (32,(2,4)):(2,(1,64))
    #   Value type:      f16
    # tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

    thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
    print(f"thr_copy_ldmatrix_A: {thr_copy_ldmatrix_A}")
    # thr_copy_ldmatrix_A: Tiled Copy
    #   Tiler MN:        (32:1,16:1)
    #   TV Layout tiled: ((4,8,2,2),((2,2,2),(1,1))):((64,1,16,0),((32,8,256),(0,0)))
    # Copy Atom
    #   ThrID:           32:1
    #   TV Layout Src:   (32,8):(8,1)
    #   TV Layout Dst:   (32,(2,4)):(2,(1,64))
    #   Value type:      f16
    # thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)  # // (CPY, CPY_M, CPY_K)
    print(f"tCsA_copy_view: {tCsA_copy_view}")
    # tCsA_copy_view: tensor<ptr<f16, smem, align<16>> o ((8,1),4,2):((1,0),1024,16)>
    # 由于MMA时已经声明了寄存器存储空间，此处直接对其进行线程级小块的retile即可，不再是大块到小块的partition。
    tCrA_copy_view = thr_copy_ldmatrix_A.retile(tArA)  # // (CPY, CPY_M, CPY_K)
    print(f"tCrA_copy_view: {tCrA_copy_view}")
    # tCrA_copy_view: tensor<ptr<f16, rmem, align<16>> o ((8,1),4,2):((1,0),16,8)>

    # tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
    # tCrB_copy_view = thr_copy_ldmatrix_B.retile(tBrB)

    # ///////////////////////////////////////////////////////////////////////////////
    # global <-> register universal copy
    # ///////////////////////////////////////////////////////////////////////////////
    load_copy_atom= cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    store_copy_atom= cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    num_tile_k = gA.shape[2]
    for tile_k_index in range(num_tile_k):
        # cute.copy(load_copy_atom, tAgA[None, None, None, tile_k_index], tArA)
        cute.copy(tiled_copy_A, tAgA_copy[None, None, None, tile_k_index], tAsA_copy[None, None, None])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(n=0)
        cute.arch.sync_threads()

        cute.copy(
            tiled_copy_s2r_A,
            tCsA_copy_view[None, None, None],
            tCrA_copy_view[None, None, None],
        )

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