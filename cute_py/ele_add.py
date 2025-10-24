import torch
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M, N = 2048, 2048


@cute.kernel
def naive_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    tid = bidx * bdimx + tidx
    M, N = a.shape
    m = tid // N
    n = tid % N

    # maps thread indices to contiguous tensor dimensions for coalesced memory access
    c[m, n] = a[m, n] + b[m, n]

    # not map to contiguous tensor dimensions, may lead to poor performance, but still correct
    # c[n, m] = a[n, m] + b[n, m]  # 512 (220 GB/s)
    # tensor layout's linear index is column-major, but memory is row-major, so it may lead to poor performance
    # c[tid] = a[tid] + b[tid]  # 512 (220 GB/s)


@cute.jit
def naive_elementwise_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    BLOCK_SIZE = 512  # 128 (1145 GB/s), 256 (2051 GB/s), 384 (2694 GB/s), 512 (2707 GB/s), 640 (2416 GB/s), 1024 (2288 GB/s)
    print(f"BLOCK_SIZE: {BLOCK_SIZE}")
    block = (BLOCK_SIZE, 1, 1)
    grid = ((cute.size(a) + block[0] - 1) // block[0], 1, 1)
    naive_kernel(a, b, c).launch(grid=grid, block=block)


def naive():
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    
    # Compile kernel
    naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)
    naive_elementwise_add_(a_, b_, c_)
    
    # verify correctness
    torch.testing.assert_close(c, a + b)

    # benchmark
    benchmark(partial(naive_elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=100, a=a)


def benchmark(callable, *, num_warmups, num_iterations, a):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    for _ in range(num_warmups):
        callable()

    start_event.record(stream=torch.cuda.current_stream())
    for _ in range(num_iterations):
        callable()
    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iterations

    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Throughput: {(3 * a.numel() * 2) / (avg_time / 1000) / 1e9:.2f} GB/s")


@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    tid = bidx * bdimx + tidx
    M, N = gA.shape[1]
    m = tid // N
    n = tid % N
    if m >= M: 
        pass
        # DSLAstPreprocessorError: Early exit (return) is not allowed.
        # If predicates are constant expression, write like `if const_expr(...)` or `for ... in range_constexpr(...)`. 
        # In that case, early exit will be executed by Python interpreter, so it's supported.
        # return  
    else:
        rA = gA[(None, (m, n))].load()
        rB = gB[(None, (m, n))].load()
        print(f"[DSL INFO] sliced gA = {gA[(None, (m, n))]}")
        print(f"[DSL INFO] sliced gB = {gB[(None, (m, n))]}")
        # rC = rA + rB
        # gC[(None, (m, n))].store(rC)
        gC[(None, (m, n))] = rA + rB


@cute.jit
def vectorized_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    BLOCK_SIZE = 384  # 256 (4766 GB/s), 384 (4638 GB/s), 512 (4853 GB/s), 640 (4617 GB/s)
    print(f"BLOCK_SIZE: {BLOCK_SIZE}")
    tensor_ssa_size = 4
    assert BLOCK_SIZE % tensor_ssa_size == 0

    # return: ((TileM,TileN), (RestM,RestN,L,...))
    gA = cute.zipped_divide(mA, (1, tensor_ssa_size))
    gB = cute.zipped_divide(mB, (1, tensor_ssa_size))
    gC = cute.zipped_divide(mC, (1, tensor_ssa_size))

    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=((cute.size(gC, mode=[1]) + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
        block=(BLOCK_SIZE, 1, 1),
    )


def vectorized():
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    
    compiled_func = cute.compile(vectorized_elementwise_add, a_, b_, c_)
    compiled_func(a_, b_, c_)
    
    # verify correctness
    torch.testing.assert_close(c, a + b)

    # benchmark
    benchmark(partial(compiled_func, a_, b_, c_), num_warmups=5, num_iterations=100, a=a)


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    #--------------------------------
    # slice for thread-block level view
    #--------------------------------
    # ((TileM, TileN), (RestM, RestN)) -> (TileM, TileN)
    # gA: ((16,256),(128,8)):((2048,1),(32768,256))
    #         256                 256  共8组，共2048个元素
    #        --------------------------
    #    16  | b0 |    |    |    |    |
    #        --------------------------
    #        | b1 |    |    |    |    |
    #        --------------------------
    #        |    |    |    |    |    |
    #        --------------------------
    #    16  |    |    |    |    |    |
    #        --------------------------
    #   共128组，共2048个元素
    # 
    block_coord = (None, bidx)
    blockA = gA[block_coord]
    blockB = gB[block_coord]
    blockC = gC[block_coord]
    print("blockA = {}", blockA.type)

    #--------------------------------
    # slice for thread level view
    #--------------------------------
    # First, compose for thread-index & value-index to physical mapping
    # blockA:    (TileM, TileN) -> physical address
    # tv_layout: (tid, vid)     -> (TileM, TileN)
    # tidfrgA = blkA o tv_layout
    # tidfrgA:   (tid, vid) -> physical address

    # tv_layout: (thread_idx, value_idx) = ((32,4),(8,4)):((128,4),(16,1))
    tidfrgA = cute.composition(blockA, tv_layout)
    tidfrgB = cute.composition(blockB, tv_layout)
    tidfrgC = cute.composition(blockC, tv_layout)
    print("tidfrgA = {}", tidfrgA.type)
    # tidfrgA = {} !cute.memref<f16, gmem, align<16>, "((32,4),(8,4)):((8,8192),(1,2048))">
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
    # 
    # Then, slice for thread-level view
    # ((32,4),(8,4)):((128,4),(16,1)) -> (8,4)
    # 由于安培架构GPU每次加载/存储操作最高支持128位，而每个元素为16位，我们可以在连续行上通过单次向量化操作加载8个元素
    thread_coord = (tidx, None)
    frgA = tidfrgA[thread_coord]
    frgB = tidfrgB[thread_coord]
    frgC = tidfrgC[thread_coord]
    print("frgA = {}", frgA.type)
    frgC.store(frgA.load() + frgB.load())


@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # mA layout: (M, N):(N, 1)
    # TV layout map thread & value index to (16, 256) logical tile
    #  - contiguous thread index maps to mode-1 because input layout is contiguous on
    #     mode-1 for coalesced load-store
    #  - each thread load 8 contiguous element each row and load 4 rows

    # 4426 GB/s
    # 4540.08 GB/s, 4566.70 GB/s, 4898.97 GB/s, 4262.27 GB/s, 4427.11 GB/s
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")  # (16, 256)
    print(f"thr_layout: {thr_layout}")
    print(f"val_layout: {val_layout}")
    print(f"TV Layout: {tv_layout}")  # ((32,4),(8,4)):((128,4),(16,1))

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print(f"Tiled Input Tensors:")
    print(f"  gA: {gA.type}")
    print(f"  gB: {gB.type}")
    print(f"  gC: {gC.type}")

    # Launch the kernel asynchronously
    # Async token(s) can also be specified as dependencies
    elementwise_add_kernel(
        gA, gB, gC, tv_layout
    ).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.kernel
def elementwise_add_mytv_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # ((TileM, TileN), (RestM, RestN))
    #     (16,256),       (128,8)
    print(f"gA.shape: {gA.shape}")
    restM, restN = gA.shape[1]  # for rest m n
    m = bidx // restN
    n = bidx % restN
    if m >= restM:
        pass
    else:
        # slice for block-level
        # ((TileM, TileN), (RestM, RestN)) -> (TileM, TileN) -> ((ThreadN, ThreadM), (ValueN, ValueM))
        # cute.printf("gA = {}", gA)
        print(f"gA.type = {gA.type}")  # ((16,256),(128,8)):((2048,1),(32768,256))
        sA = cute.composition(gA[(None, (m,n))], tv_layout)
        # cute.printf("sA = {}", sA)
        print(f"sA.type: {sA.type}")  # ((32,8),(8,2)):((8,4096),(1,2048))
        sB = cute.composition(gB[(None, (m,n))], tv_layout)
        sC = cute.composition(gC[(None, (m,n))], tv_layout)
        # slice for thread-level 
        # ((ThreadN, ThreadM), (ValueN, ValueM)) -> (ValueN, ValueM)
        # cute.printf("sA[(tidx, None)] = {}", sA[(tidx, None)])
        rA = sA[(tidx, None)]
        print(f"rA.type: {rA.type}")  # ((8,2)):((1,2048))
        rB = sB[(tidx, None)]
        rC = sC[(tidx, None)]
        rC.store(rA.load() + rB.load())


@cute.jit
def elementwise_add_mytv(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # cute.printf("--------- Start elementwise_add_mytv ---------")

    # 4508.58 GB/s, 3916.30 GB/s, 3892.65 GB/s, 4712.56 GB/s
    # 4717.93 GB/s, 4548.74 GB/s, 4699.61 GB/s, 4457.98 GB/s, 4800.00 GB/s
    M, N = mA.shape
    value_layout = cute.make_layout((2,8), stride=(8,1))
    thread_layout = cute.make_layout((8,32), stride=(32,1))
    tiler, tv_layout = cute.make_layout_tv(thread_layout, value_layout)
    # (2x8)x(8x32) = 16x256  
    print(f">>> Tiler type({type(tiler)}): {tiler}")  # tuple (16, 256)
    # compose tileA and tv_layout to get: ((Thread_n, Thread_m), (Value_n, Value_m)) 
    # Attention!!! nD is transposed to the first mode, and mD is transposed to the second mode.
    # This is for the ease of 1D coordinate to natural coordinate mapping.
    print(f">>> TV Layout: {tv_layout}")  # ((32,8),(8,2)):((128,2),(16,1))
    # gA: ((TileM, TileN), (RestM, RestN))
    gA = cute.zipped_divide(mA, tiler)  # by mode divide
    gB = cute.zipped_divide(mB, tiler)  # by mode divide
    gC = cute.zipped_divide(mC, tiler)  # by mode divide
    elementwise_add_mytv_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gA, mode=[1]), 1, 1],
        # block=[cute.size(thread_layout), 1, 1],  # both are correct
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def tvlayout():
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    
    # elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
    elementwise_add_ = cute.compile(elementwise_add_mytv, a_, b_, c_)
    elementwise_add_(a_, b_, c_)
    
    # verify correctness
    torch.testing.assert_close(c, a + b)

    # benchmark
    benchmark(partial(elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=100, a=a)


if __name__ == "__main__":
    # naive()
    # vectorized()
    tvlayout()
