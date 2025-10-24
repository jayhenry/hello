import cutlass
import cutlass.cute as cute


@cute.jit
def logical_product_1d_example():
    """
    Demonstrates 1D logical product
    """
    # Define the original layout
    layout = cute.make_layout((2, 2), stride=(4, 1))  # (2,2):(4,1)
    
    # Define the tiler
    tiler = cute.make_layout(6, stride=1)  # Apply to layout 6:1
    
    # Apply logical product
    result = cute.logical_product(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Product Result:", result)
    cute.printf(">?? Logical Product Result: {}", result)


@cute.jit
def blocked_raked_product_example():
    """
    Demonstrates blocked and raked products
    """
    # Define the original layout
    layout = cute.make_layout((2, 5), stride=(5, 1))
    
    # Define the tiler
    tiler = cute.make_layout((3, 4), stride=(1, 3))
    
    # Apply blocked product
    blocked_result = cute.blocked_product(layout, tiler=tiler)

    # Apply raked product
    raked_result = cute.raked_product(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Blocked Product Result:", blocked_result)
    print(">>> Raked Product Result:", raked_result)
    cute.printf(">?? Blocked Product Result: {}", blocked_result)
    cute.printf(">?? Raked Product Result: {}", raked_result)


@cute.jit
def zipped_tiled_flat_product_example():
    """
    Demonstrates zipped, tiled, and flat products
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>

    blocked_product : ((M,TileM), (N,TileN), L, ...)
    zipped_product  : ((M,N), (TileM,TileN,L,...))
    tiled_product   : ((M,N), TileM, TileN, L, ...)
    flat_product    : (M, N, TileM, TileN, L, ...)
    """
    # Define the original layout
    M, N = 2, 5
    layout = cute.make_layout((M, N), stride=(N, 1))
    
    # Define the tiler
    TileM, TileN = 3, 4
    tiler = cute.make_layout((TileM, TileN), stride=(1, TileM))

    # Apply blocked product
    blocked_result = cute.blocked_product(layout, tiler=tiler)

    # Apply zipped product
    zipped_result = cute.zipped_product(layout, tiler=tiler)
    
    # Apply tiled product
    tiled_result = cute.tiled_product(layout, tiler=tiler)
    
    # Apply flat product
    flat_result = cute.flat_product(layout, tiler=tiler)

    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Blocked Product Result:", blocked_result)
    print(">>> Zipped Product Result:", zipped_result)
    print(">>> Tiled Product Result:", tiled_result)
    print(">>> Flat Product Result:", flat_result)
    cute.printf(">?? Blocked Product Result: {}", blocked_result)
    cute.printf(">?? Zipped Product Result: {}", zipped_result)
    cute.printf(">?? Tiled Product Result: {}", tiled_result)
    cute.printf(">?? Flat Product Result: {}", flat_result)

    # check blocked_result Layout == zipped_result
    for m in cutlass.range_constexpr(M):
        for n in cutlass.range_constexpr(N):
            for tm in cutlass.range_constexpr(TileM):
                for tn in cutlass.range_constexpr(TileN):
                    blocked_index = blocked_result(((m, tm), (n, tn)),)
                    zipped_index = zipped_result(((m, n), (tm, tn)), )
                    print(f"coordinate: (m,n,tm,tn) = {(m, n, tm, tn)}, blocked_product's index: {blocked_index}, zipped_product's index: {zipped_index}")
                    assert blocked_index == zipped_index, \
                        f"Blocked product result at index {blocked_index} is not equal to zipped product result at index {zipped_index}"
                    tiled_index = tiled_result(((m, n), tm, tn),)
                    flat_index = flat_result((m, n, tm, tn),)
                    # print(f"coordinate: (m,n,tm,tn) = {(m, n, tm, tn)}, tiled_product's index: {tiled_index}, flat_product's index: {flat_index}")
                    assert tiled_index == flat_index, \
                        f"Tiled product result at index {tiled_index} is not equal to flat product result at index {flat_index}"


if __name__ == "__main__":
    logical_product_1d_example()
    # blocked_raked_product_example()
    # zipped_tiled_flat_product_example()