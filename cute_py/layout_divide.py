import cutlass.cute as cute


@cute.jit
def logical_divide_1d_example():
    """
    Demonstrates 1D logical divide
    """
    # Define the original layout
    layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))  # (4,2,3):(2,1,8)
    
    # Define the tiler
    tiler = cute.make_layout(4, stride=2)  # Apply to layout 4:2
    
    # Apply logical divide
    result = cute.logical_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print("After the divide, the first mode of the result is the tile of data and the second mode of the result iterates over each tile.")
    print(">>> Logical Divide Result:", result)  # ((2,2),(2,3)):((4,1),(2,8))
    cute.printf(">?? Logical Divide Result: {}", result)


@cute.jit
def logical_divide_2d_example():
    """
    Demonstrates 2D logical divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : ((TileM,RestM), (TileN,RestN), L, ...)
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply logical divide
    result = cute.logical_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(f">>> Tiler : {tiler[0]} and {tiler[1]}")
    print("After the divide, the first mode of each mode of the result is the tile of data and "
          "the second mode of each mode iterates over each tile.") 
    print("In that sense, this operation can be viewed as a kind of gather operation or as simply a permutation on the rows and cols.")
    print(">>> Logical Divide Result:", result)
    cute.printf(">?? Logical Divide Result: {}", result)


@cute.jit
def zipped_divide_example():
    """
    Demonstrates zipped divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : ((TileM,TileN), (RestM,RestN,L,...))
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply zipped divide
    result = cute.zipped_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Zipped Divide Result:", result)
    cute.printf(">?? Zipped Divide Result: {}", result)


@cute.jit
def tiled_divide_example():
    """
    Demonstrates tiled divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : ((TileM,TileN), RestM, RestN, L, ...)
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply tiled divide
    result = cute.tiled_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Tiled Divide Result:", result)
    cute.printf(">?? Tiled Divide Result: {}", result)


@cute.jit
def flat_divide_example():
    """
    Demonstrates flat divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : (TileM, TileN, RestM, RestN, L, ...)
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply flat divide
    result = cute.flat_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Flat Divide Result:", result)
    cute.printf(">?? Flat Divide Result: {}", result)


if __name__ == "__main__":
    # logical_divide_1d_example()
    # logical_divide_2d_example()
    # zipped_divide_example()
    # tiled_divide_example()
    flat_divide_example()

