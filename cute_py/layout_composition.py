import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
from cutlass.torch import dtype as torch_dtype


@cute.jit
def composition_example():
    """
    Demonstrates basic layout composition R = A ◦ B
    """
    A = cute.make_layout((6, 2), stride=(cutlass.Int32(8), 2)) # Dynamic stride
    B = cute.make_layout((4, 3), stride=(3, 1))
    R = cute.composition(A, B)

    # Print static and dynamic information
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Layout B:", B) 
    cute.printf(">?? Layout B: {}", B)
    print(">>> Composition R = A ◦ B:", R)
    cute.printf(">?? Composition R: {}", R)


@cute.jit
def composition_static_vs_dynamic_layout():
    """
    Shows difference between static and dynamic composition results
    """
    # Static version - using compile-time values
    A_static = cute.make_layout((10,2), stride=(16,4))
    B_static = cute.make_layout((5,4), stride=(1,5))
    R_static = cute.composition(A_static, B_static)

    print(">>> Static version:")
    print(">>> Layout A:", A_static)
    print(">>> Layout B:", B_static)
    print(">>> Composition R = A ◦ B:", R_static)

    # Dynamic version - using runtime Int32 values
    A_dynamic = cute.make_layout((cutlass.Int32(10), cutlass.Int32(2)), stride=(cutlass.Int32(16), cutlass.Int32(4)))
    B_dynamic = cute.make_layout((cutlass.Int32(5), cutlass.Int32(4)), stride=(cutlass.Int32(1), cutlass.Int32(5)))
    R_dynamic = cute.composition(A_dynamic, B_dynamic)

    cute.printf(">?? Dynamic version:")
    cute.printf(">?? Layout A: {}", A_dynamic)
    cute.printf(">?? Layout B: {}", B_dynamic)
    cute.printf(">?? Composition R = A ◦ B: {}", R_dynamic)


@cute.jit
def bymode_composition_example():
    """
    Demonstrates by-mode composition using a tiler
    """
    # Define the original layout A
    A = cute.make_layout(
        (cutlass.Int32(12), (cutlass.Int32(4), cutlass.Int32(8))), 
        stride=(cutlass.Int32(59), (cutlass.Int32(13), cutlass.Int32(1)))
    )

    # Define the tiler for by-mode composition
    # tiler = (3, 8) # Apply 3:1 to mode-0 and 8:1 to mode-1
    tiler = (
        cute.make_layout((3,), stride=(4,)), 
        cute.make_layout((8,), stride=(2,)),
    ) # Apply 3:4 to mode-0 and 8:2 to mode-1

    # Apply by-mode composition
    result = cute.composition(A, tiler)

    # Print static and dynamic information
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Tiler:", tiler)
    cute.printf(">?? Tiler: {} and {}", tiler[0], tiler[1])
    print(">>> By-mode Composition Result:", result)
    cute.printf(">?? By-mode Composition Result: {}", result)  # (3,(2,4)):(236,(26,1))
    cute.printf(">?? By-mode Composition Result after coalesce: {}", cute.coalesce(result))


if __name__ == "__main__":
    # composition_example()
    # composition_static_vs_dynamic_layout()
    bymode_composition_example()