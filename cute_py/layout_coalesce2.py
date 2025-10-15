import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import dtype as torch_dtype
import cutlass.cute.runtime as cute_rt


@cute.jit
def coalesce_post_conditions(ptr: cute.Pointer):
    """
    Demonstrates coalesce operation's 3 post-conditions:
    1. size(@a result) == size(@a layout)
    2. depth(@a result) <= 1
    3. for all i, 0 <= i < size(@a layout), @a result(i) == @a layout(i)
    """
    layout = cute.make_layout(
        (2,(3, 2)),
        stride=(4, (2, 6))
    )
    tensor = cute.make_tensor(ptr, layout)
    cute.printf(">?? Original tensor = {}", tensor)
    cute.print_tensor(tensor, verbose=False)
    cute.print_tensor(tensor, verbose=True)
    # addr(i0,i1,i2) = i0*4 + i1*2 + i2*6
    result = cute.coalesce(layout)
    result_tensor = cute.make_tensor(ptr, result)
    cute.printf(">?? Coalesced tensor = {}", result_tensor)
    cute.print_tensor(result_tensor, verbose=False)
    cute.print_tensor(result_tensor, verbose=True)

    print(">>> Original:", layout)  # (2,(3,2)):(4,(2,6))
    cute.printf(">?? Original: {}", layout)
    # cute.print_layout(layout)
    # cute.print_tensor(layout, verbose=True)
    print(">>> Coalesced:", result)  # (2,6):(4,2)
    cute.printf(">?? Coalesced: {}", result)
    # cute.print_layout(result)
    # cute.print_tensor(result, verbose=True)

    print(">>> Checking post-conditions:")
    print(">>> 1. Checking size remains the same after the coalesce operation:")
    original_size = cute.size(layout)
    coalesced_size = cute.size(result)
    print(f"Original size: {original_size}, Coalesced size: {coalesced_size}")
    assert coalesced_size == original_size, \
            f"Size mismatch: original {original_size}, coalesced {coalesced_size}"
    
    print(">>> 2. Checking depth of coalesced layout <= 1:")
    depth = cute.depth(result)
    print(f"Depth of coalesced layout: {depth}")
    assert depth <= 1, f"Depth of coalesced layout should be <= 1, got {depth}"

    print(">>> 3. Checking layout functionality remains the same after the coalesce operation:")
    for i in cutlass.range_constexpr(original_size):
        original_value = layout(i)
        coalesced_value = result(i)
        print(f"Index {i}: original {original_value}, coalesced {coalesced_value}")
        assert coalesced_value == original_value, \
            f"Value mismatch at index {i}: original {original_value}, coalesced {coalesced_value}"


a = torch.arange(0, 15, dtype=torch_dtype(cutlass.Float32))
ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())
coalesce_post_conditions(ptr_a)