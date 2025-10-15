import cutlass
import cutlass.cute as cute


@cute.jit
def coalesce_example():
    """
    Demonstrates coalesce operation flattening and combining modes
    """
    layout = cute.make_layout((2, (1, 6)), stride=(1, (cutlass.Int32(6), 2))) # Dynamic stride
    result = cute.coalesce(layout)

    print(f">>> Original  with type {type(layout)}: {layout}")
    cute.printf(">?? Original: {}", layout)
    print(f">>> Coalesced with type {type(result)}: {result}")
    cute.printf(">?? Coalesced: {}", result)

coalesce_example()