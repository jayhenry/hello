import cutlass
from cutlass import cute

@cute.jit
def control_flow_examples1(bound: cutlass.Int32):
    n = 3

    # ✅ This loop is Python loop, evaluated at compile time.
    for i in cutlass.range_constexpr(n):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, even when bound is Python value.
    for i in range(n):
        cute.printf("%d\\n", i)


@cute.jit
def control_flow_examples2(bound: cutlass.Int32):
    # ❌ This loop bound is a dynamic value, not allowed in Python loop.
    # Should use `range` instead.
    for i in cutlass.range_constexpr(bound):
        cute.printf("%d\\n", i)


@cute.jit
def control_flow_examples3(bound: cutlass.Int32):
    # ✅ This loop is dynamic, emitted IR loop.
    for i in range(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop with unrolling
    for i in cutlass.range(bound, unroll=2):
        cute.printf("%d\\n", i)


# control_flow_examples1(5)
# control_flow_examples2(5)
control_flow_examples3(5)
