import cutlass
from cutlass import cute


@cute.jit
def main1(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ This branch is Python branch, evaluated at compile time.
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\\n")
    else:
        cute.printf("Const else\\n")

    # ✅ This branch is dynamic branch, emitted IR branch.
    if dynamic_var == 10:
        cute.printf("Dynamic True\\n")
    else:
        cute.printf("Dynamic False\\n")


@cute.jit
def main2(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    if cutlass.const_expr(dynamic_var == 10):
        cute.printf("Bound is 10\\n")


# main1(True, 10)
# main1(False, 3)
main2(True, 10)
