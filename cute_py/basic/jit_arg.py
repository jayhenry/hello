import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack


# Example1: static and dynamic arguments
@cute.jit
def foo1(x: cutlass.Int32, y: cutlass.Constexpr):
    print("x = ", x)        # Prints x = ?
    print("y = ", y)        # Prints y = 2
    cute.printf("x: {}", x) # Prints x: 2
    cute.printf("y: {}", y) # Prints y: 2


# foo1(2, 2)


# Example2: typing check
@cute.jit
def foo2(x: cute.Tensor, y: cutlass.Float16):
    pass

a = np.random.randn(10, 10).astype(np.float16)
b = 32

# 将 numpy 数组转换为 cute.Tensor
a_tensor = from_dlpack(a)
foo2(a_tensor, b)
# foo2(b, a)  # This will fail at compile time due to type mismatch
