from typing import List

import cutlass
import cutlass.cute as cute

@cute.jit
def bar():
    a = cutlass.Float32(3.14)
    print("a(static) =", a)             # prints `a(static) = ?`
    cute.printf("a(dynamic) = {}", a)   # prints `a(dynamic) = 3.140000`

    b = cutlass.Int32(5)
    print("b(static) =", b)             # prints `b(static) = ?`
    cute.printf("b(dynamic) = {}", b)   # prints `b(dynamic) = 5`


@cute.jit
def type_conversion():
    # Convert from Int32 to Float32
    x = cutlass.Int32(42)
    y = x.to(cutlass.Float32)
    cute.printf("Int32({}) => Float32({})", x, y)

    # Convert from Float32 to Int32
    a = cutlass.Float32(3.14)
    b = a.to(cutlass.Int32)
    cute.printf("Float32({}) => Int32({})", a, b)

    # Convert from Int32 to Int8
    c = cutlass.Int32(127)
    d = c.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({})", c, d)

    # Convert from Int32 to Int8 with value exceeding Int8 range
    e = cutlass.Int32(300)
    f = e.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({}) (truncated due to range limitation)", e, f)


@cute.jit
def operator_demo():
    # Arithmetic operators
    a = cutlass.Int32(10)
    b = cutlass.Int32(3)
    cute.printf("a: Int32({}), b: Int32({})", a, b)

    x = cutlass.Float32(5.5)
    cute.printf("x: Float32({})", x)

    cute.printf("")

    sum_result = a + b
    cute.printf("a + b = {}", sum_result)

    y = x * 2  # Multiplying with Python native type
    cute.printf("x * 2 = {}", y)

    # Mixed type arithmetic (Int32 + Float32) that integer is converted into float32
    mixed_result = a + x
    cute.printf("a + x = {} (Int32 + Float32 promotes to Float32)", mixed_result)

    # Division with Int32 (note: integer division)
    div_result = a / b
    cute.printf("a / b = {}", div_result)

    # Float division
    float_div = x / cutlass.Float32(2.0)
    cute.printf("x / 2.0 = {}", float_div)

    # Comparison operators
    is_greater = a > b
    cute.printf("a > b = {}", is_greater)

    # Bitwise operators
    bit_and = a & b
    cute.printf("a & b = {}", bit_and)

    neg_a = -a
    cute.printf("-a = {}", neg_a)

    not_a = ~a
    cute.printf("~a = {}", not_a)


if __name__ == "__main__":
    bar()
    # type_conversion()
    # operator_demo()