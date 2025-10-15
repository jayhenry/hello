import cutlass
import cutlass.cute as cute


# Example1: Constexpr argument is only used at compile time, not passed in at runtime
print("Example1:".center(50, "="))
@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
   if print_result:
      cute.printf("Result: %d\n", a + b)
   return a + b


jit_executor = cute.compile(add, 1, 2, True)

# Constexpr argument is only used at compile time, not passed in at runtime
# Runtime arguments are converted to C ABI-compatible types according to argument specifications
# Then JIT executor instance Invokes the host function with the converted arguments
jit_executor(1, 2) # output: ``Result: 3``


# Example2: Custom cache
print("Example2:".center(50, "="))
@cute.jit
def add(b):
   res = a + b
   cute.printf("Result: %d\n", res)
   return res

# Define a custom cache
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
print("compiled_add_1(2) =")
compiled_add_1(2) # result = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
print("compiled_add_2(2) =")
compiled_add_2(2) # result = 4

# Use the custom cache
print("custom_cache[1](2) =")
custom_cache[1](2) # result = 3
print("custom_cache[2](2) =")
custom_cache[2](2) # result = 4
