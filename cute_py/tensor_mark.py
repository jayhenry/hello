import torch
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch
import cutlass

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
print(f"a's torch dim_order: {a.dim_order()}")  # (2, 1, 0, 3)

# (5,6,7) -> (6,7,5)
b = torch.rand((5,6,7)).permute(1,2,0)
# dim_order() 表示stride从大到小的维度顺序，(2,0,1) 表示stride最大的维度是2，其次是0，最后是1
print(f"b's torch dim_order: {b.dim_order()}, shape from (5,6,7) to {b.shape}")  # (2, 0, 1)
# The order of modes (dimensions) if the current layout were to be converted to row-major order. 
# It starts from the outermost to the innermost dimension when reading it from left to right. 

# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
# b.dim_order() is (3,2,4,0,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)

# The mode parameter determines which shape dimension becomes dynamic. 
# After calling this function, the specific shape dimension given by mode is marked as dynamic immediately. 
# The stride will be updated accordingly. For modes that have a shape of size 1, their stride are canonicalized to 0.
t = from_dlpack(a).mark_compact_shape_dynamic(
    mode=0
    # auto deduce the stride order to be [2,1,0,3]
    # stride_order=None,
)
# Tensor<0x0000000009280140@generic o (?,4,16,2):(2,?{div=2},?{div=8},1)>
print(t)

# The stride_order parameter specifies the ordering of strides in the tensor. It is consistent with torch.Tensor.dim_order()
# If stride_order is not specified, the system automatically deduces it
t0_1 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=0, 
    stride_order=(2, 1, 0, 3)
)
print(t0_1)


# The divisibility parameter specifies the divisibility of the dynamic shape. 
t0 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=0, divisibility=2
)
# (?{div=2},4,16,2):(2,?{div=4},?{div=16},1)
print(t0)

# t6 = from_dlpack(a).mark_compact_shape_dynamic(
#     mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
# )
# print(t6)
# RuntimeError: The stride_order is not consistent with the deduced stride_order


l, mode0, mode1 = 2, 3, 4
dtype = cutlass.Float16
is_mode0_major = False

shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
torch_tensor = (
    torch.empty(*shape, dtype=torch.int32)
    .random_(-2, 2)
    .to(dtype=cutlass_torch.dtype(dtype))
    .permute(permute_order)
    .cuda()
)
# torch_tensor's dim_order: (2, 0, 1), origin shape: (2, 3, 4), after permute shape: torch.Size([3, 4, 2])
print(f"torch_tensor's dim_order: {torch_tensor.dim_order()}, origin shape: {shape}, after permute shape: {torch_tensor.shape}")

# assume input is 16B aligned
cute_tensor = (
    from_dlpack(torch_tensor, assumed_align=16)
    .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
    .mark_compact_shape_dynamic(
        mode=(1 if not is_mode0_major else 0),
        # https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#mark-the-tensor-s-layout-as-dynamic-with-mark-compact-shape-dynamic
        stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
        # divisibility=(128 // dtype.width),
    )
)
print(cute_tensor)