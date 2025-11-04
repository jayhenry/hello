first = []
a=1.0
while a > 0.0:
    a = a/2.0
    first.append(a)

second = []
a=1.0
while a+1.0 > 1.0:
    a = a/2.0
    second.append(a)

"""
IEEE 754 双精度浮点数格式
类型	          数值	                 近似值
最大正数	(2 - 2⁻⁵²) × 2¹⁰²³	1.7976931348623157 × 10³⁰⁸
最小正规格化数	2⁻¹⁰²²	2.2250738585072014 × 10⁻³⁰⁸
最大正非规格化数	(1 - 2⁻⁵²) × 2⁻¹⁰²²	2.2250738585072009 × 10⁻³⁰⁸
最小正非规格化数	2⁻¹⁰⁷⁴	4.9406564584124654 × 10⁻³²⁴
"""
print(f'len(first) = {len(first)}')
print(f'first[-1] = {first [ -1]:.8e}')  # 0e0
print(f'first[-2] = {first [ -2]:.8e}')  # 4.94065646e-324
print(f'len(second) = {len(second)}')
print(f'second[-1] = {second [ -1]:.8e}')  # 1.11022302e-16
print(f'second[-2] = {second [ -2]:.20e}')  # 2.22044604925031308085e-16 == 2**-52