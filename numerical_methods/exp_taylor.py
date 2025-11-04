import math

def myexp(x):
    y = -1.  # 中间变量，保存上一次的结果
    newy = 1.  # 保存当前的结果
    term = 1  # 保存当前的Taylor项
    k = 0  # 保存当前的项数
    while newy != y:
        k = k+1  # 更新项数
        term = (term * x)/k  # 更新Taylor项
        y = newy  # 保存上一次的结果
        # 对于x=-20的情况，term有时正有时负，出现了减法，可能出现 catastrophic cancellation，从而导致较大误差
        newy = y + term  # 计算当前的结果
    return newy


print(f"myexp(-20.0) = {myexp(-20.0):.8e}")
print(f"1/myexp(20.0) = {1/myexp(20.0):.8e}")
print(f"math.exp(-20.0) = {math.exp(-20.0):.8e}")

def calculate_condition_number(func, x, delta_x=1e-5):
    y = func(x)
    delta_y = func(x + delta_x) - y
    absolute_condition_number = abs(delta_y / delta_x)
    relative_condition_number = abs(delta_y / y) / abs(delta_x / x)
    return {
        "absolute_condition_number": absolute_condition_number,
        "relative_condition_number": relative_condition_number
    }

print(f"condition number of myexp(-20.0) = {calculate_condition_number(myexp, -20.0)}")
print(f"condition number of 1/myexp(20.0) = {calculate_condition_number(lambda x: 1/myexp(x), 20.0)}")
print(f"condition number of math.exp(-20.0) = {calculate_condition_number(math.exp, -20.0)}")
