from ctypes import *

# 1. 加载C库
lib = CDLL('./libpoint.so')

# 2. 定义与C结构体内存布局完全对应的Python类
class Point(Structure):
    _fields_ = [('x', c_int),
                ('y', c_int)]

# 3. 告诉ctypes函数的参数和返回类型
lib.add_points.argtypes = [POINTER(Point), POINTER(Point), POINTER(Point)]
lib.add_points.restype = None

# 4. 创建Point结构体
p1 = Point(1, 2)
p2 = Point(3, 4)
result = Point()

# 5. 直接调用C函数！Python和C共享同一块内存。
lib.add_points(byref(result), byref(p1), byref(p2))

print(f"Result: ({result.x}, {result.y})") # 输出: Result: (4, 6)
