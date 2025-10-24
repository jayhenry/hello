import cutlass.cute as cute


@cute.jit
def example1():
    # (2,0,1) 表示 第0维排序2， 第1维排序0， 第2维排序1。所以 stride=(128,1,16)
    layout = cute.make_ordered_layout((32,16,8), order=(2,0,1))   # stride=(128,1,16)
    print(layout)


if __name__ == "__main__":
    example1()