"""
https://docs.python.org/3/library/asyncio.html
"""
import asyncio
import time

async def main():
    time.sleep(1)
    print('hello')
    return 1


def method0():
    # 错误用法，协程对象不会立即执行，需要通过 asyncio.run() 或者 runner.run() 来执行
    # 执行后会：RuntimeWarning: coroutine 'main' was never awaited
    res = main()
    print("res:", res)

def method1():
    res = asyncio.run(main())
    print("res:", res)

def method2():
    # Added in Python version 3.11
    with asyncio.Runner() as runner:
        runner.run(main())

def method3():
    async def wrapper():
        # 在协程中使用 await 把控制权交还给事件循环，来执行另一个协程main()
        res = await main()
        return res
    res = wrapper()
    print("res:", res)

if __name__ == "__main__":
    method3()
