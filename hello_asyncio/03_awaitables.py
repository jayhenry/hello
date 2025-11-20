"""
https://docs.python.org/3/library/asyncio-task.html#awaitables

We say that an object is an awaitable object if it can be used in an await expression. Many asyncio APIs are designed to accept awaitables.
There are three main types of awaitable objects: coroutines, Tasks, and Futures.
"""
import asyncio
import time


async def nested():
    return 42

async def main():
    # Nothing happens if we just call "nested()".
    # A coroutine object is created but not awaited,
    # so it *won't run at all*.
    nested()  # will raise a "RuntimeWarning".

    # Let's do it differently now and await it:
    coroutine = nested()
    res = await coroutine
    print(res)  # will print "42".

async def main2():
    # Schedule nested() to run soon concurrently
    # with "main()".
    task = asyncio.create_task(nested())

    # "task" can now be used to cancel "nested()", or
    # can simply be awaited to wait until it is complete:
    await task


def blocking_io():
    # 模拟一个耗时2秒的阻塞操作
    print(f"--> Start blocking_io at {time.strftime('%X')}")
    # 使用sync的time.sleep()会阻塞线程，导致Event Loop无法执行其他任务，这样来模拟文件IO操作
    time.sleep(2) 
    print(f"<-- End blocking_io at {time.strftime('%X')}")
    return "IO result"


async def main3():
    loop = asyncio.get_running_loop()

    future = loop.run_in_executor(None, blocking_io)
    result = await future
    print('Default thread pool result:', result)

    coroutine = nested()
    result2 = await coroutine
    print('Nested coroutine result:', result2)


async def main3_gather():
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, blocking_io)
    coroutine = nested()

    # this is also valid:
    results = await asyncio.gather(
        future,
        coroutine
    )
    print('Results:', results)


if __name__ == "__main__":
    print("main await coroutine:".center(80, '-'))
    asyncio.run(main())
    print("main2 await task:".center(80, '-'))
    asyncio.run(main2())
    print("main3 await future and coroutine:".center(80, '-'))
    asyncio.run(main3())
    print("main3_gather await future and coroutine:".center(80, '-'))
    asyncio.run(main3_gather())