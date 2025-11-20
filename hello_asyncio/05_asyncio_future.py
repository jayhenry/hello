import asyncio
import functools

async def set_after(fut, delay, value):
    # 休眠 *delay* 秒。
    await asyncio.sleep(delay)

    # 将 *value* 设为 Future 对象 *fut* 的结果。
    if not fut.cancelled():
        fut.set_result(value)

async def main():
    # 获取当前事件循环。
    loop = asyncio.get_running_loop()

    # 新建一个 Future 对象。

    fut = loop.create_future()

    # 当 "fut" 已完成时则调用 'print("Future:", fut)'。
    fut.add_done_callback(
        functools.partial(print, "Future:"))

    # 在一个并行的任务中运行 "set_after()" 协程。
    # 在这里我们使用低层级的 "loop.create_task()" API
    # 因为我们手头已经拥有一个对事件循环的引用。
    # 在其他情况下我们可以使用 "asyncio.create_task()"。
    loop.create_task(
        set_after(fut, 1, '... world'))

    print('hello ...')

    # 等待直到 *fut* 得出结果 (1 秒) 并打印它。
    print(await fut)

asyncio.run(main())
