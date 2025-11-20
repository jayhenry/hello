import asyncio



async def example_barrier():
   """ add in python 3.11
   """
   # 包含三部分的屏障
   b = asyncio.Barrier(3)

   # 新建 2 个等待任务
   asyncio.create_task(b.wait())
   asyncio.create_task(b.wait())

   await asyncio.sleep(0)
   print(b)

   # 第三个 .wait() 调用通过屏障
   await b.wait()
   print(b)
   print("barrier passed")

   await asyncio.sleep(0)
   print(b)


async def waiter(event):
    print('waiting for it ...')
    await event.wait()
    print('... got it!')

async def main():
    # 创建一个 Event 对象。
    event = asyncio.Event()

    # 产生一个任务等待直到 'event' 被设置。
    waiter_task = asyncio.create_task(waiter(event))

    # 休眠 1 秒钟并设置事件。
    await asyncio.sleep(1)
    event.set()

    # 等待直到 waiter 任务完成。
    await waiter_task


if __name__ == "__main__":
    print("example_barrier:".center(80, '-'))
    asyncio.run(example_barrier())
    print("event example:".center(80, '-'))
    asyncio.run(main())
