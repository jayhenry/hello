import asyncio
import concurrent.futures
import functools
import time

def blocking_io():
    # 模拟一个耗时2秒的阻塞操作
    print(f"--> Start blocking_io at {time.strftime('%X')}")
    time.sleep(2) 
    print(f"<-- End blocking_io at {time.strftime('%X')}")
    return "IO result"

async def blocking_io_async():
    print(f"--> Start blocking_io_async at {time.strftime('%X')}")
    time.sleep(2)
    print(f"<-- End blocking_io_async at {time.strftime('%X')}")
    return "IO result"

def cpu_bound():
    # CPU 密集型操作
    print(f"--> Start cpu_bound at {time.strftime('%X')}")
    return sum(i * i for i in range(10 ** 7))

async def cpu_bound_async():
    print(f"--> Start cpu_bound_async at {time.strftime('%X')}")
    return sum(i * i for i in range(10 ** 7))

async def print_heartbeat():
    # 这是一个“心跳”协程，用来证明主循环活着
    while True:
        print(f"💓 Heartbeat: Loop is running at {time.strftime('%X')}")
        await asyncio.sleep(0.5)

async def main():
    loop = asyncio.get_running_loop()
    
    # 1. 启动心跳任务，让它在后台一直跑
    # 如果主循环被阻塞，这个心跳就会停跳
    heartbeat_task = asyncio.create_task(print_heartbeat())

    print("\n--- 1. Testing default thread pool ---")
    # 这里的 await 会挂起 main 协程，但不会阻塞 Event Loop
    # 所以 heartbeat_task 应该能继续打印
    result = await loop.run_in_executor(
        None, blocking_io)
    print('Default thread pool result:', result)

    print("\n--- 2. Testing custom thread pool ---")
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, blocking_io)
        print('Custom thread pool result:', result)

    print("\n--- 3. Testing custom process pool ---")
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, cpu_bound)
        print('Custom process pool result:', result)
    
    # 停止心跳任务，否则程序不会退出
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


async def main_without_pool():
    # 1. 启动心跳任务，让它在后台一直跑
    # 如果主循环被阻塞，这个心跳就会停跳
    heartbeat_task = asyncio.create_task(print_heartbeat())

    print("\n--- 1. Testing default thread pool ---")
    # 这里的 await 会挂起 main 协程，但不会阻塞 Event Loop
    # 所以 heartbeat_task 应该能继续打印
    result = await blocking_io_async()
    print('Default thread pool result:', result)

    print("\n--- 2. Testing custom thread pool ---")
    result = await blocking_io_async()
    print('Custom thread pool result:', result)

    print("\n--- 3. Testing custom process pool ---")
    result = await cpu_bound_async()
    print('Custom process pool result:', result)

    print("\n--- 4. async sleep for 3 seconds ---")
    # 不能使用 time.sleep(3) 因为它是sync阻塞的，会阻塞Event Loop
    await asyncio.sleep(3)  # 模拟阻塞操作, 让心跳任务打印
    
    # 停止心跳任务，否则程序不会退出
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


if __name__ == '__main__':
    # asyncio.run(main())  # compare with main_without_pool()
    asyncio.run(main_without_pool())
