# https://docs.python.org/3/library/asyncio-task.html#coroutine
import asyncio
import time

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main_baseline():
    # The following snippet of code will print “hello” after waiting for 1 second, and then print “world” after waiting for another 2 seconds:
    print(f"started at {time.strftime('%X')}")

    await say_after(1, 'hello')
    await say_after(2, 'world')

    print(f"finished at {time.strftime('%X')}")


async def main_with_task():
    # The asyncio.create_task() function to run coroutines concurrently as asyncio Tasks.
    task1 = asyncio.create_task(
        say_after(1, 'hello'))

    task2 = asyncio.create_task(
        say_after(2, 'world'))

    print(f"started at {time.strftime('%X')}")

    # Wait until both tasks are completed (should take
    # around 2 seconds.)
    await task1
    await task2

    print(f"finished at {time.strftime('%X')}")


async def main_with_task_group():
    # The asyncio.TaskGroup class provides a more modern alternative to create_task(). 
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(
            say_after(1, 'hello'))

        task2 = tg.create_task(
            say_after(2, 'world'))

        print(f"started at {time.strftime('%X')}")

    # The await is implicit when the context manager exits.
    # Wait until both tasks are completed (should also take
    # around 2 seconds.)

    print(f"finished at {time.strftime('%X')}")


if __name__ == "__main__":
    print("main_baseline(3s):".center(80, '-'))
    asyncio.run(main_baseline())
    print("main_with_task(2s):".center(80, '-'))
    asyncio.run(main_with_task())
    print("main_with_task_group(2s):".center(80, '-'))
    asyncio.run(main_with_task_group())

