# https://docs.ray.io/en/latest/ray-core/actors/async_api.html
import ray
import asyncio

ray.init()

@ray.remote
class AsyncActor:
    # multiple invocation of this method can be running in
    # the event loop at the same time
    async def run_concurrent(self):
        print("started")
        await asyncio.sleep(2) # concurrent workload here. For example, Network, I/O task here
        print("finished")
        return 10


def main_async_actor():
    actor = AsyncActor.remote()

    # regular ray.get
    # All 5 tasks should start at once. After 2 second they should all finish.
    # they should finish at the same time
    reg_get = ray.get([actor.run_concurrent.remote() for _ in range(4)])
    print("regular get:", reg_get)

    # async ray.get
    async def async_get():
        res = await actor.run_concurrent.remote()
        return res

    aget = asyncio.run(async_get())
    print("async get:", aget)


@ray.remote
def some_task():
    return 1


async def await_obj_ref():
    print(await some_task.remote())
    print(await asyncio.wait([some_task.remote()]))

def main_objref_as_future():
    # one of the ways to get a future object from an objref
    print(ray.get(some_task.remote()))
    print(ray.wait([some_task.remote()]))

    # another way to get a future object from an objref
    asyncio.run(await_obj_ref())

    # wrapped in concurrent.futures.Future
    print("wrapped in concurrent.futures.Future")
    import concurrent

    refs = [some_task.remote() for _ in range(4)]
    futs = [ref.future() for ref in refs]
    for fut in concurrent.futures.as_completed(futs):
        assert fut.done()
        print(fut.result())


def main_set_conc():
    actor = AsyncActor.options(max_concurrency=2).remote()
    # Only 2 tasks will be running concurrently. Once 2 finish, the next 2 should run.
    res = ray.get([actor.run_concurrent.remote() for _ in range(8)])
    print(res)

@ray.remote
class ThreadedActor:
    def task_1(self): print("I'm running in a thread!")
    def task_2(self): print("I'm running in another thread!")


def main_threaded_actor():
    # You can use the max_concurrency Actor options without any async methods, 
    # allowing you to achieve threaded concurrency (like a thread pool).
    # When there is at least one async def method in actor definition, 
    # Ray will recognize the actor as AsyncActor instead of ThreadedActor.
    a = ThreadedActor.options(max_concurrency=2).remote()

    # Each invocation of the threaded actor will be running in a thread pool. 
    # The size of the threadpool is limited by the max_concurrency value.
    ray.get([a.task_1.remote(), a.task_2.remote()])

if __name__ == "__main__":
    # main_async_actor()
    # main_objref_as_future()
    # main_set_conc()
    main_threaded_actor()
