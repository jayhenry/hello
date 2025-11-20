"""
https://docs.ray.io/en/latest/ray-core/scheduling/index.html
"""
import ray

ray.init()

@ray.remote
def func():
    return 1


@ray.remote(num_cpus=1)
class Actor:
    pass


print("If unspecified, \"DEFAULT\" scheduling strategy is used.".center(80, '-'))
future = func.remote()
print("future: ", future)
print("result: ", ray.get(future))
actor = Actor.remote()
# print("actor: ", actor)

print("Explicitly set scheduling strategy to \"DEFAULT\".".center(80, '-'))
future = func.options(scheduling_strategy="DEFAULT").remote()
print("result: ", ray.get(future))
actor = Actor.options(scheduling_strategy="DEFAULT").remote()
# print("actor: ", actor)

print("Zero-CPU (and no other resources) actors are randomly assigned to nodes.".center(80, '-'))
actor = Actor.options(num_cpus=0).remote()
# print("actor: ", actor)

@ray.remote(scheduling_strategy="SPREAD")
def spread_func():
    return 2


print("Spread tasks across the cluster.".center(80, '-'))
@ray.remote(num_cpus=1)
class SpreadActor:
    pass


# Spread tasks across the cluster.
futures = [spread_func.remote() for _ in range(10)]
print("futures: ", futures)
print("results: ", ray.get(futures))
# Spread actors across the cluster.
actors = [SpreadActor.options(scheduling_strategy="SPREAD").remote() for _ in range(10)]

print("Complete scheduling strategy examples".center(80, '-'))