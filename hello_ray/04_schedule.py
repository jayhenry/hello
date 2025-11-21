"""
https://docs.ray.io/en/latest/ray-core/scheduling/index.html
"""
import ray

ray.init()

# https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#resource-requirements
# By default, Ray tasks use 1 logical CPU resource
@ray.remote
def func():
    return 1


# By default, Ray actors use 1 logical CPU for scheduling, and 0 logical CPU for running. 
# (This means, by default, actors cannot get scheduled on a zero-cpu node, but an infinite number of 
# them can run on any non-zero cpu node. The default resource requirements for actors was chosen for historical reasons. 
# It’s recommended to always explicitly set num_cpus for actors to avoid any surprises. 
# If resources are specified explicitly, they are required both at schedule time and at execution time.)
@ray.remote(num_cpus=1)
class Actor:
    pass


print("If unspecified, \"DEFAULT\" scheduling strategy is used.".center(80, '-'))
future = func.remote()
print("future: ", future)
print("result: ", ray.get(future))
actor = Actor.remote()
ray.get(actor.__ray_ready__.remote())  # Wait for the actor to be ready
# print("actor: ", actor)

print("Explicitly set scheduling strategy to \"DEFAULT\".".center(80, '-'))
future = func.options(scheduling_strategy="DEFAULT").remote()
print("result: ", ray.get(future))
actor = Actor.options(scheduling_strategy="DEFAULT").remote()
ray.get(actor.__ray_ready__.remote())  # Wait for the actor to be ready
# print("actor: ", actor)

print("Zero-CPU (and no other resources) actors are randomly assigned to nodes.".center(80, '-'))
actor = Actor.options(num_cpus=0).remote()
ray.get(actor.__ray_ready__.remote())  # Wait for the actor to be ready
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
ray.get([actor.__ray_ready__.remote() for actor in actors])  # Wait for the actor to be ready

print("Complete scheduling strategy examples".center(80, '-'))