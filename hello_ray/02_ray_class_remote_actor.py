# https://docs.ray.io/en/latest/ray-core/actors.html
import ray
ray.init() # Only call this once.

@ray.remote(num_cpus=2)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

# Methods called on different actors execute in parallel, and 
# methods called on the same actor execute serially in the order you call them. 
# Methods on the same actor share state with one another
counters = [Counter.remote() for i in range(4)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures)) # [1, 1, 1, 1]