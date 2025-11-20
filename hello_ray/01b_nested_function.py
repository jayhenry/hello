import ray
ray.init()

@ray.remote
def f():
    return 1


@ray.remote
def g():
    # Call f 4 times and return the resulting object refs.
    return [f.remote() for _ in range(4)]


@ray.remote
def h():
    # Call f 4 times, block until those 4 tasks finish,
    # retrieve the results, and return the values.
    return ray.get([f.remote() for _ in range(4)])

print("g()")
objref_list = ray.get(g.remote())
print(objref_list)
# print([ray.get(obj_ref) for obj_ref in objref_list])
print(ray.get(objref_list))

print("h()")
ret_ref = h.remote()
print(ret_ref)
print(ray.get(ret_ref))
