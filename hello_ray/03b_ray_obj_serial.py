import ray
from ray import cloudpickle

FILE = "external_store.pickle"

ray.init()


def main_ser_deser():
    my_dict = {"hello": "world"}
    
    obj_ref = ray.put(my_dict)
    with open(FILE, "wb+") as f:
        cloudpickle.dump(obj_ref, f)
    
    # ObjectRef remains pinned in memory because
    # it was serialized with ray.cloudpickle.
    del obj_ref
    
    with open(FILE, "rb") as f:
        new_obj_ref = cloudpickle.load(f)
    
    # The deserialized ObjectRef works as expected.
    assert ray.get(new_obj_ref) == my_dict
    
    # Explicitly free the object.
    ray._private.internal_api.free(new_obj_ref)

@ray.remote
def f(arr):
    # arr = arr.copy()  # Adding a copy will fix the error.
    arr[0] = 1


def main_use_numpy_amsp():
    import numpy as np
    
    obj = [np.zeros(42)] * 99
    print(obj[0] is obj[1])
    l = ray.get(ray.put(obj))
    print(l[0] is l[1])  # no problem!
    print(obj[0] is l[0])  # False

    # np.array argument is read-only
    # To avoid this issue, you can manually copy the array at the destination if you need to mutate it 
    # (arr = arr.copy()). 
    # Note that this is effectively like disabling the zero-copy deserialization feature provided by Ray.
    try:
        ray.get(f.remote(np.zeros(100)))
    except ray.exceptions.RayTaskError as e:
        print(e)
    # ray.exceptions.RayTaskError(ValueError): ray::f()
    #   File "test.py", line 6, in f
    #     arr[0] = 1
    # ValueError: assignment destination is read-only


if __name__ == "__main__":
    # main_ser_deser()
    main_use_numpy_amsp()
