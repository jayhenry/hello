"""
# https://docs.ray.io/en/latest/ray-core/objects.html

In Ray, tasks and actors create and compute on objects. We refer to these objects as remote objects because they can be stored anywhere in a Ray cluster, and we use object refs to refer to them. Remote objects are cached in Ray’s distributed shared-memory object store, and there is one object store per node in the cluster. In the cluster setting, a remote object can live on one or many nodes, independent of who holds the object ref(s).

An object ref is essentially a pointer or a unique ID that can be used to refer to a remote object without seeing its value. If you’re familiar with futures, Ray object refs are conceptually similar.

Object refs can be created in two ways.

They are returned by remote function calls.

They are returned by ray.put().
"""

import ray
import numpy as np

ray.init()

# Define a task that sums the values in a matrix.
@ray.remote
def sum_matrix(matrix):
    return np.sum(matrix)

def main_task_arg():
    # Call the task with a literal argument value.
    print("literal argument:", ray.get(sum_matrix.remote(np.ones((100, 100)))))
    # -> 10000.0
    
    # Put a large array into the object store.
    matrix_ref = ray.put(np.ones((1000, 1000)))
    
    # Call the task with the object reference as an argument.
    print("object ref arg:", ray.get(sum_matrix.remote(matrix_ref)))
    # -> 1000000.0

import ray


@ray.remote
def echo(a: int, b: int, c: int):
    """This function prints its input values to stdout."""
    print(a, b, c)

@ray.remote
def echo_and_get(x_list):  # List[ObjectRef]
    """This function prints its input values to stdout."""
    print("args:", x_list)
    print("values:", ray.get(x_list))

def main_pass_object_arg():
    # method 1
    # Passing the literal values (1, 2, 3) to `echo`.
    ray.get(echo.remote(1, 2, 3))
    # -> prints "1 2 3"
    
    # Put the values (1, 2, 3) into Ray's object store.
    a, b, c = ray.put(1), ray.put(2), ray.put(3)
    
    # method 2
    # Passing an object as a top-level argument to `echo`. Ray will de-reference top-level
    # arguments, so `echo` will see the literal values (1, 2, 3) in this case as well.
    ray.get(echo.remote(a, b, c))
    # -> prints "1 2 3"

    # Passing an object as a nested argument to `echo_and_get`. Ray does not
    # de-reference nested args, so `echo_and_get` sees the references.
    ray.get(echo_and_get.remote([a, b, c]))
    # -> prints args: [ObjectRef(...), ObjectRef(...), ObjectRef(...)]
    #           values: [1, 2, 3]


# Put the values (1, 2, 3) into Ray's object store.
a, b, c = ray.put(1), ray.put(2), ray.put(3)


@ray.remote
def print_via_capture():
    """This function prints the values of (a, b, c) to stdout."""
    print(ray.get([a, b, c]))


def main_object_in_closure():
    # Passing object references via closure-capture. Inside the `print_via_capture`
    # function, the global object refs (a, b, c) can be retrieved and printed.
    print_via_capture.remote()
    # -> prints [1, 2, 3]

if __name__ == "__main__":
    # main_task_arg()
    # main_pass_object_arg()
    main_object_in_closure()
