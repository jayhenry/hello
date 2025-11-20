import ray

@ray.remote
def node_affinity_func():
    return ray.get_runtime_context().get_node_id()


@ray.remote(num_cpus=1)
class NodeAffinityActor:
    pass


def main_node_affinity():
    # Only run the task on the local node.
    node_affinity_func.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
    ).remote()
    
    # Run the two node_affinity_func tasks on the same node if possible.
    node_affinity_func.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=ray.get(node_affinity_func.remote()),
            soft=True,
        )
    ).remote()
    
    # Only run the actor on the local node.
    actor = NodeAffinityActor.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
    ).remote()

@ray.remote
def large_object_func():
    # Large object is stored in the local object store
    # and available in the distributed memory,
    # instead of returning inline directly to the caller.
    return [1] * (1024 * 1024)


@ray.remote
def small_object_func():
    # Small object is returned inline directly to the caller,
    # instead of storing in the distributed memory.
    return [1]


@ray.remote
def consume_func(data):
    return len(data)


def main_locality():
    large_object = large_object_func.remote()
    small_object = small_object_func.remote()
    print("large object", large_object)
    print("small object", small_object)
    
    # Ray will try to run consume_func on the same node
    # where large_object_func runs.
    print(ray.get(consume_func.remote(large_object)))
    
    # Ray will try to spread consume_func across the entire cluster
    # instead of only running on the node where large_object_func runs.
    print(ray.get([
        consume_func.options(scheduling_strategy="SPREAD").remote(large_object)
        for i in range(10)
    ]))
    
    # Ray won't consider locality for scheduling consume_func
    # since the argument is small and will be sent to the worker node inline directly.
    print(ray.get(consume_func.remote(small_object)))
    



if __name__ == "__main__":
    ray.init()
    # main_node_affinity()
    main_locality()