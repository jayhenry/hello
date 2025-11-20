"""
https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html
"""
from pprint import pprint
import time

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Initialize Ray.
import ray

# Create a single node Ray cluster with 2 CPUs and 2 GPUs.
ray.init(num_cpus=2, num_gpus=2)

@ray.remote(num_cpus=1)
class Actor:
    def __init__(self):
        pass

    def ready(self):
        pass


@ray.remote(num_cpus=0, num_gpus=1)
class Actor2:
    def __init__(self):
        pass

    def ready(self):
        pass


def main():
    print("## 1. Create a placement group.".center(80, '-'))
    
    # Reserve a placement group of 1 bundle that reserves 1 CPU and 1 GPU.
    pg = placement_group([{"CPU": 1, "GPU": 1}])
    
    # Wait until placement group is created.
    print("pg.ready:", ray.get(pg.ready(), timeout=10))
    
    # You can also use ray.wait.
    ready, unready = ray.wait([pg.ready()], timeout=10)
    print("ray.wait:", ready, unready)
    
    # You can look at placement group states using this API.
    pprint(placement_group_table(pg))
    
    # Cannot create this placement group because we
    # cannot create a {"GPU": 2} bundle.
    pending_pg = placement_group([{"CPU": 1}, {"GPU": 2}],
            # Reserve a placement group of 2 bundles
            # that have to be packed on the same node.
                                 strategy="PACK")
    
    # This raises the timeout exception!
    try:
        ray.get(pending_pg.ready(), timeout=5)
    except Exception as e:
        print(
            "Cannot create a placement group because "
            "{'GPU': 2} bundle cannot be created."
        )
        print(e)

    print("## 2. Schedule tasks using placement group.".center(80, '-'))
    print("Create an actor to a placement group.".center(80, '-'))
    actor = Actor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
        )
    ).remote()
    
    # Verify the actor is scheduled.
    print("actor.ready:", ray.get(actor.ready.remote(), timeout=10))

    print("Create a GPU actor on the first bundle of index 0.".center(80, '-'))
    actor2 = Actor2.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        )
    ).remote()
    
    # Verify that the GPU actor is scheduled.
    print("actor2.ready:", ray.get(actor2.ready.remote(), timeout=10))

    print("## 3. Remove a placement group.".center(80, '-'))
    # This API is asynchronous.
    remove_placement_group(pg)
    
    # Wait until placement group is killed.
    time.sleep(1)
    # Check that the placement group has died.
    pprint(placement_group_table(pg))


import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

def main_child_schedule():
    ray.init(num_cpus=2)
    
    # Create a placement group.
    pg = placement_group([{"CPU": 2}])
    ray.get(pg.ready())
    
    
    @ray.remote(num_cpus=1)
    def child():
        import time
    
        time.sleep(5)
    
    
    @ray.remote(num_cpus=1)
    def parent():
        # The child task is scheduled to the same placement group as its parent,
        # although it didn't specify the PlacementGroupSchedulingStrategy.
        ray.get(child.remote())
    
    
    # Since the child and parent use 1 CPU each, the placement group
    # bundle {"CPU": 2} is fully occupied.
    ray.get(
        parent.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True
            )
        ).remote()
    )


if __name__ == "__main__":
    main()
    # main_child_schedule()