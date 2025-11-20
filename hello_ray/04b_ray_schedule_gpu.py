import os
# Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.
os.environ["RAY_DEDUP_LOGS"] = "0"

import ray


@ray.remote(num_gpus=1)
class GPUActor:
    def ping(self):
        print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

@ray.remote(num_gpus=1)
def gpu_task():
    print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

def main_gpu():
    gpu_actor = GPUActor.remote()
    print(f"gpu_actor: {gpu_actor}".center(80, '-'))
    ray.get(gpu_actor.ping.remote())
    print("gpu_task:".center(80, '-'))
    # The actor uses the first GPU so the task uses the second one.
    ray.get(gpu_task.remote())


@ray.remote(num_gpus=0.5)
class FractionalGPUActor:
    def ping(self):
        gpu_id = ray.get_runtime_context().get_accelerator_ids()["GPU"]
        print(f"GPU id: {gpu_id}".center(20, '*'))
        return gpu_id

def main_frac():
    print("fractional actors:")
    actor_num = 5
    fractional_gpu_actors = [FractionalGPUActor.remote() for _ in range(actor_num)]
    # Ray tries to pack GPUs if possible.
    print([ray.get(fractional_gpu_actors[i].ping.remote()) for i in range(actor_num)])


# By default, ray does not reuse workers for GPU tasks to prevent
# GPU resource leakage.
@ray.remote(num_gpus=1)
def leak_gpus():
    import tensorflow as tf

    # This task allocates memory on the GPU and then never release it.
    tf.Session()

if __name__ == "__main__":
    ray.init(num_gpus=3)
    # main_gpu()
    main_frac()