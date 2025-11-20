import os
# export NCCL_SOCKET_IFNAME=xgbe1
# export NCCL_DEBUG=INFO
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # "xgbe1"
os.environ["NCCL_SOCKET_IFNAME"] = "xgbe1"  # run in 39,36 etc. A100
os.environ["NCCL_DEBUG"] = "INFO"

import asyncio
import torch
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

model_path = "/big_model/models/Qwen2.5/Qwen2.5-1.5B-Instruct"  #"facebook/opt-125m"
# 配置异步引擎参数
engine_args = AsyncEngineArgs(
    enforce_eager=True,
    model=model_path,
    tensor_parallel_size=2,
    pipeline_parallel_size=2,  # 流水线并行需配合异步接口
    trust_remote_code=True,    # 若使用自定义模型需开启
    max_model_len=256         # 根据模型调整最大序列长度
)

# 初始化异步引擎
llm = AsyncLLMEngine.from_engine_args(engine_args)

# 异步生成逻辑
async def generate(prompts):
    # 为每个prompt生成唯一ID（网页7）
    requests = [{"prompt": p, "request_id": f"req_{i}"} 
               for i, p in enumerate(prompts)]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # outputs = await llm.generate(prompts, sampling_params)
    # outputs = await asyncio.gather(*[
    #     llm.generate(request["prompt"], 
    #                 SamplingParams(temperature=0.8, top_p=0.95),
    #                 request["request_id"])
    #     for request in requests
    # ])

    example_input = requests[0]
    results_generator = llm.generate(example_input["prompt"], sampling_params,example_input["request_id"])
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # 关闭引擎释放资源
    # await llm.engine.shutdown()
    # 清理CUDA上下文（网页5）
    torch.cuda.empty_cache()  
    return [final_output]

# 执行异步函数（需在事件循环中运行）

def gen_demo():
    print("Gen Demo".center(50, "-"))
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    results = asyncio.run(generate(prompts))

    print("Output".center(50, "-"))
    print(results)

    print("Formated Output".center(50, "-"))
    for output in results:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    

if __name__ == "__main__":
    gen_demo()
