import os
# export NCCL_SOCKET_IFNAME=xgbe1
# export NCCL_DEBUG=INFO
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # "xgbe1"
os.environ["NCCL_SOCKET_IFNAME"] = "xgbe1"  # run in 39 etc. A100
os.environ["NCCL_DEBUG"] = "INFO"

from vllm import LLM, SamplingParams

model_path = "/big_model/models/Qwen2.5/Qwen2.5-1.5B-Instruct"  #"facebook/opt-125m"
model_path = "/big_model/models/Qwen2.5/Qwen2.5-Math-7B-Instruct/"  #"facebook/opt-125m"
tp = 2
pp = 2
llm = LLM(model=model_path,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp
        )

# llm.apply_model(lambda model: print("Current model type is:", type(model)))
# If it is TransformersForCausalLM then it means it’s based on Transformers!
# ref: https://docs.vllm.ai/en/latest/models/supported_models.html#vllm
# If vLLM natively supports a model, its implementation can be found in vllm/model_executor/models.
# These models are what we list in List of Text-only Language Models and List of Multimodal Language Models.


def gen_demo():
    print("Gen Demo".center(50, "-"))
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def chat_demo():
    """
    https://docs.vllm.ai/en/latest/models/generative_models.html#llm-chat
    """
    print("Chat Demo".center(50, "-"))
    conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
    ]
    outputs = llm.chat(conversation)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    gen_demo()
    chat_demo()
