from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-30B-Instruct")

print(model)
