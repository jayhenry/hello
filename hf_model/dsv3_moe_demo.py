# from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

model = DeepseekV3ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3")

print(model)
