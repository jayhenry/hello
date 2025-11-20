import os

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import torch

device = torch.device('cuda')

model_dir = os.environ["LLAMA_MODEL_DIR"]

tokenizer = AutoTokenizer.from_pretrained(model_dir)
# tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaForCausalLM.from_pretrained(model_dir).to(device)

prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

prompts = [
    """请扮演医生""",

]
# prompts = "Hey, are you conscious? Can you talk to me?"
tokenizer.pad_token = tokenizer.eos_token


def generate(prompts):
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        # [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    
        generate_ids = model.generate(inputs.input_ids, max_length=1000, temperature=0.6)
    
        res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
        print(f"问题：\n{p} \n\n 答案:\n{res}\n")
# for prompt, result in zip(prompts, res):
#         print(prompt)
#         print(f"> {result['generation']}")
#         print("\n==================================\n")

generate(prompts)


import pdb; pdb.set_trace()

