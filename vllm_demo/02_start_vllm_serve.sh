
NCCL_DEBUG=WARN vllm serve /big_model/models/Qwen2.5/Qwen2.5-Math-7B-Instruct/ \
 --tensor-parallel-size 4  --pipeline-parallel-size 4 --enforce-eager