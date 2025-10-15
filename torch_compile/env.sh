# ref: https://docs.pytorch.org/docs/stable/logging.html

# export TORCH_LOGS="bytecode,aot_graphs,aot_joint_graph,graph,graph_code,recompiles,output_code,schedule"
export TORCH_LOGS="graph,graph_code,recompiles,output_code,schedule"
# export TORCH_LOGS="graph,graph_code,guards,recompiles,output_code,schedule"
# export TORCH_LOGS="+dynamo,schedule"
# unset TORCH_LOGS