from torch.distributed.device_mesh import init_device_mesh
mesh_3d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp"))
print(f"mesh_3d.mesh: {mesh_3d.mesh}")

# Users can slice child meshes from the parent mesh.
hsdp_mesh = mesh_3d["replicate", "shard"]
tp_mesh = mesh_3d["tp"]
print("tp_mesh", tp_mesh)
print("shard_mesh", mesh_3d["shard"])
print("replicate_mesh", mesh_3d["replicate"])
print(f"replicate_mesh info: global_rank {mesh_3d['replicate'].get_rank()}, world size {mesh_3d['replicate'].size()}, local_rank {mesh_3d['replicate'].get_local_rank()}")

# Users can access the underlying process group thru `get_group` API.
rep_shard_group = hsdp_mesh.get_all_groups()
# print("rep_shard_group", rep_shard_group)
# replicate_group = hsdp_mesh["replicate"].get_group()
# print("replicate_group", replicate_group)
# shard_group = hsdp_mesh["shard"].get_group()
# print("shard_group", shard_group)
tp_group = tp_mesh.get_group()
# print("tp_group", tp_group)