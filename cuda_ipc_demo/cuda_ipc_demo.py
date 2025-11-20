import sys
import torch
import os
import pickle
import time

def sender(output_file, device_id):
    # 创建一个 CUDA 张量
    tensor = torch.randn(3, 3).cuda(device_id)
    print("tensor:", tensor)
    # 打印tensor的device id
    print("tensor device id:", tensor.device.index)

    
    # 确保张量是叶节点且不需要梯度
    if tensor.requires_grad and not tensor.is_leaf:
        tensor = tensor.detach()
    
    # 使用 reduce_tensor 序列化张量
    reduced = torch.multiprocessing.reductions.reduce_tensor(tensor)
    
    # 提取 IPC 相关信息
    func, args = reduced
    # ipc_handles = {torch.cuda.current_device(): (func, args)}
    ipc_handles = {device_id: (func, args)}
    print("ipc handles:", ipc_handles)
    print("current device index:", torch.cuda.current_device())
    
    # 准备要保存的数据
    data = {
        'name': 'example_tensor',
        'dtype': tensor.dtype,
        'shape': tensor.shape,
        'ipc_handles': ipc_handles
    }
    
    # 将数据保存到文件
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Sender: Tensor sent and saved to {output_file}")
    time.sleep(60000)


def get_physical_gpu_id():
    """获取物理 GPU ID"""
    return torch.cuda.current_device()

def receiver(input_file, device_id=None):
    # 从文件加载 IPC 句柄信息
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    name = data['name']
    dtype = data['dtype']
    shape = data['shape']
    ipc_handles = data['ipc_handles']
    
    # 获取当前设备 ID
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    # 使用 IPC 句柄重建张量
    # handle = ipc_handles[get_physical_gpu_id()]
    handle = ipc_handles[device_id]
    func, args = handle
    list_args = list(args)
    
    # 关键步骤：修改设备 ID 为当前设备的 ID
    # 处理不同进程可能有不同的 CUDA_VISIBLE_DEVICES 设置的情况
    list_args[6] = device_id
    
    # 重建张量
    tensor = func(*list_args)
    
    # 确保张量符合预期
    assert tensor.dtype == dtype, f"dtype mismatch: {tensor.dtype} vs {dtype}"
    assert tensor.shape == shape, f"shape mismatch: {tensor.shape} vs {shape}"
    
    print(f"Receiver: Tensor '{name}' received on device {device_id}")
    print(f"Type: {tensor.dtype}, Shape: {tensor.shape}")
    print(tensor)
    
    # 可选：同步 CUDA 操作
    torch.cuda.synchronize()
    
    # 清理文件
    os.remove(input_file)
    
    return tensor


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(f'Usage: python {sys.argv[0]} send|receive device_id')
    
    role = sys.argv[1]
    assert role in ['send', 'receive']
    device_id = int(sys.argv[2])

    output_file = input_file = 'tensor_ipc.pkl'
    if role == 'send':
        if os.path.exists(output_file):
            os.remove(output_file)
    
        sender(output_file, device_id)
    else:
        while not os.path.exists(input_file):
            pass  # 等待文件出现
    
        receiver(input_file, device_id)
        