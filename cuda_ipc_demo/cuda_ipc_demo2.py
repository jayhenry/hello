# 发送端
import torch
import pickle
import os
import torch.multiprocessing
import torch.multiprocessing as mp
from torch.multiprocessing import Lock
import ctypes
from ctypes import c_void_p


def sender(lock):
    import time
    tensor = torch.ones(3, 3).cuda()
    tensor[1, 1] = 5.0  # 修改一个值
    
    reduced = torch.multiprocessing.reductions.reduce_tensor(tensor)
    with lock:
        with open('ipc_handle.pkl', 'wb') as f:
            pickle.dump(reduced, f)
    
    sender_ptr = c_void_p(tensor.data_ptr())
    print("sender_ptr", sender_ptr)
    # print(f"[Sender] Tensor ptr: {tensor.data_ptr()}, cptr: {ctypes.cast(sender_ptr, ctypes.py_object).value}, value at [1,1]: {tensor[1,1].item()}")
    print(f"[Sender] Tensor ptr: {tensor.storage().data_ptr()}, value at [1,1]: {tensor[1,1].item()}")
    torch.cuda.synchronize()
    time.sleep(10)
    print(f"[Sender] Tensor ptr: {tensor.data_ptr()}, value at [1,1]: {tensor[1,1].item()}")
    return tensor

# 接收端
def receiver(lock):
    import time
    while not os.path.exists('ipc_handle.pkl'):
        time.sleep(1)
    with lock:
        with open('ipc_handle.pkl', 'rb') as f:
            reduced = pickle.load(f)
        # 清理文件
        os.remove('ipc_handle.pkl')
    
    func, args = reduced
    tensor = func(*args)
    
    torch.cuda.synchronize()
    # 获取原始CUDA指针
    sender_ptr = c_void_p(tensor.data_ptr())
    print("receiver_ptr", sender_ptr)
    # print(f"[Receiver] Tensor ptr: {tensor.data_ptr()}, cptr: {ctypes.cast(sender_ptr, ctypes.py_object).value}, value at [1,1]: {tensor[1,1].item()}")
    print(f"[Receiver] Tensor ptr: {tensor.storage().data_ptr()}, value at [1,1]: {tensor[1,1].item()}")
    
    # 修改接收端数据
    tensor[1, 1] = 10.0
    print(f"[Receiver] Modified value: {tensor[1,1].item()}")

# 实际使用时分开运行
if __name__ == "__main__":
    mp.set_start_method("spawn")
    lock = Lock()
    # 运行发送端
    p1 = mp.Process(target=sender, args=(lock,))
    p1.start()
    # 运行接收端
    import time
    time.sleep(3)
    # 在子进程中运行接收端
    p = mp.Process(target=receiver, args=(lock,))
    p.start()
    p.join()
    # with mp.get_context("spawn").Process(target=receiver) as p:
        # p.start()
    # receiver()
    p1.join()