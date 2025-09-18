import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T

import torch.profiler
from torchvision import models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.cuda()
cudnn.benchmark = True

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
# trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        # download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0")
model.train()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_resnet50_v2', worker_name='worker0'),
    record_shapes=True,
    # profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True,
    # experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as p:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 >= 4:
            break
        p.step()


sort_key = "self_cuda_time_total"
# sort_key = "self_cuda_memory_usage"
print(p.key_averages().table(sort_by=sort_key, row_limit=10))
print("这里是分割线--------------------------------")
print(p.key_averages(group_by_stack_n=5).table(sort_by=sort_key, row_limit=10))
