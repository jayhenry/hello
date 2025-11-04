import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR

input_dim = 2

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 4) 
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x): 
        return self.layer2(self.layer1(x))


model = LinearModel()

# learning_rates = { 
#     'layer1.weight': 0.01,
#     'layer1.bias': 0.1,
#     'layer2.weight': 0.001,
#     'layer2.bias': 1.0}

# Build param_group where each group consists of a single parameter.
# `param_group_names` is created so we can keep track of which param_group
# corresponds to which parameter.
# param_groups = []
# param_group_names = []
params = []
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        params.append(parameter)
        # param_groups.append({'params': [parameter], 'lr': learning_rates[name]})
        # param_group_names.append(name)

lr = 1e-2
# optimizer requires default learning rate even if its overridden by all param groups
# optimizer = optim.AdamW(param_groups, lr=lr)
optimizer = optim.AdamW(params, lr=lr)
# print(f"When initialized, optimizer.param_groups: {optimizer.param_groups}")
print(f"When initialized, optimizer.: {optimizer.state_dict()}")

total_steps = 10
# scheduler's iter(=epoch) starts from 0, and ends at total_iters. so scheduler iters belongs to [0, total_iters]
scheduler = LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=total_steps,
    last_epoch=-1,
)

# print(f"After scheduler initialization, optimizer.param_groups: {optimizer.param_groups}")
print(f"After scheduler initialization, optimizer.state_dict(): {optimizer.state_dict()}")

for i in range(total_steps + 1):
    output = model(torch.zeros(1, input_dim))
    loss = output.sum()
    loss.backward()

    optimizer.step()
    param_groups = optimizer.param_groups
    if i == 0:
        # print(f"after step {i}, optimizer.param_groups: {param_groups}")
        print(f"after step {i}, optimizer.state_dict(): {optimizer.state_dict()}")

    cur_lr = scheduler.get_last_lr()[0]
    print(f'step {i} learning rate: {cur_lr}')
    scheduler.step()
