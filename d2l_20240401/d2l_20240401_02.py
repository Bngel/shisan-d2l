import torch
from torch import nn
from d2l_20240401_func import try_gpu

if torch.cuda.is_available():
    torch.cuda.device('cuda')

print(torch.device)
print(torch.cuda.device_count())

x = torch.tensor([1, 2, 3])
print(x.device)

X = torch.ones(2, 3, device=try_gpu())
print(X)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

net(X)