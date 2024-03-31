import torch
from torch import nn
from d2l_20240330_func import train_ch3, load_data_fashion_mnist

batch_size = 256
num_epochs = 10
train_iter, test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)