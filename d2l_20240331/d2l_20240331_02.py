import torch
from torch import nn
from d2l_20240331_func import load_data_fashion_mnist, train_ch3

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hidden = 784, 10, 256

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_inputs, num_hidden),
                    nn.ReLU(),
                    nn.Linear(256, 10))

batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
