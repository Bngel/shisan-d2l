import torch
from torch import nn
from d2l_20240331_func import dropout_layer, load_data_fashion_mnist, train_ch3

num_inputs, num_outputs, num_hidden_1, num_hidden_2 = 784, 10, 256, 256

dropout_1, dropout_2 = 0.3, 0.5


class Net(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_hidden_1, num_hidden_2, is_training=True, *args, **kwargs):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hidden_1)
        self.lin2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.lin3 = nn.Linear(num_hidden_2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
        if self.training:
            dropout_layer(H1, dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            dropout_layer(H2, dropout_2)
        out = self.relu(self.lin3(H2))
        return out


# net = Net(num_inputs, num_outputs, num_hidden_1, num_hidden_2, True)
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hidden_1),
    nn.ReLU(),
    nn.Dropout(dropout_1),
    nn.Linear(num_hidden_1, num_hidden_2),
    nn.ReLU(),
    nn.Dropout(dropout_2),
    nn.Linear(num_hidden_2, num_outputs)
)
num_epochs, lr, batch_size = 20, 0.03, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
