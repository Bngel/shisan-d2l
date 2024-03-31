import torch
from torch import nn
from d2l_20240331_func import synthetic_data, load_array, linreg, squared_loss, sgd

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
test_data = synthetic_data(true_w, true_b, n_test)
train_iter = load_array(train_data, batch_size)
test_iter = load_array(test_data, batch_size, is_train=False)


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    # sigma(w²) / 2
    return torch.sum(w.pow(2)) / 2


def train(lamb):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # with torch.enable_grad():
            l = loss(net(X), y) + lamb * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
    print(f"w的L2范数是: {torch.norm(w).item()}")


train(3)

