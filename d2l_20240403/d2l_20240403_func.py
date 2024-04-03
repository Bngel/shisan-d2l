import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils import data
from torchvision import transforms
import torch
import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(url, folder='./data', sha1_hash=None):
    if not url.startswith('http'):
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    f_name = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(f_name) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(f_name, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return f_name
    # Download
    print(f'Downloading {f_name} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(f_name, 'wb') as f:
        f.write(r.content)
    return f_name
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if len(y_hat.shape) > 0 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    # 评估维度 3
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            # 总损失值, 预测正确个数, 总标签个数
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y))
    # 总损失值 / 总标签个数 = 平均损失值
    # 预测正确个数 / 总标签个数 = 预测正确率
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        print(f'epoch:{epoch + 1} test_acc:{test_acc:f}, train_loss:{train_loss:f}, train_acc:{train_acc:f}')


def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
        'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    fig_size = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
            ax.axis('off')
            ax.set_title(titles[i])
        else:
            ax.imshow(img)


def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    true_label = get_fashion_mnist_labels(y)
    predict_label = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [pre for true, pre in zip(true_label, predict_label)]
    show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])
    plt.show()


def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.normal_(m.weight, mean=0, std=0.01)
        torch.nn.zeros_(m.bias)

def init_constant(m):
    if type(m) == torch.nn.Linear:
        torch.nn.constant_(m.weight, 1)
        torch.nn.zeros_(m.bias)

def init_xavier(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.constant_(m.weight, 42)

def try_gpu(i = 0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device['cpu']]

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j: j + w] * K).sum()
    return Y

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j: j + p_w].mean()
    return Y