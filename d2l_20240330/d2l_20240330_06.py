import torch
from torchvision import datasets
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time

# Fashion-MNIST

trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

print(len(mnist_train), len(mnist_test))


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


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
plt.show()

batch_size = 256


def get_dataloader_workers():
    # 使用4个worker（进程）
    return 4

