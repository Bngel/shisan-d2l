import torch
from torch import nn
from d2l_20240404_func import load_data_fashion_mnist,train_ch6, try_gpu

def vgg_block(num_conv, in_channels, out_channels):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        ))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blk = []
    in_channels = 1
    for (num_conv, out_channels) in conv_arch:
        conv_blk.append(vgg_block(
            num_conv, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blk, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )

net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
# loss 0.182, train acc 0.933, test acc 0.921
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())