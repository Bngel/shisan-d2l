import numpy
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

loss = nn.CrossEntropyLoss()
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 176))

net.load_state_dict(torch.load('./model/net.pth'))
img_data = pd.read_csv('./data/train.csv')
label_arr = np.array(img_data.iloc[:, 1])
label_dict = { value: key for (value, key) in enumerate(set(label_arr)) }

test_frame = pd.read_csv('./data/test.csv')
test_data = [
    './data/' + path for path in np.array(test_frame.iloc[:, 0])
]
output = []
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

net.eval()
for i in range(0, len(test_data), 256):
    X = [transform(Image.open(x)) for x in test_data[i:min(len(test_data), i+256)]]
    X = torch.stack(X)
    output_item = torch.argmax(F.softmax(net(X), dim=0), dim=1)
    tensor = [label_dict[key.item()] for key in output_item]
    output += tensor

output_frame = pd.Series(output, name='label')
test_frame = pd.concat([test_frame, output_frame], axis=1)
test_frame.to_csv('./data/answer.csv', index=False)